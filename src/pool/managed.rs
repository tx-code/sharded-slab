use super::{OwnedLease, Pool};
use crate::{
    cfg,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    },
    Clear, DefaultConfig, Pack,
};
use std::{
    fmt, mem, ops,
    sync::{Arc, Weak},
};

/// A clearable value that reports the dynamic payload it retains while idle.
///
/// The report is taken **after** [`Clear::clear`]. Include all dynamic storage
/// still owned by the value; a conservative overestimate is permitted. The
/// report must stay conservative while idle; external aliases must not grow
/// cached storage behind that report.
///
/// The cache bounds the sum of these reports, not allocator overhead, inline value
/// storage, slab pages, other owners' data, or process RSS. This is a safe
/// accounting contract, not a memory-safety assertion.
pub trait Recyclable: Clear + Default {
    /// Reports retained dynamic payload bytes, without overflowing the sum.
    /// Called outside cache locks. A panic propagates and prevents caching.
    fn retained_bytes(&self) -> usize;
}

impl<T> Recyclable for Vec<T> {
    fn retained_bytes(&self) -> usize {
        self.capacity().saturating_mul(mem::size_of::<T>())
    }
}
impl Recyclable for String {
    fn retained_bytes(&self) -> usize {
        self.capacity()
    }
}

/// Fixed limits for a managed pool's idle payload cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheLimits {
    /// Maximum sum of idle values' reported payload bytes.
    pub max_cached_bytes: usize,
    /// Maximum idle values; zero disables caching, including zero-byte values.
    pub max_cached_values: usize,
    /// Maximum reported payload of one value admitted to the cache.
    pub max_value_bytes: usize,
}
impl CacheLimits {
    /// Sets a total byte/count limit and allows one value up to the byte limit.
    pub const fn new(max_cached_bytes: usize, max_cached_values: usize) -> Self {
        Self {
            max_cached_bytes,
            max_cached_values,
            max_value_bytes: max_cached_bytes,
        }
    }
}

/// Diagnostics for a managed pool. No field is an active-memory hard limit.
///
/// Idle fields are read with every return shard locked. Active and returning
/// records change independently; this is not a consistent whole-heap snapshot.
/// Hit/miss counters count admitted requests and saturate at `usize::MAX`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheStats {
    /// Whether admission has been closed.
    pub closed: bool,
    /// Values currently stored idle, excluding active and returning values.
    pub cached_values: usize,
    /// Sum of cached payload reports, bounded by `max_cached_bytes`.
    pub cached_bytes: usize,
    /// Capacity bytes of the cache's own value/metadata vector, outside the budget.
    pub cache_storage_bytes: usize,
    /// Admitted values still in use, initializing, or being cleared.
    /// Shared clones together count as one lease, until last-reader clearing.
    pub active_leases: usize,
    /// Admitted requests that obtained an idle value.
    pub cache_hits: usize,
    /// Admitted misses, including backing-capacity rejection or panicking Default.
    pub cache_misses: usize,
    /// Credits reserved across idle shards, including temporarily unused credit.
    pub reserved_cache_bytes: usize,
    /// Reserved idle value credits across shards; bounded by the count limit.
    pub reserved_cache_values: usize,
    /// Values undergoing return or detached trim/close disposal. These diagnostic
    /// counts can overlap `active_leases` while clear/report is still running.
    pub returning_values: usize,
    /// Saturating sum of completed reports for observed returns not yet disposed
    /// or stored idle. Unreported returns are explicitly counted separately.
    pub reported_returning_bytes: usize,
    /// Observed returns whose clear/report is incomplete or intentionally skipped
    /// after closure/disabled retention. Their byte size is not reported as zero.
    pub unreported_returning_values: usize,
    /// Known reclaim metadata storage excluding allocator/Arc headers, inline
    /// backing entries and slab pages. Not included in the idle payload budget.
    pub reclaim_storage_bytes: usize,
}

/// Idle values removed by trim/close, not bytes proven returned to the OS.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Trimmed {
    /// Number of values removed from idle ownership.
    pub values: usize,
    /// Their total reported payload bytes.
    pub bytes: usize,
}

/// A managed checkout rejection, separate from application or payload errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedAcquireError {
    /// The cache controller has closed admission.
    Closed,
    /// The backing pool could not provide a slot under its configuration.
    Capacity,
}
impl fmt::Display for ManagedAcquireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Closed => "pool admission is closed",
            Self::Capacity => "pool slot capacity is exhausted",
        })
    }
}
impl std::error::Error for ManagedAcquireError {}

/// Bounded idle ownership on the existing sharded pool and exclusive leases.
///
/// Idle entries retain a reserved backing lease. Warm reuse does not allocate a
/// new slot or clone the backing pool. Internal return shards borrow a single
/// controller's byte/count credits; credits never exceed the configured total.
/// Returns may discard on contention. Neither active memory nor slab pages are
/// covered by the idle payload budget.
///
/// Share the controller with `Arc`. Its last owner closes and drains idle
/// ownership, without invalidating active or shared readers. Payload callbacks
/// run outside locks. Their panics propagate; detached payloads and admission
/// permits are unwound before the empty backing slot is reclaimed.
///
/// ```
/// use sharded_slab::pool::{CacheLimits, ManagedAcquireError, ManagedPool};
/// let pool = ManagedPool::<Vec<u64>>::new(CacheLimits::new(4096, 4));
/// let mut block = pool.try_lease_owned().unwrap();
/// block.extend_from_slice(&[1, 2, 3]);
/// let reader = block.freeze();
/// pool.close();
/// assert_eq!(pool.try_lease_owned().err(), Some(ManagedAcquireError::Closed));
/// assert_eq!(&**reader, &[1, 2, 3]);
/// drop(reader);
/// assert_eq!(pool.stats().active_leases, 0);
/// assert_eq!(pool.stats().cached_bytes, 0);
/// ```
pub struct ManagedPool<T: Recyclable, C: cfg::Config = DefaultConfig> {
    slots: Arc<Pool<Entry<T, C>, C>>,
    caches: Vec<Arc<Cache<T, C>>>,
    policy: Arc<Policy<T, C>>,
}

/// Exclusive ownership. Drop clears, then caches the reserved lease or disposes it.
#[must_use]
pub struct ManagedLease<T: Recyclable, C: cfg::Config = DefaultConfig> {
    inner: Option<ReservedLease<T, C>>,
}

// 大的底层guard只在首次reserve时装箱，热借还只移动这个拥有句柄。
type ReservedLease<T, C> = Box<OwnedLease<Entry<T, C>, C>>;

/// Read-only shared ownership; only the last reader clears and returns the value.
#[must_use]
pub struct ManagedSharedLease<T: Recyclable, C: cfg::Config = DefaultConfig> {
    inner: crate::sync::Arc<ManagedLease<T, C>>,
}

const RECORD_IDLE: usize = 0;
const RECORD_ACTIVE: usize = 1;
const RECORD_CLEARING: usize = 2;
const RECORD_REPORTED: usize = 3;
const RECORD_DEAD: usize = 4;

struct ReturnRecord {
    phase: AtomicUsize,
    bytes: AtomicUsize,
}

struct Idle<T: Recyclable, C: cfg::Config> {
    lease: ReservedLease<T, C>,
    bytes: usize,
}
struct State<T: Recyclable, C: cfg::Config> {
    idle: Vec<Idle<T, C>>,
    bytes: usize,
    credit_bytes: usize,
    credit_values: usize,
    hits: usize,
    misses: usize,
}
struct Credits<T: Recyclable, C: cfg::Config> {
    bytes: usize,
    values: usize,
    donors: Vec<Weak<Cache<T, C>>>,
}
struct Policy<T: Recyclable, C: cfg::Config> {
    limits: CacheLimits,
    closed: AtomicBool,
    operations: Mutex<()>,
    credits: Mutex<Credits<T, C>>,
    records: Mutex<Vec<Weak<ReturnRecord>>>,
}

// 热借还只写所属分片；避免无关分片的计数/锁共享缓存行。
#[repr(align(128))]
struct Cache<T: Recyclable, C: cfg::Config> {
    index: usize,
    policy: Arc<Policy<T, C>>,
    state: Mutex<State<T, C>>,
    initializing: AtomicUsize,
}
struct Permit<T: Recyclable, C: cfg::Config> {
    cache: Weak<Cache<T, C>>,
    index: usize,
    active: bool,
    initializing: bool,
}
struct Entry<T: Recyclable, C: cfg::Config> {
    value: Option<T>,
    permit: Option<Permit<T, C>>,
    record: Option<Arc<ReturnRecord>>,
}

impl<T: Recyclable, C: cfg::Config> Default for Entry<T, C> {
    fn default() -> Self {
        Self {
            value: None,
            permit: None,
            record: None,
        }
    }
}

struct RetiringRecord(Option<Arc<ReturnRecord>>);
impl Drop for RetiringRecord {
    fn drop(&mut self) {
        if let Some(ref record) = self.0 {
            record.phase.store(RECORD_DEAD, Ordering::Release);
        }
    }
}

impl<T: Recyclable, C: cfg::Config> Clear for Entry<T, C> {
    fn clear(&mut self) {
        // 值进入idle前已经Clear；底层槽释放只析构，不再次调用用户Clear。
        // 回调在ManagedLease中先移出值；异常路径的空槽也可安全归还。
        let permit = self.permit.take();
        let mut record = RetiringRecord(self.record.take());
        drop(self.value.take());
        drop(permit);
        if let Some(record) = record.0.take() {
            record.phase.store(RECORD_DEAD, Ordering::Release);
            self.record = Some(record);
        }
    }
}

impl<T: Recyclable, C: cfg::Config> Permit<T, C> {
    #[inline(always)]
    fn release(&mut self, cache: &Cache<T, C>) {
        if self.active {
            self.active = false;
            if self.initializing {
                self.initializing = false;
                cache.initializing.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}
impl<T: Recyclable, C: cfg::Config> Drop for Permit<T, C> {
    fn drop(&mut self) {
        if self.active && self.initializing {
            if let Some(cache) = self.cache.upgrade() {
                self.release(&cache);
            }
        }
    }
}

impl<T: Recyclable, C: cfg::Config> State<T, C> {
    #[inline(always)]
    fn store_idle(&mut self, idle: Idle<T, C>, total: usize) {
        if let Some(ref record) = idle.lease.record {
            record.phase.store(RECORD_IDLE, Ordering::Release);
        }
        debug_assert!(self.idle.len() < self.idle.capacity());
        self.idle.push(idle);
        self.bytes = total;
    }
}

impl<T: Recyclable, C: cfg::Config> Policy<T, C> {
    // 调用方持operation，目录不含载荷，预留的8个句柄足以覆盖所有分片。
    fn cache_handles(&self, handles: &mut Vec<Arc<Cache<T, C>>>) {
        let credits = self.credits.lock().unwrap_or_else(|e| e.into_inner());
        handles.extend(credits.donors.iter().filter_map(Weak::upgrade));
    }
}

impl<T: Recyclable, C: cfg::Config> Cache<T, C> {
    #[inline(always)]
    fn checkout(&self) -> Result<Option<ReservedLease<T, C>>, ManagedAcquireError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // close也持有此锁；读取closed是这个请求的准入点。
        if self.policy.closed.load(Ordering::Acquire) {
            return Err(ManagedAcquireError::Closed);
        }
        if let Some(idle) = state.idle.pop() {
            state.bytes -= idle.bytes;
            state.hits = state.hits.saturating_add(1);
            if let Some(ref record) = idle.lease.record {
                record.phase.store(RECORD_ACTIVE, Ordering::Release);
            }
            return Ok(Some(idle.lease));
        }
        self.initializing.fetch_add(1, Ordering::Relaxed);
        drop(state);
        Ok(self.borrow_idle())
    }

    #[cold]
    #[inline(never)]
    fn borrow_idle(&self) -> Option<ReservedLease<T, C>> {
        // 冷路径释放本分片锁后才取全局operation；所有多锁都按同一升序。
        let mut caches = Vec::with_capacity(8);
        let mut states = Vec::with_capacity(8);
        let operation = self
            .policy
            .operations
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.policy.cache_handles(&mut caches);
        for cache in &caches {
            states.push(cache.state.lock().unwrap_or_else(|e| e.into_inner()));
        }
        let mut source = self.index;
        if states[source].idle.is_empty() {
            if let Some(index) = states.iter().position(|state| !state.idle.is_empty()) {
                source = index;
            }
        }
        let lease = if let Some(idle) = states[source].idle.pop() {
            states[source].bytes -= idle.bytes;
            states[self.index].hits = states[self.index].hits.saturating_add(1);
            if let Some(ref record) = idle.lease.record {
                record.phase.store(RECORD_ACTIVE, Ordering::Release);
            }
            self.initializing.fetch_sub(1, Ordering::Relaxed);
            Some(idle.lease)
        } else {
            states[self.index].misses = states[self.index].misses.saturating_add(1);
            None
        };
        drop(states);
        drop(operation);
        lease
    }

    #[inline(always)]
    fn put_clean(&self, lease: ReservedLease<T, C>, bytes: usize) {
        let mut rejected = Some(Idle { lease, bytes });
        let mut needs_credit = false;
        {
            if let Ok(mut state) = self.state.try_lock() {
                if !self.policy.closed.load(Ordering::Acquire)
                    && bytes <= self.policy.limits.max_value_bytes
                {
                    if let Some(total) = state.bytes.checked_add(bytes) {
                        let count = state.idle.len() + 1;
                        if total <= state.credit_bytes && count <= state.credit_values {
                            state.store_idle(
                                rejected.take().expect("owned idle reservation"),
                                total,
                            );
                        } else {
                            needs_credit = true;
                        }
                    }
                }
            }
        }
        if needs_credit {
            self.put_with_credit(&mut rejected);
        }
        // 已拒收和待销毁对象的record保持reported，实际Drop后才标DEAD。
        drop(rejected);
    }

    #[cold]
    #[inline(never)]
    fn put_with_credit(&self, rejected: &mut Option<Idle<T, C>>) {
        let mut caches = Vec::with_capacity(8);
        let mut states = Vec::with_capacity(8);
        let operation = match self.policy.operations.try_lock() {
            Ok(operation) => operation,
            Err(_) => return,
        };
        if self.policy.closed.load(Ordering::Acquire) {
            return;
        }
        self.policy.cache_handles(&mut caches);
        for cache in &caches {
            match cache.state.try_lock() {
                Ok(state) => states.push(state),
                Err(_) => return,
            }
        }
        let bytes = rejected.as_ref().expect("pending return").bytes;
        let total = match states[self.index].bytes.checked_add(bytes) {
            Some(total) => total,
            None => return,
        };
        let count = states[self.index].idle.len() + 1;
        let need_bytes = total.saturating_sub(states[self.index].credit_bytes);
        let need_values = count.saturating_sub(states[self.index].credit_values);
        let mut credits = self
            .policy
            .credits
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let limits = self.policy.limits;
        if need_bytes > limits.max_cached_bytes - credits.bytes
            || need_values > limits.max_cached_values - credits.values
        {
            for (index, state) in states.iter_mut().enumerate() {
                if index == self.index {
                    continue;
                }
                let free_bytes = state.credit_bytes - state.bytes;
                let free_values = state.credit_values - state.idle.len();
                state.credit_bytes -= free_bytes;
                state.credit_values -= free_values;
                credits.bytes -= free_bytes;
                credits.values -= free_values;
            }
        }
        if need_bytes <= limits.max_cached_bytes - credits.bytes
            && need_values <= limits.max_cached_values - credits.values
        {
            credits.bytes += need_bytes;
            credits.values += need_values;
            let state = &mut states[self.index];
            state.credit_bytes += need_bytes;
            state.credit_values += need_values;
            state.store_idle(rejected.take().expect("owned idle reservation"), total);
        }
        drop(credits);
        drop(states);
        drop(operation);
    }
}

impl<T: Recyclable> ManagedPool<T> {
    /// Creates an optional retention policy on the default backing pool.
    pub fn new(limits: CacheLimits) -> Self {
        Self::new_with_config(limits)
    }

    /// Creates a retention policy with a custom backing-pool configuration.
    pub fn new_with_config<C: cfg::Config>(limits: CacheLimits) -> ManagedPool<T, C> {
        let count = limits.max_cached_values.max(1).min(8).next_power_of_two();
        let policy = Arc::new(Policy {
            limits,
            closed: AtomicBool::new(false),
            operations: Mutex::new(()),
            credits: Mutex::new(Credits {
                bytes: 0,
                values: 0,
                donors: Vec::with_capacity(count),
            }),
            records: Mutex::new(Vec::new()),
        });
        let caches: Vec<_> = (0..count)
            .map(|index| {
                Arc::new(Cache {
                    index,
                    policy: policy.clone(),
                    state: Mutex::new(State {
                        idle: Vec::with_capacity(limits.max_cached_values),
                        bytes: 0,
                        credit_bytes: 0,
                        credit_values: 0,
                        hits: 0,
                        misses: 0,
                    }),
                    initializing: AtomicUsize::new(0),
                })
            })
            .collect();
        policy
            .credits
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .donors = caches.iter().map(Arc::downgrade).collect();
        ManagedPool {
            slots: Arc::new(Pool::<Entry<T, C>>::new_with_config::<C>()),
            caches,
            policy,
        }
    }
}

impl<T: Recyclable, C: cfg::Config> ManagedPool<T, C> {
    /// Acquires a cleared value. Cached reserved slots can cross workers.
    ///
    /// A pre-close admission can finish lazy Default after close. Callbacks and
    /// allocation failures are not converted into capacity/closed rejections.
    #[inline(always)]
    pub fn try_lease_owned(&self) -> Result<ManagedLease<T, C>, ManagedAcquireError> {
        if self.policy.closed.load(Ordering::Acquire) {
            return Err(ManagedAcquireError::Closed);
        }
        let index = crate::Tid::<C>::current().as_usize() & (self.caches.len() - 1);
        self.lease_from_cache(&self.caches[index])
    }

    #[inline(always)]
    fn lease_from_cache(
        &self,
        cache: &Arc<Cache<T, C>>,
    ) -> Result<ManagedLease<T, C>, ManagedAcquireError> {
        let index = cache.index;
        let cached = cache.checkout()?;
        let mut inner = if let Some(mut inner) = cached {
            let permit = inner.permit.as_mut().expect("cached admission route");
            if permit.index != index {
                permit.cache = Arc::downgrade(cache);
                permit.index = index;
            }
            debug_assert!(!permit.active);
            permit.active = true;
            inner
        } else {
            self.acquire_cold(cache)?
        };
        if inner.value.is_none() {
            inner.value = Some(T::default());
        }
        Ok(ManagedLease { inner: Some(inner) })
    }

    #[cold]
    #[inline(never)]
    fn acquire_cold(
        &self,
        cache: &Arc<Cache<T, C>>,
    ) -> Result<ReservedLease<T, C>, ManagedAcquireError> {
        let permit = Permit {
            cache: Arc::downgrade(cache),
            index: cache.index,
            active: true,
            initializing: true,
        };
        let mut inner = self
            .slots
            .clone()
            .lease_owned()
            .ok_or(ManagedAcquireError::Capacity)?;
        inner.permit = Some(permit);
        if inner.record.is_none() {
            let record = Arc::new(ReturnRecord {
                phase: AtomicUsize::new(RECORD_ACTIVE),
                bytes: AtomicUsize::new(0),
            });
            {
                let mut records = self
                    .policy
                    .records
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                records.retain(|record| record.upgrade().is_some());
                records.push(Arc::downgrade(&record));
            }
            inner.record = Some(record);
        }
        if let Some(ref record) = inner.record {
            record.phase.store(RECORD_ACTIVE, Ordering::Release);
        }
        inner
            .permit
            .as_mut()
            .expect("initializing permit")
            .initializing = false;
        cache.initializing.fetch_sub(1, Ordering::Relaxed);
        Ok(Box::new(inner))
    }

    /// Removes exactly then-idle ownership at one all-shard snapshot. Racing
    /// returns can refill afterward. Backing slab pages are not evicted.
    pub fn trim(&self) -> Trimmed {
        self.drain(false)
    }

    /// Closes admission and drains idle ownership at one all-shard snapshot.
    /// Does not wait for, clear or invalidate active/shared readers.
    pub fn close(&self) -> Trimmed {
        self.drain(true)
    }

    fn drain(&self, close: bool) -> Trimmed {
        if !close && self.policy.closed.load(Ordering::Acquire) {
            return Trimmed::default();
        }
        let mut replacements: Vec<_> = (0..self.caches.len())
            .map(|_| {
                if close {
                    Vec::new()
                } else {
                    Vec::with_capacity(self.policy.limits.max_cached_values)
                }
            })
            .collect();
        let mut states = Vec::with_capacity(self.caches.len());
        let mut drained = Vec::with_capacity(self.caches.len());
        let mut removed = Trimmed::default();
        {
            let operation = self
                .policy
                .operations
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for cache in &self.caches {
                states.push(cache.state.lock().unwrap_or_else(|e| e.into_inner()));
            }
            if close {
                self.policy.closed.store(true, Ordering::Release);
            } else if self.policy.closed.load(Ordering::Acquire) {
                return Trimmed::default();
            }
            for (state, replacement) in states.iter_mut().zip(replacements.drain(..)) {
                for idle in &state.idle {
                    if let Some(ref record) = idle.lease.record {
                        record.phase.store(RECORD_REPORTED, Ordering::Release);
                    }
                }
                removed.values += state.idle.len();
                removed.bytes += state.bytes;
                drained.push(mem::replace(&mut state.idle, replacement));
                state.bytes = 0;
                state.credit_bytes = 0;
                state.credit_values = 0;
            }
            let mut credits = self
                .policy
                .credits
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            credits.bytes = 0;
            credits.values = 0;
            drop(credits);
            states.clear();
            drop(operation);
        }
        drop(drained);
        removed
    }

    /// Observes all idle shards together, plus separately changing active/return
    /// diagnostics. This is not a transactionally consistent heap measurement.
    pub fn stats(&self) -> CacheStats {
        let mut states = Vec::with_capacity(self.caches.len());
        for cache in &self.caches {
            states.push(cache.state.lock().unwrap_or_else(|e| e.into_inner()));
        }
        let mut stats = CacheStats {
            closed: self.policy.closed.load(Ordering::Acquire),
            cached_values: 0,
            cached_bytes: 0,
            cache_storage_bytes: 0,
            active_leases: 0,
            cache_hits: 0,
            cache_misses: 0,
            reserved_cache_bytes: 0,
            reserved_cache_values: 0,
            returning_values: 0,
            reported_returning_bytes: 0,
            unreported_returning_values: 0,
            reclaim_storage_bytes: self
                .caches
                .len()
                .saturating_mul(mem::size_of::<Cache<T, C>>())
                .saturating_add(mem::size_of::<Policy<T, C>>())
                .saturating_add(
                    self.caches
                        .capacity()
                        .saturating_mul(mem::size_of::<Arc<Cache<T, C>>>()),
                ),
        };
        for (state, cache) in states.iter().zip(&self.caches) {
            stats.cached_values += state.idle.len();
            stats.cached_bytes += state.bytes;
            stats.cache_storage_bytes = stats.cache_storage_bytes.saturating_add(
                state
                    .idle
                    .capacity()
                    .saturating_mul(mem::size_of::<Idle<T, C>>()),
            );
            stats.reserved_cache_bytes += state.credit_bytes;
            stats.reserved_cache_values += state.credit_values;
            stats.active_leases = stats
                .active_leases
                .saturating_add(cache.initializing.load(Ordering::Relaxed));
            stats.cache_hits = stats.cache_hits.saturating_add(state.hits);
            stats.cache_misses = stats.cache_misses.saturating_add(state.misses);
        }
        states.clear();
        {
            let credits = self
                .policy
                .credits
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            stats.reclaim_storage_bytes = stats.reclaim_storage_bytes.saturating_add(
                credits
                    .donors
                    .capacity()
                    .saturating_mul(mem::size_of::<Weak<Cache<T, C>>>()),
            );
        }
        let records = self
            .policy
            .records
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        stats.reclaim_storage_bytes = stats.reclaim_storage_bytes.saturating_add(
            records
                .capacity()
                .saturating_mul(mem::size_of::<Weak<ReturnRecord>>()),
        );
        for record in records.iter().filter_map(Weak::upgrade) {
            stats.reclaim_storage_bytes = stats
                .reclaim_storage_bytes
                .saturating_add(mem::size_of::<ReturnRecord>());
            let phase = record.phase.load(Ordering::Acquire);
            if phase != RECORD_DEAD {
                stats.reclaim_storage_bytes = stats
                    .reclaim_storage_bytes
                    .saturating_add(mem::size_of::<OwnedLease<Entry<T, C>, C>>());
            }
            if phase == RECORD_ACTIVE || phase == RECORD_CLEARING {
                stats.active_leases = stats.active_leases.saturating_add(1);
            }
            if phase == RECORD_CLEARING || phase == RECORD_REPORTED {
                stats.returning_values = stats.returning_values.saturating_add(1);
                if phase == RECORD_CLEARING {
                    stats.unreported_returning_values =
                        stats.unreported_returning_values.saturating_add(1);
                } else {
                    stats.reported_returning_bytes = stats
                        .reported_returning_bytes
                        .saturating_add(record.bytes.load(Ordering::Relaxed));
                }
            }
        }
        stats
    }
}

impl<T: Recyclable, C: cfg::Config> Drop for ManagedPool<T, C> {
    fn drop(&mut self) {
        self.close();
    }
}

impl<T: Recyclable, C: cfg::Config> ManagedLease<T, C> {
    /// Shares read-only ownership of the private exclusive reservation using
    /// one Arc control block. No raw key is published or exposed.
    pub fn freeze(self) -> ManagedSharedLease<T, C> {
        ManagedSharedLease {
            inner: crate::sync::Arc::new(self),
        }
    }
}
impl<T: Recyclable, C: cfg::Config> Drop for ManagedLease<T, C> {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            if let Some(mut value) = inner.value.take() {
                if let Some(ref record) = inner.record {
                    record.phase.store(RECORD_CLEARING, Ordering::Release);
                }
                value.clear();
                if let Some(mut permit) = inner.permit.take() {
                    if let Some(cache) = permit.cache.upgrade() {
                        if !cache.policy.closed.load(Ordering::Acquire)
                            && cache.policy.limits.max_cached_values != 0
                        {
                            let bytes = value.retained_bytes();
                            if let Some(ref record) = inner.record {
                                record.bytes.store(bytes, Ordering::Relaxed);
                                record.phase.store(RECORD_REPORTED, Ordering::Release);
                            }
                            permit.release(&cache);
                            inner.value = Some(value);
                            inner.permit = Some(permit);
                            cache.put_clean(inner, bytes);
                            return;
                        }
                        permit.release(&cache);
                    }
                }
            }
        }
    }
}
impl<T: Recyclable, C: cfg::Config> ops::Deref for ManagedLease<T, C> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        self.inner
            .as_ref()
            .and_then(|inner| inner.value.as_ref())
            .expect("managed lease value")
    }
}
impl<T: Recyclable, C: cfg::Config> ops::DerefMut for ManagedLease<T, C> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        self.inner
            .as_mut()
            .and_then(|inner| inner.value.as_mut())
            .expect("managed lease value")
    }
}
impl<T: Recyclable, C: cfg::Config> ops::Deref for ManagedSharedLease<T, C> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.inner
    }
}
impl<T: Recyclable, C: cfg::Config> Clone for ManagedSharedLease<T, C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl<T: Recyclable + fmt::Debug, C: cfg::Config> fmt::Debug for ManagedLease<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ManagedLease").field(&&**self).finish()
    }
}
impl<T: Recyclable + fmt::Debug, C: cfg::Config> fmt::Debug for ManagedSharedLease<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ManagedSharedLease").field(&&**self).finish()
    }
}
impl<T: Recyclable, C: cfg::Config> fmt::Debug for ManagedPool<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManagedPool")
            .field("limits", &self.policy.limits)
            .field("stats", &self.stats())
            .finish()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Probe {
        pool: Weak<ManagedPool<Probe>>,
        observed: Option<Arc<AtomicUsize>>,
    }
    impl Probe {
        fn check_unlocked(&self, bit: usize) {
            if let (Some(pool), Some(ref observed)) = (self.pool.upgrade(), &self.observed) {
                for cache in &pool.caches {
                    assert!(cache.state.try_lock().is_ok());
                }
                assert!(pool.policy.credits.try_lock().is_ok());
                assert!(pool.policy.operations.try_lock().is_ok());
                assert!(pool.policy.records.try_lock().is_ok());
                let stats = pool.stats();
                assert!(stats.returning_values >= 1);
                observed.fetch_or(bit, Ordering::SeqCst);
            }
        }
    }
    impl Clear for Probe {
        fn clear(&mut self) {
            self.check_unlocked(1);
        }
    }
    impl Recyclable for Probe {
        fn retained_bytes(&self) -> usize {
            self.check_unlocked(2);
            0
        }
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            self.check_unlocked(4);
            if let Some(pool) = self.pool.upgrade() {
                pool.close();
                pool.trim();
            }
        }
    }

    #[test]
    fn payload_callbacks_reenter_stats_and_close_outside_all_policy_locks() {
        for close in [false, true].iter().copied() {
            let pool = Arc::new(ManagedPool::<Probe>::new(CacheLimits::new(0, 2)));
            let observed = Arc::new(AtomicUsize::new(0));
            let mut value = pool.try_lease_owned().unwrap();
            value.pool = Arc::downgrade(&pool);
            value.observed = Some(observed.clone());
            drop(value);
            assert_eq!(observed.load(Ordering::SeqCst), 3);
            if close {
                pool.close();
            } else {
                pool.trim();
            }
            assert_eq!(observed.load(Ordering::SeqCst), 7);
        }
    }

    #[test]
    fn contended_return_discards_without_waiting_for_return_shard() {
        let pool = ManagedPool::<Vec<u8>>::new(CacheLimits::new(1024, 2));
        let mut value = pool.try_lease_owned().unwrap();
        value.resize(128, 0);
        let index = value.inner.as_ref().unwrap().permit.as_ref().unwrap().index;
        let state = pool.caches[index].state.lock().unwrap();
        drop(value);
        assert!(state.idle.is_empty());
        drop(state);
        let stats = pool.stats();
        assert_eq!(stats.active_leases, 0);
        assert_eq!(stats.returning_values, 0);
    }

    #[test]
    fn unpublished_cached_reservation_does_not_retire_another_admission() {
        let pool = ManagedPool::<Vec<u8>>::new(CacheLimits::new(1024, 2));
        drop(pool.lease_from_cache(&pool.caches[0]).unwrap());
        let abandoned = pool.caches[0].checkout().unwrap().unwrap();
        let mut survivor = pool.lease_from_cache(&pool.caches[0]).unwrap();
        survivor.push(73);
        pool.close();
        drop(abandoned);
        assert_eq!(pool.stats().active_leases, 1);
        let cache = Arc::downgrade(&pool.caches[0]);
        drop(pool);
        assert!(cache.upgrade().is_none());
        assert_eq!(&**survivor, &[73]);
        std::thread::spawn(move || drop(survivor)).join().unwrap();
    }

    #[test]
    fn cached_reservations_and_unused_credit_can_move_between_return_shards() {
        let pool = ManagedPool::<Vec<u8>>::new(CacheLimits::new(512, 4));
        let mut values = Vec::new();
        let mut pointers = Vec::new();
        for _ in 0..4 {
            let mut value = pool.lease_from_cache(&pool.caches[0]).unwrap();
            value.resize(128, 7);
            pointers.push(value.as_ptr());
            values.push(value);
        }
        drop(values);
        let value = pool.lease_from_cache(&pool.caches[1]).unwrap();
        assert!(pointers.contains(&value.as_ptr()));
        assert_eq!(pool.stats().cache_misses, 4);
        drop(value);
        let stats = pool.stats();
        assert_eq!(stats.cached_values, 4);
        assert_eq!(stats.cached_bytes, 512);
        assert_eq!(stats.reserved_cache_values, 4);
        assert_eq!(stats.reserved_cache_bytes, 512);
        assert_eq!(pool.caches[0].state.lock().unwrap().credit_values, 3);
        assert_eq!(pool.caches[1].state.lock().unwrap().credit_values, 1);
        assert_eq!(
            pool.trim(),
            Trimmed {
                values: 4,
                bytes: 512
            }
        );
        assert_eq!(pool.stats().reserved_cache_values, 0);
        assert_eq!(pool.stats().reserved_cache_bytes, 0);
    }
}
