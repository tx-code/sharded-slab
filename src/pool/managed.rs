use super::{OwnedLease, Pool, SharedLease};
use crate::{
    cfg,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    },
    Clear, DefaultConfig,
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
/// Idle fields are read under the cache lock. The independent active count may
/// change concurrently; this is not a transactionally consistent heap snapshot.
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
    /// Admitted requests that needed `T::default`, including panicking defaults.
    pub cache_misses: usize,
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

/// An optional bounded idle-payload policy on the existing sharded [`Pool`].
///
/// Slab slots still use the backing pool's shards and local/remote free lists.
/// Payloads move to a bounded, shared idle cache only after last-owner clearing.
/// Cache hits can therefore reuse payloads returned by other workers. Returns
/// use a weak controller route and may discard on lock contention or overflow.
/// This does not evict backing slab pages or bound active allocations.
///
/// Share the controller with `Arc<ManagedPool<_>>`. Dropping its last owner
/// closes admission and drains idle payloads; active leases remain valid.
/// Allocation and payload callbacks can panic; no recoverable OOM is implied.
/// A panicking clear/report quarantines that backing slot until pool destruction.
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
    slots: Arc<Pool<Entry<T>, C>>,
    cache: Arc<Cache<T>>,
}

/// Exclusive managed value; dropping it clears then caches or discards payload.
#[must_use]
pub struct ManagedLease<T: Recyclable, C: cfg::Config = DefaultConfig> {
    inner: OwnedLease<Entry<T>, C>,
}

/// Cloneable managed read ownership; only the last reader returns its payload.
#[must_use]
pub struct ManagedSharedLease<T: Recyclable, C: cfg::Config = DefaultConfig> {
    inner: SharedLease<Entry<T>, C>,
}

struct Idle<T> {
    value: T,
    bytes: usize,
}
struct State<T> {
    idle: Vec<Idle<T>>,
    bytes: usize,
    closed: bool,
    hits: usize,
    misses: usize,
}
struct Cache<T: Recyclable> {
    limits: CacheLimits,
    state: Mutex<State<T>>,
    closed: AtomicBool,
    active: AtomicUsize,
}
struct Permit<T: Recyclable> {
    cache: Weak<Cache<T>>,
    active: bool,
}
struct Entry<T: Recyclable> {
    value: Option<T>,
    permit: Option<Permit<T>>,
}

impl<T: Recyclable> Default for Entry<T> {
    fn default() -> Self {
        Self {
            value: None,
            permit: None,
        }
    }
}
impl<T: Recyclable> Clear for Entry<T> {
    fn clear(&mut self) {
        // 先移出资源；回调panic时permit仍会释放统计，槽不能被误发为干净。
        let permit = self.permit.take();
        if let Some(mut value) = self.value.take() {
            value.clear();
            if let Some(mut permit) = permit {
                permit.recycle(value);
                // 同一槽复用弱归还路由，避免每次借还都修改全局weak计数。
                // 回调panic时局部permit仍会销毁；成功后仅留下失活路由。
                self.permit = Some(permit);
            }
        }
    }
}

impl<T: Recyclable> Permit<T> {
    fn deactivate(&mut self) -> Option<Arc<Cache<T>>> {
        if !self.active {
            return None;
        }
        let cache = self.cache.upgrade();
        if self.active {
            self.active = false;
            if let Some(ref cache) = cache {
                cache.active.fetch_sub(1, Ordering::Relaxed);
            }
        }
        cache
    }
    fn recycle(&mut self, value: T) {
        // Clear已成功；后续容量计算或Drop panic不能使active许可遗留。
        if let Some(cache) = self.deactivate() {
            cache.put_clean(value);
        }
    }
}
impl<T: Recyclable> Drop for Permit<T> {
    fn drop(&mut self) {
        self.deactivate();
    }
}

impl<T: Recyclable> Cache<T> {
    fn checkout(&self) -> Result<Option<T>, ManagedAcquireError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            return Err(ManagedAcquireError::Closed);
        }
        // 此锁下授予许可是admission线性化点；Default在解锁后执行。
        self.active.fetch_add(1, Ordering::Relaxed);
        let value = state.idle.pop().map(|idle| {
            state.bytes -= idle.bytes;
            idle.value
        });
        if value.is_some() {
            state.hits = state.hits.saturating_add(1);
        } else {
            state.misses = state.misses.saturating_add(1);
        }
        Ok(value)
    }

    fn put_clean(&self, value: T) {
        if self.closed.load(Ordering::Acquire) || self.limits.max_cached_values == 0 {
            return;
        }
        let bytes = value.retained_bytes();
        if bytes > self.limits.max_value_bytes {
            return;
        }
        let mut rejected = Some(Idle { value, bytes });
        {
            // 归还可放弃缓存；不等待读者或持锁运行用户回调。
            if let Ok(mut state) = self.state.try_lock() {
                if !state.closed && state.idle.len() < self.limits.max_cached_values {
                    if let Some(total) = state.bytes.checked_add(bytes) {
                        if total <= self.limits.max_cached_bytes {
                            state.idle.push(rejected.take().expect("clean idle value"));
                            state.bytes = total;
                        }
                    }
                }
            }
        }
        drop(rejected);
    }

    fn close(&self) -> Trimmed {
        let (idle, removed) = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            self.closed.store(true, Ordering::Release);
            state.closed = true;
            let removed = Trimmed {
                values: state.idle.len(),
                bytes: state.bytes,
            };
            state.bytes = 0;
            (mem::take(&mut state.idle), removed)
        };
        drop(idle);
        removed
    }

    fn trim(&self) -> Trimmed {
        if self.closed.load(Ordering::Acquire) {
            return Trimmed::default();
        }
        // 预留接收目录在锁外，后续Drop归还push不得引起锁内扩容。
        let replacement = Vec::with_capacity(self.limits.max_cached_values);
        let (idle, removed) = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.closed {
                return Trimmed::default();
            }
            let removed = Trimmed {
                values: state.idle.len(),
                bytes: state.bytes,
            };
            state.bytes = 0;
            (mem::replace(&mut state.idle, replacement), removed)
        };
        drop(idle);
        removed
    }
}

impl<T: Recyclable> ManagedPool<T> {
    /// Creates a controller with the default backing-pool configuration.
    pub fn new(limits: CacheLimits) -> Self {
        Self::new_with_config(limits)
    }

    /// Creates a controller with a custom backing-pool configuration.
    pub fn new_with_config<C: cfg::Config>(limits: CacheLimits) -> ManagedPool<T, C> {
        let idle = Vec::with_capacity(limits.max_cached_values);
        ManagedPool {
            slots: Arc::new(Pool::<Entry<T>>::new_with_config::<C>()),
            cache: Arc::new(Cache {
                limits,
                state: Mutex::new(State {
                    idle,
                    bytes: 0,
                    closed: false,
                    hits: 0,
                    misses: 0,
                }),
                closed: AtomicBool::new(false),
                active: AtomicUsize::new(0),
            }),
        }
    }
}

impl<T: Recyclable, C: cfg::Config> ManagedPool<T, C> {
    /// Acquires an exclusive value or reports closure/backing-slot exhaustion.
    ///
    /// Admission linearizes under the controller lock. A request admitted
    /// before close may finish its lazy `Default` and return afterward. Such a
    /// lease remains valid but cannot cache payload on return. Defaults run
    /// outside locks; their panics and allocation failures are not rejections.
    pub fn try_lease_owned(&self) -> Result<ManagedLease<T, C>, ManagedAcquireError> {
        if self.cache.closed.load(Ordering::Acquire) {
            return Err(ManagedAcquireError::Closed);
        }
        let mut inner = self
            .slots
            .clone()
            .lease_owned()
            .ok_or(ManagedAcquireError::Capacity)?;
        let value = self.cache.checkout()?;
        // Default panic时底层独占租约清理Entry，已安装的permit自动退回许可。
        if let Some(ref mut permit) = inner.permit {
            debug_assert!(!permit.active);
            permit.active = true;
        } else {
            inner.permit = Some(Permit {
                cache: Arc::downgrade(&self.cache),
                active: true,
            });
        }
        inner.value = Some(value.unwrap_or_default());
        Ok(ManagedLease { inner })
    }

    /// Removes payloads idle at the trim linearization point, then drops them
    /// outside locks. Racing returns may refill the cache. Active/shared values
    /// and backing slab pages are not trimmed. Replacement metadata can allocate.
    pub fn trim(&self) -> Trimmed {
        self.cache.trim()
    }

    /// Closes admission and drains then-current idle payloads outside locks.
    /// Idempotent. Does not invalidate admitted or shared leases, wait for them,
    /// or evict backing pages. Future returns are cleared and discarded.
    pub fn close(&self) -> Trimmed {
        self.cache.close()
    }

    /// Reads cache accounting. This does not measure process memory.
    pub fn stats(&self) -> CacheStats {
        let state = self.cache.state.lock().unwrap_or_else(|e| e.into_inner());
        CacheStats {
            closed: state.closed,
            cached_values: state.idle.len(),
            cached_bytes: state.bytes,
            cache_storage_bytes: state
                .idle
                .capacity()
                .saturating_mul(mem::size_of::<Idle<T>>()),
            active_leases: self.cache.active.load(Ordering::Relaxed),
            cache_hits: state.hits,
            cache_misses: state.misses,
        }
    }
}
impl<T: Recyclable, C: cfg::Config> Drop for ManagedPool<T, C> {
    fn drop(&mut self) {
        self.cache.close();
    }
}

impl<T: Recyclable, C: cfg::Config> ManagedLease<T, C> {
    /// Consumes exclusive access. Last-reader clearing uses the managed policy.
    /// The backing shared lease allocates one `Arc` control block.
    pub fn freeze(self) -> ManagedSharedLease<T, C> {
        ManagedSharedLease {
            inner: self.inner.freeze(),
        }
    }
}
impl<T: Recyclable, C: cfg::Config> ops::Deref for ManagedLease<T, C> {
    type Target = T;
    fn deref(&self) -> &T {
        self.inner.value.as_ref().expect("managed lease value")
    }
}
impl<T: Recyclable, C: cfg::Config> ops::DerefMut for ManagedLease<T, C> {
    fn deref_mut(&mut self) -> &mut T {
        self.inner.value.as_mut().expect("managed lease value")
    }
}
impl<T: Recyclable, C: cfg::Config> ops::Deref for ManagedSharedLease<T, C> {
    type Target = T;
    fn deref(&self) -> &T {
        self.inner.value.as_ref().expect("managed shared value")
    }
}
impl<T: Recyclable, C: cfg::Config> Clone for ManagedSharedLease<T, C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl<T: Recyclable, C: cfg::Config> fmt::Debug for ManagedPool<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManagedPool")
            .field("limits", &self.cache.limits)
            .field("stats", &self.stats())
            .finish()
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

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Probe {
        cache: Weak<Cache<Probe>>,
        observed: Option<Arc<AtomicUsize>>,
    }
    impl Probe {
        fn check_unlocked(&self, bit: usize) {
            if let (Some(cache), Some(ref observed)) = (self.cache.upgrade(), &self.observed) {
                assert!(
                    cache.state.try_lock().is_ok(),
                    "payload callback ran with cache locked"
                );
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
        }
    }

    #[test]
    fn all_payload_callbacks_run_outside_cache_lock() {
        for close in [false, true].iter().copied() {
            let pool = ManagedPool::<Probe>::new(CacheLimits::new(0, 2));
            let observed = Arc::new(AtomicUsize::new(0));
            let mut value = pool.try_lease_owned().unwrap();
            value.cache = Arc::downgrade(&pool.cache);
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
    fn contended_return_discards_without_waiting_for_cache_lock() {
        let pool = ManagedPool::<Vec<u8>>::new(CacheLimits::new(1024, 2));
        let mut value = pool.try_lease_owned().unwrap();
        value.resize(128, 0);
        let state = pool.cache.state.lock().unwrap();
        drop(value);
        assert_eq!(pool.cache.active.load(Ordering::Relaxed), 0);
        assert!(state.idle.is_empty());
        drop(state);
        assert_eq!(pool.stats().cached_bytes, 0);
    }

    #[test]
    fn abandoned_inactive_slot_does_not_release_another_admission() {
        let pool = ManagedPool::<Vec<u8>>::new(CacheLimits::new(1024, 2));
        drop(pool.try_lease_owned().unwrap());
        // 模拟槽已取得但随后close拒绝准入；旧失活路由不能扣别人的许可。
        let abandoned = pool.slots.clone().lease_owned().unwrap();
        let mut survivor = pool.try_lease_owned().unwrap();
        survivor.push(73);
        pool.close();
        assert_eq!(
            pool.cache.checkout().err(),
            Some(ManagedAcquireError::Closed)
        );
        drop(abandoned);
        assert_eq!(pool.stats().active_leases, 1);
        let cache = Arc::downgrade(&pool.cache);
        drop(pool);
        assert!(cache.upgrade().is_none());
        assert_eq!(&**survivor, &[73]);
        std::thread::spawn(move || drop(survivor)).join().unwrap();
    }
}
