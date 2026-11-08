use sharded_slab::{
    pool::{CacheLimits, ManagedAcquireError, ManagedPool, Recyclable},
    Clear, Config,
};
use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
};

#[test]
fn idle_count_and_reported_byte_limits_bound_retention() {
    let pool = ManagedPool::<Vec<u8>>::new(CacheLimits::new(256, 2));
    let mut values = Vec::new();
    for _ in 0..4 {
        let mut value = pool.try_lease_owned().unwrap();
        value.resize(128, 1);
        values.push(value);
    }
    assert_eq!(pool.stats().active_leases, 4);
    drop(values);
    let stats = pool.stats();
    assert_eq!(stats.active_leases, 0);
    assert!(stats.cached_values <= 2);
    assert!(stats.cached_bytes <= 256);
    assert!(stats.cached_values > 0);
    let hit = pool.try_lease_owned().unwrap();
    assert!(hit.is_empty());
    assert!(hit.capacity() >= 128);
    assert_eq!(pool.stats().cache_hits, 1);
}

#[test]
fn oversized_values_are_dropped_instead_of_cached() {
    let mut limits = CacheLimits::new(4096, 4);
    limits.max_value_bytes = 16;
    let pool = ManagedPool::<Vec<u8>>::new(limits);
    let mut value = pool.try_lease_owned().unwrap();
    value.resize(128, 1);
    drop(value);
    assert_eq!(pool.stats().cached_values, 0);
    let next = pool.try_lease_owned().unwrap();
    assert_eq!(next.capacity(), 0);
}

#[test]
fn checked_byte_accounting_rejects_overflow_and_caps_zero_byte_values() {
    #[derive(Default)]
    struct Report(usize);
    impl Clear for Report {
        fn clear(&mut self) {}
    }
    impl Recyclable for Report {
        fn retained_bytes(&self) -> usize {
            self.0
        }
    }
    let pool = ManagedPool::<Report>::new(CacheLimits::new(usize::MAX, 2));
    let mut large = pool.try_lease_owned().unwrap();
    let mut small = pool.try_lease_owned().unwrap();
    large.0 = usize::MAX;
    small.0 = 1;
    drop(large);
    drop(small);
    assert_eq!(pool.stats().cached_values, 1);
    assert_eq!(pool.stats().cached_bytes, usize::MAX);
    assert_eq!(pool.trim().bytes, usize::MAX);

    let zero = ManagedPool::<Vec<()>>::new(CacheLimits::new(0, 2));
    let values = (0..5)
        .map(|_| zero.try_lease_owned().unwrap())
        .collect::<Vec<_>>();
    drop(values);
    assert_eq!(zero.stats().cached_values, 2);
    assert_eq!(zero.stats().cached_bytes, 0);
}

#[test]
fn values_are_lazy_defaulted_and_reused() {
    static DEFAULTS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Debug)]
    struct Value;
    impl Default for Value {
        fn default() -> Self {
            DEFAULTS.fetch_add(1, Ordering::SeqCst);
            Self
        }
    }
    impl Clear for Value {
        fn clear(&mut self) {}
    }
    impl Recyclable for Value {
        fn retained_bytes(&self) -> usize {
            0
        }
    }
    let pool = ManagedPool::<Value>::new(CacheLimits::new(0, 2));
    assert_eq!(DEFAULTS.load(Ordering::SeqCst), 0);
    let first = pool.try_lease_owned().unwrap();
    assert_eq!(DEFAULTS.load(Ordering::SeqCst), 1);
    drop(first);
    drop(pool.try_lease_owned().unwrap());
    assert_eq!(DEFAULTS.load(Ordering::SeqCst), 1);
    pool.close();
    assert_eq!(
        pool.try_lease_owned().err(),
        Some(ManagedAcquireError::Closed)
    );
}

#[test]
fn close_and_trim_do_not_touch_active_readers() {
    let pool = ManagedPool::<Vec<u8>>::new(CacheLimits::new(1024, 4));
    let mut active = pool.try_lease_owned().unwrap();
    active.extend_from_slice(&[1, 2, 3]);
    let root = active.freeze();
    let later = root.clone();
    let mut idle = pool.try_lease_owned().unwrap();
    idle.resize(128, 4);
    drop(idle);
    assert_eq!(pool.trim().values, 1);
    assert_eq!(pool.stats().cached_values, 0);
    assert_eq!(&**root, &[1, 2, 3]);
    let mut idle = pool.try_lease_owned().unwrap();
    idle.resize(64, 5);
    drop(idle);
    assert_eq!(pool.close().values, 1);
    assert_eq!(pool.close().values, 0);
    assert!(pool.stats().closed);
    assert_eq!(pool.stats().active_leases, 1);
    assert_eq!(
        pool.try_lease_owned().err(),
        Some(ManagedAcquireError::Closed)
    );
    drop(root);
    assert_eq!(&**later, &[1, 2, 3]);
    drop(later);
    assert_eq!(pool.stats().active_leases, 0);
    assert_eq!(pool.stats().cached_bytes, 0);
}

#[derive(Default)]
struct Watched {
    data: Vec<u8>,
    cleared: Option<Arc<AtomicUsize>>,
    dropped: Option<Arc<AtomicUsize>>,
    panic_clear: bool,
    panic_report: bool,
}
impl Clear for Watched {
    fn clear(&mut self) {
        if let Some(count) = self.cleared.take() {
            count.fetch_add(1, Ordering::SeqCst);
        }
        if self.panic_clear {
            self.panic_clear = false;
            panic!("injected clear panic");
        }
        self.data.clear();
    }
}
impl Recyclable for Watched {
    fn retained_bytes(&self) -> usize {
        if self.panic_report {
            panic!("injected report panic");
        }
        self.data.capacity()
    }
}
impl Drop for Watched {
    fn drop(&mut self) {
        if let Some(ref count) = self.dropped {
            count.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[test]
fn controller_drop_drains_idle_but_owned_readers_survive() {
    let pool = Arc::new(ManagedPool::<Watched>::new(CacheLimits::new(1024, 2)));
    let cleared = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut active = pool.try_lease_owned().unwrap();
    active.data.push(7);
    active.cleared = Some(cleared.clone());
    active.dropped = Some(dropped.clone());
    let root = active.freeze();
    let mut idle = pool.try_lease_owned().unwrap();
    idle.data.resize(128, 0);
    idle.dropped = Some(dropped.clone());
    drop(idle);
    let sibling = pool.clone();
    drop(sibling);
    assert!(!pool.stats().closed);
    let weak = Arc::downgrade(&pool);
    drop(pool);
    assert!(weak.upgrade().is_none());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(root.data, [7]);
    std::thread::spawn(move || drop(root)).join().unwrap();
    assert_eq!(cleared.load(Ordering::SeqCst), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
}

#[test]
fn callback_panics_never_enter_idle_and_release_active_accounting() {
    for report in [false, true].iter().copied() {
        let pool = ManagedPool::<Watched>::new(CacheLimits::new(1024, 2));
        let cleared = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut value = pool.try_lease_owned().unwrap();
        value.cleared = Some(cleared.clone());
        value.dropped = Some(dropped.clone());
        value.panic_report = report;
        value.panic_clear = !report;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(value)));
        assert!(result.is_err());
        assert_eq!(pool.stats().active_leases, 0);
        assert_eq!(pool.stats().cached_values, 0);
        assert_eq!(cleared.load(Ordering::SeqCst), 1);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert!(pool.try_lease_owned().is_ok());
    }
}

#[test]
fn default_panic_rolls_back_admission_without_caching() {
    struct Fails;
    impl Default for Fails {
        fn default() -> Self {
            panic!("injected default panic");
        }
    }
    impl Clear for Fails {
        fn clear(&mut self) {}
    }
    impl Recyclable for Fails {
        fn retained_bytes(&self) -> usize {
            0
        }
    }
    let pool = ManagedPool::<Fails>::new(CacheLimits::new(0, 1));
    for _ in 0..2 {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.try_lease_owned()))
                .is_err()
        );
        assert_eq!(pool.stats().active_leases, 0);
        assert_eq!(pool.stats().cached_values, 0);
    }
    pool.close();
    assert_eq!(
        pool.try_lease_owned().err(),
        Some(ManagedAcquireError::Closed)
    );
}

#[test]
fn reused_admission_panics_leave_independent_readers_and_controller_drop_safe() {
    for report in [false, true].iter().copied() {
        let pool = Arc::new(ManagedPool::<Watched>::new(CacheLimits::new(1024, 2)));
        let cleared = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        for _ in 0..8 {
            let mut value = pool.try_lease_owned().unwrap();
            value.data.push(79);
            value.cleared = Some(cleared.clone());
            value.dropped = Some(dropped.clone());
            drop(value);
            assert_eq!(pool.stats().active_leases, 0);
        }
        let mut failing = pool.try_lease_owned().unwrap();
        failing.cleared = Some(cleared.clone());
        failing.panic_report = report;
        failing.panic_clear = !report;
        let mut survivor = pool.try_lease_owned().unwrap();
        survivor.data.push(83);
        survivor.cleared = Some(cleared.clone());
        survivor.dropped = Some(dropped.clone());
        let reader = survivor.freeze();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(failing))).is_err());
        assert_eq!(pool.stats().active_leases, 1);
        assert_eq!(pool.stats().cached_values, 0);
        pool.close();
        drop(pool);
        assert_eq!(reader.data, [83]);
        std::thread::spawn(move || drop(reader)).join().unwrap();
        assert_eq!(cleared.load(Ordering::SeqCst), 10);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }
}

#[derive(Clone, Copy)]
enum ReturnStage {
    Clear,
    Report,
    Dispose,
}
struct ReturnBarrier {
    stage: ReturnStage,
    started: mpsc::Sender<()>,
    proceed: Arc<std::sync::Mutex<mpsc::Receiver<()>>>,
}
impl ReturnBarrier {
    fn wait(&self, stage: ReturnStage) {
        if std::mem::discriminant(&self.stage) == std::mem::discriminant(&stage) {
            self.started.send(()).unwrap();
            self.proceed.lock().unwrap().recv().unwrap();
        }
    }
}
fn return_barrier(stage: ReturnStage) -> (ReturnBarrier, mpsc::Receiver<()>, mpsc::Sender<()>) {
    let (started, ready) = mpsc::channel();
    let (resume, proceed) = mpsc::channel();
    (
        ReturnBarrier {
            stage,
            started,
            proceed: Arc::new(std::sync::Mutex::new(proceed)),
        },
        ready,
        resume,
    )
}
#[derive(Default)]
struct ReturningProbe {
    data: Vec<u8>,
    barrier: Option<ReturnBarrier>,
}
impl Clear for ReturningProbe {
    fn clear(&mut self) {
        if let Some(ref barrier) = self.barrier {
            barrier.wait(ReturnStage::Clear);
        }
        self.data.clear();
    }
}
impl Recyclable for ReturningProbe {
    fn retained_bytes(&self) -> usize {
        if let Some(ref barrier) = self.barrier {
            barrier.wait(ReturnStage::Report);
        }
        self.data.capacity()
    }
}
impl Drop for ReturningProbe {
    fn drop(&mut self) {
        if let Some(ref barrier) = self.barrier {
            barrier.wait(ReturnStage::Dispose);
        }
    }
}

#[test]
fn incomplete_clear_and_report_are_visible_until_callback_completion() {
    for stage in [ReturnStage::Clear, ReturnStage::Report].iter().copied() {
        let pool = ManagedPool::<ReturningProbe>::new(CacheLimits::new(256, 2));
        let (barrier, ready, resume) = return_barrier(stage);
        let mut value = pool.try_lease_owned().unwrap();
        value.data.resize(128, 1);
        value.barrier = Some(barrier);
        let worker = std::thread::spawn(move || drop(value));
        ready
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        let stats = pool.stats();
        assert_eq!(stats.cached_values, 0);
        assert_eq!(stats.returning_values, 1);
        assert_eq!(stats.unreported_returning_values, 1);
        assert_eq!(stats.reported_returning_bytes, 0);
        assert_eq!(stats.active_leases, 1);
        pool.close();
        resume.send(()).unwrap();
        worker.join().unwrap();
        assert_eq!(pool.stats().returning_values, 0);
        assert_eq!(pool.stats().active_leases, 0);
    }
}

#[test]
fn rejected_reported_payload_stays_accounted_until_destructor_finishes() {
    let mut limits = CacheLimits::new(256, 2);
    limits.max_value_bytes = 0;
    let pool = ManagedPool::<ReturningProbe>::new(limits);
    let (barrier, ready, resume) = return_barrier(ReturnStage::Dispose);
    let mut value = pool.try_lease_owned().unwrap();
    value.data.resize(128, 1);
    value.barrier = Some(barrier);
    let worker = std::thread::spawn(move || drop(value));
    ready
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    let stats = pool.stats();
    assert_eq!(stats.cached_values, 0);
    assert_eq!(stats.active_leases, 0);
    assert_eq!(stats.returning_values, 1);
    assert_eq!(stats.reported_returning_bytes, 128);
    assert_eq!(stats.unreported_returning_values, 0);
    resume.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(pool.stats().returning_values, 0);
}

#[test]
fn detached_trim_and_close_batches_stay_accounted_during_payload_drop() {
    for close in [false, true].iter().copied() {
        let pool = Arc::new(ManagedPool::<ReturningProbe>::new(CacheLimits::new(256, 2)));
        let (barrier, ready, resume) = return_barrier(ReturnStage::Dispose);
        let mut first = pool.try_lease_owned().unwrap();
        let mut second = pool.try_lease_owned().unwrap();
        first.data.resize(128, 1);
        second.data.resize(128, 2);
        first.barrier = Some(barrier);
        drop(first);
        drop(second);
        let control = pool.clone();
        let worker = std::thread::spawn(move || {
            if close {
                control.close()
            } else {
                control.trim()
            }
        });
        ready
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        let stats = pool.stats();
        assert_eq!(stats.closed, close);
        assert_eq!(stats.cached_values, 0);
        assert_eq!(stats.active_leases, 0);
        assert_eq!(stats.returning_values, 2);
        assert_eq!(stats.reported_returning_bytes, 256);
        assert_eq!(stats.reserved_cache_values, 0);
        resume.send(()).unwrap();
        let removed = worker.join().unwrap();
        assert_eq!(removed.values, 2);
        assert_eq!(removed.bytes, 256);
        assert_eq!(pool.stats().returning_values, 0);
    }
}

struct OneSlot;
impl Config for OneSlot {
    const INITIAL_PAGE_SIZE: usize = 1;
    const MAX_PAGES: usize = 1;
}

#[test]
fn clear_and_report_panics_restore_exact_single_slot_capacity() {
    for report in [false, true].iter().copied() {
        let pool = ManagedPool::<Watched>::new_with_config::<OneSlot>(CacheLimits::new(256, 1));
        let mut value = pool.try_lease_owned().unwrap();
        value.panic_clear = !report;
        value.panic_report = report;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(value))).is_err());
        assert_eq!(pool.stats().active_leases, 0);
        assert_eq!(pool.stats().returning_values, 0);
        let next = pool
            .try_lease_owned()
            .expect("detached callback panic must return the sole backing slot");
        assert_eq!(
            pool.try_lease_owned().err(),
            Some(ManagedAcquireError::Capacity)
        );
        drop(next);
    }
}

type Gate = (mpsc::Sender<()>, mpsc::Receiver<()>);
thread_local! { static DEFAULT_GATE: RefCell<Option<Gate>> = RefCell::new(None); }

#[test]
fn preclose_admission_can_finish_default_after_close() {
    struct Paused;
    impl Default for Paused {
        fn default() -> Self {
            DEFAULT_GATE.with(|gate| {
                let (ready, proceed) = gate.borrow_mut().take().expect("admitted factory gate");
                ready.send(()).unwrap();
                proceed.recv().unwrap();
            });
            Self
        }
    }
    impl Clear for Paused {
        fn clear(&mut self) {}
    }
    impl Recyclable for Paused {
        fn retained_bytes(&self) -> usize {
            0
        }
    }
    let pool = Arc::new(ManagedPool::<Paused>::new(CacheLimits::new(0, 1)));
    let (ready, observed) = mpsc::channel();
    let (resume, proceed) = mpsc::channel();
    let worker_pool = pool.clone();
    let worker = std::thread::spawn(move || {
        DEFAULT_GATE.with(|gate| *gate.borrow_mut() = Some((ready, proceed)));
        worker_pool.try_lease_owned().unwrap()
    });
    observed.recv().unwrap();
    assert_eq!(pool.stats().active_leases, 1);
    pool.close();
    assert_eq!(
        pool.try_lease_owned().err(),
        Some(ManagedAcquireError::Closed)
    );
    resume.send(()).unwrap();
    let admitted = worker.join().unwrap();
    assert_eq!(pool.stats().active_leases, 1);
    drop(admitted);
    assert_eq!(pool.stats().active_leases, 0);
    assert_eq!(pool.stats().cached_values, 0);
}

struct TwoSlots;
impl Config for TwoSlots {
    const INITIAL_PAGE_SIZE: usize = 2;
    const MAX_PAGES: usize = 1;
}

#[test]
fn slot_capacity_failure_and_other_controller_rejection_leave_no_partial_group() {
    let a = ManagedPool::<Vec<u8>>::new_with_config::<TwoSlots>(CacheLimits::new(1024, 2));
    let b = ManagedPool::<String>::new(CacheLimits::new(1024, 2));
    let first = a.try_lease_owned().unwrap();
    let second = a.try_lease_owned().unwrap();
    assert_eq!(
        a.try_lease_owned().err(),
        Some(ManagedAcquireError::Capacity)
    );
    assert_eq!(a.stats().active_leases, 2);
    drop(first);
    drop(second);
    b.close();
    let group = || {
        let first = a.try_lease_owned()?;
        let second = b.try_lease_owned()?;
        Ok::<_, ManagedAcquireError>((first, second))
    };
    assert_eq!(group().err(), Some(ManagedAcquireError::Closed));
    assert_eq!(a.stats().active_leases, 0);
    let all = (a.try_lease_owned().unwrap(), a.try_lease_owned().unwrap());
    drop(all);
}

#[test]
fn zero_count_disables_payload_caching() {
    let pool = ManagedPool::<Vec<u8>>::new(CacheLimits::new(usize::MAX, 0));
    let mut value = pool.try_lease_owned().unwrap();
    value.resize(128, 2);
    drop(value);
    assert_eq!(pool.stats().cached_values, 0);
    assert_eq!(pool.try_lease_owned().unwrap().capacity(), 0);
}

#[test]
fn panicking_cached_destructor_does_not_restore_idle_ownership() {
    #[derive(Default)]
    struct Value {
        panic_drop: bool,
    }
    impl Clear for Value {
        fn clear(&mut self) {}
    }
    impl Recyclable for Value {
        fn retained_bytes(&self) -> usize {
            0
        }
    }
    impl Drop for Value {
        fn drop(&mut self) {
            if self.panic_drop {
                self.panic_drop = false;
                panic!("injected cached destructor failure");
            }
        }
    }
    for close in [false, true].iter().copied() {
        let pool = ManagedPool::<Value>::new(CacheLimits::new(0, 2));
        let mut value = pool.try_lease_owned().unwrap();
        value.panic_drop = true;
        drop(value);
        assert_eq!(pool.stats().cached_values, 1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if close {
                pool.close()
            } else {
                pool.trim()
            }
        }));
        assert!(result.is_err());
        assert_eq!(pool.stats().cached_values, 0);
        assert_eq!(pool.stats().active_leases, 0);
        assert_eq!(pool.stats().closed, close);
        if !close {
            assert!(pool.try_lease_owned().is_ok());
        }
    }
}

#[test]
fn concurrent_returns_and_trim_keep_budget_and_clear_counts() {
    let pool = Arc::new(ManagedPool::<Watched>::new(CacheLimits::new(512, 4)));
    let cleared = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::new();
    for worker in 0..4u8 {
        let pool = pool.clone();
        let count = cleared.clone();
        workers.push(std::thread::spawn(move || {
            for index in 0..128u8 {
                let mut value = pool.try_lease_owned().unwrap();
                assert!(value.data.is_empty());
                value.data.extend_from_slice(&[worker, index]);
                value.data.resize(128, worker);
                value.cleared = Some(count.clone());
                let root = value.freeze();
                let reader = root.clone();
                drop(root);
                assert_eq!(&reader.data[..2], &[worker, index]);
                drop(reader);
                let stats = pool.stats();
                assert!(stats.cached_bytes <= 512);
                assert!(stats.cached_values <= 4);
            }
        }));
    }
    let trim_pool = pool.clone();
    let trimmer = std::thread::spawn(move || {
        for _ in 0..64 {
            trim_pool.trim();
            std::thread::yield_now();
        }
    });
    for worker in workers {
        worker.join().unwrap();
    }
    trimmer.join().unwrap();
    assert_eq!(cleared.load(Ordering::SeqCst), 512);
    assert_eq!(pool.stats().active_leases, 0);
    pool.close();
    assert_eq!(pool.stats().cached_bytes, 0);
}
