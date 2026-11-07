use sharded_slab::{pool::OwnedLease, Clear, Config, Pool};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

struct TwoSlots;
impl Config for TwoSlots {
    const INITIAL_PAGE_SIZE: usize = 2;
    const MAX_PAGES: usize = 1;
}

#[derive(Clone, Copy)]
enum Strategy {
    Collect,
    Reserved,
}

fn checkout<T: Clear + Default, C: Config>(
    pool: &Arc<Pool<T, C>>,
    count: usize,
    strategy: Strategy,
) -> Option<Vec<OwnedLease<T, C>>> {
    match strategy {
        Strategy::Collect => (0..count).map(|_| pool.clone().lease_owned()).collect(),
        Strategy::Reserved => {
            let mut leases = Vec::with_capacity(count);
            for _ in 0..count {
                leases.push(pool.clone().lease_owned()?);
            }
            Some(leases)
        }
    }
}

#[test]
fn partial_capacity_failure_clears_all_acquired_slots_and_recovers() {
    static CLEARS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Default)]
    struct Counted;
    impl Clear for Counted {
        fn clear(&mut self) {
            CLEARS.fetch_add(1, Ordering::SeqCst);
        }
    }
    for strategy in [Strategy::Collect, Strategy::Reserved].iter().copied() {
        let pool = Arc::new(Pool::<Counted>::new_with_config::<TwoSlots>());
        let before = CLEARS.load(Ordering::SeqCst);
        for attempt in 0..20 {
            assert!(checkout(&pool, 3, strategy).is_none());
            assert_eq!(CLEARS.load(Ordering::SeqCst), before + (attempt + 1) * 2);
        }
        let all = checkout(&pool, 2, strategy).unwrap();
        assert!(checkout(&pool, 1, strategy).is_none());
        drop(all);
        assert_eq!(CLEARS.load(Ordering::SeqCst), before + 42);
    }
}

#[test]
fn partial_failure_does_not_clear_an_existing_shared_value() {
    for strategy in [Strategy::Collect, Strategy::Reserved].iter().copied() {
        let pool = Arc::new(Pool::<Vec<usize>>::new_with_config::<TwoSlots>());
        let mut value = pool.clone().lease_owned().unwrap();
        value.push(43);
        let reader = value.freeze();
        assert!(checkout(&pool, 2, strategy).is_none());
        let later_reader = reader.clone();
        let remaining = checkout(&pool, 1, strategy).unwrap();
        assert!(remaining[0].is_empty());
        assert_eq!(&**later_reader, &[43]);
        drop(remaining);
        drop(reader);
        assert_eq!(&**later_reader, &[43]);
        drop(later_reader);
        assert_eq!(checkout(&pool, 2, strategy).unwrap().len(), 2);
    }
}

#[test]
fn second_typed_pool_failure_releases_first_pool_batch() {
    type NumberBatch = Vec<OwnedLease<Vec<usize>, TwoSlots>>;
    type TextBatch = Vec<OwnedLease<String, TwoSlots>>;
    fn pair(
        a: &Arc<Pool<Vec<usize>, TwoSlots>>,
        b: &Arc<Pool<String, TwoSlots>>,
        strategy: Strategy,
    ) -> Option<(NumberBatch, TextBatch)> {
        let first = checkout(a, 2, strategy)?;
        let second = checkout(b, 1, strategy)?;
        Some((first, second))
    }
    for strategy in [Strategy::Collect, Strategy::Reserved].iter().copied() {
        let a = Arc::new(Pool::<Vec<usize>>::new_with_config::<TwoSlots>());
        let b = Arc::new(Pool::<String>::new_with_config::<TwoSlots>());
        let occupied = checkout(&b, 2, strategy).unwrap();
        assert!(pair(&a, &b, strategy).is_none());
        let recovered = checkout(&a, 2, strategy).unwrap();
        assert!(recovered.iter().all(|v| v.is_empty()));
        drop(recovered);
        drop(occupied);
        assert!(pair(&a, &b, strategy).is_some());
    }
}

#[test]
fn empty_batch_does_not_construct_any_value() {
    struct NoDefault;
    impl Default for NoDefault {
        fn default() -> Self {
            panic!("empty batch must not allocate a slot");
        }
    }
    impl Clear for NoDefault {
        fn clear(&mut self) {}
    }
    let pool = Arc::new(Pool::<NoDefault>::new());
    for strategy in [Strategy::Collect, Strategy::Reserved].iter().copied() {
        assert!(checkout(&pool, 0, strategy).unwrap().is_empty());
    }
}

#[derive(Default)]
struct Buffer {
    values: Vec<usize>,
    cleared: Option<Arc<AtomicUsize>>,
}
impl Clear for Buffer {
    fn clear(&mut self) {
        if let Some(counter) = self.cleared.take() {
            counter.fetch_add(1, Ordering::SeqCst);
        }
        self.values.clear();
    }
}

#[test]
fn batch_can_move_and_freeze_after_creator_exits() {
    for strategy in [Strategy::Collect, Strategy::Reserved].iter().copied() {
        let pool = Arc::new(Pool::<Buffer>::new_with_config::<TwoSlots>());
        let count = Arc::new(AtomicUsize::new(0));
        let creator_pool = pool.clone();
        let observer = count.clone();
        let leases = std::thread::spawn(move || {
            let mut leases = checkout(&creator_pool, 2, strategy).unwrap();
            for (index, value) in leases.iter_mut().enumerate() {
                value.values.push(index);
                value.cleared = Some(observer.clone());
            }
            leases
        })
        .join()
        .unwrap();
        let shared = std::thread::spawn(move || {
            leases
                .into_iter()
                .map(|value| value.freeze())
                .collect::<Vec<_>>()
        })
        .join()
        .unwrap();
        drop(pool);
        assert_eq!(count.load(Ordering::SeqCst), 0);
        let later = shared.clone();
        drop(shared);
        std::thread::spawn(move || {
            for (index, value) in later.iter().enumerate() {
                assert_eq!(value.values, [index]);
            }
            drop(later);
        })
        .join()
        .unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn initializer_unwind_releases_current_and_earlier_leases() {
    let pool = Arc::new(Pool::<Buffer>::new_with_config::<TwoSlots>());
    let count = Arc::new(AtomicUsize::new(0));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut acquired = Vec::new();
        for index in 0..2 {
            let mut value = pool.clone().lease_owned().unwrap();
            value.cleared = Some(count.clone());
            value.values.push(index);
            if index == 1 {
                panic!("injected private initialization failure");
            }
            acquired.push(value);
        }
    }));
    assert!(result.is_err());
    assert_eq!(count.load(Ordering::SeqCst), 2);
    assert!(checkout(&pool, 2, Strategy::Reserved).is_some());
}

#[test]
fn rollback_clear_panic_is_not_converted_to_capacity_none() {
    static PANIC_ONCE: AtomicBool = AtomicBool::new(true);
    static CLEARS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Default)]
    struct PanicsOnce;
    impl Clear for PanicsOnce {
        fn clear(&mut self) {
            CLEARS.fetch_add(1, Ordering::SeqCst);
            if PANIC_ONCE.swap(false, Ordering::SeqCst) {
                panic!("injected rollback clear failure");
            }
        }
    }
    for strategy in [Strategy::Collect, Strategy::Reserved].iter().copied() {
        PANIC_ONCE.store(true, Ordering::SeqCst);
        let before = CLEARS.load(Ordering::SeqCst);
        let pool = Arc::new(Pool::<PanicsOnce>::new_with_config::<TwoSlots>());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            checkout(&pool, 3, strategy)
        }));
        assert!(result.is_err());
        assert_eq!(CLEARS.load(Ordering::SeqCst), before + 2);
        assert!(checkout(&pool, 2, strategy).is_none());
        // 只有成功清理的槽可再发，另一个仍隔离，不把panic当正常None。
        assert!(checkout(&pool, 1, strategy).is_some());
        assert_eq!(CLEARS.load(Ordering::SeqCst), before + 4);
    }
}
