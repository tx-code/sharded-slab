use sharded_slab::{Clear, Pool};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Default)]
struct Buffer {
    values: Vec<usize>,
    cleared: Option<Arc<AtomicUsize>>,
    panic_on_clear: bool,
}

impl Clear for Buffer {
    fn clear(&mut self) {
        if let Some(count) = self.cleared.take() {
            count.fetch_add(1, Ordering::SeqCst);
        }
        if self.panic_on_clear {
            self.panic_on_clear = false;
            panic!("injected clear failure");
        }
        self.values.clear();
    }
}

#[test]
fn exclusive_drop_clears_once_and_reuses_capacity() {
    let pool = Arc::new(Pool::<Buffer>::new());
    let count = Arc::new(AtomicUsize::new(0));
    let mut capacity = 0;
    for i in 0..20 {
        let mut value = pool.clone().lease_owned().unwrap();
        assert!(value.values.is_empty());
        assert!(value.values.capacity() >= capacity);
        value.values.extend(0..128);
        value.cleared = Some(count.clone());
        capacity = value.values.capacity();
        drop(value);
        assert_eq!(count.load(Ordering::SeqCst), i + 1);
    }
}

#[test]
fn frozen_readers_outlive_pool_handle_and_clear_only_after_last_clone() {
    let pool = Arc::new(Pool::<Buffer>::new());
    let count = Arc::new(AtomicUsize::new(0));
    let mut value = pool.clone().lease_owned().unwrap();
    value.values.extend_from_slice(&[3, 5, 7]);
    value.cleared = Some(count.clone());
    let reader = value.freeze();
    let second = reader.clone();
    drop(pool);
    drop(reader);
    let third = second.clone();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(second);
    assert_eq!(third.values, [3, 5, 7]);
    drop(third);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn creator_exit_and_cross_thread_transfer_preserve_owned_storage() {
    let pool = Arc::new(Pool::<Buffer>::new());
    let count = Arc::new(AtomicUsize::new(0));
    let creator_pool = pool.clone();
    let observer = count.clone();
    let value = std::thread::spawn(move || {
        let mut value = creator_pool.lease_owned().unwrap();
        value.values.push(11);
        value.cleared = Some(observer);
        value
    })
    .join()
    .unwrap();
    let reader = std::thread::spawn(move || {
        let mut value = value;
        value.values.push(13);
        value.freeze()
    })
    .join()
    .unwrap();
    drop(pool);
    std::thread::spawn(move || {
        assert_eq!(reader.values, [11, 13]);
        drop(reader);
    })
    .join()
    .unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn unwind_discards_exclusive_storage() {
    let pool = Arc::new(Pool::<Buffer>::new());
    let count = Arc::new(AtomicUsize::new(0));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut value = pool.clone().lease_owned().unwrap();
        value.values.push(17);
        value.cleared = Some(count.clone());
        panic!("injected caller failure");
    }));
    assert!(result.is_err());
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(pool.lease_owned().unwrap().values.is_empty());
}

#[test]
fn panicking_clear_does_not_requeue_dirty_storage() {
    let pool = Arc::new(Pool::<Buffer>::new());
    let count = Arc::new(AtomicUsize::new(0));
    let mut value = pool.clone().lease_owned().unwrap();
    value.cleared = Some(count.clone());
    value.values.push(19);
    value.panic_on_clear = true;
    let dirty_address = &*value as *const Buffer;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(value)));
    assert!(result.is_err());
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let clean = pool.lease_owned().unwrap();
    assert_ne!(&*clean as *const Buffer, dirty_address);
    assert!(clean.values.is_empty());
}

#[test]
fn concurrent_producers_can_leave_shared_results_after_exiting() {
    let pool = Arc::new(Pool::<Buffer>::new());
    let count = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::new();
    for worker in 0..4 {
        let pool = pool.clone();
        let count = count.clone();
        workers.push(std::thread::spawn(move || {
            let mut output = Vec::new();
            for record in 0..32 {
                let mut value = pool.clone().lease_owned().unwrap();
                value.values.extend_from_slice(&[worker, record]);
                value.cleared = Some(count.clone());
                output.push(value.freeze());
            }
            output
        }));
    }
    let mut results = Vec::new();
    for worker in workers {
        results.extend(worker.join().unwrap());
    }
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(pool);
    for result in results {
        let reader = result.clone();
        drop(result);
        assert_eq!(reader.values.len(), 2);
    }
    assert_eq!(count.load(Ordering::SeqCst), 128);
}
