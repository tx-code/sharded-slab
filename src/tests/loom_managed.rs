use super::util::run_model;
use crate::{
    pool::{CacheLimits, ManagedAcquireError, ManagedPool, Recyclable},
    Clear,
};
use loom::{
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};
use std::sync::Arc;

#[derive(Default)]
struct Buffer {
    data: Vec<usize>,
    cleared: Option<Arc<AtomicUsize>>,
}
impl Clear for Buffer {
    fn clear(&mut self) {
        if let Some(counter) = self.cleared.take() {
            counter.fetch_add(1, Ordering::SeqCst);
        }
        self.data.clear();
    }
}
impl Recyclable for Buffer {
    fn retained_bytes(&self) -> usize {
        self.data
            .capacity()
            .saturating_mul(std::mem::size_of::<usize>())
    }
}

#[test]
fn close_races_last_remote_reader() {
    run_model("close_races_last_remote_reader", || {
        let pool = Arc::new(ManagedPool::<Buffer>::new(CacheLimits::new(1024, 2)));
        let count = Arc::new(AtomicUsize::new(0));
        let mut value = pool.try_lease_owned().unwrap();
        value.data.push(59);
        value.cleared = Some(count.clone());
        let reader = value.freeze();
        let closer = pool.clone();
        let a = thread::spawn(move || {
            closer.close();
        });
        let b = thread::spawn(move || {
            assert_eq!(reader.data, [59]);
            drop(reader);
        });
        a.join().unwrap();
        b.join().unwrap();
        let stats = pool.stats();
        assert!(stats.closed);
        assert_eq!(stats.cached_values, 0);
        assert_eq!(stats.active_leases, 0);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn reused_admission_returns_once_when_close_races_remote_readers() {
    run_model(
        "reused_admission_returns_once_when_close_races_remote_readers",
        || {
            let pool = Arc::new(ManagedPool::<Buffer>::new(CacheLimits::new(1024, 2)));
            let count = Arc::new(AtomicUsize::new(0));
            for _ in 0..2 {
                let mut value = pool.try_lease_owned().unwrap();
                value.data.push(71);
                value.cleared = Some(count.clone());
                drop(value);
            }
            let mut value = pool.try_lease_owned().unwrap();
            value.data.push(73);
            value.cleared = Some(count.clone());
            let root = value.freeze();
            let reader = root.clone();
            let worker = thread::spawn(move || {
                assert_eq!(reader.data, [73]);
                drop(reader);
            });
            let closer = pool.clone();
            let close = thread::spawn(move || {
                closer.close();
            });
            drop(root);
            worker.join().unwrap();
            close.join().unwrap();
            assert_eq!(count.load(Ordering::SeqCst), 3);
            assert_eq!(pool.stats().active_leases, 0);
            assert_eq!(pool.stats().cached_values, 0);
        },
    );
}

#[test]
fn concurrent_returns_cannot_exceed_idle_budget() {
    run_model("concurrent_returns_cannot_exceed_idle_budget", || {
        let pool = Arc::new(ManagedPool::<Buffer>::new(CacheLimits::new(64, 1)));
        let count = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        for index in 0..2 {
            let pool = pool.clone();
            let count = count.clone();
            threads.push(thread::spawn(move || {
                let mut value = pool.try_lease_owned().unwrap();
                value.data.resize(8, index);
                value.cleared = Some(count);
                drop(value);
                let stats = pool.stats();
                assert!(stats.cached_bytes <= 64);
                assert!(stats.cached_values <= 1);
            }));
        }
        for worker in threads {
            worker.join().unwrap();
        }
        assert_eq!(count.load(Ordering::SeqCst), 2);
        assert_eq!(pool.stats().active_leases, 0);
        assert!(pool.stats().cached_bytes <= 64);
    });
}

#[test]
fn admission_races_close_without_invalidating_admitted_value() {
    run_model(
        "admission_races_close_without_invalidating_admitted_value",
        || {
            let pool = Arc::new(ManagedPool::<Vec<usize>>::new(CacheLimits::new(1024, 2)));
            let borrower = pool.clone();
            let a = thread::spawn(move || match borrower.try_lease_owned() {
                Ok(mut value) => {
                    value.push(61);
                    assert_eq!(&**value, &[61]);
                    drop(value);
                }
                Err(error) => assert_eq!(error, ManagedAcquireError::Closed),
            });
            let closer = pool.clone();
            let b = thread::spawn(move || {
                closer.close();
            });
            a.join().unwrap();
            b.join().unwrap();
            assert_eq!(
                pool.try_lease_owned().err(),
                Some(ManagedAcquireError::Closed)
            );
            assert_eq!(pool.stats().active_leases, 0);
            assert_eq!(pool.stats().cached_bytes, 0);
        },
    );
}

#[test]
fn trim_does_not_clear_a_retained_read_version() {
    run_model("trim_does_not_clear_a_retained_read_version", || {
        let pool = Arc::new(ManagedPool::<Buffer>::new(CacheLimits::new(1024, 2)));
        let count = Arc::new(AtomicUsize::new(0));
        let mut active = pool.try_lease_owned().unwrap();
        active.data.push(67);
        active.cleared = Some(count.clone());
        let root = active.freeze();
        let reader = root.clone();
        let mut idle = pool.try_lease_owned().unwrap();
        idle.data.resize(8, 0);
        drop(idle);
        let trimmer = pool.clone();
        let a = thread::spawn(move || assert_eq!(trimmer.trim().values, 1));
        let b = thread::spawn(move || {
            assert_eq!(reader.data, [67]);
            drop(reader);
        });
        a.join().unwrap();
        b.join().unwrap();
        assert_eq!(root.data, [67]);
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert_eq!(pool.stats().cached_values, 0);
        drop(root);
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(pool.stats().active_leases, 0);
    });
}
