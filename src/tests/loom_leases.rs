use super::util::run_model;
use crate::{Clear, Pool};
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

#[test]
fn exclusive_remote_return() {
    run_model("exclusive_remote_return", || {
        let pool = Arc::new(Pool::<Buffer>::new());
        let count = Arc::new(AtomicUsize::new(0));
        let mut value = pool.clone().lease_owned().unwrap();
        value.data.push(31);
        value.cleared = Some(count.clone());
        thread::spawn(move || {
            assert_eq!(value.data, [31]);
            drop(value);
        })
        .join()
        .unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);
        let next = pool.lease_owned().unwrap();
        assert!(next.data.is_empty());
    });
}

#[test]
fn shared_readers_outlive_external_pool() {
    run_model("shared_readers_outlive_external_pool", || {
        let pool = Arc::new(Pool::<Buffer>::new());
        let count = Arc::new(AtomicUsize::new(0));
        let mut value = pool.clone().lease_owned().unwrap();
        value.data.push(37);
        value.cleared = Some(count.clone());
        let first = value.freeze();
        let second = first.clone();
        drop(pool);
        let a = thread::spawn(move || {
            assert_eq!(first.data, [37]);
            drop(first);
        });
        let b = thread::spawn(move || {
            assert_eq!(second.data, [37]);
            drop(second);
        });
        a.join().unwrap();
        b.join().unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn retained_root_can_create_reader_after_other_reader_exits() {
    run_model(
        "retained_root_can_create_reader_after_other_reader_exits",
        || {
            let pool = Arc::new(Pool::<Buffer>::new());
            let count = Arc::new(AtomicUsize::new(0));
            let mut value = pool.lease_owned().unwrap();
            value.data.push(41);
            value.cleared = Some(count.clone());
            let root = value.freeze();
            let first = root.clone();
            thread::spawn(move || drop(first)).join().unwrap();
            assert_eq!(count.load(Ordering::SeqCst), 0);
            let later = root.clone();
            drop(root);
            assert_eq!(later.data, [41]);
            drop(later);
            assert_eq!(count.load(Ordering::SeqCst), 1);
        },
    );
}
