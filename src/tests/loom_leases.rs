use super::util::run_model;
use crate::{Clear, Pool};
use loom::{
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};
use std::sync::Arc;

struct TwoSlots;
impl crate::Config for TwoSlots {
    const INITIAL_PAGE_SIZE: usize = 2;
    const MAX_PAGES: usize = 1;
}

#[test]
fn failed_batch_keeps_existing_reader_and_returns_partial_slots() {
    run_model(
        "failed_batch_keeps_existing_reader_and_returns_partial_slots",
        || {
            let pool = Arc::new(Pool::<Buffer>::new_with_config::<TwoSlots>());
            let count = Arc::new(AtomicUsize::new(0));
            let mut value = pool.clone().lease_owned().unwrap();
            value.data.push(47);
            value.cleared = Some(count.clone());
            let root = value.freeze();
            let reader = root.clone();
            let worker = thread::spawn(move || {
                assert_eq!(reader.data, [47]);
                drop(reader);
            });
            let failed: Option<Vec<_>> = (0..2).map(|_| pool.clone().lease_owned()).collect();
            assert!(failed.is_none());
            let free = pool.clone().lease_owned().unwrap();
            assert!(free.data.is_empty());
            assert_eq!(count.load(Ordering::SeqCst), 0);
            drop(free);
            worker.join().unwrap();
            assert_eq!(root.data, [47]);
            drop(root);
            assert_eq!(count.load(Ordering::SeqCst), 1);
            let all: Option<Vec<_>> = (0..2).map(|_| pool.clone().lease_owned()).collect();
            assert_eq!(all.unwrap().len(), 2);
        },
    );
}

#[test]
fn complete_batch_can_be_transferred_and_frozen_on_another_thread() {
    run_model(
        "complete_batch_can_be_transferred_and_frozen_on_another_thread",
        || {
            let pool = Arc::new(Pool::<Buffer>::new_with_config::<TwoSlots>());
            let count = Arc::new(AtomicUsize::new(0));
            let batch: Option<Vec<_>> = (0..2).map(|_| pool.clone().lease_owned()).collect();
            let mut batch = batch.unwrap();
            for (index, value) in batch.iter_mut().enumerate() {
                value.data.push(index);
                value.cleared = Some(count.clone());
            }
            let roots = thread::spawn(move || {
                batch
                    .into_iter()
                    .map(|value| value.freeze())
                    .collect::<Vec<_>>()
            })
            .join()
            .unwrap();
            drop(pool);
            assert_eq!(count.load(Ordering::SeqCst), 0);
            thread::spawn(move || {
                for (index, value) in roots.iter().enumerate() {
                    assert_eq!(value.data, [index]);
                }
                drop(roots);
            })
            .join()
            .unwrap();
            assert_eq!(count.load(Ordering::SeqCst), 2);
        },
    );
}

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
