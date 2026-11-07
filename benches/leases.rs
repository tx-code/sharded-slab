use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sharded_slab::{pool::OwnedLease, Pool};
use std::{
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};

fn reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("lease_reuse");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    for &size in &[64usize, 1024] {
        let input = vec![7u64; size];
        let pool = Arc::new(Pool::<Vec<u64>>::new());
        group.bench_with_input(BenchmarkId::new("owned", size), &size, |b, _| {
            b.iter(|| {
                let mut value = pool.clone().lease_owned().unwrap();
                value.extend_from_slice(black_box(&input));
                black_box(value.len());
                drop(value);
            });
        });
        let pool = Arc::new(Pool::<Vec<u64>>::new());
        group.bench_with_input(BenchmarkId::new("manual_owned", size), &size, |b, _| {
            b.iter(|| {
                let mut value = pool.clone().create_owned().unwrap();
                value.extend_from_slice(black_box(&input));
                black_box(value.len());
                let key = value.key();
                drop(value);
                assert!(pool.clear(key));
            });
        });
        let mut direct = Vec::with_capacity(size);
        group.bench_with_input(
            BenchmarkId::new("exclusive_vec_floor", size),
            &size,
            |b, _| {
                b.iter(|| {
                    direct.extend_from_slice(black_box(&input));
                    black_box(direct.len());
                    direct.clear();
                });
            },
        );
    }
    group.finish();
}

fn sharing(c: &mut Criterion) {
    let mut group = c.benchmark_group("lease_sharing");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    let pool = Arc::new(Pool::<Vec<u64>>::new());
    group.bench_function("freeze_two_readers", |b| {
        b.iter(|| {
            let mut value = pool.clone().lease_owned().unwrap();
            value.extend_from_slice(&[1, 2, 3, 4]);
            let first = value.freeze();
            let second = first.clone();
            black_box(second.len());
            drop(first);
            drop(second);
        })
    });
    let pool = Arc::new(Pool::<Vec<u64>>::new());
    group.bench_function("manual_two_readers", |b| {
        b.iter(|| {
            let mut value = pool.clone().create_owned().unwrap();
            value.extend_from_slice(&[1, 2, 3, 4]);
            let key = value.key();
            let first = Arc::new(value.downgrade());
            let second = first.clone();
            black_box(second.len());
            drop(first);
            assert!(pool.clear(key));
            drop(second);
        })
    });
    group.bench_function("cold_checkout_drop", |b| {
        b.iter(|| {
            let pool = Arc::new(Pool::<Vec<u64>>::new());
            let mut value = pool.lease_owned().unwrap();
            value.extend_from_slice(&[1, 2, 3, 4]);
            black_box(value.len());
            drop(value);
        })
    });
    group.finish();
}

fn handoff(c: &mut Criterion) {
    let mut group = c.benchmark_group("lease_handoff");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    let pool = Arc::new(Pool::<Vec<u64>>::new());
    let (tx, rx) = mpsc::channel::<OwnedLease<Vec<u64>>>();
    let (done, ack) = mpsc::channel();
    let worker = thread::spawn(move || {
        for value in rx {
            black_box(value.len());
            drop(value);
            done.send(()).unwrap();
        }
    });
    group.bench_function("owned_remote_return", |b| {
        b.iter(|| {
            let mut value = pool.clone().lease_owned().unwrap();
            value.extend(0..64);
            tx.send(value).unwrap();
            ack.recv().unwrap();
        })
    });
    drop(tx);
    worker.join().unwrap();

    let pool = Arc::new(Pool::<Vec<u64>>::new());
    let worker_pool = pool.clone();
    let (tx, rx) = mpsc::channel::<sharded_slab::pool::OwnedRefMut<Vec<u64>>>();
    let (done, ack) = mpsc::channel();
    let worker = thread::spawn(move || {
        for value in rx {
            black_box(value.len());
            let key = value.key();
            drop(value);
            assert!(worker_pool.clear(key));
            done.send(()).unwrap();
        }
    });
    group.bench_function("manual_remote_return", |b| {
        b.iter(|| {
            let mut value = pool.clone().create_owned().unwrap();
            value.extend(0..64);
            tx.send(value).unwrap();
            ack.recv().unwrap();
        })
    });
    drop(tx);
    worker.join().unwrap();
    group.finish();
}

fn parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("lease_parallel");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    for &workers in &[1usize, 4] {
        let pool = Arc::new(Pool::<Vec<u64>>::new());
        let mut commands = Vec::new();
        let mut threads = Vec::new();
        for _ in 0..workers {
            let pool = pool.clone();
            let (tx, rx) = mpsc::channel::<()>();
            let (done, ack) = mpsc::channel();
            threads.push(thread::spawn(move || {
                for () in rx {
                    for _ in 0..64 {
                        let mut value = pool.clone().lease_owned().unwrap();
                        value.extend(0..64);
                        black_box(value.len());
                    }
                    done.send(()).unwrap();
                }
            }));
            commands.push((tx, ack));
        }
        group.bench_with_input(
            BenchmarkId::new("batch_64_per_worker", workers),
            &workers,
            |b, _| {
                b.iter(|| {
                    for (tx, _) in &commands {
                        tx.send(()).unwrap();
                    }
                    for (_, ack) in &commands {
                        ack.recv().unwrap();
                    }
                })
            },
        );
        drop(commands);
        for worker in threads {
            worker.join().unwrap();
        }
    }
    group.finish();
}

criterion_group!(benches, reuse, sharing, handoff, parallel);
criterion_main!(benches);
