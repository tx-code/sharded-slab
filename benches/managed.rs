use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use criterion::{measurement::WallTime, BenchmarkGroup};
use sharded_slab::{
    pool::{CacheLimits, ManagedLease, ManagedPool},
    Pool,
};
use std::{
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};

fn warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("managed_reuse");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    for &size in &[64usize, 1024] {
        let input = vec![3u64; size];
        let native = Arc::new(Pool::<Vec<u64>>::new());
        group.bench_with_input(BenchmarkId::new("native_owned", size), &size, |b, _| {
            b.iter(|| {
                let mut value = native.clone().lease_owned().unwrap();
                value.extend_from_slice(black_box(&input));
                black_box(&value[..]);
                drop(value);
            })
        });
        for &cached in &[true, false] {
            let pool = ManagedPool::<Vec<u64>>::new(CacheLimits::new(
                size * 8 * 8,
                if cached { 8 } else { 0 },
            ));
            group.bench_with_input(
                BenchmarkId::new(if cached { "cached" } else { "discarded" }, size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let mut value = pool.try_lease_owned().unwrap();
                        value.extend_from_slice(black_box(&input));
                        black_box(&value[..]);
                        drop(value);
                    })
                },
            );
            black_box(pool.stats());
        }
    }
    group.finish();
}

fn lifecycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("managed_lifecycle");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    let pool = ManagedPool::<Vec<u64>>::new(CacheLimits::new(8192, 8));
    group.bench_function("freeze_two_readers", |b| {
        b.iter(|| {
            let mut value = pool.try_lease_owned().unwrap();
            value.extend_from_slice(black_box(&[3u64; 64]));
            let root = value.freeze();
            let reader = root.clone();
            black_box(&reader[..]);
            drop(root);
            drop(reader);
        })
    });
    let native = Arc::new(Pool::<Vec<u64>>::new());
    group.bench_function("native_freeze_two_readers", |b| {
        b.iter(|| {
            let mut value = native.clone().lease_owned().unwrap();
            value.extend_from_slice(black_box(&[3u64; 64]));
            let root = value.freeze();
            let reader = root.clone();
            black_box(&reader[..]);
            drop(root);
            drop(reader);
        })
    });
    let pool = ManagedPool::<Vec<u64>>::new(CacheLimits::new(8192, 8));
    group.bench_function("fill_four_and_trim", |b| {
        b.iter(|| {
            let mut values = Vec::with_capacity(4);
            for _ in 0..4 {
                let mut value = pool.try_lease_owned().unwrap();
                value.extend_from_slice(black_box(&[3u64; 64]));
                black_box(&value[..]);
                values.push(value);
            }
            drop(values);
            black_box(pool.trim());
        })
    });
    group.bench_function("cold_close_with_live_reader", |b| {
        b.iter(|| {
            let pool = ManagedPool::<Vec<u64>>::new(CacheLimits::new(8192, 8));
            let mut active = pool.try_lease_owned().unwrap();
            active.extend_from_slice(black_box(&[3u64; 64]));
            let reader = active.freeze();
            let mut idle = pool.try_lease_owned().unwrap();
            idle.extend_from_slice(black_box(&[5u64; 64]));
            drop(idle);
            black_box(pool.close());
            black_box(&reader[..]);
            drop(reader);
            drop(pool);
        })
    });
    group.finish();
}

fn handoff(c: &mut Criterion) {
    let mut group = c.benchmark_group("managed_handoff");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    let pool = Arc::new(ManagedPool::<Vec<u64>>::new(CacheLimits::new(8192, 8)));
    let (tx, rx) = mpsc::channel::<ManagedLease<Vec<u64>>>();
    let (done, ack) = mpsc::channel();
    let worker = thread::spawn(move || {
        for value in rx {
            black_box(&value[..]);
            drop(value);
            done.send(()).unwrap();
        }
    });
    group.bench_function("cached_remote_return", |b| {
        b.iter(|| {
            let mut value = pool.try_lease_owned().unwrap();
            value.extend_from_slice(black_box(&[3u64; 64]));
            tx.send(value).unwrap();
            ack.recv().unwrap();
        })
    });
    drop(tx);
    worker.join().unwrap();

    let native = Arc::new(Pool::<Vec<u64>>::new());
    let (tx, rx) = mpsc::channel::<sharded_slab::pool::OwnedLease<Vec<u64>>>();
    let (done, ack) = mpsc::channel();
    let worker = thread::spawn(move || {
        for value in rx {
            black_box(&value[..]);
            drop(value);
            done.send(()).unwrap();
        }
    });
    group.bench_function("native_remote_return", |b| {
        b.iter(|| {
            let mut value = native.clone().lease_owned().unwrap();
            value.extend_from_slice(black_box(&[3u64; 64]));
            tx.send(value).unwrap();
            ack.recv().unwrap();
        })
    });
    drop(tx);
    worker.join().unwrap();
    group.finish();
}

fn parallel_case<F: Fn(usize) + Send + Sync + 'static>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    workers: usize,
    action: F,
) {
    let action = Arc::new(action);
    let mut commands = Vec::new();
    let mut threads = Vec::new();
    for index in 0..workers {
        let action = action.clone();
        let (tx, rx) = mpsc::channel::<()>();
        let (done, ack) = mpsc::channel();
        threads.push(thread::spawn(move || {
            for () in rx {
                for _ in 0..64 {
                    action(index);
                }
                done.send(()).unwrap();
            }
        }));
        commands.push((tx, ack));
    }
    group.bench_with_input(BenchmarkId::new(name, workers), &workers, |b, _| {
        b.iter(|| {
            for (tx, _) in &commands {
                tx.send(()).unwrap();
            }
            for (_, ack) in &commands {
                ack.recv().unwrap();
            }
        })
    });
    drop(commands);
    for worker in threads {
        worker.join().unwrap();
    }
}

fn parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("managed_parallel");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    for &workers in &[1usize, 4] {
        let pool = Arc::new(ManagedPool::<Vec<u64>>::new(CacheLimits::new(65536, 16)));
        parallel_case(
            &mut group,
            "cached_batch64_per_worker",
            workers,
            move |_| {
                let mut value = pool.try_lease_owned().unwrap();
                value.extend_from_slice(black_box(&[3u64; 64]));
                black_box(&value[..]);
                drop(value);
            },
        );
        let native = Arc::new(Pool::<Vec<u64>>::new());
        parallel_case(
            &mut group,
            "native_batch64_per_worker",
            workers,
            move |_| {
                let mut value = native.clone().lease_owned().unwrap();
                value.extend_from_slice(black_box(&[3u64; 64]));
                black_box(&value[..]);
                drop(value);
            },
        );
        // 总idle上限不变；仅把控制器分开，用于区分共享元数据与载荷工作。
        let pools: Vec<_> = (0..workers)
            .map(|_| ManagedPool::<Vec<u64>>::new(CacheLimits::new(65536 / workers, 16 / workers)))
            .collect();
        parallel_case(
            &mut group,
            "isolated_cached_batch64_per_worker",
            workers,
            move |index| {
                let mut value = pools[index].try_lease_owned().unwrap();
                value.extend_from_slice(black_box(&[3u64; 64]));
                black_box(&value[..]);
                drop(value);
            },
        );
    }
    group.finish();
}

criterion_group!(benches, warm, lifecycle, handoff, parallel);
criterion_main!(benches);
