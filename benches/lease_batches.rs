use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sharded_slab::{pool::OwnedLease, Clear, Config, Pool};
use std::{
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};

#[derive(Clone, Copy)]
enum Strategy {
    Collect,
    Reserved,
}
impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Self::Collect => "collect",
            Self::Reserved => "reserved_loop",
        }
    }
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

fn successful_batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("lease_batch_success");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    for &count in &[1usize, 8, 32] {
        for strategy in [Strategy::Collect, Strategy::Reserved].iter().copied() {
            let pool = Arc::new(Pool::<Vec<u64>>::new());
            group.bench_with_input(
                BenchmarkId::new(strategy.name(), count),
                &count,
                |b, &count| {
                    b.iter(|| {
                        let mut batch = checkout(&pool, count, strategy).unwrap();
                        for value in &mut batch {
                            value.extend_from_slice(black_box(&[3u64; 64]));
                            black_box(&**value);
                        }
                        black_box(batch.iter().map(|v| v.len()).sum::<usize>());
                        drop(batch);
                    })
                },
            );
        }
    }
    group.finish();
}

struct TwoSlots;
impl Config for TwoSlots {
    const INITIAL_PAGE_SIZE: usize = 2;
    const MAX_PAGES: usize = 1;
}

fn failed_batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("lease_batch_capacity_failure");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    for strategy in [Strategy::Collect, Strategy::Reserved].iter().copied() {
        let pool = Arc::new(Pool::<Vec<u64>>::new_with_config::<TwoSlots>());
        group.bench_function(strategy.name(), |b| {
            b.iter(|| {
                let failed = checkout(&pool, black_box(3), strategy);
                assert!(failed.is_none());
            })
        });
    }
    group.finish();
}

fn remote_batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("lease_batch_remote_return");
    group
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    for strategy in [Strategy::Collect, Strategy::Reserved].iter().copied() {
        let pool = Arc::new(Pool::<Vec<u64>>::new());
        let (tx, rx) = mpsc::channel::<Vec<OwnedLease<Vec<u64>>>>();
        let (done, ack) = mpsc::channel();
        let worker = thread::spawn(move || {
            for batch in rx {
                for value in &batch {
                    black_box(&**value);
                }
                drop(batch);
                done.send(()).unwrap();
            }
        });
        group.bench_function(strategy.name(), |b| {
            b.iter(|| {
                let mut batch = checkout(&pool, 8, strategy).unwrap();
                for value in &mut batch {
                    value.extend_from_slice(black_box(&[3u64; 64]));
                }
                tx.send(batch).unwrap();
                ack.recv().unwrap();
            })
        });
        drop(tx);
        worker.join().unwrap();
    }
    group.finish();
}

criterion_group!(benches, successful_batches, failed_batches, remote_batches);
criterion_main!(benches);
