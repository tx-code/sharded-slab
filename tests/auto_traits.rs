use sharded_slab::{pool, Clear, Entry, OwnedEntry, Pool, Slab};
use std::{cell::Cell, sync::MutexGuard};

// 编译期反例，不构造或跨线程执行具有线程绑定析构的值。
macro_rules! assert_not_impl {
    ($ty:ty: $bound:path) => {
        const _: fn() = || {
            trait AmbiguousIfImpl<A> {
                fn check() {}
            }
            impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
            struct Implemented;
            impl<T: ?Sized + $bound> AmbiguousIfImpl<Implemented> for T {}
            let _ = <$ty as AmbiguousIfImpl<_>>::check;
        };
    };
}

type ThreadBound = Option<MutexGuard<'static, ()>>;

#[derive(Default)]
struct SendOnly(Cell<usize>);
impl Clear for SendOnly {
    fn clear(&mut self) {
        self.0.set(0);
    }
}

assert_not_impl!(Slab<ThreadBound>: Sync);
assert_not_impl!(Pool<ThreadBound>: Sync);
assert_not_impl!(Entry<'static, ThreadBound>: Send);
assert_not_impl!(OwnedEntry<ThreadBound>: Send);
assert_not_impl!(pool::OwnedRef<ThreadBound>: Send);
assert_not_impl!(pool::OwnedRefMut<ThreadBound>: Send);
assert_not_impl!(pool::OwnedRefMut<SendOnly>: Send);
assert_not_impl!(Pool<SendOnly>: Sync);
assert_not_impl!(pool::OwnedLease<ThreadBound>: Send);
assert_not_impl!(pool::SharedLease<ThreadBound>: Send);
assert_not_impl!(pool::SharedLease<ThreadBound>: Sync);

#[derive(Default)]
struct Reported<T: Clear + Default>(T);
impl<T: Clear + Default> Clear for Reported<T> {
    fn clear(&mut self) {
        self.0.clear();
    }
}
impl<T: Clear + Default> pool::Recyclable for Reported<T> {
    fn retained_bytes(&self) -> usize {
        0
    }
}
assert_not_impl!(pool::ManagedPool<Reported<ThreadBound>>: Sync);
assert_not_impl!(pool::ManagedLease<Reported<ThreadBound>>: Send);
assert_not_impl!(pool::ManagedSharedLease<Reported<ThreadBound>>: Send);
assert_not_impl!(pool::ManagedSharedLease<Reported<ThreadBound>>: Sync);
assert_not_impl!(pool::ManagedPool<Reported<SendOnly>>: Sync);
assert_not_impl!(pool::ManagedLease<Reported<SendOnly>>: Send);
assert_not_impl!(pool::ManagedSharedLease<Reported<SendOnly>>: Send);
assert_not_impl!(pool::ManagedSharedLease<Reported<SendOnly>>: Sync);

#[test]
fn supported_auto_traits() {
    fn send<T: Send>() {}
    fn sync<T: Sync>() {}
    send::<Slab<Cell<usize>>>();
    send::<Pool<SendOnly>>();
    send::<Entry<'static, String>>();
    send::<OwnedEntry<String>>();
    send::<pool::OwnedRef<String>>();
    send::<pool::OwnedRefMut<String>>();
    send::<pool::OwnedLease<String>>();
    send::<pool::SharedLease<String>>();
    sync::<pool::SharedLease<String>>();
    send::<pool::ManagedPool<Vec<u8>>>();
    sync::<pool::ManagedPool<Vec<u8>>>();
    send::<pool::ManagedLease<Vec<u8>>>();
    send::<pool::ManagedSharedLease<Vec<u8>>>();
    sync::<pool::ManagedSharedLease<Vec<u8>>>();
    sync::<Slab<String>>();
    sync::<Pool<String>>();
    // &guard 只暴露共享读取；共享借用不移动拥有者及其析构责任。
    sync::<OwnedEntry<ThreadBound>>();
    sync::<pool::OwnedRef<ThreadBound>>();
    sync::<pool::OwnedRefMut<ThreadBound>>();
}
