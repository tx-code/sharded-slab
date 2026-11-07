use super::{OwnedRef, OwnedRefMut};
use crate::{cfg, sync::Arc, Clear, DefaultConfig};
use std::{fmt, ops};

/// An owned exclusive pool value, automatically cleared when dropped.
///
/// The lease may outlive the caller's pool handle. It can move between threads
/// when `T: Send + Sync`. Ordinary reads and writes through this lease do not
/// perform a pool lookup. Dropping it clears the value without publishing it.
/// Use [`freeze`](Self::freeze) to transfer ownership to read-only readers.
///
/// Pool pages and any capacity retained by [`Clear`] remain allocated until
/// the pool is dropped. A lease is not a byte-budget or page-trimming policy.
#[must_use]
pub struct OwnedLease<T, C = DefaultConfig>
where
    T: Clear + Default,
    C: cfg::Config,
{
    inner: Option<OwnedRefMut<T, C>>,
}

/// A cloneable read-only lease that clears its value after its last reader.
///
/// Clones share one owned pool guard. Dropping the caller's pool handle or
/// another lease does not invalidate the value. Clearing happens on the thread
/// that drops the last owner, unless another guard obtained through the raw
/// pool API is still reading it. Such guards also defer clearing.
///
/// This type intentionally does not expose a raw pool key or mutable access.
#[must_use]
pub struct SharedLease<T, C = DefaultConfig>
where
    T: Clear + Default,
    C: cfg::Config,
{
    inner: Arc<SharedInner<T, C>>,
}

struct SharedInner<T, C>
where
    T: Clear + Default,
    C: cfg::Config,
{
    guard: OwnedRef<T, C>,
}

impl<T, C> OwnedLease<T, C>
where
    T: Clear + Default,
    C: cfg::Config,
{
    pub(super) fn new(inner: OwnedRefMut<T, C>) -> Self {
        Self { inner: Some(inner) }
    }

    /// Consumes exclusive ownership and shares the value with read-only leases.
    ///
    /// This publishes the pool slot and allocates one `Arc` control block.
    /// Cloning a [`SharedLease`] does not allocate another control block or
    /// acquire another pool guard. The last reader requests clearing.
    pub fn freeze(mut self) -> SharedLease<T, C> {
        let guard = self.inner.take().expect("lease has an exclusive guard");
        SharedLease {
            inner: Arc::new(SharedInner {
                guard: guard.downgrade(),
            }),
        }
    }
}

impl<T, C> ops::Deref for OwnedLease<T, C>
where
    T: Clear + Default,
    C: cfg::Config,
{
    type Target = T;
    fn deref(&self) -> &T {
        self.inner.as_ref().expect("lease has an exclusive guard")
    }
}

impl<T, C> ops::DerefMut for OwnedLease<T, C>
where
    T: Clear + Default,
    C: cfg::Config,
{
    fn deref_mut(&mut self) -> &mut T {
        self.inner.as_mut().expect("lease has an exclusive guard")
    }
}

impl<T, C> Drop for OwnedLease<T, C>
where
    T: Clear + Default,
    C: cfg::Config,
{
    fn drop(&mut self) {
        if let Some(guard) = self.inner.take() {
            guard.discard();
        }
    }
}

impl<T, C> Clone for SharedLease<T, C>
where
    T: Clear + Default,
    C: cfg::Config,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T, C> ops::Deref for SharedLease<T, C>
where
    T: Clear + Default,
    C: cfg::Config,
{
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner.guard
    }
}

impl<T, C> Drop for SharedInner<T, C>
where
    T: Clear + Default,
    C: cfg::Config,
{
    fn drop(&mut self) {
        // 只读guard已计入refcount；先标清理，再由字段Drop释放最后引用。
        self.guard.pool.clear(self.guard.key());
    }
}

impl<T, C> fmt::Debug for OwnedLease<T, C>
where
    T: Clear + Default + fmt::Debug,
    C: cfg::Config,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("OwnedLease").field(&&**self).finish()
    }
}

impl<T, C> fmt::Debug for SharedLease<T, C>
where
    T: Clear + Default + fmt::Debug,
    C: cfg::Config,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SharedLease").field(&&**self).finish()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use crate::Pool;
    use std::sync::Arc;

    #[test]
    fn exclusive_discard_invalidates_key_and_advances_reuse_generation() {
        let pool = Arc::new(Pool::<Vec<usize>>::new());
        let mut value = pool.clone().lease_owned().unwrap();
        let key = value.inner.as_ref().unwrap().key();
        value.push(23);
        drop(value);
        assert!(pool.get(key).is_none());
        let next = pool.lease_owned().unwrap();
        assert_ne!(key, next.inner.as_ref().unwrap().key());
        assert!(next.is_empty());
    }

    #[test]
    fn raw_reader_also_defers_final_shared_lease_cleanup() {
        let pool = Arc::new(Pool::<Vec<usize>>::new());
        let mut value = pool.clone().lease_owned().unwrap();
        let key = value.inner.as_ref().unwrap().key();
        value.push(29);
        let shared = value.freeze();
        let raw = pool.get(key).unwrap();
        drop(shared);
        assert_eq!(&**raw, &[29]);
        assert!(pool.get(key).is_none());
        drop(raw);
        assert!(pool.get(key).is_none());
    }
}
