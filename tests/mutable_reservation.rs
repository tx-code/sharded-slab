use sharded_slab::Pool;

#[test]
fn take_does_not_requeue_a_vacant_reservation() {
    let slab = sharded_slab::Slab::new();
    let first = slab.vacant_entry().unwrap();
    let first_key = first.key();
    assert!(slab.take(first_key).is_none());
    let second = slab.vacant_entry().unwrap();
    let second_key = second.key();
    first.insert(7usize);
    second.insert(9usize);
    assert_eq!(*slab.get(first_key).unwrap(), 7);
    assert_eq!(*slab.get(second_key).unwrap(), 9);
    assert_eq!(slab.take(first_key), Some(7));
    assert!(slab.take(first_key).is_none());
    assert_eq!(slab.take(second_key), Some(9));
}

#[test]
fn reused_mutable_slot_cannot_be_cleared_before_publication() {
    let pool = Pool::<String>::new();
    let first = pool.create().unwrap();
    let first_key = first.key();
    drop(first);
    assert!(pool.clear(first_key));

    let mut reserved = pool.create().unwrap();
    let key = reserved.key();
    assert_ne!(key, first_key);
    // 旧实现若错误回收了保留槽，不再运行该guard的析构路径。
    if pool.clear(key) {
        std::mem::forget(reserved);
        panic!("a mutable reservation was cleared before publication");
    }
    assert!(pool.get(key).is_none());
    reserved.push_str("reserved");
    let reader = reserved.downgrade();
    assert!(pool.clear(key));
    assert_eq!(&*reader, "reserved");
    drop(reader);
    assert!(pool.get(key).is_none());
}
