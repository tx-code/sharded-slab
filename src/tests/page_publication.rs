use crate::{test_util, Slab};
use loom::{sync::Arc, thread};

#[test]
fn arbitrary_key_lookup_during_initial_page_allocation() {
    test_util::run_model(
        "arbitrary_key_lookup_during_initial_page_allocation",
        || {
            let slab = Arc::new(Slab::<bool>::new());
            let shared = slab.clone();
            let writer = thread::spawn(move || {
                shared.insert(true);
            });
            // Yield is needed to explore the initialization race reported in upstream #116.
            thread::yield_now();
            assert!(slab.get(20).is_none());
            writer.join().unwrap();
        },
    );
}
