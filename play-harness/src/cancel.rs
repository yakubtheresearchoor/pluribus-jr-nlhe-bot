//! Cooperative cancellation for solve work whose requester is gone.
//!
//! The hammer post-mortem (2026-07-02): when an HTTP client disconnects, axum
//! drops the handler FUTURE, but the `spawn_blocking` solve keeps running —
//! zombie work. 17 abandoned river resolves were caught burning ~13 cores for
//! clients that had timed out minutes earlier. The server can't preempt a
//! blocking thread; the solve loops must check a flag and bail.
//!
//! Mechanism: the handler creates an `Arc<AtomicBool>` and a drop-guard that
//! sets it when the future is dropped (client gone OR normal completion —
//! setting it after completion is harmless). The blocking closure installs the
//! flag in a THREAD-LOCAL for the duration of the solve; the chunked solve
//! loops poll `cancelled()` between chunks. No signature changes anywhere.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

thread_local! {
    static CANCEL_FLAG: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Run `f` with `flag` installed as this thread's cancellation flag.
/// Always uninstalls afterwards (even on unwind: the previous value is
/// restored via a guard so a panicking solve can't leak the flag into the
/// blocking thread's next task).
pub fn with_cancel_flag<F: FnOnce() -> R, R>(flag: Arc<AtomicBool>, f: F) -> R {
    struct Restore(Option<Arc<AtomicBool>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CANCEL_FLAG.with(|c| *c.borrow_mut() = self.0.take());
        }
    }
    let prev = CANCEL_FLAG.with(|c| c.borrow_mut().replace(flag));
    let _restore = Restore(prev);
    f()
}

/// True iff the current thread's installed flag (if any) is set — i.e. the
/// requester abandoned this work. Solve loops poll this between chunks and
/// return their best-so-far (the result is discarded upstream anyway).
pub fn cancelled() -> bool {
    CANCEL_FLAG.with(|c| {
        c.borrow()
            .as_ref()
            .map_or(false, |f| f.load(Ordering::Relaxed))
    })
}

/// Handler-side guard: sets the flag when dropped. Hold it across the
/// `.await` on the blocking task; if the client disconnects the future (and
/// this guard) is dropped mid-await and the solve sees `cancelled()`.
pub struct CancelOnDrop(pub Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
