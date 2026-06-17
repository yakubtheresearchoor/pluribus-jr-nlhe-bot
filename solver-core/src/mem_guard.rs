//! MEMORY GUARD: a background watchdog on this process's resident set size
//! (RSS), so the full-scale joint cold-CFR solve can be RUN AT REAL SCALE on a
//! 36 GB box without OOM-killing the machine. The whole point of the joint test
//! is to measure reality, not extrapolate from a toy — so we point the real
//! workload at the hardware and let this guard be the net.
//!
//! Two ceilings (both env-configurable, GB):
//!   - SOFT (`MEM_GUARD_SOFT_GB`): set an atomic abort flag the solver polls
//!     between subgames / iterations and exits cleanly (flush + report peak).
//!   - HARD (`MEM_GUARD_HARD_GB`): last-resort `process::exit` BEFORE the OS
//!     starts swap-thrashing — protects the machine when the soft check was
//!     not polled in time (e.g. a single allocation overshoots mid-subgame).
//!
//! RSS is read via `ps -o rss= -p <pid>` — macOS-friendly, no extra deps, and
//! reports the real unified-memory footprint (GPU `StorageModeShared` buffers
//! are part of this process's RSS on Apple Silicon, so device buffers count).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Classification of an RSS reading against the two ceilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Breach {
    None,
    Soft,
    Hard,
}

/// Pure ceiling classification — unit-testable without a thread or a process.
#[inline]
pub fn classify(rss_gb: f64, soft_gb: f64, hard_gb: f64) -> Breach {
    if rss_gb >= hard_gb {
        Breach::Hard
    } else if rss_gb >= soft_gb {
        Breach::Soft
    } else {
        Breach::None
    }
}

/// Read this process's resident set size in KiB via `ps`. `None` if the call
/// fails (the watchdog then simply skips that tick rather than guessing).
pub fn read_rss_kb() -> Option<u64> {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse::<u64>().ok()
}

/// Handle the solver holds. Cloneable (shared atomics) so it can be polled from
/// any thread / passed into the per-subgame loop.
#[derive(Clone)]
pub struct MemGuard {
    abort: Arc<AtomicBool>,
    peak_kb: Arc<AtomicU64>,
    soft_gb: f64,
    hard_gb: f64,
}

impl MemGuard {
    /// Arm the watchdog. `soft_gb`/`hard_gb` are the ceilings; `poll_ms` the
    /// sampling cadence. The thread is detached and runs for the process
    /// lifetime. A hard breach calls `process::exit(137)` after flushing.
    pub fn spawn(soft_gb: f64, hard_gb: f64, poll_ms: u64) -> Self {
        let g = MemGuard {
            abort: Arc::new(AtomicBool::new(false)),
            peak_kb: Arc::new(AtomicU64::new(0)),
            soft_gb,
            hard_gb,
        };
        let abort = g.abort.clone();
        let peak_kb = g.peak_kb.clone();
        thread::Builder::new()
            .name("mem-guard".into())
            .spawn(move || {
                let mut warned_soft = false;
                loop {
                    if let Some(kb) = read_rss_kb() {
                        peak_kb.fetch_max(kb, Ordering::Relaxed);
                        let gb = kb as f64 / 1_048_576.0;
                        match classify(gb, soft_gb, hard_gb) {
                            Breach::Hard => {
                                eprintln!(
                                    "MEM_GUARD HARD breach: RSS {gb:.2} GB >= {hard_gb:.1} GB — \
                                     aborting process to protect the machine"
                                );
                                use std::io::Write;
                                let _ = std::io::stderr().flush();
                                std::process::exit(137);
                            }
                            Breach::Soft => {
                                if !warned_soft {
                                    eprintln!(
                                        "MEM_GUARD SOFT breach: RSS {gb:.2} GB >= {soft_gb:.1} GB — \
                                         signalling clean abort to the solver"
                                    );
                                    warned_soft = true;
                                }
                                abort.store(true, Ordering::SeqCst);
                            }
                            Breach::None => {}
                        }
                    }
                    thread::sleep(Duration::from_millis(poll_ms));
                }
            })
            .expect("spawn mem-guard thread");
        g
    }

    /// Arm from environment: `MEM_GUARD_SOFT_GB` (default 26), `MEM_GUARD_HARD_GB`
    /// (default 31 — leaves ~5 GB headroom on a 36 GB box), 500 ms cadence.
    pub fn from_env() -> Self {
        let soft = env_gb("MEM_GUARD_SOFT_GB", 26.0);
        let hard = env_gb("MEM_GUARD_HARD_GB", 31.0);
        let poll: u64 = std::env::var("MEM_GUARD_POLL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(500);
        eprintln!("MEM_GUARD: armed — soft {soft:.1} GB, hard {hard:.1} GB (RSS watchdog, {poll} ms)");
        Self::spawn(soft, hard, poll)
    }

    /// The solver polls this between subgames / iterations; `true` ⇒ unwind and
    /// exit cleanly (soft ceiling crossed).
    #[inline]
    pub fn should_abort(&self) -> bool {
        self.abort.load(Ordering::SeqCst)
    }

    /// Peak RSS observed so far, in GB — the measurement we actually care about.
    pub fn peak_gb(&self) -> f64 {
        self.peak_kb.load(Ordering::Relaxed) as f64 / 1_048_576.0
    }

    pub fn soft_gb(&self) -> f64 {
        self.soft_gb
    }
    pub fn hard_gb(&self) -> f64 {
        self.hard_gb
    }
}

fn env_gb(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_boundaries() {
        // below soft → None
        assert_eq!(classify(10.0, 26.0, 31.0), Breach::None);
        // exactly soft → Soft (>= is the trip)
        assert_eq!(classify(26.0, 26.0, 31.0), Breach::Soft);
        // between → Soft
        assert_eq!(classify(28.5, 26.0, 31.0), Breach::Soft);
        // exactly hard → Hard (hard wins over soft)
        assert_eq!(classify(31.0, 26.0, 31.0), Breach::Hard);
        // above hard → Hard
        assert_eq!(classify(40.0, 26.0, 31.0), Breach::Hard);
    }

    /// The real watchdog path: read RSS, classify, flip the soft flag. Soft=0
    /// trips on any nonzero RSS; hard is set unreachably high so the test
    /// process is never `exit`ed. Proves read_rss_kb + the loop + the flag wire
    /// together end-to-end.
    #[test]
    fn soft_trip_sets_abort_flag() {
        let g = MemGuard::spawn(0.0, 1.0e12, 10);
        assert!(!g.should_abort(), "flag must start clear");
        // give the watchdog a few poll cycles to sample RSS and trip
        let mut tripped = false;
        for _ in 0..50 {
            if g.should_abort() {
                tripped = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(tripped, "soft ceiling 0 GB must trip the abort flag");
        assert!(g.peak_gb() > 0.0, "peak RSS must be observed (nonzero)");
    }

    #[test]
    fn read_rss_is_sane() {
        let kb = read_rss_kb().expect("ps RSS read should succeed on this platform");
        // a running test process is comfortably between 1 MB and 100 GB
        assert!(kb > 1_024, "RSS suspiciously low: {kb} KiB");
        assert!(kb < 100 * 1_048_576, "RSS suspiciously high: {kb} KiB");
    }
}
