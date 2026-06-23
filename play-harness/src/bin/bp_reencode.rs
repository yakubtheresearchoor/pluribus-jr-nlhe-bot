//! RETROACTIVE BLUEPRINT RE-ENCODER: rewrite raw-f32 SSBP1 `.bp` cells as compressed
//! SSBP2 (cum_* = [lo:f32][scale:f32][zstd(u8-quant)], maps = zstd(u16)) — a ~16×
//! shrink (21 GB → ~1.3 GB) with NO re-solve. Operates purely on bytes (parse
//! sections, requantize, recompress) — no table rebuild. u8 quant is the same
//! per-buffer linear scale `Blueprint::quantize_roundtrip` uses (money-test-proven
//! play-safe). Idempotent (skips files already SSBP2). Writes atomically (tmp+rename).
//!
//! Env: ROOT (dir to recurse for *.bp, default blueprint_out_v1), DRY=1 (measure, no
//! write), LIMIT (max files, 0=all), THREADS (default rayon default), ZLEVEL (9).
//!
//! Run: ROOT=blueprint_out_v1 cargo run --release -p play-harness --bin bp_reencode

use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

fn list_bp_files(root: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                list_bp_files(&p, out);
            } else if p.extension().map(|x| x == "bp").unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

/// Returns (old_len, new_len) or None on parse failure / non-SSBP1.
fn reencode(path: &Path, dry: bool, zlevel: i32) -> Option<(u64, u64)> {
    let raw = std::fs::read(path).ok()?;
    if raw.len() >= 6 && &raw[..6] == b"SSBP2\n" {
        return Some((raw.len() as u64, raw.len() as u64)); // already compressed
    }
    let out = play_harness::blueprint::reencode_to_v2(&raw, zlevel)?;
    if !dry {
        let tmp = path.with_extension("bp.tmp");
        std::fs::write(&tmp, &out).ok()?;
        std::fs::rename(&tmp, path).ok()?;
    }
    Some((raw.len() as u64, out.len() as u64))
}

fn main() {
    let root = std::env::var("ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let dry = std::env::var("DRY").as_deref() == Ok("1");
    let limit: usize = std::env::var("LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let zlevel: i32 = std::env::var("ZLEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(9);
    if let Ok(t) = std::env::var("THREADS").map(|s| s.parse::<usize>().unwrap_or(0)) {
        if t > 0 {
            rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok();
        }
    }

    let mut files = Vec::new();
    list_bp_files(Path::new(&root), &mut files);
    files.sort();
    if limit > 0 {
        files.truncate(limit);
    }
    eprintln!(
        "bp_reencode: {} .bp files under {root}  (dry={dry}, zlevel={zlevel})",
        files.len()
    );

    let old_total = AtomicU64::new(0);
    let new_total = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let t0 = std::time::Instant::now();
    files.par_iter().for_each(|p| {
        match reencode(p, dry, zlevel) {
            Some((o, n)) => {
                old_total.fetch_add(o, Ordering::Relaxed);
                new_total.fetch_add(n, Ordering::Relaxed);
            }
            None => {
                failed.fetch_add(1, Ordering::Relaxed);
            }
        }
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 20000 == 0 {
            eprintln!("  {d}/{} ...", files.len());
        }
    });

    let (o, n) = (old_total.load(Ordering::Relaxed), new_total.load(Ordering::Relaxed));
    eprintln!(
        "DONE {} files in {:.1}s | {:.2} GB → {:.2} GB ({:.1}× smaller) | failed/skipped {}",
        files.len(),
        t0.elapsed().as_secs_f64(),
        o as f64 / 1e9,
        n as f64 / 1e9,
        o as f64 / (n.max(1)) as f64,
        failed.load(Ordering::Relaxed),
    );
}
