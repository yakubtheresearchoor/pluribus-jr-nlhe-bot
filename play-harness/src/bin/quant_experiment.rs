//! Quantization experiment: does u8-quantizing the blueprint's cum_strategy
//! shrink it meaningfully without hurting strategy fidelity?
//!
//! The .bp files are ~85% cum_flop/turn/river (cumulative regret-matched
//! strategy). Those are NOT arbitrary floats: they are sums over `iters`
//! regret-matching steps, so every value lives in a NARROW range [0, ~iters]
//! — which means a simple per-array linear u8 quant preserves the within-
//! infoset ratios (what normalization actually uses) without crushing small
//! values. The loader renormalizes per infoset, so absolute scale is free.
//!
//! Run: cargo run --release -p play-harness --bin quant_experiment -- <flop_NNNN.bp>
//! Measures, on REAL strategy data:
//!   - size: f32 vs u8 vs (each)+zstd  (the empirical unknown: how well do
//!     near-pure quantized strategies compress?)
//!   - error: u8 round-trip abs/rel error, and the rel error restricted to
//!     high-magnitude (high-reach ⇒ EV-relevant) values.

use std::io::Write;
use std::process::Command;

use play_harness::blueprint::Blueprint;

fn zstd_size(bytes: &[u8], level: u32) -> usize {
    let tmp = std::env::temp_dir().join(format!("quant_exp_{level}.bin"));
    std::fs::File::create(&tmp).unwrap().write_all(bytes).unwrap();
    let out = Command::new("zstd")
        .args(["-q", "-f", &format!("-{level}"), tmp.to_str().unwrap(), "-o"])
        .arg(tmp.with_extension("zst"))
        .output()
        .expect("zstd");
    assert!(out.status.success(), "zstd failed: {}", String::from_utf8_lossy(&out.stderr));
    let sz = std::fs::metadata(tmp.with_extension("zst")).unwrap().len() as usize;
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(tmp.with_extension("zst"));
    sz
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: quant_experiment <flop_NNNN.bp>");
    let bp = Blueprint::load(&path).expect("load .bp");

    // All cum_strategy (the prize — ~85% of the file).
    let mut cum: Vec<f32> = Vec::new();
    cum.extend_from_slice(&bp.cum_flop);
    cum.extend_from_slice(&bp.cum_turn);
    cum.extend_from_slice(&bp.cum_river);
    let n = cum.len();
    println!("blueprint {path}  (nb={}, nh={})", bp.nb, bp.nh);
    println!(
        "cum_strategy: {n} f32 values (flop {}, turn {}, river {})",
        bp.cum_flop.len(),
        bp.cum_turn.len(),
        bp.cum_river.len()
    );

    // ── Per-array linear u8 quant: q = round((v-min)/(max-min)·255). ──
    let vmin = cum.iter().cloned().fold(f32::INFINITY, f32::min);
    let vmax = cum.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scale = (vmax - vmin).max(1e-12);
    let q: Vec<u8> = cum.iter().map(|&v| (((v - vmin) / scale) * 255.0).round() as u8).collect();
    let deq: Vec<f32> = q.iter().map(|&b| vmin + (b as f32 / 255.0) * scale).collect();

    // ── Error ──
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut zeros = 0usize;
    // rel error on EV-relevant (high-magnitude ⇒ high-reach) values only.
    let thresh = vmax * 0.1;
    let (mut rel_num, mut rel_den, mut rel_max, mut hi_n) = (0.0f64, 0.0f64, 0.0f32, 0usize);
    for (&v, &d) in cum.iter().zip(&deq) {
        let e = (v - d).abs();
        max_abs = max_abs.max(e);
        sum_abs += e as f64;
        if v == 0.0 {
            zeros += 1;
        }
        if v.abs() >= thresh {
            rel_num += e as f64;
            rel_den += v.abs() as f64;
            rel_max = rel_max.max(e / v.abs());
            hi_n += 1;
        }
    }
    println!("\nvalue range [{vmin:.4}, {vmax:.4}]  ({:.1}% exactly zero — sparse)", 100.0 * zeros as f64 / n as f64);
    println!("u8 round-trip error: max abs {max_abs:.5}, mean abs {:.6}", sum_abs / n as f64);
    println!(
        "  on high-reach values (|v| ≥ {thresh:.3}, n={hi_n}): weighted-mean rel {:.4}%, max rel {:.4}%",
        100.0 * rel_num / rel_den.max(1e-30),
        100.0 * rel_max as f64
    );

    // ── Size ──
    let f32_bytes: Vec<u8> = cum.iter().flat_map(|x| x.to_le_bytes()).collect();
    let u8_bytes: Vec<u8> = q.clone();
    let f32_raw = f32_bytes.len();
    let u8_raw = u8_bytes.len();
    let f32_z3 = zstd_size(&f32_bytes, 3);
    let u8_z3 = zstd_size(&u8_bytes, 3);
    let u8_z19 = zstd_size(&u8_bytes, 19);

    let mb = |b: usize| b as f64 / 1_048_576.0;
    println!("\n── cum_strategy size ({n} values) ──");
    println!("  f32 raw          : {:8.3} MB   (1.00×)", mb(f32_raw));
    println!("  f32 + zstd-3     : {:8.3} MB   ({:.2}×)", mb(f32_z3), f32_raw as f64 / f32_z3 as f64);
    println!("  u8  raw          : {:8.3} MB   ({:.2}×)", mb(u8_raw), f32_raw as f64 / u8_raw as f64);
    println!("  u8  + zstd-3     : {:8.3} MB   ({:.2}×)", mb(u8_z3), f32_raw as f64 / u8_z3 as f64);
    println!("  u8  + zstd-19    : {:8.3} MB   ({:.2}×)", mb(u8_z19), f32_raw as f64 / u8_z19 as f64);
    println!(
        "\n→ u8 quant alone: 4.00× ; u8+zstd: {:.2}× off f32 (vs {:.2}× for zstd-alone on f32)",
        f32_raw as f64 / u8_z19 as f64,
        f32_raw as f64 / f32_z3 as f64
    );
}
