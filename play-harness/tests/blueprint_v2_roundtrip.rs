//! BLUEPRINT SSBP2 ROUND-TRIP GATE: re-encode a real raw (SSBP1) cell to compressed
//! SSBP2 and confirm the loaded strategy EXACTLY equals the SSBP1 cell's u8-quantized
//! strategy (quantize_roundtrip) — proving the retroactive compression is the same
//! play-safe quantization, just stored compressed. Also reports the size shrink.
//!
//! Run: BP_ROOT=$PWD/blueprint_out_v1 cargo test --release -p play-harness \
//!   --test blueprint_v2_roundtrip -- --ignored --nocapture

use play_harness::blueprint::{reencode_to_v2, Blueprint};

#[test]
#[ignore = "needs blueprint_out_v1; --ignored --nocapture --release"]
fn blueprint_v2_roundtrip() {
    let bp_root = std::env::var("BP_ROOT").unwrap_or_else(|_| "blueprint_out_v1".into());
    let cell = std::fs::read_dir(&bp_root)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .find(|n| n.starts_with("live3_")
            && std::path::Path::new(&format!("{bp_root}/{n}/flop_0000.bp")).exists());
    let cell = match cell {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no live3 cell under {bp_root}");
            return;
        }
    };
    let src = format!("{bp_root}/{cell}/flop_0000.bp");

    // Expected: SSBP1 load + the in-memory u8 quantization (what the bot would play).
    let mut expected = Blueprint::load(&src).unwrap();
    expected.quantize_roundtrip();

    // Re-encode SSBP1 → SSBP2 to a temp file, load it back.
    let raw = std::fs::read(&src).unwrap();
    let v2 = reencode_to_v2(&raw, 9).expect("reencode");
    let dst = format!("{}/bp_v2_rt.bp", std::env::temp_dir().display());
    std::fs::write(&dst, &v2).unwrap();
    let got = Blueprint::load(&dst).unwrap();

    // The SSBP2-loaded cum must equal the SSBP1 cell's quantize_roundtrip EXACTLY
    // (identical quant + lossless zstd).
    for (name, e, g) in [
        ("flop", &expected.cum_flop, &got.cum_flop),
        ("turn", &expected.cum_turn, &got.cum_turn),
        ("river", &expected.cum_river, &got.cum_river),
    ] {
        assert_eq!(e.len(), g.len(), "{name} cum length");
        let maxd = e.iter().zip(g).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(maxd < 1e-6, "{name} cum mismatch after SSBP2 round-trip: max Δ = {maxd}");
    }
    // bucket maps must survive losslessly too (used for continuation lookups).
    assert_eq!(expected.nb, got.nb);
    assert_eq!(expected.turns, got.turns);

    eprintln!(
        "SSBP2 round-trip OK: {} ({:.1} KB) → {:.1} KB ({:.1}× smaller); cum bit-matches quantize_roundtrip",
        cell,
        raw.len() as f64 / 1e3,
        v2.len() as f64 / 1e3,
        raw.len() as f64 / v2.len() as f64
    );
    std::fs::remove_file(&dst).ok();
}
