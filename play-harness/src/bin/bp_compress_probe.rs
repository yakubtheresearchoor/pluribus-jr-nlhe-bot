//! BLUEPRINT COMPRESSION PROBE: on a real .bp cell, measure how much the cum_flop/
//! turn/river f32 strategy sections shrink under u8-quant + zstd (the live-2 trick),
//! to scope a retroactive re-encode of the ~21 GB raw-f32 blueprint. Operates at the
//! byte/section level (no table rebuild) — the same fast path a re-encoder would use.
//!
//! Run: BP=blueprint_out_v1/live3_c2_p7_b15/flop_0000.bp \
//!      cargo run --release -p play-harness --bin bp_compress_probe

use std::collections::HashMap;

fn parse_sections(raw: &[u8]) -> (String, HashMap<String, Vec<u8>>) {
    assert_eq!(&raw[..6], b"SSBP1\n", "bad magic");
    let hdr_end = 6 + raw[6..].iter().position(|&b| b == b'\n').unwrap();
    let header = std::str::from_utf8(&raw[6..hdr_end]).unwrap().to_string();
    let mut sections = HashMap::new();
    let mut pos = hdr_end + 1;
    while pos < raw.len() {
        let name_end = pos + raw[pos..].iter().position(|&b| b == b'\n').unwrap();
        let name = std::str::from_utf8(&raw[pos..name_end]).unwrap().to_string();
        let mut len8 = [0u8; 8];
        len8.copy_from_slice(&raw[name_end + 1..name_end + 9]);
        let len = u64::from_le_bytes(len8) as usize;
        sections.insert(name, raw[name_end + 9..name_end + 9 + len].to_vec());
        pos = name_end + 9 + len;
    }
    (header, sections)
}

/// Per-buffer u8 quant (global lo/scale linear) → the bytes a re-encoder would store
/// (8 header bytes lo+scale + 1 byte/value), matching Blueprint::quantize_roundtrip.
fn u8_quant(f32_bytes: &[u8]) -> Vec<u8> {
    let vals: Vec<f32> = f32_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &vals { lo = lo.min(v); hi = hi.max(v); }
    let scale = (hi - lo).max(1e-12);
    let mut out = Vec::with_capacity(8 + vals.len());
    out.extend_from_slice(&lo.to_le_bytes());
    out.extend_from_slice(&scale.to_le_bytes());
    for &v in &vals {
        out.push((((v - lo) / scale) * 255.0).round().clamp(0.0, 255.0) as u8);
    }
    out
}

fn main() {
    let path = std::env::var("BP").unwrap_or_else(|_| "blueprint_out_v1/live3_c2_p7_b15/flop_0000.bp".into());
    let raw = std::fs::read(&path).expect("read .bp");
    let total = raw.len();
    let (_header, sections) = parse_sections(&raw);

    println!("cell {path}  ({:.1} KB raw)\n", total as f64 / 1e3);
    println!("{:<12} {:>9} {:>9} {:>9} {:>10} {:>10}", "section", "rawKB", "u8KB", "zstdf32", "zstd(u8)", "ratio");
    let mut new_total = 0usize;
    let mut keys: Vec<&String> = sections.keys().collect();
    keys.sort();
    for name in keys {
        let data = &sections[name];
        let is_cum = name.starts_with("cum_");
        let encoded = if is_cum {
            // re-encoder would store zstd(u8-quant) for cum sections.
            let u8b = u8_quant(data);
            let zf = zstd::encode_all(&data[..], 9).unwrap().len();
            let zu = zstd::encode_all(&u8b[..], 9).unwrap().len();
            println!("{:<12} {:>8.1} {:>8.1} {:>8.1} {:>9.1} {:>9.1}x",
                name, data.len() as f64/1e3, u8b.len() as f64/1e3, zf as f64/1e3, zu as f64/1e3,
                data.len() as f64 / zu as f64);
            zu
        } else {
            // non-cum sections (maps etc.): zstd as-is.
            let z = zstd::encode_all(&data[..], 9).unwrap().len();
            println!("{:<12} {:>8.1} {:>9} {:>9} {:>9.1} {:>9.1}x",
                name, data.len() as f64/1e3, "-", "-", z as f64/1e3, data.len() as f64 / z.max(1) as f64);
            z
        };
        new_total += encoded;
    }
    let cells_disk_gb = 21.0;
    println!("\nper-cell: {:.1} KB raw → ~{:.1} KB re-encoded  ({:.1}× smaller)",
        total as f64/1e3, new_total as f64/1e3, total as f64 / new_total.max(1) as f64);
    println!("PROJECTED full blueprint: {cells_disk_gb} GB → ~{:.1} GB",
        cells_disk_gb * new_total as f64 / total as f64);
}
