//! Probe: device threadgroup-memory limit → max feasible MAX_BUCKETS_GPU.
//! Threadgroup bytes for the bucketed_vcfr kernel = 16·B² + 40·B + 128
//! (bucket_reach 9·B + fw/ft/fl/fn 4·B² + partials 32 + cfv_bucket B, ×4 B/float).
#![cfg(feature = "metal")]

use metal::Device;

#[test]
#[ignore = "device probe; run with --ignored --nocapture --features metal"]
fn gpu_tg_mem_limit() {
    let device = Device::system_default().expect("metal device");
    let tg_mem = device.max_threadgroup_memory_length() as u64;
    eprintln!("device: {:?}", device.name());
    eprintln!("maxThreadgroupMemoryLength = {tg_mem} bytes ({:.1} KB)", tg_mem as f64 / 1024.0);
    let mtt = device.max_threads_per_threadgroup();
    eprintln!("maxThreadsPerThreadgroup = ({}, {}, {})", mtt.width, mtt.height, mtt.depth);

    let bytes = |b: u64| -> u64 { 16 * b * b + 40 * b + 128 };
    eprintln!("\nB  | threadgroup bytes | fits {tg_mem}B? | occupancy note");
    for b in [16u64, 20, 24, 28, 32, 36, 40, 43, 44, 48] {
        let by = bytes(b);
        // Apple GPU cores expose 32 KB tg-mem; concurrent threadgroups/core ≈ 32KB/by.
        let conc = (32 * 1024) / by.max(1);
        eprintln!(
            "B{b:<3}| {by:>6} ({:>4.1} KB) | {:>3} | ~{conc} concurrent tg/core",
            by as f64 / 1024.0,
            if by <= tg_mem { "yes" } else { "NO" }
        );
    }
    // Largest B that fits the hard limit.
    let mut max_b = 16u64;
    for b in 16..=128 {
        if bytes(b) <= tg_mem { max_b = b; }
    }
    eprintln!("\n→ HARD max B (fits {tg_mem}B threadgroup mem): B{max_b} ({} bytes)", bytes(max_b));
}
