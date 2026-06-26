use std::env;
use std::path::PathBuf;

fn main() {
    // ── CUDA feature ──
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_shaders();
    }

    // ── Metal feature ──
    if env::var("CARGO_FEATURE_METAL").is_ok() {
        build_metal_shaders();
    }
}

// ============================================================================
// CUDA shader compilation (Windows/Linux, requires nvcc)
// ============================================================================

fn build_cuda_shaders() {
    let nvcc = match find_nvcc() {
        Some(n) => n,
        None => {
            eprintln!("cargo:warning=CUDA feature enabled but nvcc not found. Skipping CUDA compilation.");
            return;
        }
    };

    let msvc_bindir = find_msvc_bindir();
    if let Some(ref bindir) = msvc_bindir {
        let path = env::var("PATH").unwrap_or_default();
        env::set_var("PATH", format!("{};{}", bindir.display(), path));
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/gpu/kernels");

    let kernel_files = ["mccfr.cu", "test_showdown.cu", "vcfr.cu"];

    for kernel in &kernel_files {
        let src = kernels_dir.join(kernel);
        if !src.exists() {
            continue;
        }

        let ptx_name = kernel.replace(".cu", ".ptx");
        let ptx_path = out_dir.join(&ptx_name);

        let arch = gpu_arch();

        let mut cmd = std::process::Command::new(&nvcc);
        cmd.arg(format!("--gpu-architecture={}", arch))
            .arg("--ptx")
            .arg(format!("-o={}", ptx_path.display()))
            .arg(&src)
            .arg("-use_fast_math")
            .arg("-O3");

        if let Some(ref bindir) = msvc_bindir {
            cmd.arg(format!("--compiler-bindir={}", bindir.display()));
        }

        let status = cmd.status().unwrap_or_else(|e| {
            panic!(
                "Failed to execute nvcc at '{}': {}. \
                 Ensure CUDA Toolkit is installed and nvcc is in PATH.",
                nvcc, e
            )
        });

        if !status.success() {
            panic!(
                "nvcc failed to compile {}. \
                 Check the CUDA kernel source for errors.",
                kernel
            );
        }

        println!("cargo:rerun-if-changed=src/gpu/kernels/{}", kernel);
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn find_nvcc() -> Option<String> {
    let candidates = [
        "nvcc".to_string(),
        format!(
            "{}/bin/nvcc",
            env::var("CUDA_PATH").unwrap_or_default()
        ),
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe".to_string(),
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe".to_string(),
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe".to_string(),
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe".to_string(),
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvcc.exe".to_string(),
    ];

    for candidate in &candidates {
        if std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some(candidate.clone());
        }
    }

    None
}

fn gpu_arch() -> String {
    if let Ok(arch) = env::var("CUDA_ARCH") {
        return arch;
    }

    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        let cap = String::from_utf8_lossy(&output.stdout);
        let cap = cap.trim().replace('.', "");
        if !cap.is_empty() {
            return format!("sm_{}", cap);
        }
    }

    "sm_89".to_string()
}

fn find_msvc_bindir() -> Option<PathBuf> {
    if which("cl.exe") {
        return None;
    }

    let msvc_base = PathBuf::from(
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
    );

    if let Ok(entries) = std::fs::read_dir(&msvc_base) {
        let mut versions: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map_or(false, |t| t.is_dir()))
            .collect();
        versions.sort_by_key(|e| e.file_name());

        if let Some(latest) = versions.last() {
            let bin = latest.path().join("bin").join("Hostx64").join("x64");
            if bin.exists() {
                return Some(bin);
            }
        }
    }

    None
}

fn which(cmd: &str) -> bool {
    std::process::Command::new(cmd)
        .arg("/?")
        .output()
        .is_ok()
}

// ============================================================================
// Metal shader compilation (macOS, requires Xcode + Metal Toolchain)
// ============================================================================

fn build_metal_shaders() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shaders_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src/gpu_metal/shaders");

    // ── Per-stage MAX_NA: codegen Metal header from Rust source of truth ──
    // The constants live in src/tree/flat.rs. This step reads them out and
    // writes max_na_generated.metal into OUT_DIR. The Metal compiler invocation
    // below adds -I OUT_DIR so flat_tree.metal can `#include "max_na_generated.metal"`.
    // The build dependency below (rerun-if-changed on src/tree/flat.rs) guarantees
    // the generated header is regenerated whenever the constants change —
    // making "stale generated header" structurally impossible.
    generate_metal_max_na_header(&out_dir);
    println!("cargo:rerun-if-changed=src/tree/flat.rs");
    generate_metal_bucketed_header(&out_dir);
    println!("cargo:rerun-if-changed=src/solver/bucketed_showdown.rs");

    if !shaders_dir.exists() {
        eprintln!("cargo:warning=Metal shaders directory not found: {}", shaders_dir.display());
        return;
    }

    // Collect all .metal files
    // Collect all .metal files — compile as a single unit to resolve includes
    let metal_source_files =
        ["vcfr.metal", "bucketed_vcfr.metal", "mccfr_bench.metal"].iter().map(|s| s.to_string()).collect::<Vec<_>>();

    // Compile each .metal file to .air, then link all into one .metallib
    let mut air_files: Vec<PathBuf> = Vec::new();

    for metal_file in &metal_source_files {
        let src = shaders_dir.join(metal_file);
        if !src.exists() {
            eprintln!("cargo:warning=Metal shader not found: {}", src.display());
            continue;
        }

        // Compile the single source file (it #includes flat_tree.metal)
        let air_name = metal_file.replace(".metal", ".air");
        let air_path = out_dir.join(&air_name);

        let metal_compiler = find_metal_compiler();
        
        // Add the shaders directory as an include path so #include works
        let mut cmd = std::process::Command::new(&metal_compiler);
        cmd.arg("-c")
            .arg(&src)
            .arg("-o").arg(&air_path)
            .arg("-I").arg(&shaders_dir)
            // -I OUT_DIR so flat_tree.metal can resolve the generated header
            // (max_na_generated.metal). Without this the include fails.
            .arg("-I").arg(&out_dir)
            .arg("-target").arg("air64-apple-macos")
            // STEP 2.A.2 BUG FIX 2026-06: disable FMA contraction so that
            // `a * b + c` is NOT folded into a fused multiply-add. The
            // Metal compiler's default fp-contract policy is "on" (allow
            // contraction within statements). CPU Rust f32 NEVER contracts
            // (no implicit FMA). The mismatch produces 1-ULP terminal CFV
            // divergences under non-uniform reach (`accum += ra * net`
            // contracts on GPU, 2-op on CPU), which compound 2.8x per iter
            // through the CFR feedback loop into the order-of-magnitude
            // multi-iter divergence observed at stratum 1 iter 5+.
            // Localized via tests/p1_5_4_step2a2_iter1_noise_source.rs
            // (terminal node 13, contributions=[0,0], h0 diverges by 1 ULP).
            .arg("-ffp-contract=off")
            .arg("-fno-fast-math")
            .arg("-O3");

        let result = cmd.output();
        match result {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    panic!(
                        "Metal shader compilation failed for {}:\n{}",
                        metal_file, stderr
                    );
                }
                air_files.push(air_path);
            }
            Err(e) => {
                panic!(
                    "Failed to execute Metal compiler at '{}': {}.\n\
                     Ensure Xcode and Metal Toolchain are installed.",
                    metal_compiler, e
                );
            }
        }

        // Note: path corrected from "src/gpu/metal/shaders/" (wrong, never
        // matched any file) to "src/gpu_metal/shaders/" (actual location).
        // Discovered during Slice 2 Phase B Site (d) part 2: kernel edits
        // were not triggering rebuilds — the build script was watching a
        // non-existent path. Earlier Phase B edits happened to recompile
        // because adjacent Rust struct changes invalidated other build
        // artifacts; standalone .metal edits would NOT have triggered
        // rebuild before this fix.
        println!("cargo:rerun-if-changed=src/gpu_metal/shaders/{}", metal_file);
    }

    // Link all .air files into a single .metallib
    if !air_files.is_empty() {
        let metallib_path = out_dir.join("solver.metallib");
        
        let metallib_tool = find_metallib_tool();
        
        let mut cmd = std::process::Command::new(&metallib_tool);
        for air in &air_files {
            cmd.arg(air);
        }
        cmd.arg("-o").arg(&metallib_path);

        let result = cmd.output();
        match result {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    panic!("Metal library linking failed:\n{}", stderr);
                }
                eprintln!("cargo:warning=Built Metal library: {}", metallib_path.display());
            }
            Err(e) => {
                panic!(
                    "Failed to execute metallib tool at '{}': {}.\n\
                     Ensure Xcode and Metal Toolchain are installed.",
                    metallib_tool, e
                );
            }
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn find_metal_compiler() -> String {
    // Try xcrun first (standard way)
    if let Ok(output) = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "-f", "metal"])
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return path;
            }
        }
    }

    // Direct path fallback
    let candidates = [
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal",
    ];
    for candidate in &candidates {
        if std::path::Path::new(candidate).exists() {
            return candidate.to_string();
        }
    }

    "xcrun".to_string()
}

fn find_metallib_tool() -> String {
    if let Ok(output) = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "-f", "metallib"])
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return path;
            }
        }
    }

    "metallib".to_string()
}

// ============================================================================
// Per-stage MAX_NA codegen (Phase 2)
// ============================================================================
//
// Reads MAX_NA_PREFLOP and MAX_NA_POSTFLOP from src/tree/flat.rs (the single
// source of truth) and emits a Metal header at OUT_DIR/max_na_generated.metal.
// The build dependency declared above (rerun-if-changed on src/tree/flat.rs)
// keeps this regenerated whenever the constants change.
//
// Failure modes this function MUST prevent:
//   - Silent stride desync between Rust and Metal (the core Phase 2 goal)
//   - Stale generated header (caught by rerun-if-changed)
//   - Successful build with wrong values (caught by panic on parse failure)

fn generate_metal_max_na_header(out_dir: &std::path::Path) {
    let flat_rs = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src/tree/flat.rs");
    let source = std::fs::read_to_string(&flat_rs)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", flat_rs.display(), e));

    let preflop = parse_const(&source, "MAX_NA_PREFLOP");
    let postflop = parse_const(&source, "MAX_NA_POSTFLOP");

    let header = format!(
        "// AUTO-GENERATED by build.rs from src/tree/flat.rs.\n\
         // Do NOT edit this file directly — change the Rust constants and rebuild.\n\
         // Phase 2 per-stage MAX_NA: this header is the Metal-side source of truth\n\
         // for the action-slot strides used by all dispatched kernels.\n\
         #ifndef MAX_NA_GENERATED_METAL_H\n\
         #define MAX_NA_GENERATED_METAL_H\n\
         \n\
         constant int MAX_NA_PREFLOP  = {};\n\
         constant int MAX_NA_POSTFLOP = {};\n\
         \n\
         #endif\n",
        preflop, postflop
    );

    let out_path = out_dir.join("max_na_generated.metal");
    std::fs::write(&out_path, header)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", out_path.display(), e));
    println!("cargo:warning=Generated {}", out_path.display());
}

// ============================================================================
// Bucketed-solver codegen (G2 step 2 — GPU bucketed port)
// ============================================================================
//
// Mirrors the bucketed terminal's constants from
// src/solver/bucketed_showdown.rs into bucketed_generated.metal, through
// the same panic-guarded exact-form parsing as the MAX_NA header. Two
// new constants (MAX_OPP, MAX_BUCKETS_GPU) plus the relation codes and
// the state-vector layout — the five-item desync surface's items (2),
// (3), and (5) from the G1 report, each killed at build time.

fn generate_metal_bucketed_header(out_dir: &std::path::Path) {
    let src_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src/solver/bucketed_showdown.rs");
    let source = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", src_path.display(), e));

    let max_opp = parse_const(&source, "MAX_OPP");
    let max_b = parse_const_typed(&source, "MAX_BUCKETS_GPU", "usize");
    let rel_win = parse_const_typed(&source, "REL_WIN", "u8");
    let rel_tie = parse_const_typed(&source, "REL_TIE", "u8");
    let rel_lose = parse_const_typed(&source, "REL_LOSE", "u8");
    let rel_folded = parse_const_typed(&source, "REL_FOLDED", "u8");

    let header = format!(
        "// AUTO-GENERATED by build.rs from src/solver/bucketed_showdown.rs.\n\
         // Do NOT edit — change the Rust constants and rebuild.\n\
         // G2 step 2: Metal-side mirror of the bucketed terminal's\n\
         // constants. State vector layout: state[0] = beaten mass,\n\
         // state[1 + j] = mass with j ties (j in 0..=num_opp) — length\n\
         // MAX_OPP + 2, identical to the CPU arm's [f32; MAX_OPP + 2].\n\
         #ifndef BUCKETED_GENERATED_METAL_H\n\
         #define BUCKETED_GENERATED_METAL_H\n\
         \n\
         constant int MAX_OPP_BUCKETED = {max_opp};\n\
         constant int MAX_BUCKETS_GPU  = {max_b};\n\
         constant int STATE_LEN        = {state_len};\n\
         \n\
         constant uchar REL_WIN    = {rel_win};\n\
         constant uchar REL_TIE    = {rel_tie};\n\
         constant uchar REL_LOSE   = {rel_lose};\n\
         constant uchar REL_FOLDED = {rel_folded};\n\
         \n\
         #endif\n",
        state_len = max_opp + 2,
    );

    let out_path = out_dir.join("bucketed_generated.metal");
    std::fs::write(&out_path, header)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", out_path.display(), e));
    println!("cargo:warning=Generated {}", out_path.display());
}

/// parse_const with an explicit type token (e.g. `u8`), same strict
/// single-line exact-form match and panic-on-miss as `parse_const`.
fn parse_const_typed(source: &str, name: &str, ty: &str) -> usize {
    let needle = format!("pub const {}: {} = ", name, ty);
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(&needle) {
            // Strip any trailing line comment, then the semicolon.
            let no_comment = rest.split("//").next().unwrap_or(rest).trim();
            let value_str = no_comment.trim_end_matches(';').trim();
            return value_str.parse().unwrap_or_else(|e| {
                panic!("Failed to parse {} value '{}': {}", name, value_str, e)
            });
        }
    }
    panic!(
        "Could not find `pub const {}: {} = N;` in bucketed_showdown.rs. \
         build.rs codegen depends on this exact form; update parse_const_typed \
         if the declaration moved.",
        name, ty
    );
}

fn parse_const(source: &str, name: &str) -> usize {
    // Match lines like: `pub const MAX_NA_PREFLOP: usize = 16;`
    // The match is intentionally strict — single-line, exact name, no expressions.
    // A loose match would let a stale or wrong declaration through; the panic on
    // miss is the desync guard.
    let needle = format!("pub const {}: usize = ", name);
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(&needle) {
            let value_str = rest.trim_end_matches(';').trim();
            return value_str.parse().unwrap_or_else(|e| {
                panic!("Failed to parse {} value '{}': {}", name, value_str, e)
            });
        }
    }
    panic!(
        "Could not find `pub const {}: usize = N;` in src/tree/flat.rs. \
         The build.rs codegen depends on this exact form. If the declaration was \
         moved or rewritten, update parse_const to match the new form.",
        name
    );
}
