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

    if !shaders_dir.exists() {
        eprintln!("cargo:warning=Metal shaders directory not found: {}", shaders_dir.display());
        return;
    }

    // Collect all .metal files
    // Collect all .metal files — compile as a single unit to resolve includes
    let metal_source_files = ["vcfr.metal"].iter().map(|s| s.to_string()).collect::<Vec<_>>();

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
            .arg("-target").arg("air64-apple-macos")
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

        println!("cargo:rerun-if-changed=src/gpu/metal/shaders/{}", metal_file);
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
