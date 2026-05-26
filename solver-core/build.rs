use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let nvcc = find_nvcc();
    let msvc_bindir = find_msvc_bindir();
    if let Some(ref bindir) = msvc_bindir {
        let path = env::var("PATH").unwrap_or_default();
        env::set_var("PATH", format!("{};{}", bindir.display(), path));
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/gpu/kernels");

    let kernel_files = ["mccfr.cu", "test_showdown.cu"];

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

fn find_nvcc() -> String {
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
            return candidate.clone();
        }
    }

    panic!(
        "CUDA Toolkit required for 'cuda' feature, but nvcc not found.\n\
         \n\
         Install CUDA Toolkit 12.x or later:\n\
           winget install Nvidia.CUDA\n\
           or download from https://developer.nvidia.com/cuda-toolkit\n\
         \n\
         Alternatively, build without CUDA: cargo build (without --features cuda)"
    )
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

    panic!(
        "MSVC cl.exe not found in PATH, and could not locate it automatically.\n\
         nvcc requires the MSVC C++ compiler.\n\
         \n\
         Install Visual Studio Build Tools:\n\
           winget install Microsoft.VisualStudio.2022.BuildTools\n\
         \n\
         Or run from a 'Developer Command Prompt for VS 2022'."
    )
}

fn which(cmd: &str) -> bool {
    std::process::Command::new(cmd)
        .arg("/?")
        .output()
        .is_ok()
}
