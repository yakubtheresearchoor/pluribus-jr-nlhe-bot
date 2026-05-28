#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaStream, sys, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Minimal CUDA graph capture test to diagnose cudarc compatibility.
#[test]
fn minimal_cuda_graph() {
    let ctx = CudaContext::new(0).expect("context");
    let stream = ctx.new_stream().expect("stream");

    // Allocate a small buffer
    let mut buf: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros(256).expect("alloc");

    // Load the VCFR module
    let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
    let ptx_path = std::path::PathBuf::from(out_dir).join("vcfr.ptx");
    let ptx = cudarc::nvrtc::Ptx::from_file(&ptx_path);
    let module = ctx.load_module(ptx).expect("module");
    let zero_fn = module.load_function("vcfr_zero_buffer").expect("function");

    // Test 1: Can we capture a single kernel launch?
    println!("Test 1: Single kernel launch capture");
    stream.synchronize().expect("sync");
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).expect("begin");
    
    let size = 256i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0
    };
    unsafe {
        let mut b = stream.launch_builder(&zero_fn);
        b.arg(&buf).arg(&size);
        b.launch(cfg).expect("launch");
    }

    let graph = stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture");
    let graph = graph.expect("graph");
    println!("Test 1: PASSED — captured single kernel launch");

    // Test 2: Can we replay it?
    graph.launch().expect("replay");
    stream.synchronize().expect("sync after replay");
    println!("Test 2: PASSED — replayed graph");

    // Test 3: Can we capture with memset_zeros?
    let mut buf2: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros(256).expect("alloc");
    println!("Test 3: memset_zeros during capture");
    stream.synchronize().expect("sync");
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).expect("begin2");
    
    match stream.memset_zeros(&mut buf2) {
        Ok(()) => println!("  memset_zeros: OK"),
        Err(e) => println!("  memset_zeros: FAILED — {:?}", e),
    }

    match stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH) {
        Ok(g) => { println!("  end_capture: OK"); if let Some(g) = g { g.launch().expect("replay2"); } },
        Err(e) => println!("  end_capture: FAILED — {:?}", e),
    }
    println!();

    // Test 4: Can we capture with memcpy_htod?
    println!("Test 4: memcpy_htod during capture");
    stream.synchronize().expect("sync");
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).expect("begin3");
    
    let data = vec![1.0f32; 256];
    match stream.memcpy_htod(&data[..], &mut buf2) {
        Ok(()) => println!("  memcpy_htod: OK"),
        Err(e) => println!("  memcpy_htod: FAILED — {:?}", e),
    }

    match stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH) {
        Ok(g) => { println!("  end_capture: OK"); if let Some(g) = g { g.launch().expect("replay3"); } },
        Err(e) => println!("  end_capture: FAILED — {:?}", e),
    }

    println!("\nDone.");
}
