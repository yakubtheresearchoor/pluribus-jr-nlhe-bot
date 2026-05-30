/// Phase 1 validation test: verifies the Metal compute infrastructure end-to-end.
///
/// Tests:
/// 1. MetalContext creation (device, queue, library loading)
/// 2. Buffer allocation and data transfer (MetalBuffer)
/// 3. Kernel launch (vcfr_compute_strategies)
/// 4. Result verification against CPU reference

use solver_core::gpu_metal::{MetalContext, MetalBuffer};
use solver_core::tree::flat::FlatNode;
use metal::MTLSize;

#[test]
fn test_metal_context_creation() {
    let ctx = MetalContext::new();
    assert!(ctx.is_ok(), "Failed to create MetalContext: {:?}", ctx.err());
    let ctx = ctx.unwrap();
    
    let pipeline = ctx.create_pipeline("vcfr_compute_strategies");
    assert!(pipeline.is_ok(), "Failed to create pipeline: {:?}", pipeline.err());
    
    let pipeline = pipeline.unwrap();
    println!("Pipeline thread execution width: {}", pipeline.thread_execution_width());
    println!("Pipeline max total threads per threadgroup: {}", pipeline.max_total_threads_per_threadgroup());
    
    assert!(pipeline.thread_execution_width() > 0);
    assert!(pipeline.max_total_threads_per_threadgroup() > 0);
}

#[test]
fn test_metal_buffer_allocation() {
    let ctx = MetalContext::new().unwrap();
    
    // Test zero allocation
    let buf: MetalBuffer<f32> = ctx.alloc_zeros(100);
    assert_eq!(buf.len(), 100);
    let data = buf.to_vec();
    assert!(data.iter().all(|&v| v == 0.0f32), "Buffer should be zero-initialized");
    
    // Test upload + download
    let input: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let buf = ctx.upload(&input);
    let output = buf.to_vec();
    assert_eq!(input, output, "Upload/download roundtrip should preserve data");
    
    // Test larger buffer (1M elements)
    let big: Vec<f32> = (0..1_000_000).map(|i| i as f32 * 0.001).collect();
    let buf = ctx.upload(&big);
    let out = buf.to_vec();
    let max_diff = big.iter().zip(out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-6, "Max diff in large buffer: {}", max_diff);
}

#[test]
fn test_metal_vcfr_compute_strategies() {
    let ctx = MetalContext::new().unwrap();
    let pipeline = ctx.create_pipeline("vcfr_compute_strategies").unwrap();
    
    // Create a simple test case:
    // 1 infoset, 2 actions, 4 hands
    let num_infosets = 1usize;
    let nh = 4usize;
    let na = 2usize;
    let max_na = 4usize;
    let stride = max_na * nh;
    
    // Regrets: [infoset 0][action 0: 4 hands, action 1: 4 hands, action 2: zeros, action 3: zeros]
    let mut regrets = vec![0.0f32; num_infosets * stride];
    // infoset 0, action 0: [3.0, 1.0, 0.0, 2.0]
    regrets[0 * nh + 0] = 3.0;
    regrets[0 * nh + 1] = 1.0;
    regrets[0 * nh + 2] = 0.0;
    regrets[0 * nh + 3] = 2.0;
    // infoset 0, action 1: [1.0, 2.0, -1.0, 0.0]
    regrets[1 * nh + 0] = 1.0;
    regrets[1 * nh + 1] = 2.0;
    regrets[1 * nh + 2] = -1.0; // negative → 0
    regrets[1 * nh + 3] = 0.0;
    
    // Expected strategy:
    // Hand 0: pos_sum = 3.0 + 1.0 = 4.0, strat = [3/4, 1/4]
    // Hand 1: pos_sum = 1.0 + 2.0 = 3.0, strat = [1/3, 2/3]
    // Hand 2: pos_sum = 0.0,          strat = [1/2, 1/2] (uniform fallback)
    // Hand 3: pos_sum = 2.0 + 0.0 = 2.0, strat = [1.0, 0.0]
    let expected = vec![
        0.75, 1.0/3.0, 0.5, 1.0,   // action 0
        0.25, 2.0/3.0, 0.5, 0.0,   // action 1
        0.0, 0.0, 0.0, 0.0,        // action 2 (unused)
        0.0, 0.0, 0.0, 0.0,        // action 3 (unused)
    ];
    
    // Create a minimal FlatNode
    let mut node = FlatNode::player(0, solver_core::tree::action::BoardState::Flop, 0);
    node.num_children = na as u16;
    
    let decision_node_ids = vec![0u32];
    let nodes = vec![node];
    let infoset_offsets = vec![0u32];
    
    // Upload to GPU
    let d_regrets = ctx.upload(&regrets);
    let d_strategy: MetalBuffer<f32> = ctx.alloc_zeros(num_infosets * stride);
    let d_decision_node_ids = ctx.upload(&decision_node_ids);
    let d_nodes = ctx.upload(&nodes);
    let d_infoset_offsets = ctx.upload(&infoset_offsets);
    
    // Params struct: { num_infosets: i32, nh: i32 }
    let params: [i32; 2] = [num_infosets as i32, nh as i32];
    let d_params = ctx.upload(&params);
    
    // Dispatch
    let cmd_buffer = ctx.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(d_regrets.as_ref()), 0);
    encoder.set_buffer(1, Some(d_strategy.as_ref()), 0);
    encoder.set_buffer(2, Some(d_decision_node_ids.as_ref()), 0);
    encoder.set_buffer(3, Some(d_nodes.as_ref()), 0);
    encoder.set_buffer(4, Some(d_infoset_offsets.as_ref()), 0);
    encoder.set_buffer(5, Some(d_params.as_ref()), 0);
    
    let max_tpg = pipeline.max_total_threads_per_threadgroup() as usize;
    let (grid_size, group_size) = ctx.dispatch_2d(num_infosets, nh, max_tpg);
    encoder.dispatch_thread_groups(grid_size, group_size);
    encoder.end_encoding();
    
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
    
    let result = d_strategy.to_vec();
    
    println!("Regrets:  {:?}", regrets);
    println!("Expected: {:?}", expected);
    println!("Got:      {:?}", result);
    
    for i in 0..expected.len() {
        let diff = (result[i] - expected[i]).abs();
        assert!(diff < 1e-5, 
            "Mismatch at index {}: expected {}, got {}, diff {}", 
            i, expected[i], result[i], diff);
    }
    
    println!("✓ vcfr_compute_strategies kernel produces correct output on Metal");
}

#[test]
fn test_metal_zero_buffer_kernel() {
    let ctx = MetalContext::new().unwrap();
    let pipeline = ctx.create_pipeline("vcfr_zero_buffer").unwrap();
    
    let size = 1000usize;
    let initial: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
    let d_buf = ctx.upload(&initial);
    
    let pre = d_buf.to_vec();
    assert!(pre[0] != 0.0);
    
    let params: [i32; 1] = [size as i32];
    let d_params = ctx.upload(&params);
    
    let cmd_buffer = ctx.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(d_buf.as_ref()), 0);
    encoder.set_buffer(1, Some(d_params.as_ref()), 0);
    
    let (grid_size, group_size) = ctx.dispatch_1d(size, 256);
    encoder.dispatch_thread_groups(grid_size, group_size);
    encoder.end_encoding();
    
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
    
    let result = d_buf.to_vec();
    assert!(result.iter().all(|&v| v == 0.0f32), "Buffer should be all zeros after kernel");
    
    println!("✓ vcfr_zero_buffer kernel works correctly");
}

#[test]
fn test_metal_init_reach_kernel() {
    let ctx = MetalContext::new().unwrap();
    let pipeline = ctx.create_pipeline("vcfr_init_reach").unwrap();
    
    let np = 2usize;
    let nh = 4usize;
    let nn = 3usize;
    let total_reach = nn * np * nh;
    let np_nh = np * nh;
    
    let initial_weight: Vec<f32> = (0..np_nh).map(|i| (i + 1) as f32 * 0.1).collect();
    
    let d_reach: MetalBuffer<f32> = ctx.alloc_zeros(total_reach);
    let d_initial_weight = ctx.upload(&initial_weight);
    let params: [i32; 2] = [total_reach as i32, np_nh as i32];
    let d_params = ctx.upload(&params);
    
    let cmd_buffer = ctx.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(d_reach.as_ref()), 0);
    encoder.set_buffer(1, Some(d_initial_weight.as_ref()), 0);
    encoder.set_buffer(2, Some(d_params.as_ref()), 0);
    
    let (grid_size, group_size) = ctx.dispatch_1d(total_reach, 256);
    encoder.dispatch_thread_groups(grid_size, group_size);
    encoder.end_encoding();
    
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
    
    let result = d_reach.to_vec();
    
    for i in 0..np_nh {
        assert!((result[i] - initial_weight[i]).abs() < 1e-6,
            "Node 0 reach mismatch at {}: expected {}, got {}", i, initial_weight[i], result[i]);
    }
    for i in np_nh..total_reach {
        assert!(result[i] == 0.0, "Non-node-0 reach should be 0, got {} at {}", result[i], i);
    }
    
    println!("✓ vcfr_init_reach kernel works correctly");
}
