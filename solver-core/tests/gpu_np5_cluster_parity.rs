//! Standalone parity: the np5 CLUSTER-MASS Metal kernels vs the validated CPU
//! reference cluster_mass::mass_cluster_pairs_fast. Dispatches the 5 prep + main
//! kernels directly on one synthetic lone-fold terminal (np=5, K=4 opponents),
//! uniform+random reaches, and compares cfv·nc/payoff to the CPU mass.
#![cfg(feature = "metal")]

use solver_core::gpu_metal::{MetalBuffer, MetalContext};
use solver_core::solver::cluster_mass::mass_cluster_pairs_fast;
use solver_core::tree::flat::FlatNode;

const STRIDE: usize = 28200;

#[test]
fn gpu_np5_cluster_matches_cpu() {
    let ctx = MetalContext::new().expect("Metal");
    let dev = ctx.device();
    let np = 5usize;
    // river-ish hand universe: valid 2-card combos off a 5-card board (deck 47).
    let board = [51u8, 46, 20, 9, 30];
    let on: Vec<bool> = (0..52).map(|c| board.contains(&(c as u8))).collect();
    let deck: Vec<u8> = (0..52u8).filter(|&c| !on[c as usize]).collect();
    let mut hand_cards: Vec<u8> = Vec::new();
    for i in 0..deck.len() { for j in (i+1)..deck.len() { hand_cards.push(deck[i]); hand_cards.push(deck[j]); } }
    let nh = hand_cards.len()/2;

    // reach [np*nh]: player 0 = traverser (unused in mass), 1..5 = opponents.
    let mut lcg = 0x1234_5678u64;
    let mut rf = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); ((lcg>>33) as f32)/(1u64<<31) as f32 };
    let mut reach = vec![0.0f32; np*nh];
    for p in 0..np { for h in 0..nh { reach[p*nh+h] = if rf()<0.3 {0.0} else {rf()}; } }

    // p2h[52*52]
    let mut p2h = vec![-1i32; 52*52];
    for h in 0..nh { let (a,b)=(hand_cards[h*2] as usize, hand_cards[h*2+1] as usize); p2h[a*52+b]=h as i32; p2h[b*52+a]=h as i32; }

    // one terminal: node 0, no rake (board_state 3 makes flop_seen false? kernel:
    // flop_seen = board_state!=3 → false → no rake), trav=0 not folded, equal contribs.
    let mut node = FlatNode::terminal();
    node.board_state = 3; // → no rake
    let nodes = vec![node];
    let contributions = vec![10i32; np]; // equal
    let folded = vec![0u16]; // nobody folded (trav wins uncontested here — payoff = pot-stake)
    let term_nodes = vec![0u32];
    let starting_pot = 15i32;
    let nc = 1.0f32;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct P { nh:i32, np:i32, traverser:i32, n_term:i32, starting_pot:i32, rake_rate:f32, rake_cap:f32, num_combinations:f32, a:i32, b:i32, c:u32 }
    let params = P { nh:nh as i32, np:np as i32, traverser:0, n_term:1, starting_pot, rake_rate:0.0, rake_cap:0.0, num_combinations:nc, a:0, b:0, c:0 };
    let pbytes = std::mem::size_of::<P>() as u64;
    let pptr = &params as *const _ as *const std::ffi::c_void;

    let d_tab = MetalBuffer::<f32>::zeros(dev, STRIDE);
    let d_reach = MetalBuffer::from_slice(dev, &reach);
    let d_hc = MetalBuffer::from_slice(dev, &hand_cards);
    let d_p2h = MetalBuffer::from_slice(dev, &p2h);
    let d_term = MetalBuffer::from_slice(dev, &term_nodes);
    let d_nodes = MetalBuffer::from_slice(dev, &nodes);
    let d_contrib = MetalBuffer::from_slice(dev, &contributions);
    let d_fold = MetalBuffer::from_slice(dev, &folded);
    let d_cfv = MetalBuffer::<f32>::zeros(dev, nh);

    let pm = ctx.create_pipeline("vcfr_np5cl_prep_pm").unwrap();
    let scc = ctx.create_pipeline("vcfr_np5cl_prep_scc").unwrap();
    let dota = ctx.create_pipeline("vcfr_np5cl_prep_dota").unwrap();
    let pb = ctx.create_pipeline("vcfr_np5cl_prep_b").unwrap();
    let main = ctx.create_pipeline("vcfr_np5cl_main").unwrap();

    let cmd = ctx.new_command_buffer();
    // prep_pm (1,52): buffers tab,term,reach,hc,params
    let enc2 = |pipe:&metal::ComputePipelineState, bufs:&[&metal::BufferRef], pidx:u64, gy:usize| {
        let e = cmd.new_compute_command_encoder();
        e.set_compute_pipeline_state(pipe);
        for (i,b) in bufs.iter().enumerate() { e.set_buffer(i as u64, Some(b), 0); }
        e.set_bytes(pidx, pbytes, pptr);
        let max = pipe.max_total_threads_per_threadgroup() as usize;
        let (g,t) = ctx.dispatch_2d(1, gy, max);
        e.dispatch_thread_groups(g,t); e.end_encoding();
    };
    enc2(&pm, &[d_tab.as_ref(), d_term.as_ref(), d_reach.as_ref(), d_hc.as_ref()], 4, 52);
    // scc (1): grid 1d
    {
        let e = cmd.new_compute_command_encoder();
        e.set_compute_pipeline_state(&scc);
        e.set_buffer(0, Some(d_tab.as_ref()),0); e.set_buffer(1, Some(d_term.as_ref()),0);
        e.set_buffer(2, Some(d_reach.as_ref()),0); e.set_buffer(3, Some(d_hc.as_ref()),0);
        e.set_bytes(4, pbytes, pptr);
        let (g,t)=ctx.dispatch_1d(1, scc.max_total_threads_per_threadgroup() as usize);
        e.dispatch_thread_groups(g,t); e.end_encoding();
    }
    enc2(&dota, &[d_tab.as_ref(), d_term.as_ref()], 2, 52);
    enc2(&pb, &[d_tab.as_ref(), d_term.as_ref()], 2, 52);
    // main (1,nh): tab is buffer 8
    {
        let e = cmd.new_compute_command_encoder();
        e.set_compute_pipeline_state(&main);
        e.set_buffer(0, Some(d_cfv.as_ref()),0); e.set_buffer(1, Some(d_term.as_ref()),0);
        e.set_buffer(2, Some(d_nodes.as_ref()),0); e.set_buffer(3, Some(d_contrib.as_ref()),0);
        e.set_buffer(4, Some(d_fold.as_ref()),0); e.set_buffer(5, Some(d_reach.as_ref()),0);
        e.set_buffer(6, Some(d_hc.as_ref()),0); e.set_buffer(7, Some(d_p2h.as_ref()),0);
        e.set_buffer(8, Some(d_tab.as_ref()),0); e.set_bytes(9, pbytes, pptr);
        let max = main.max_total_threads_per_threadgroup() as usize;
        let (g,t)=ctx.dispatch_2d(1, nh, max);
        e.dispatch_thread_groups(g,t); e.end_encoding();
    }
    cmd.commit(); cmd.wait_until_completed();

    // CPU reference
    let opp: Vec<&[f32]> = (1..np).map(|p| &reach[p*nh..(p+1)*nh]).collect();
    let mass = mass_cluster_pairs_fast(&opp, &hand_cards, nh);
    // payoff for trav=0, uncontested win, no rake: total_pot - stake
    let total_pot = starting_pot + contributions.iter().sum::<i32>();
    let stake = starting_pot as f32/np as f32 + contributions[0] as f32;
    let payoff = total_pot as f32 - stake;

    let gpu = d_cfv.as_slice();
    let mut scale = 1e-9f64;
    for h in 0..nh { scale = scale.max((payoff as f64 * mass[h] as f64).abs()); }
    let mut worst = 0.0f64;
    for h in 0..nh {
        let expect = payoff * mass[h] / nc;
        worst = worst.max((gpu[h] as f64 - expect as f64).abs()/scale);
    }
    eprintln!("np5 cluster GPU vs CPU-fast: worst scale-rel = {worst:.3e} (nh={nh})");
    assert!(worst < 1e-3, "cluster kernel diverges: {worst}");
}
