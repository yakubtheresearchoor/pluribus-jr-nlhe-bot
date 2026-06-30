//! GPU MULTIWAY continuation parity (np=3): the MC-sampled GPU continuation
//! (vcfr_continuation_*_mw) must match the CPU EXHAUSTIVE Arm-1 collapsed
//! showdown within sampling noise. We install synthetic valid tables, run one
//! batched pass, read back the leaf reach + cfv, compute the exact CPU value on
//! that exact reach, and compare RMS-relative within MC tolerance.

#![cfg(feature = "metal")]

use solver_core::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
use solver_core::solver::bucketed_showdown::{
    bucketed_showdown_cfv_design1_collapsed, BucketedRunoutTables,
};
use solver_core::solver::flop_start_game::FlopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree_depth_limited;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
}

#[test]
fn gpu_mw_continuation_matches_cpu_exhaustive() {
    let np = 3u8;
    let board: Vec<Card> = vec![3, 19, 35];
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let valid: Vec<u16> = (0..NUM_POSSIBLE_HANDS)
        .filter(|&hi| { let (c1, c2) = index_to_card_pair(hi);
            board_mask & (1u64 << c1) == 0 && board_mask & (1u64 << c2) == 0 })
        .map(|hi| hi as u16).collect();
    let nh_target = 36usize;
    let step = valid.len() / nh_target;
    let hands: Vec<u16> = valid.iter().step_by(step).copied().take(nh_target).collect();
    let nbc: Vec<u8> = (0..52u8).filter(|&c| board_mask & (1u64 << c) == 0).collect();
    let turn = nbc[0];
    let mut rd: Vec<Vec<u8>> = vec![vec![]; 52];
    rd[turn as usize] = vec![nbc[1]];
    let ranges: Vec<Vec<f32>> = (0..np).map(|_| vec![1.0f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS]).collect();
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(&board, &ranges, np, &hands, &[turn], &rd);
    let nh = table.num_valid;
    let nc = table.num_combinations;

    let cfg = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 30,
        starting_stacks: vec![400; np as usize],
        initial_contributions: vec![0; np as usize],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0,
        merging_threshold: 0.0, button_player: None,
        max_bets_per_street: None, no_open_limp: false, threebet_or_fold: false,
    };
    let tree = build_tree_depth_limited(&cfg).expect("tree");
    let leaf_nodes: Vec<u32> = (0..tree.num_nodes())
        .filter(|&n| tree.nodes[n].is_chance() && tree.node_children(n).is_empty())
        .map(|n| n as u32).collect();
    assert!(!leaf_nodes.is_empty(), "no continuation leaves");

    let mut rng = Lcg(0x0BAD_F00D_CAFE_1234);
    let nb = 8usize;
    let (mut f_w, mut f_t, mut f_l, mut f_n) =
        (vec![0.0f32; nb*nb], vec![0.0f32; nb*nb], vec![0.0f32; nb*nb], vec![0.0f32; nb*nb]);
    for i in 0..nb*nb {
        let compat = 0.5 + 0.5 * rng.f(); // keep compat high → little blocking, lower variance
        let (a, b, c) = (rng.f(), rng.f(), rng.f());
        let s = a + b + c;
        f_w[i] = compat*a/s; f_t[i] = compat*b/s; f_l[i] = compat*c/s; f_n[i] = f_w[i]+f_t[i]+f_l[i];
    }
    let map: Vec<u16> = (0..nh).map(|h| (h % nb) as u16).collect();
    let tables = BucketedRunoutTables { nb, f_w: f_w.clone(), f_t: f_t.clone(), f_l: f_l.clone(), f_n: f_n.clone() };

    let (sos, soi, sps, spi, _) = table.sorted_opp_arrays_base();
    let iw: Vec<Vec<f32>> = (0..np as usize).map(|p| table.initial_weights[p].clone()).collect();
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
    let sample_m = 60_000u32; // low MC noise for the parity check
    gpu.set_continuation(&ctx, &leaf_nodes, &map, nb, &f_w, &f_t, &f_l, &f_n, 0.0, 0.0, sample_m, 12345);

    gpu.run_batched(&ctx, &tree, 1);

    let reach = gpu.reach_slice();
    let cfv = gpu.cfv_slice();
    let traverser = (np - 1) as usize; // 2
    let opps: Vec<usize> = (0..np as usize).filter(|&p| p != traverser).collect();

    let mut sse = 0.0f64;
    let mut ss = 0.0f64;
    let mut worst_rel = 0.0f32;
    let mut compared = 0usize;
    let mut sse_a1 = 0.0f64; let mut ss_a1 = 0.0f64;
    let mut sse_a2 = 0.0f64; let mut ss_a2 = 0.0f64;
    let mut max_abs = 0.0f32; let mut global_scale = 0.0f32;
    let mut arm1_leaves = 0usize;
    let mut arm2_leaves = 0usize;
    for &lnode in &leaf_nodes {
        let ln = lnode as usize;
        let fm = tree.get_folded_mask(ln);
        let active_eq = {
            let mut r = None; let mut eq = true;
            for pp in 0..np { if fm & (1 << pp) != 0 { continue; }
                let cc = tree.get_contribution(ln, pp);
                match r { None => r = Some(cc), Some(v) => if v != cc { eq = false; } } }
            eq
        };
        let is_arm1 = fm == 0 && active_eq;
        if is_arm1 { arm1_leaves += 1; } else { arm2_leaves += 1; }
        let dbg = false;
        // per-opp bucket reach
        let mut br: Vec<Vec<f32>> = Vec::new();
        for &opp in &opps {
            let r = &reach[(ln * np as usize + opp) * nh..(ln * np as usize + opp) * nh + nh];
            let mut b = vec![0.0f32; nb];
            for h in 0..nh { b[map[h] as usize] += r[h]; }
            br.push(b);
        }
        let views: Vec<&[f32]> = br.iter().map(|v| v.as_slice()).collect();
        let contribs: Vec<i32> = (0..np).map(|p| tree.get_contribution(ln, p)).collect();
        let cpu_bucket = bucketed_showdown_cfv_design1_collapsed(
            &views, &tables, &contribs, fm, traverser, np, cfg.starting_pot, 0.0, 0.0, true,
        );
        let _ = dbg;
        for h in 0..nh {
            let want = if nc > 0.0 { cpu_bucket[map[h] as usize] / nc as f32 } else { cpu_bucket[map[h] as usize] };
            let got = cfv[ln * nh + h];
            let e = ((want - got) as f64).powi(2);
            let s = (want as f64).powi(2);
            sse += e; ss += s;
            if is_arm1 { sse_a1 += e; ss_a1 += s; } else { sse_a2 += e; ss_a2 += s; }
            max_abs = max_abs.max((want - got).abs());
            global_scale = global_scale.max(want.abs());
            compared += 1;
        }
    }
    let _ = (worst_rel, sse_a2, ss_a2);
    // Global-scale metric: per-element relative error is meaningless on the
    // tree's near-zero Arm-2 leaves (tiny/tinier). Arm-1 end-to-end and the
    // Arm-2 net_expected port logic are pinned by their own tests
    // (RMS_a1 below + mw_arm2_port_parity); here we require the GPU output to
    // track the CPU exhaustive on a global pot scale, which still fails hard on
    // any structural bug (those produce O(scale) divergence).
    let rms_a1 = (sse_a1 / ss_a1.max(1e-12)).sqrt();
    let rel_global = max_abs / global_scale.max(1e-9);
    eprintln!("MW continuation parity: leaves={} (arm1={arm1_leaves}, arm2={arm2_leaves}), compared={compared}, M={sample_m}",
              leaf_nodes.len());
    eprintln!("  arm1 RMS_rel={rms_a1:.4}  max_abs={max_abs:.3e}  global_scale={global_scale:.3e}  max_abs/scale={rel_global:.4}");
    assert!(rms_a1 < 0.03, "GPU Arm-1 continuation diverges: RMS_a1={rms_a1}");
    assert!(rel_global < 0.05, "GPU multiway continuation diverges on global scale: {rel_global}");
}
