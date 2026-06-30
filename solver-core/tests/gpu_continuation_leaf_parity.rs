//! GPU continuation-leaf parity: the `MetalVectorCfr` HU continuation kernel
//! (vcfr_continuation_reduce + _fill) must reproduce the closed-form HU Arm-1
//! showdown on the SAME reach it used. We install synthetic-but-valid runout
//! tables + a synthetic map, run one batched pass, read back the leaf reach +
//! cfv, recompute the closed form on CPU from that exact reach, and compare.
//! (The closed form itself is validated against the CPU recursion in
//! hu_continuation_closed_form.rs.)

#![cfg(feature = "metal")]

use solver_core::card::{index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::{MetalContext, MetalVectorCfr};
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
fn gpu_hu_continuation_matches_closed_form() {
    let np = 2u8;
    // Small flop game (subset nh) so the test is fast.
    let board: Vec<Card> = vec![3, 19, 35]; // arbitrary non-colliding flop
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let valid: Vec<u16> = (0..NUM_POSSIBLE_HANDS)
        .filter(|&hi| { let (c1, c2) = index_to_card_pair(hi);
            board_mask & (1u64 << c1) == 0 && board_mask & (1u64 << c2) == 0 })
        .map(|hi| hi as u16).collect();
    let nh_target = 48usize;
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

    // Depth-limited flop tree: childless chance leaves = continuation points.
    let cfg = TreeConfig {
        num_players: np,
        initial_state: BoardState::Flop,
        starting_pot: 20,
        starting_stacks: vec![100; np as usize],
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
    assert!(!leaf_nodes.is_empty(), "no continuation leaves in tree");

    // Synthetic valid tables + map.
    let mut rng = Lcg(0xDEAD_BEEF_1234_5678);
    let nb = 8usize;
    let (mut f_w, mut f_t, mut f_l, mut f_n) =
        (vec![0.0f32; nb*nb], vec![0.0f32; nb*nb], vec![0.0f32; nb*nb], vec![0.0f32; nb*nb]);
    for i in 0..nb*nb {
        let compat = 0.3 + 0.7 * rng.f();
        let (a, b, c) = (rng.f(), rng.f(), rng.f());
        let s = a + b + c;
        f_w[i] = compat*a/s; f_t[i] = compat*b/s; f_l[i] = compat*c/s; f_n[i] = f_w[i]+f_t[i]+f_l[i];
    }
    let map: Vec<u16> = (0..nh).map(|h| (h % nb) as u16).collect();

    // GPU solver from the flop table's HU sorted arrays.
    let (sos, soi, sps, spi, _) = table.sorted_opp_arrays_base();
    let iw: Vec<Vec<f32>> = (0..np as usize).map(|p| table.initial_weights[p].clone()).collect();
    let ctx = MetalContext::new().expect("Metal");
    let mut gpu = MetalVectorCfr::new(&ctx, &tree, nh, &iw, &sos, &soi, &sps, &spi, &table.hand_cards, nc);
    gpu.set_continuation(&ctx, &leaf_nodes, &map, nb, &f_w, &f_t, &f_l, &f_n, 0.0, 0.0, 0, 7);

    gpu.run_batched(&ctx, &tree, 1);

    // After the pass, d_reach = last traverser's (np-1=1) reach; cfv[leaf] = its
    // continuation. Recompute the closed form from that exact opp(=0) reach.
    let reach = gpu.reach_slice();
    let cfv = gpu.cfv_slice();
    let traverser = (np - 1) as usize;
    let opp = if traverser == 0 { 1 } else { 0 };

    let mut worst = 0.0f32;
    for &lnode in &leaf_nodes {
        let ln = lnode as usize;
        let opp_r = &reach[(ln * np as usize + opp) * nh..(ln * np as usize + opp) * nh + nh];
        // reduce → bucket reach
        let mut br = vec![0.0f32; nb];
        for h in 0..nh { br[map[h] as usize] += opp_r[h]; }
        // closed form per bucket (rake 0)
        let c_t = tree.get_contribution(ln, traverser as u8);
        let half_pot = cfg.starting_pot as f32 / np as f32 + c_t as f32;
        let mut cfv_b = vec![0.0f32; nb];
        for bt in 0..nb {
            let mut accum = 0.0f32;
            for bo in 0..nb {
                let r = br[bo];
                if r == 0.0 { continue; }
                let i = bt*nb + bo;
                if f_n[i] == 0.0 { continue; }
                accum += r * (f_w[i] - f_l[i]); // rps = 0
            }
            cfv_b[bt] = half_pot * accum;
        }
        // expand + /nc, compare to GPU cfv at this leaf
        for h in 0..nh {
            let want = if nc > 0.0 { cfv_b[map[h] as usize] / nc as f32 } else { cfv_b[map[h] as usize] };
            let got = cfv[ln * nh + h];
            worst = worst.max((want - got).abs());
        }
    }
    let scale = cfv.iter().map(|x| x.abs()).fold(1e-6, f32::max);
    assert!(worst / scale < 1e-4, "GPU continuation diverges from closed form: worst={worst} scale={scale}");
    eprintln!("GPU continuation parity OK: {} leaves, nb={nb}, nh={nh}, worst_abs={worst:.3e}", leaf_nodes.len());
}
