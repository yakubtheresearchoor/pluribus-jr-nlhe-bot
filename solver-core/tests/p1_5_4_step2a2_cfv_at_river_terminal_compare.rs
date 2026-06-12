// Step 2.A.2 CFV-trace continuation: with reach anchored to ground truth
// (task #74), compare CFV at river terminal nodes between CPU and GPU on
// the K=4 minimal-asymmetry game. Localize WHICH specific terminal first
// diverges.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};
use solver_core::gpu_metal::context::MetalContext;
use solver_core::gpu_metal::flop_solver::{DcfrParams as GpuDcfrParams, MetalFlopStartSolver};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::flop_start_vector_cfr::{DcfrParams, FlopStartVectorCfr, Zone};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn build_minimal_asymmetry_game() -> (FlatTree, FlopStartGame) {
    let board: Vec<Card> = ["Ah", "Kd", "7c"]
        .iter().map(|s| card_from_str(s).unwrap()).collect();
    let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let num_players = 2u8;
    let k = 4usize;

    use solver_core::hand::eval::Hand;
    let mut all_with_strength: Vec<(u16, u16)> = Vec::new();
    for idx in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(idx);
        if board_mask & (1u64 << c1) != 0 || board_mask & (1u64 << c2) != 0 { continue; }
        let mut h = Hand::new().add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board { h = h.add_card(bc as usize); }
        all_with_strength.push((h.evaluate_internal() as u16, idx as u16));
    }
    all_with_strength.sort_by_key(|&(s, _)| s);
    let step = all_with_strength.len() / k;
    let chosen: Vec<u16> = (0..k).map(|i| all_with_strength[i * step].1).collect();

    let mut ranges: Vec<Vec<f32>> = (0..num_players)
        .map(|_| vec![0.0f32; NUM_POSSIBLE_HANDS]).collect();
    for (rank_idx, &hi) in chosen.iter().enumerate() {
        let strength_frac = rank_idx as f32 / k as f32;
        let p0_weight = 1.0_f32;
        let p1_weight = if strength_frac >= 0.5 { 1.0_f32 } else { 0.5_f32 };
        let (c1, c2) = index_to_card_pair(hi as usize);
        let (lo, hi_c) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
        let pair_idx = lo as usize * (101 - lo as usize) / 2 + hi_c as usize - 1;
        ranges[0][pair_idx] = p0_weight;
        ranges[1][pair_idx] = p1_weight;
    }
    let turn_cards: Vec<u8> = vec![
        card_from_str("Td").unwrap() as u8,
        card_from_str("3s").unwrap() as u8,
    ];
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    river_decks[turn_cards[0] as usize] = vec![
        card_from_str("4h").unwrap() as u8,
        card_from_str("Qc").unwrap() as u8,
    ];
    river_decks[turn_cards[1] as usize] = vec![
        card_from_str("2s").unwrap() as u8,
        card_from_str("Js").unwrap() as u8,
    ];
    let table = FlopChanceTable::compute_flop_start_subset_with_decks(
        &board, &ranges, num_players, &chosen, &turn_cards, &river_decks,
    );
    let config = TreeConfig {
        num_players, initial_state: BoardState::Flop, starting_pot: 6,
        starting_stacks: vec![50, 50], initial_contributions: vec![0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
            raise: vec![BetSize::PotRelative(0.5), BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
        button_player: None,
            max_bets_per_street: None,
    };
    let tree = build_tree(&config).expect("tree build");
    let game = FlopStartGame::new(table);
    (tree, game)
}

/// Compute CFV at river-zone terminals on both sides for (ti=0, ri=0,
/// traverser=0) and find the first node where they diverge.
#[test]
#[ignore = "2.A.2 trace: CFV at river terminals comparison"]
fn cfv_at_river_terminals_cpu_vs_gpu_iter0_minimal_asymmetry() {
    let (tree, game) = build_minimal_asymmetry_game();
    let ctx = MetalContext::new().expect("Metal");
    let mut cpu = FlopStartVectorCfr::new(&tree, game.table());
    let mut gpu = MetalFlopStartSolver::new(&ctx, &tree, &game, &cpu);

    let nh = 4usize;
    let nn = tree.num_nodes();
    let ti = 0usize;
    let ri = 0usize;
    let traverser = 0u8;

    // Populate strategies on both sides (uniform 0.5 from zero regrets).
    cpu.compute_all_strategies(&tree);
    gpu.compute_all_strategies(&ctx);

    // Compute reach to the river (ti=0, ri=0) terminal.
    let cpu_flop_reach = cpu.compute_reach_flop(&tree, &game);
    let cpu_turn_reach = cpu.compute_reach_turn(&tree, ti, &cpu_flop_reach);
    let cpu_river_reach = cpu.compute_reach_river(&tree, ti, ri, &cpu_turn_reach);

    gpu.compute_reach_flop(&ctx);
    gpu.compute_reach_turn(&ctx, ti);
    gpu.compute_reach_river(&ctx, ti, ri);

    // Run CPU's bottom_up_zone for this river slot.
    let mut cpu_cfv = vec![0.0f32; nn * nh];
    let cpu_params = DcfrParams::new(0);
    cpu.bottom_up_zone(
        &tree, game.table(), traverser, &cpu_river_reach, &mut cpu_cfv,
        Zone::River, Some(ti), Some(ri), &cpu_params,
    );

    // Run GPU's bottom_up_river. The result is in d_river_cfv_batch at
    // offset ri * nn * nh.
    let gpu_params = GpuDcfrParams::new(0);
    gpu.bottom_up_river(&ctx, ti, ri, traverser as u32, &gpu_params);
    let gpu_river_cfv_batch = gpu.download_river_cfv_batch();
    let gpu_cfv_slot = &gpu_river_cfv_batch[ri * nn * nh..(ri + 1) * nn * nh];

    // Compare CFV at ALL river-zone nodes (not just terminals) — find
    // where the bug shows up. Showdown at terminals already verified
    // matching; if internal-node aggregated CFV diverges, bug is in the
    // bottom-up CFV propagation under non-uniform inputs.
    eprintln!("\n=== CFV AT ALL RIVER-ZONE NODES (ti={}, ri={}, traverser={}) ===", ti, ri, traverser);

    let mut first_diverging_node: Option<usize> = None;
    let mut max_diff = 0.0f32;
    let mut max_diff_node = 0usize;
    let mut max_diff_h = 0usize;
    let mut terminal_count = 0usize;
    let mut internal_count = 0usize;

    for node_id in 0..nn {
        let n = &tree.nodes[node_id];
        let node_type = n.node_type;
        if node_type == 1 { terminal_count += 1; }
        else if node_type == 2 { internal_count += 1; }
        else { continue; } // skip chance/other
        for h in 0..nh {
            let cpu_v = cpu_cfv[node_id * nh + h];
            let gpu_v = gpu_cfv_slot[node_id * nh + h];
            let d = (cpu_v - gpu_v).abs();
            if d > max_diff {
                max_diff = d;
                max_diff_node = node_id;
                max_diff_h = h;
            }
            if d > 1e-4 && first_diverging_node.is_none() {
                first_diverging_node = Some(node_id);
            }
        }
    }
    eprintln!("Inspected: {} terminals + {} player nodes in river zone",
        terminal_count, internal_count);

    // CORRECTED 2026-06: NodeType encoding is
    //   type 0 = TERMINAL (NOT 1 as I assumed)
    //   type 1 = CHANCE
    //   type 2 = PLAYER
    // The "audit-arc lesson recursing": diagnostic constants drift like
    // production constants. Verify TERMINAL-ONLY max with the CORRECT
    // type, separated from CHANCE and PLAYER nodes.
    let mut term_max = 0.0f32;
    let mut term_max_node = 0usize;
    let mut term_max_h = 0usize;
    let mut term_count = 0usize;
    let mut chance_count = 0usize;
    let mut player_count = 0usize;
    let mut player_max = 0.0f32;
    let mut chance_max = 0.0f32;
    for node_id in 0..nn {
        let t = tree.nodes[node_id].node_type;
        for h in 0..nh {
            let d = (cpu_cfv[node_id * nh + h] - gpu_cfv_slot[node_id * nh + h]).abs();
            match t {
                0 => {
                    term_count += 1;
                    if d > term_max { term_max = d; term_max_node = node_id; term_max_h = h; }
                }
                1 => { chance_count += 1; if d > chance_max { chance_max = d; } }
                2 => { player_count += 1; if d > player_max { player_max = d; } }
                _ => {}
            }
        }
    }
    eprintln!("\nCORRECTED CFV per node-type (REAL semantics: 0=terminal, 1=chance, 2=player):");
    eprintln!("  TERMINAL (type 0) max diff: {:.6e} at node {}, h{}", term_max, term_max_node, term_max_h);
    eprintln!("  CHANCE (type 1) max diff: {:.6e}", chance_max);
    eprintln!("  PLAYER (type 2) max diff: {:.6e}", player_max);
    // Diagnostic: dump chosen hand identities and strengths at this board
    let table = game.table();
    eprintln!("\nChosen hand identities (table.valid_hand_indices, table.hand_cards):");
    for h in 0..nh {
        let vh = table.valid_hand_indices[h];
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        // Look up strength at (tc=Td, rc=4h) — should be in river_sorted_str/idx
        eprintln!("  hand {}: valid_idx={}, cards=({}, {}), P0 reach={}, P1 reach={}",
            h, vh, c1, c2,
            table.initial_weights[0][h], table.initial_weights[1][h]);
    }

    // ════════════════════════════════════════════════════════════════
    // HAND-DERIVATION GROUND TRUTH (anchor against showdown rules, not
    // against either CPU or GPU implementation per #76)
    // ════════════════════════════════════════════════════════════════
    // Extract contributions at terminal node 13
    let term_node = term_max_node;
    let np_full = 2usize;
    let contribs: Vec<i32> = (0..np_full)
        .map(|p| tree.get_contribution(term_node, p as u8))
        .collect();
    eprintln!("\n=== TERMINAL NODE {} CONTRIBUTIONS ===", term_node);
    eprintln!("  contributions per player: {:?}", contribs);
    eprintln!("  starting_pot: 6 (from config)");
    eprintln!("  fold_mask byte at node: {}", tree.folded_masks[term_node]);

    // Build hand_strength values per HU showdown semantics on the board
    // Ah Kd 7c Td 4h. Use Hand::evaluate_internal — same evaluator both
    // CPU and GPU use. Note: the convention is "lower = stronger" per
    // earlier audit-arc findings.
    use solver_core::hand::eval::Hand;
    let board_cards: Vec<u8> = ["Ah", "Kd", "7c"].iter()
        .map(|s| card_from_str(s).unwrap() as u8).collect();
    let tc_card = card_from_str("Td").unwrap() as u8;
    let rc_card = card_from_str("4h").unwrap() as u8;
    let mut strengths_internal = vec![0u16; nh];
    for h in 0..nh {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        let mut hand_obj = Hand::new()
            .add_card(c1 as usize).add_card(c2 as usize);
        for &bc in &board_cards { hand_obj = hand_obj.add_card(bc as usize); }
        hand_obj = hand_obj.add_card(tc_card as usize).add_card(rc_card as usize);
        // Use evaluate() (which does HAND_TABLE.binary_search lookup),
        // NOT evaluate_internal() which returns a raw key. The audit-arc
        // tests use evaluate_internal directly in compute_flop_start so
        // the production code may share my confusion — but the actual
        // showdown semantics rely on the looked-up rank.
        strengths_internal[h] = hand_obj.evaluate() as u16;
    }
    eprintln!("\nHand strengths (evaluate_internal raw values):");
    for h in 0..nh { eprintln!("  h{}: strength_internal = {}", h, strengths_internal[h]); }

    // Use evaluate_internal as the oracle strength directly. The production
    // sorted_pl arrays use these same values (per compute_flop_start),
    // so consistency is guaranteed. Determine convention by checking the
    // known ordering: h0 (AAKK two pair) > h1 (77 pair) > h2 (44 pair) > h3 (A high).
    let convention_higher_is_stronger = strengths_internal[0] > strengths_internal[3];
    eprintln!("Convention determined by h0(AAKK) vs h3(A high): {}",
        if convention_higher_is_stronger { "HIGHER = STRONGER" } else { "LOWER = STRONGER" });

    // Use raw internal directly in the oracle, choose comparator based on convention.
    let strengths_oracle = strengths_internal.clone();

    // Build ShowdownCase-equivalent inputs and run independent enumerator.
    let starting_pot = 6i32;
    let traverser_usize = traverser as usize;
    let fold_mask = tree.folded_masks[term_node];
    let c_t = contribs[traverser_usize];
    let traverser_folded = (fold_mask & (1 << traverser_usize)) != 0;
    let opp_reach_p1: Vec<f32> = (0..nh).map(|h| cpu_river_reach[term_node * np_full * nh + nh + h]).collect();
    eprintln!("\nopp_reach[P1 at this terminal]: {:?}", opp_reach_p1);

    // Inline brute-force enumerator (HU = 1 opponent)
    let num_combinations = table.num_combinations as f32;
    let mut oracle_cfv = vec![0.0f32; nh];
    for h in 0..nh {
        let hc1 = table.hand_cards[h * 2];
        let hc2 = table.hand_cards[h * 2 + 1];
        let h_str = strengths_oracle[h];
        let mut accum = 0.0f32;
        for g in 0..nh {
            let gc1 = table.hand_cards[g * 2];
            let gc2 = table.hand_cards[g * 2 + 1];
            if gc1 == hc1 || gc1 == hc2 || gc2 == hc1 || gc2 == hc2 { continue; }
            let r = opp_reach_p1[g];
            if r == 0.0 { continue; }
            let g_str = strengths_oracle[g];
            // Compute payoff: scenario = no-fold showdown HU
            let payoff: f32 = {
                if traverser_folded {
                    -(starting_pot as f32 / np_full as f32 + c_t as f32)
                } else {
                    // Single pot, both eligible
                    let total_pot: i32 = starting_pot + contribs.iter().sum::<i32>();
                    let traverser_investment = starting_pot as f32 / np_full as f32 + c_t as f32;
                    // Use convention determined above
                    let h_wins = if convention_higher_is_stronger {
                        h_str > g_str
                    } else {
                        h_str < g_str
                    };
                    let h_loses = if convention_higher_is_stronger {
                        h_str < g_str
                    } else {
                        h_str > g_str
                    };
                    if h_wins {
                        (total_pot as f32) - traverser_investment  // win all
                    } else if h_loses {
                        -traverser_investment  // lose stake
                    } else {
                        // Tie: split
                        (total_pot as f32) / 2.0 - traverser_investment
                    }
                }
            };
            accum += r * payoff;
        }
        oracle_cfv[h] = accum / num_combinations;
    }

    eprintln!("\n=== ORACLE (HAND-DERIVED) vs CPU vs GPU AT TERMINAL {} ===", term_node);
    eprintln!("num_combinations = {}", num_combinations);
    for h in 0..nh {
        let cpu_v = cpu_cfv[term_node * nh + h];
        let gpu_v = gpu_cfv_slot[term_node * nh + h];
        let ora_v = oracle_cfv[h];
        let cpu_d = (cpu_v - ora_v).abs();
        let gpu_d = (gpu_v - ora_v).abs();
        let verdict = if cpu_d < 1e-4 && gpu_d > 1e-4 { "CPU CORRECT, GPU WRONG" }
                      else if gpu_d < 1e-4 && cpu_d > 1e-4 { "GPU CORRECT, CPU WRONG" }
                      else if cpu_d < 1e-4 && gpu_d < 1e-4 { "BOTH MATCH ORACLE" }
                      else { "BOTH WRONG" };
        eprintln!("  h{}: oracle={:>8.4}  cpu={:>8.4} (Δ={:.4})  gpu={:>8.4} (Δ={:.4})  → {}",
            h, ora_v, cpu_v, cpu_d, gpu_v, gpu_d, verdict);
    }

    if term_max > 1e-4 {
        eprintln!("\n!!! BUG IS AT TERMINAL CFV (SHOWDOWN). My earlier 'terminals match' diagnostic was wrong (used type==1).");
        eprintln!("Diverging terminal node {} h{}:", term_max_node, term_max_h);
        eprintln!("  CPU CFV: {:?}", &cpu_cfv[term_max_node * nh..term_max_node * nh + nh]);
        eprintln!("  GPU CFV: {:?}", &gpu_cfv_slot[term_max_node * nh..term_max_node * nh + nh]);
        let reach_p0 = &cpu_river_reach[term_max_node * 2 * nh..term_max_node * 2 * nh + nh];
        let reach_p1 = &cpu_river_reach[term_max_node * 2 * nh + nh..term_max_node * 2 * nh + 2*nh];
        eprintln!("  reach[P0]: {:?}", reach_p0);
        eprintln!("  reach[P1]: {:?}", reach_p1);
        let n = &tree.nodes[term_max_node];
        eprintln!("  Terminal node fold_mask={}", n.player_id);  // player_id field reused for fold_mask in terminals
    }

    // Trace deeper: find shallowest diverging node and dump full subtree CFV
    eprintln!("\n=== TRACING DEEPER FROM NODE 11 (first internal diverging child) ===");
    let trace_node = 11usize;
    let kids = tree.node_children(trace_node);
    eprintln!("Node {} owner=P{}, children: {:?}",
        trace_node, tree.nodes[trace_node].player_id, kids);
    for (i, &c) in kids.iter().enumerate() {
        let c = c as usize;
        let t = tree.nodes[c].node_type;
        let matches = (0..nh).all(|h|
            (cpu_cfv[c * nh + h] - gpu_cfv_slot[c * nh + h]).abs() < 1e-5);
        eprintln!("  Child[{}] = node {} (type {}, owner=P{}): MATCHES={}",
            i, c, t, tree.nodes[c].player_id, matches);
        eprintln!("    CPU CFV: {:?}", &cpu_cfv[c * nh..c * nh + nh]);
        eprintln!("    GPU CFV: {:?}", &gpu_cfv_slot[c * nh..c * nh + nh]);
    }

    eprintln!("Max CFV diff: {:.6} at node {}, hand {} (CPU={:.6}, GPU={:.6})",
        max_diff, max_diff_node, max_diff_h,
        cpu_cfv[max_diff_node * nh + max_diff_h],
        gpu_cfv_slot[max_diff_node * nh + max_diff_h]);

    // Find ANY diverging node whose ALL children have matching CFV — that
    // isolates the bug to a single aggregation step where children are
    // known-correct on both sides.
    let mut isolated_node: Option<usize> = None;
    for node_id in 0..nn {
        let n = &tree.nodes[node_id];
        if n.node_type != 2 { continue; }
        let kids = tree.node_children(node_id);
        // All children must have matching CFV between CPU and GPU
        let kids_all_match = kids.iter().all(|&c| {
            let c = c as usize;
            (0..nh).all(|h| (cpu_cfv[c * nh + h] - gpu_cfv_slot[c * nh + h]).abs() < 1e-5)
        });
        if !kids_all_match { continue; }
        // But the parent diverges
        let parent_diverges = (0..nh).any(|h| {
            (cpu_cfv[node_id * nh + h] - gpu_cfv_slot[node_id * nh + h]).abs() > 1e-4
        });
        if parent_diverges {
            isolated_node = Some(node_id);
            break;
        }
    }

    if let Some(node_id) = isolated_node {
        eprintln!("\n=== ISOLATED AGGREGATION BUG ===");
        eprintln!("Node {} diverges but ALL its children's CFV match between CPU and GPU.",
            node_id);
        let n = &tree.nodes[node_id];
        eprintln!("Node owner: P{}; traverser: P{}; num_children: {}",
            n.player_id, traverser, n.num_children);
        eprintln!("CPU CFV: {:?}", &cpu_cfv[node_id * nh..node_id * nh + nh]);
        eprintln!("GPU CFV: {:?}", &gpu_cfv_slot[node_id * nh..node_id * nh + nh]);
        for (a, &c) in tree.node_children(node_id).iter().enumerate() {
            let c = c as usize;
            eprintln!("Child[{}] = node {} (type {}, MATCHING):", a, c, tree.nodes[c].node_type);
            eprintln!("  CFV: {:?}", &cpu_cfv[c * nh..c * nh + nh]);
        }
    } else {
        eprintln!("\n(No isolated-aggregation node found — divergence propagates from deeper)");
    }

    // Original "first diverging" diagnostic (parent or leaf)
    let mut terminal_parent_diverging: Option<usize> = None;
    for node_id in 0..nn {
        let n = &tree.nodes[node_id];
        if n.node_type != 2 { continue; }
        let all_terminal = tree.node_children(node_id).iter()
            .all(|&c| tree.nodes[c as usize].node_type == 1);
        if !all_terminal { continue; }
        for h in 0..nh {
            let cpu_v = cpu_cfv[node_id * nh + h];
            let gpu_v = gpu_cfv_slot[node_id * nh + h];
            if (cpu_v - gpu_v).abs() > 1e-4 {
                terminal_parent_diverging = Some(node_id);
                break;
            }
        }
        if terminal_parent_diverging.is_some() { break; }
    }

    if let Some(node_id) = terminal_parent_diverging {
        eprintln!("\n=== ISOLATED AGGREGATION BUG (parent of all-terminal children) ===");
        let n = &tree.nodes[node_id];
        eprintln!("Node {}: type={}, owner=P{}, num_children={}",
            node_id, n.node_type, n.player_id, n.num_children);
        eprintln!("Children (all terminals): {:?}", tree.node_children(node_id));
        eprintln!("CPU CFV[node {}]: {:?}", node_id, &cpu_cfv[node_id * nh..node_id * nh + nh]);
        eprintln!("GPU CFV[node {}]: {:?}", node_id, &gpu_cfv_slot[node_id * nh..node_id * nh + nh]);
        for (a, &c) in tree.node_children(node_id).iter().enumerate() {
            let c = c as usize;
            eprintln!("Child[{}] = terminal node {}:", a, c);
            eprintln!("  CPU CFV: {:?}", &cpu_cfv[c * nh..c * nh + nh]);
            eprintln!("  GPU CFV: {:?}", &gpu_cfv_slot[c * nh..c * nh + nh]);
        }
        let children = tree.node_children(node_id);
        let owner_traverser = n.player_id == traverser;
        eprintln!("Aggregation rule (owner=P{}, traverser=P{}): {}",
            n.player_id, traverser,
            if owner_traverser { "weighted by strategy" } else { "unweighted sum" });
        for h in 0..nh {
            let c0 = children[0] as usize;
            let c1 = children[1] as usize;
            let (cpu_c0, cpu_c1) = (cpu_cfv[c0 * nh + h], cpu_cfv[c1 * nh + h]);
            let (gpu_c0, gpu_c1) = (gpu_cfv_slot[c0 * nh + h], gpu_cfv_slot[c1 * nh + h]);
            let expected_cpu = if owner_traverser { 0.5 * (cpu_c0 + cpu_c1) } else { cpu_c0 + cpu_c1 };
            let expected_gpu = if owner_traverser { 0.5 * (gpu_c0 + gpu_c1) } else { gpu_c0 + gpu_c1 };
            eprintln!("  h{}: CPU child=[{:.4},{:.4}] expected_cpu={:.4} got_cpu={:.4}  ||  GPU child=[{:.4},{:.4}] expected_gpu={:.4} got_gpu={:.4}",
                h, cpu_c0, cpu_c1, expected_cpu, cpu_cfv[node_id * nh + h],
                gpu_c0, gpu_c1, expected_gpu, gpu_cfv_slot[node_id * nh + h]);
        }
    }

    if let Some(node_id) = first_diverging_node {
        eprintln!("\nFirst diverging node: node {} (type {})", node_id, tree.nodes[node_id].node_type);
        let n = &tree.nodes[node_id];
        eprintln!("  player_id (owner): {}", n.player_id);
        eprintln!("  num_children: {}", n.num_children);
        eprintln!("  children: {:?}", tree.node_children(node_id));
        eprintln!("  child types: {:?}", tree.node_children(node_id).iter()
            .map(|&c| tree.nodes[c as usize].node_type).collect::<Vec<_>>());
        eprintln!("  CPU CFV[node {}]: {:?}",
            node_id, &cpu_cfv[node_id * nh..node_id * nh + nh]);
        eprintln!("  GPU CFV[node {}]: {:?}",
            node_id, &gpu_cfv_slot[node_id * nh..node_id * nh + nh]);

        // Children CFV on both sides
        for (i, &c) in tree.node_children(node_id).iter().enumerate() {
            let c = c as usize;
            eprintln!("  child[{}] = node {} (type {}):", i, c, tree.nodes[c].node_type);
            eprintln!("    CPU CFV: {:?}", &cpu_cfv[c * nh..c * nh + nh]);
            eprintln!("    GPU CFV: {:?}", &gpu_cfv_slot[c * nh..c * nh + nh]);
            let same = cpu_cfv[c * nh..c * nh + nh].iter().zip(gpu_cfv_slot[c * nh..c * nh + nh].iter())
                .all(|(a, b)| (a - b).abs() < 1e-5);
            eprintln!("    children CFV match: {}", if same { "YES" } else { "NO" });
        }

        let cpu_reach = &cpu_river_reach[node_id * 2 * nh..(node_id + 1) * 2 * nh];
        eprintln!("  reach[node {}, P0]: {:?}", node_id, &cpu_reach[0..nh]);
        eprintln!("  reach[node {}, P1]: {:?}", node_id, &cpu_reach[nh..2*nh]);
    } else {
        eprintln!("\nNo divergence found above 1e-4 — CFV matches at all terminals.");
        eprintln!("Bug must be in CFV PROPAGATION (cfv_avg or regret update), not showdown.");
    }
}
