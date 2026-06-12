// Phase 1.A memory check: what's the distribution of action counts per
// player infoset on the corrected tree? If MAX_NA_POSTFLOP=4 padding wastes most
// of the regret/strategy buffer (most infosets have 2 actions), packing
// to actual NA could halve the storage.

use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::MAX_NA_POSTFLOP;

#[test]
fn action_count_distribution_on_corrected_tree() {
    let cfg = TreeConfig {
        num_players: 6, initial_state: BoardState::Flop, starting_pot: 30,
        starting_stacks: vec![200; 6],
        initial_contributions: vec![10, 5, 5, 5, 5, 5],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0, force_allin_threshold: 1.0, merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };
    let tree = build_tree(&cfg).unwrap();
    eprintln!("\n=== Action count distribution on corrected tree ===");
    eprintln!("MAX_NA_POSTFLOP constant = {} (regret buffer pads to this per infoset)\n", MAX_NA_POSTFLOP);

    let mut hist: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
    let mut total_actions = 0usize;
    let mut total_max_na_slots = 0usize;
    let mut player_node_count = 0usize;
    for n in &tree.nodes {
        if !n.is_player() { continue; }
        player_node_count += 1;
        let nc = n.num_children;
        *hist.entry(nc).or_insert(0) += 1;
        total_actions += nc as usize;
        total_max_na_slots += MAX_NA_POSTFLOP;
    }

    eprintln!("Player infosets: {}", player_node_count);
    eprintln!("\nNum-children histogram:");
    let mut keys: Vec<u16> = hist.keys().copied().collect();
    keys.sort();
    for k in &keys {
        let count = hist[k];
        let pct = count as f64 / player_node_count as f64 * 100.0;
        eprintln!("  {} children: {:>10} infosets ({:5.1}%)", k, count, pct);
    }

    let actual_storage_factor = total_actions as f64 / total_max_na_slots as f64;
    let waste_factor = 1.0 - actual_storage_factor;
    eprintln!("\nStorage analysis:");
    eprintln!("  Total actual actions: {}", total_actions);
    eprintln!("  Total MAX_NA_POSTFLOP slots:   {}", total_max_na_slots);
    eprintln!("  Actual / padded:      {:.3} ({:.1}% utilized)",
        actual_storage_factor, actual_storage_factor * 100.0);
    eprintln!("  Wasted padding:       {:.1}%", waste_factor * 100.0);
    eprintln!();
    let savings_at_nh50 = waste_factor * (13.8e6_f64 * 4.0 * 50.0 * 4.0 / 1024.0 / 1024.0 / 1024.0);
    eprintln!("  At nh=50: regret class buffer storage waste ≈ {:.1} GB per buffer", savings_at_nh50);
    eprintln!("  ×3 buffers (regret + strategy + cum_strategy): ≈ {:.1} GB total", savings_at_nh50 * 3.0);
}
