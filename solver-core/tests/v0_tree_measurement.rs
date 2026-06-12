use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;

fn make_config(
    np: u8,
    street: BoardState,
    starting_pot: i32,
    stacks: Vec<i32>,
    contributions: Vec<i32>,
    bet: Vec<f64>,
    raise: Vec<f64>,
) -> TreeConfig {
    TreeConfig {
        num_players: np,
        initial_state: street,
        starting_pot,
        starting_stacks: stacks,
        initial_contributions: contributions,
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: bet.iter().map(|&f| BetSize::PotRelative(f)).collect(),
            raise: raise.iter().map(|&f| BetSize::PotRelative(f)).collect(),
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,
    }

}

fn build_hu_turn(bet: &[f64], raise: &[f64]) -> FlatTree {
    build_tree(&make_config(
        2, BoardState::Turn, 200,
        vec![9500, 9500], vec![0, 0],
        bet.to_vec(), raise.to_vec(),
    )).unwrap()
}

fn build_hu_flop(bet: &[f64], raise: &[f64], stacks: Vec<i32>) -> FlatTree {
    let np = 2;
    build_tree(&make_config(
        np, BoardState::Flop, 100 * np as i32,
        stacks, vec![0; np as usize],
        bet.to_vec(), raise.to_vec(),
    )).unwrap()
}

fn build_np_river(np: u8, bet: &[f64], raise: &[f64], stack: i32) -> FlatTree {
    let starting_pot = 100 * np as i32;
    build_tree(&make_config(
        np, BoardState::River, starting_pot,
        vec![stack; np as usize], vec![0; np as usize],
        bet.to_vec(), raise.to_vec(),
    )).unwrap()
}

fn build_np_turn(np: u8, bet: &[f64], raise: &[f64], stack: i32) -> FlatTree {
    let starting_pot = 100 * np as i32;
    build_tree(&make_config(
        np, BoardState::Turn, starting_pot,
        vec![stack; np as usize], vec![0; np as usize],
        bet.to_vec(), raise.to_vec(),
    )).unwrap()
}

fn build_np_flop(np: u8, bet: &[f64], raise: &[f64], stack: i32) -> FlatTree {
    let starting_pot = 100 * np as i32;
    build_tree(&make_config(
        np, BoardState::Flop, starting_pot,
        vec![stack; np as usize], vec![0; np as usize],
        bet.to_vec(), raise.to_vec(),
    )).unwrap()
}

fn build_6max_river(bet: &[f64], raise: &[f64], stack: i32, pot: i32) -> FlatTree {
    build_tree(&make_config(
        6, BoardState::River, pot,
        vec![stack; 6], vec![0; 6],
        bet.to_vec(), raise.to_vec(),
    )).unwrap()
}

fn build_6max_turn(bet: &[f64], raise: &[f64], stack: i32, pot: i32) -> FlatTree {
    build_tree(&make_config(
        6, BoardState::Turn, pot,
        vec![stack; 6], vec![0; 6],
        bet.to_vec(), raise.to_vec(),
    )).unwrap()
}

fn build_6max_flop(bet: &[f64], raise: &[f64], stack: i32, pot: i32) -> FlatTree {
    build_tree(&make_config(
        6, BoardState::Flop, pot,
        vec![stack; 6], vec![0; 6],
        bet.to_vec(), raise.to_vec(),
    )).unwrap()
}

#[derive(Debug)]
struct TreeStats {
    name: String,
    id: String,
    np: u8,
    street: String,
    total_nodes: usize,
    terminal_nodes: usize,
    chance_nodes: usize,
    player_nodes: usize,
    max_depth: usize,
    max_level_nodes: usize,
    min_level_nodes: usize,
}

fn measure_tree(name: &str, id: &str, tree: &FlatTree) -> TreeStats {
    let terminal_nodes = tree.nodes.iter().filter(|n| n.is_terminal()).count();
    let chance_nodes = tree.nodes.iter().filter(|n| n.is_chance()).count();
    let player_nodes = tree.nodes.iter().filter(|n| n.is_player()).count();

    let mut depths = vec![0usize; tree.num_nodes()];
    let mut stack = vec![(0usize, 0usize)];
    while let Some((node_idx, depth)) = stack.pop() {
        depths[node_idx] = depth;
        for &child in tree.node_children(node_idx) {
            stack.push((child as usize, depth + 1));
        }
    }
    let max_depth = *depths.iter().max().unwrap_or(&0);

    let mut level_counts = vec![0usize; max_depth + 1];
    for &d in &depths {
        level_counts[d] += 1;
    }
    let max_level_nodes = *level_counts.iter().max().unwrap_or(&0);
    let min_level_nodes = *level_counts.iter().filter(|&&c| c > 0).min().unwrap_or(&0);

    let street_name = match tree.nodes[0].board_state {
        0 => "Flop",
        1 => "Turn",
        2 => "River",
        _ => "?",
    };

    TreeStats {
        name: name.to_string(),
        id: id.to_string(),
        np: tree.num_players,
        street: street_name.to_string(),
        total_nodes: tree.num_nodes(),
        terminal_nodes,
        chance_nodes,
        player_nodes,
        max_depth,
        max_level_nodes,
        min_level_nodes,
    }
}

fn estimate_memory(stats: &TreeStats, nh: usize) -> MemoryEstimate {
    let v = stats.total_nodes;
    let n = stats.np as usize;
    let decision_nodes = stats.player_nodes;

    let reach_bytes = v * n * nh * 4;
    let cfv_bytes = 2 * stats.max_level_nodes * nh * 4;
    let regret_bytes = decision_nodes * 4 * nh * 4;
    let strategy_bytes = regret_bytes;
    let cum_strategy_bytes = regret_bytes;

    let total = reach_bytes + cfv_bytes + regret_bytes + strategy_bytes + cum_strategy_bytes;

    MemoryEstimate {
        reach_mb: reach_bytes as f64 / 1e6,
        cfv_mb: cfv_bytes as f64 / 1e6,
        regret_mb: regret_bytes as f64 / 1e6,
        strategy_mb: strategy_bytes as f64 / 1e6,
        cum_strategy_mb: cum_strategy_bytes as f64 / 1e6,
        total_mb: total as f64 / 1e6,
        total_gb: total as f64 / 1e9,
        fits_8gb: total < 8_000_000_000,
    }
}

#[derive(Debug)]
struct MemoryEstimate {
    reach_mb: f64,
    cfv_mb: f64,
    regret_mb: f64,
    strategy_mb: f64,
    cum_strategy_mb: f64,
    total_mb: f64,
    total_gb: f64,
    fits_8gb: bool,
}

fn classify(_stats: &TreeStats, mem: &MemoryEstimate) -> &'static str {
    if mem.fits_8gb && mem.total_gb < 3.0 {
        "PROCEED"
    } else if mem.fits_8gb {
        "PROCEED-CAUTIOUS"
    } else {
        "DEFER"
    }
}

#[test]
fn v0_tree_measurement() {
    let nh_river = 1081;
    let nh_turn = 1128;

    let mut all_stats: Vec<(TreeStats, usize, MemoryEstimate)> = Vec::new();

    let configs: Vec<(&str, &str, FlatTree, usize)> = vec![
        // === N=2 (HU) — proven configurations ===
        ("HU turn 1bet",           "T2C", build_hu_turn(&[0.5], &[]),                        nh_turn),
        ("HU turn 2bet+raise",     "T2D", build_hu_turn(&[0.5, 1.0], &[0.5]),                nh_turn),
        ("HU flop 1bet+raise",     "T2E", build_hu_flop(&[0.5], &[1.0], vec![500, 500]),     nh_river),
        ("HU flop 2bet+raise",     "T2F", build_hu_flop(&[0.33, 0.5, 0.75], &[0.5, 1.0],
                                                         vec![500, 500]),                    nh_river),
        ("HU flop deep 2bet",      "T2X", build_hu_flop(&[0.5, 1.0], &[0.5, 1.0],
                                                         vec![2000, 2000]),                  nh_river),

        // === N=3 — common multiway (BTN+blinds) ===
        ("3p river 1bet",          "T3A", build_np_river(3, &[0.5], &[], 500),                nh_river),
        ("3p river 2bet",          "T3R2", build_np_river(3, &[0.5, 1.0], &[], 500),          nh_river),
        ("3p turn 1bet",           "T3B", build_np_turn(3, &[0.5], &[], 500),                 nh_turn),
        ("3p flop 1bet noraixe",   "T3C", build_np_flop(3, &[0.5], &[], 500),                 nh_river),

        // === N=6 — worst case ===
        ("6p river 1bet",          "T6A", build_6max_river(&[0.5], &[], 400, 600),            nh_river),
        ("6p river 2bet",          "T6B", build_6max_river(&[0.5, 1.0], &[], 400, 600),       nh_river),
        ("6p turn 1bet",           "T6C", build_6max_turn(&[0.5], &[], 400, 600),             nh_turn),
    ];

    for (name, id, tree, nh) in &configs {
        let stats = measure_tree(name, id, tree);
        let mem = estimate_memory(&stats, *nh);
        all_stats.push((stats, *nh, mem));
    }

    // === Print comprehensive report ===
    println!("\n{}", "=".repeat(70));
    println!("V0 TREE MEASUREMENT REPORT");
    println!("{}", "=".repeat(70));

    println!("\n--- Node Counts ---");
    println!("{:<28} {:>4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>5} {:>5} {:>5}",
        "Config", "N", "St", "Total", "Terminal", "Chance", "Player", "Depth", "MaxLv", "MinLv");
    for (stats, _, _) in &all_stats {
        println!("{:<28} {:>4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>5} {:>5} {:>5}",
            stats.name, stats.np, stats.street, stats.total_nodes,
            stats.terminal_nodes, stats.chance_nodes, stats.player_nodes,
            stats.max_depth, stats.max_level_nodes, stats.min_level_nodes);
    }

    println!("\n--- Memory Estimates (Vector CFR, per-traverser) ---");
    println!("{:<28} {:>4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>12}",
        "Config", "N", "NH", "Reach", "CFV", "Regret", "Strat", "Total", "GB", "Verdict");
    for (stats, nh, mem) in &all_stats {
        let verdict = classify(stats, mem);
        println!("{:<28} {:>4} {:>5} {:>7.1}M {:>7.1}M {:>7.1}M {:>7.1}M {:>7.1}M {:>5.2}G {:>12}",
            stats.name, stats.np, nh,
            mem.reach_mb, mem.cfv_mb, mem.regret_mb, mem.strategy_mb,
            mem.total_mb, mem.total_gb, verdict);
    }

    println!("\n--- Go/No-Go Decisions ---");
    for (stats, nh, mem) in &all_stats {
        let verdict = classify(stats, mem);
        let reason = match verdict {
            "PROCEED" => "fits comfortably, no optimization needed",
            "PROCEED-CAUTIOUS" => "fits but tight, may need memory optimization",
            "DEFER" => "does not fit in 8GB, needs abstraction or depth-limiting",
            _ => "unknown",
        };
        println!("{} [{}]: {} — {}", stats.id, stats.name, verdict, reason);
    }

    // === Assertions for basic sanity ===
    // N=2 configs should always build
    assert!(all_stats[0].0.total_nodes > 0, "T2C should have nodes");
    assert!(all_stats[3].0.total_nodes > 0, "T2F should have nodes");

    // N=3 and N=6 should build without error
    for (stats, _, _) in &all_stats {
        assert!(stats.total_nodes > 0, "{} should have nodes", stats.name);
    }
}

#[test]
fn v0_level_grouping_correctness() {
    use solver_core::tree::flat::{NODE_TYPE_PLAYER, NODE_TYPE_TERMINAL, NODE_TYPE_CHANCE};

    let tree = build_hu_flop(&[0.5], &[1.0], vec![500, 500]);
    assert!(!tree.level_nodes.is_empty(), "level_nodes should be populated");
    assert!(!tree.level_offsets.is_empty(), "level_offsets should be populated");
    assert!(tree.max_depth > 0, "max_depth should be > 0");
    assert!(!tree.decision_node_ids.is_empty(), "decision_node_ids should be populated");
    assert!(tree.num_infosets > 0, "num_infosets should be > 0");

    // Every node should appear exactly once in level_nodes
    let mut seen = vec![false; tree.num_nodes()];
    for &node_idx in &tree.level_nodes {
        let idx = node_idx as usize;
        assert!(idx < tree.num_nodes(), "level_nodes contains invalid index {}", idx);
        assert!(!seen[idx], "node {} appears twice in level_nodes", idx);
        seen[idx] = true;
    }
    assert!(seen.iter().all(|&s| s), "not all nodes appear in level_nodes");

    // Root should be at level 0
    let level0 = tree.nodes_at_level(0);
    assert_eq!(level0.len(), 1, "level 0 should have exactly 1 node (root)");
    assert_eq!(level0[0], 0, "level 0 should contain node 0 (root)");

    // level_offsets should be monotonically increasing
    for i in 1..tree.level_offsets.len() {
        assert!(tree.level_offsets[i] >= tree.level_offsets[i-1],
            "level_offsets not monotonic at index {}", i);
    }

    // Total nodes across all levels should equal tree size
    let total: usize = (0..=tree.max_depth).map(|l| tree.level_size(l)).sum();
    assert_eq!(total, tree.num_nodes(), "sum of level sizes != num_nodes");

    // infoset_offsets: every player node should have a valid offset
    for idx in 0..tree.num_nodes() {
        if tree.nodes[idx].node_type == NODE_TYPE_PLAYER {
            assert!(tree.infoset_offsets[idx] != solver_core::tree::flat::VCFR_NO_INFOSET,
                "player node {} should have valid infoset_offset", idx);
        } else {
            assert_eq!(tree.infoset_offsets[idx], solver_core::tree::flat::VCFR_NO_INFOSET,
                "non-player node {} should have VCFR_NO_INFOSET", idx);
        }
    }

    // decision_node_ids should match player nodes
    assert_eq!(tree.decision_node_ids.len(), tree.num_infosets as usize,
        "decision_node_ids length != num_infosets");
    for &node_idx in &tree.decision_node_ids {
        let idx = node_idx as usize;
        assert!(tree.nodes[idx].node_type == NODE_TYPE_PLAYER,
            "decision_node_ids contains non-player node {}", idx);
    }

    // Children depth should be parent depth + 1
    for idx in 0..tree.num_nodes() {
        let parent_level = tree.level_of(idx);
        for &child in tree.node_children(idx) {
            let child_level = tree.level_of(child as usize);
            assert_eq!(child_level, parent_level + 1,
                "child {} at level {} but parent {} at level {}",
                child, child_level, idx, parent_level);
        }
    }

    println!("Level grouping correctness: PASS ({} nodes, {} levels, {} infosets)",
        tree.num_nodes(), tree.max_depth + 1, tree.num_infosets);
}

#[test]
fn v0_6max_two_active_players() {
    // In 6max, when 4 players fold preflop leaving BTN vs BB:
    // The bot creates a 2-player tree with the correct pot/stack state.
    // Position labels (BTN, BB) are handled at the CLI layer, not the tree builder.
    // The tree builder receives num_players=2 with the postflop pot and stacks.
    let config = TreeConfig {
        num_players: 2,
        initial_state: BoardState::Flop,
        starting_pot: 200,
        starting_stacks: vec![4900, 4900],
        initial_contributions: vec![100, 100],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(0.5)],
            raise: vec![BetSize::PotRelative(1.0)],
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.0,
    button_player: None,
            max_bets_per_street: None,

    };

    let tree = build_tree(&config).unwrap();

    assert_eq!(tree.num_players, 2, "should be 2-player tree");

    for node in &tree.nodes {
        if node.is_player() {
            assert!(node.player_id < 2,
                "player_id {} exceeds num_players=2", node.player_id);
        }
    }

    assert!(tree.num_nodes() > 10, "tree should have nodes");
    assert!(tree.num_nodes() < 5000, "2-player tree should be small, got {}", tree.num_nodes());

    assert!(!tree.level_nodes.is_empty(), "level grouping should be populated");
    assert!(tree.max_depth > 0, "max_depth should be set");
    assert!(tree.num_infosets > 0, "should have infosets");

    println!("6max HU spot (2p tree): {} nodes, {} levels, {} infosets",
        tree.num_nodes(), tree.max_depth + 1, tree.num_infosets);
}
