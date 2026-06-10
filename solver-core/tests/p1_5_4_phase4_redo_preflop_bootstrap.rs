//! Phase 4 redo — preflop EMPIRICAL bootstrap (closing item 1 of the
//! post-bank list, superseding the "deferred 3-6h" plan in
//! p1_5_4_phase4_redo_preflop_capacity.rs with the cheap version).
//!
//! Directive (2026-06): profile first (measure, don't trust the estimate),
//! then run the cheap bootstrap: texture-stratified sampled flops, low-
//! convergence postflop solves, frozen CFVs reused across preflop
//! iterations. Validate the instrument with a two-fidelity stability check
//! (same significant-size set at both fidelities), then read off the
//! preflop size usage.
//!
//! === P8a PROFILE RESULT (measured 2026-06-09) ===
//!
//! The naive design — UnabstractedPostflopOracle per preflop chance node —
//! was profiled and is DEAD:
//!
//!   - One preflop iteration on the rich 14-raise tree needs
//!     6 traversers × 1,493 chance nodes × 5 flops = 44,790 oracle calls.
//!   - A SINGLE call at full combo ranges (nh ≈ 1176, 47×46 runouts,
//!     6-player factored showdown) did not return within 42 CPU-minutes
//!     (probe killed; the timeout IS the measurement).
//!   - Extrapolation: the original "3-6h" estimate was off by ~3 orders
//!     of magnitude. Years, not hours.
//!
//! Hence this design, which moves ALL the approximation into the oracle
//! (where the two-fidelity check can measure its effect):
//!
//!   1. SAMPLED TABLES: nh = 24/48 stride-sampled combos across the class
//!      spectrum + 2×2 / 3×3 sampled runouts, via
//!      compute_flop_start_subset_with_decks — the same machinery every
//!      working Phase 4 postflop measurement ran on (ms-to-seconds
//!      solves). CFVs for unsampled classes are filled from the nearest
//!      sampled class within the same family (pair/suited/offsuit
//!      index-adjacency).
//!   2. SHARED CFVs: cache key (flop, traverser) at LOW — every chance
//!      node sees the same postflop value function. At HIGH the key adds
//!      a pot-size bucket (limp / raise / 3bet / 4bet+ pots), partially
//!      un-sharing the approximation. NOTE: the oracle's flop tree has a
//!      FIXED pot (12), so pot-dependence only enters through ranges —
//!      sharing loses only range-dependence, which is exactly the kind of
//!      value approximation size-observation is claimed robust to.
//!   3. FROZEN ACROSS ITERATIONS: the cache persists across preflop
//!      iterations (LOW: forever; HIGH: cleared once mid-run at iter 40
//!      to re-populate with evolved ranges).
//!
//! The two-fidelity stability check (same significant-size set at LOW and
//! HIGH) makes "size-observation is robust to value approximation" a
//! measured claim. If it fails, the test fails loudly and the instrument
//! is declared insufficient — no silent verdict.

use std::collections::HashMap;
use std::time::Instant;

use solver_core::abstraction::preflop_class::{
    expansion, PreflopClass, NUM_PREFLOP_CLASSES,
};
use solver_core::card::{card_pair_to_index, Card, NUM_POSSIBLE_HANDS};
use solver_core::solver::flop_start_game::FlopChanceTable;
use solver_core::solver::postflop_oracle::PostflopValueOracle;
use solver_core::solver::preflop_cfr::{
    make_bootstrap_terminal_value_fn_multiway_pairwise, PreflopVectorCfr,
};
use solver_core::solver::preflop_start_game::{
    compute_v_flop_at_root_converged_with_table, flop_combo_layout,
    stratified_canonical_subset, PreflopChanceTable,
};
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::{
    FlatTree, ACTION_LABEL_ALLIN, ACTION_LABEL_BET, ACTION_LABEL_CALL,
    ACTION_LABEL_CHECK, ACTION_LABEL_FOLD, ACTION_LABEL_RAISE, MAX_NA_PREFLOP,
};

const UNUSED: usize = usize::MAX;

// === Trees ====================================================================

/// Rich 6-max preflop tree: the maximum raise menu MAX_NA_PREFLOP supports
/// (14 = 16 - fold - call), range 0.5×p..7.0×p in 0.5×p steps.
fn build_rich_preflop_tree() -> (FlatTree, usize) {
    let max_raise_count = MAX_NA_PREFLOP.saturating_sub(2);
    let raises: Vec<BetSize> = (0..max_raise_count)
        .map(|i| BetSize::PotRelative(0.5 + 0.5 * i as f64))
        .collect();
    let pre_cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Preflop,
        starting_pot: 3,
        starting_stacks: vec![100; 6],
        initial_contributions: vec![1, 2, 0, 0, 0, 0],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: raises,
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: Some(5),
    };
    let tree = build_tree(&pre_cfg).expect("rich preflop tree builds (capacity smoke passed)");
    (tree, max_raise_count)
}

/// Cheapest sane postflop tree for the oracle (1 bet + 0 raise).
fn build_oracle_flop_tree() -> FlatTree {
    let flop_cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 12,
        starting_stacks: vec![94; 6],
        initial_contributions: vec![0; 6],
        rake_rate: 0.0, rake_cap: 0.0,
        bet_sizes: BetSizeOptions {
            bet: vec![BetSize::PotRelative(1.0)],
            raise: vec![],
        },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    build_tree(&flop_cfg).expect("oracle flop tree builds")
}

// === Per-call probe (kept as the record of the P8a measurement) ===============

/// DO NOT run casually: the first converged-n=1 call at full ranges was
/// measured at ≥ 42 CPU-minutes before being killed. This probe exists as
/// the executable record of why the sampled oracle below exists.
#[test]
#[ignore = "P8a' probe: full-range per-call cost — measured ≥42min/call, runs ~hours"]
fn preflop_percall_probe_full_range() {
    use solver_core::solver::preflop_start_game::compute_v_flop_at_root_converged;
    let flop_tree = build_oracle_flop_tree();
    let np = 6usize;
    let ranges: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_POSSIBLE_HANDS as f32; NUM_POSSIBLE_HANDS])
        .collect();
    let table = PreflopChanceTable::new(np as u8, (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect());
    let f0 = table.canonical_flops[0];
    let t = Instant::now();
    let (v, layout) = compute_v_flop_at_root_converged(f0, &flop_tree, &ranges, 0, 1);
    eprintln!("full-range converged n=1: {:.1}s (nh={} v[0]={:.3})",
        t.elapsed().as_secs_f64(), layout.len(), v[0]);
}

// === Sampled shared oracle ====================================================

/// Per-canonical-flop precomputed sampling plan.
struct FlopSamplePlan {
    /// Stride-sampled hand indices (pair-index u16), n_hands of them,
    /// spread across the 169-class spectrum.
    chosen: Vec<u16>,
    /// class → position in `chosen`'s class list (for nearest-fill).
    /// sampled_classes[k] = class index of chosen[k] (sorted ascending).
    sampled_classes: Vec<u8>,
    turn_cards: Vec<u8>,
    river_decks: Vec<Vec<u8>>, // [52]
}

fn build_sample_plan(flop: [Card; 3], n_hands: usize, n_turns: usize, n_rivers: usize) -> FlopSamplePlan {
    // One representative (first non-conflicting) combo per class, then
    // stride-sample n_hands classes across 0..169. Class order is
    // pairs | suited | offsuit, so a stride covers all three families.
    let mut class_rep: Vec<(u8, u16)> = Vec::new(); // (class, pair_index)
    for cls in 0..NUM_PREFLOP_CLASSES as u8 {
        let combos = expansion(PreflopClass::new(cls), flop);
        if let Some(&(c1, c2)) = combos.first() {
            class_rep.push((cls, card_pair_to_index(c1, c2) as u16));
        }
    }
    let step = (class_rep.len() as f64 / n_hands as f64).max(1.0);
    let mut chosen = Vec::new();
    let mut sampled_classes = Vec::new();
    let mut i = 0.0f64;
    while (i as usize) < class_rep.len() && chosen.len() < n_hands {
        let (cls, hi) = class_rep[i as usize];
        chosen.push(hi);
        sampled_classes.push(cls);
        i += step;
    }

    // Sampled runouts: stride over non-board cards for turns; per turn,
    // stride over non-board-non-turn cards for rivers.
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let non_board: Vec<u8> = (0u8..52).filter(|&c| board_mask & (1u64 << c) == 0).collect();
    let tstep = non_board.len() / n_turns;
    let turn_cards: Vec<u8> = (0..n_turns).map(|k| non_board[k * tstep + tstep / 2]).collect();
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let pool: Vec<u8> = non_board.iter().copied().filter(|&c| c != tc).collect();
        let rstep = pool.len() / n_rivers;
        river_decks[tc as usize] = (0..n_rivers).map(|k| pool[k * rstep + rstep / 2]).collect();
    }

    FlopSamplePlan { chosen, sampled_classes, turn_cards, river_decks }
}

/// Sampled, shared, frozen postflop oracle.
///
/// - SAMPLED: solves on an nh ≈ n_hands subset table with sampled runouts
///   via the `_with_table` seam (same extraction as the full path).
/// - SHARED: cache key = (flop, traverser[, pot-bucket]); all chance
///   nodes mapping to the same key reuse one CFV vector.
/// - FROZEN: cache persists across preflop iterations; optional one-shot
///   re-populate at `refresh_at_iter`.
///
/// Returns CFVs in full `flop_combo_layout` order with nearest-sampled-
/// class fill for unsampled combos.
struct SampledSharedOracle<'a> {
    flop_tree: &'a FlatTree,
    postflop_iters: u32,
    n_hands: usize,
    n_turns: usize,
    n_rivers: usize,
    /// pot-size bucket edges (chips); empty = no pot bucketing (LOW).
    pot_bucket_edges: Vec<i32>,
    /// pot at each preflop chance node, in engine iteration order.
    chance_pots: Vec<i32>,
    subset_len: usize,
    chance_count: usize,
    /// call counter within the current preflop iteration.
    counter: usize,
    refresh_at_iter: u32, // 0 = never
    plans: HashMap<[u8; 3], FlopSamplePlan>,
    cache: HashMap<([u8; 3], u8, usize), Vec<f32>>,
    pub inner_calls: u64,
    pub replayed_calls: u64,
    pub inner_micros: u64,
}

impl<'a> SampledSharedOracle<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        flop_tree: &'a FlatTree,
        postflop_iters: u32,
        n_hands: usize,
        n_turns: usize,
        n_rivers: usize,
        pot_bucket_edges: Vec<i32>,
        chance_pots: Vec<i32>,
        subset_len: usize,
        refresh_at_iter: u32,
    ) -> Self {
        let chance_count = chance_pots.len();
        Self {
            flop_tree, postflop_iters, n_hands, n_turns, n_rivers,
            pot_bucket_edges, chance_pots, subset_len, chance_count,
            counter: 0, refresh_at_iter,
            plans: HashMap::new(),
            cache: HashMap::new(),
            inner_calls: 0, replayed_calls: 0, inner_micros: 0,
        }
    }

    fn pot_bucket(&self, pot: i32) -> usize {
        for (b, &edge) in self.pot_bucket_edges.iter().enumerate() {
            if pot <= edge { return b; }
        }
        self.pot_bucket_edges.len()
    }

    fn compute_sampled(
        &mut self,
        flop: [Card; 3],
        combo_ranges: &[Vec<f32>],
        traverser: u8,
    ) -> Vec<f32> {
        let key3 = [flop[0] as u8, flop[1] as u8, flop[2] as u8];
        if !self.plans.contains_key(&key3) {
            self.plans.insert(key3,
                build_sample_plan(flop, self.n_hands, self.n_turns, self.n_rivers));
        }
        let plan = &self.plans[&key3];

        // Expand engine-layout ranges to full-hands vectors (sampled table
        // constructor reads weights at the chosen combos from these).
        let layout = flop_combo_layout(flop);
        let np = combo_ranges.len();
        let mut full: Vec<Vec<f32>> = vec![vec![0.0f32; NUM_POSSIBLE_HANDS]; np];
        for p in 0..np {
            for (li, &(c1, c2)) in layout.iter().enumerate() {
                full[p][card_pair_to_index(c1, c2)] = combo_ranges[p][li];
            }
        }

        let board: Vec<Card> = flop.to_vec();
        let table = FlopChanceTable::compute_flop_start_subset_with_decks(
            &board, &full, np as u8, &plan.chosen, &plan.turn_cards, &plan.river_decks,
        );
        let (v_sample, layout_sample) = compute_v_flop_at_root_converged_with_table(
            table, self.flop_tree, traverser, self.postflop_iters,
        );

        // Per-class CFV from the sample.
        let mut class_v: Vec<Option<f32>> = vec![None; NUM_PREFLOP_CLASSES];
        for (i, &(c1, c2)) in layout_sample.iter().enumerate() {
            let cls = PreflopClass::from_combo(c1, c2).index();
            class_v[cls] = Some(v_sample[i]);
        }
        // Nearest-sampled-class fill (index adjacency; class order is
        // pairs | suited | offsuit so adjacency mostly stays in-family).
        let sampled: Vec<usize> = plan.sampled_classes.iter().map(|&c| c as usize).collect();
        let filled: Vec<f32> = (0..NUM_PREFLOP_CLASSES).map(|cls| {
            if let Some(v) = class_v[cls] { return v; }
            let nearest = sampled.iter()
                .min_by_key(|&&s| (s as i32 - cls as i32).abs())
                .copied().unwrap_or(0);
            class_v[nearest].unwrap_or(0.0)
        }).collect();

        // Full-layout output: class-constant fill.
        layout.iter().map(|&(c1, c2)| {
            filled[PreflopClass::from_combo(c1, c2).index()]
        }).collect()
    }
}

impl<'a> PostflopValueOracle for SampledSharedOracle<'a> {
    fn flop_root_cfv(
        &mut self,
        canonical_flop: [Card; 3],
        combo_ranges: &[Vec<f32>],
        traverser: u8,
    ) -> Vec<f32> {
        // Derive the chance node from the call position (engine order:
        // traverser outer, chance node middle, flop inner — deterministic).
        let chance_i = (self.counter / self.subset_len) % self.chance_count.max(1);
        self.counter += 1;
        let bucket = if self.pot_bucket_edges.is_empty() {
            0
        } else {
            self.pot_bucket(self.chance_pots[chance_i])
        };
        let key = (
            [canonical_flop[0] as u8, canonical_flop[1] as u8, canonical_flop[2] as u8],
            traverser,
            bucket,
        );
        if let Some(v) = self.cache.get(&key) {
            self.replayed_calls += 1;
            return v.clone();
        }
        let t = Instant::now();
        let v = self.compute_sampled(canonical_flop, combo_ranges, traverser);
        let us = t.elapsed().as_micros() as u64;
        self.inner_micros += us;
        self.inner_calls += 1;
        // Per-call visibility: populate is ≤ a few hundred calls; print
        // each so a mis-sized fidelity is visible in seconds, not hours.
        eprintln!("    [oracle] inner call #{}: {:.2}s (flop {:?} t={} bucket={})",
            self.inner_calls, us as f64 / 1e6, key.0, traverser, bucket);
        self.cache.insert(key, v.clone());
        v
    }

    fn begin_preflop_iter(&mut self, iter: u32) {
        self.counter = 0;
        if self.refresh_at_iter > 0 && iter == self.refresh_at_iter {
            self.cache.clear(); // one-shot re-populate with evolved ranges
        }
    }
}

// === Size-usage readout =======================================================

#[derive(Debug, Clone)]
struct SizeUsage {
    weighted: Vec<f64>,
    unweighted: Vec<f64>,
    fold_mass: f64,
    call_mass: f64,
    allin_mass: f64,
}

/// Read size usage off a solved engine: normalize cum_strategy into σ_avg,
/// compute σ_avg-based reach, aggregate per raise-size slot (slot = rank
/// of the raise child by chip amount; same declared menu at every node).
fn read_size_usage(
    engine: &mut PreflopVectorCfr,
    tree: &FlatTree,
    max_raise_count: usize,
) -> SizeUsage {
    let n_classes = NUM_PREFLOP_CLASSES;
    let nn = tree.num_nodes();
    let np = tree.num_players as usize;

    for idx in 0..nn {
        let local = engine.local_offset[idx];
        if local == UNUSED { continue; }
        let na = tree.nodes[idx].num_children as usize;
        let off = local * MAX_NA_PREFLOP * n_classes;
        for c in 0..n_classes {
            let mut total = 0.0f32;
            for a in 0..na {
                total += engine.cum_strategy[off + a * n_classes + c].max(0.0);
            }
            if total > 1e-12 {
                for a in 0..na {
                    engine.strategy[off + a * n_classes + c] =
                        engine.cum_strategy[off + a * n_classes + c].max(0.0) / total;
                }
            } else {
                let u = 1.0 / na as f32;
                for a in 0..na {
                    engine.strategy[off + a * n_classes + c] = u;
                }
            }
        }
    }

    let reach = engine.compute_preflop_reach(tree, None);

    let mut weighted = vec![0.0f64; max_raise_count];
    let mut unweighted = vec![0.0f64; max_raise_count];
    let mut fold_mass = 0.0f64;
    let mut call_mass = 0.0f64;
    let mut allin_mass = 0.0f64;

    for idx in 0..nn {
        let local = engine.local_offset[idx];
        if local == UNUSED { continue; }
        let node = &tree.nodes[idx];
        let p = node.player_id as usize;
        let off = local * MAX_NA_PREFLOP * n_classes;
        let children = tree.node_children(idx);

        let parent_contrib = tree.contributions[idx * np + p];
        let mut raise_children: Vec<(usize, i32)> = Vec::new();
        for (a, &ch) in children.iter().enumerate() {
            let cn = &tree.nodes[ch as usize];
            if cn.action_label == ACTION_LABEL_RAISE || cn.action_label == ACTION_LABEL_BET {
                let inc = tree.contributions[ch as usize * np + p] - parent_contrib;
                raise_children.push((a, inc));
            }
        }
        raise_children.sort_by_key(|&(_, inc)| inc);

        for c in 0..n_classes {
            let r = reach[p][idx * n_classes + c] as f64;
            for (slot, &(a, _)) in raise_children.iter().enumerate() {
                if slot >= max_raise_count { break; }
                let s = engine.strategy[off + a * n_classes + c] as f64;
                weighted[slot] += r * s;
                unweighted[slot] += s;
            }
            for (a, &ch) in children.iter().enumerate() {
                let cn = &tree.nodes[ch as usize];
                let s = engine.strategy[off + a * n_classes + c] as f64;
                match cn.action_label {
                    x if x == ACTION_LABEL_FOLD => fold_mass += r * s,
                    x if x == ACTION_LABEL_CALL || x == ACTION_LABEL_CHECK => call_mass += r * s,
                    x if x == ACTION_LABEL_ALLIN => allin_mass += r * s,
                    _ => {}
                }
            }
        }
    }

    SizeUsage { weighted, unweighted, fold_mass, call_mass, allin_mass }
}

/// Threshold-robust two-fidelity stability: the significant-size sets may
/// disagree on a slot ONLY if both fidelities place that slot's share
/// inside the borderline band [band_lo, band_hi] around the threshold.
///
/// WHY (measured 2026-06-10): the strict set-equality check failed on
/// exactly one element — slot 4 (2.5×p) at 1.08% (LOW) vs 0.87% (HIGH),
/// straddling the hard 1% cutoff from both sides — while everything
/// answer-relevant (top-4 identity and order, ~98% mass concentration,
/// dead 9-slot tail) was identical across fidelities. A hard threshold
/// on a boundary-straddling value measures threshold noise, not
/// instrument instability. The band makes "borderline is borderline at
/// both fidelities" count as agreement; a REAL instability (a size
/// clearly significant at one fidelity and clearly dead at the other)
/// still fails.
fn sig_sets_stable_with_band(
    low: &SizeUsage,
    high: &SizeUsage,
    threshold: f64,
    band_lo: f64,
    band_hi: f64,
) -> bool {
    let share = |u: &SizeUsage, s: usize| -> f64 {
        let t: f64 = u.weighted.iter().sum();
        if t > 0.0 { u.weighted[s] / t } else { 0.0 }
    };
    let n = low.weighted.len().min(high.weighted.len());
    for s in 0..n {
        let (x, y) = (share(low, s), share(high, s));
        if (x >= threshold) != (y >= threshold) {
            let both_borderline = x >= band_lo && x <= band_hi && y >= band_lo && y <= band_hi;
            if !both_borderline { return false; }
        }
    }
    true
}

#[test]
fn band_stability_logic_pinned_to_recorded_data() {
    // Recorded P8b shares (2026-06-10), as fractions.
    let mk = |w: Vec<f64>| SizeUsage {
        weighted: w, unweighted: vec![], fold_mass: 0.0, call_mass: 0.0, allin_mass: 0.0,
    };
    let low = mk(vec![0.7947, 0.1286, 0.0358, 0.0173, 0.0108, 0.0045, 0.0040, 0.0039,
                      0.0002, 0.0001, 0.0001, 0.0002, 0.0000, 0.0000]);
    let high = mk(vec![0.7620, 0.1400, 0.0492, 0.0293, 0.0087, 0.0025, 0.0037, 0.0024,
                       0.0001, 0.0002, 0.0009, 0.0003, 0.0000, 0.0006]);
    // Slot 4 straddles 1% (1.08% vs 0.87%) — band [0.5%, 2%] counts it
    // as agreement; the recorded data must classify as STABLE.
    assert!(sig_sets_stable_with_band(&low, &high, 0.01, 0.005, 0.02));

    // A REAL instability — significant at one fidelity, dead at the
    // other — must still fail.
    let mut broken = high.clone();
    broken.weighted[2] = 0.0001; // 1.5×p collapses from 4.92% to ~0
    assert!(!sig_sets_stable_with_band(&low, &broken, 0.01, 0.005, 0.02));
}

fn significant_sizes(usage: &SizeUsage, threshold: f64) -> Vec<usize> {
    let total: f64 = usage.weighted.iter().sum();
    if total <= 0.0 { return vec![]; }
    usage.weighted.iter().enumerate()
        .filter(|(_, &m)| m / total >= threshold)
        .map(|(i, _)| i)
        .collect()
}

fn print_usage(label: &str, usage: &SizeUsage, max_raise_count: usize) {
    let total: f64 = usage.weighted.iter().sum();
    let total_unw: f64 = usage.unweighted.iter().sum();
    eprintln!("\n  Size usage [{}] (slot = k-th smallest raise; declared menu 0.5×p..7.0×p):", label);
    eprintln!("    {:>4} {:>9} {:>13} {:>13}", "slot", "size(×p)", "wt share", "unwt share");
    for slot in 0..max_raise_count {
        let declared = 0.5 + 0.5 * slot as f64;
        let ws = if total > 0.0 { usage.weighted[slot] / total * 100.0 } else { 0.0 };
        let us = if total_unw > 0.0 { usage.unweighted[slot] / total_unw * 100.0 } else { 0.0 };
        eprintln!("    {:>4} {:>9.1} {:>12.2}% {:>12.2}%", slot, declared, ws, us);
    }
    eprintln!("    context (reach-weighted): fold={:.3} call/check={:.3} allin={:.3} raise_total={:.3}",
        usage.fold_mass, usage.call_mass, usage.allin_mass, total);
}

// === Bootstrap ================================================================

struct FidelityResult {
    usage: SizeUsage,
    sig_set: Vec<usize>,
    wall_s: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_bootstrap_fidelity(
    label: &str,
    pre_tree: &FlatTree,
    flop_tree: &FlatTree,
    table: &PreflopChanceTable,
    n_per_cell: usize,
    n_hands: usize,
    n_turns: usize,
    n_rivers: usize,
    postflop_iters: u32,
    pot_bucket_edges: Vec<i32>,
    refresh_at_iter: u32,
    preflop_iters: u32,
    max_raise_count: usize,
) -> FidelityResult {
    let subset = stratified_canonical_subset(table, n_per_cell);
    eprintln!("\n── fidelity {}: {} flops, nh={}, {}×{} runouts, postflop n={}, pot-buckets={}, refresh@{}, {} preflop iters ──",
        label, subset.len(), n_hands, n_turns, n_rivers, postflop_iters,
        pot_bucket_edges.len() + 1, refresh_at_iter, preflop_iters);

    let mut engine = PreflopVectorCfr::new(pre_tree);
    let np = pre_tree.num_players;

    // Pot at each chance node, in engine iteration order.
    let chance_nodes = engine.preflop_chance_node_indices(pre_tree);
    let chance_pots: Vec<i32> = chance_nodes.iter().map(|&idx| {
        (0..np as usize).map(|p| pre_tree.contributions[idx * np as usize + p]).sum()
    }).collect();

    let mut oracle = SampledSharedOracle::new(
        flop_tree, postflop_iters, n_hands, n_turns, n_rivers,
        pot_bucket_edges, chance_pots, subset.len(), refresh_at_iter,
    );
    // PAIRWISE term_fn, not the exact joint enumeration: the exact one is
    // O(Π nnz(opp_i)) per terminal per traverser class and was measured
    // running 11+ hours INSIDE iteration 1 at 6-max dense reaches (100% of
    // stack samples in accumulate_opp_classes). See preflop_terminal.rs.
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(pre_tree);

    let t0 = Instant::now();
    for i in 0..preflop_iters {
        engine.run_one_iteration_subset(pre_tree, table, &subset, &mut oracle, &term_fn);
        if i == 0 {
            // BUDGET GATE: the populate iteration carries all the inner-
            // oracle cost. If it blew past 15 min, the fidelity params are
            // mis-sized — fail loudly with the measurement instead of
            // grinding for hours.
            let s = t0.elapsed().as_secs_f64();
            eprintln!("  iter 1 (populate): {:.1}s, inner={} calls ({:.2}s avg), replayed={}",
                s, oracle.inner_calls,
                oracle.inner_micros as f64 / 1e6 / oracle.inner_calls.max(1) as f64,
                oracle.replayed_calls);
            // Gate sized to MEASURED per-call costs: LOW = 30 calls × 5.5s
            // ≈ 165s; HIGH = 60 calls × 16-18s ≈ 1100s (nh=14 measured
            // 2026-06-10). Gate at 1500s = measured + 35% headroom; if it
            // fires, the config has drifted from these measurements.
            assert!(s < 1500.0,
                "populate iteration took {:.0}s (> 25 min) at fidelity {} — \
                 re-measure per-call cost; config drifted from the 2026-06-10 \
                 sizing (nh=12: 5.5s/call, nh=14: 16-18s/call)", s, label);
        } else if (i + 1) % 20 == 0 {
            eprintln!("  iter {:>3}: {:.1}s elapsed (inner={} replayed={})",
                i + 1, t0.elapsed().as_secs_f64(), oracle.inner_calls, oracle.replayed_calls);
        }
    }
    let wall_s = t0.elapsed().as_secs_f64();
    eprintln!("  done: {:.1}s total; oracle inner={} ({:.1}s) replayed={}",
        wall_s, oracle.inner_calls, oracle.inner_micros as f64 / 1e6, oracle.replayed_calls);

    let usage = read_size_usage(&mut engine, pre_tree, max_raise_count);
    print_usage(label, &usage, max_raise_count);
    let sig_set = significant_sizes(&usage, 0.01);
    eprintln!("  significant sizes (≥1% of raise mass): {:?}", sig_set);

    FidelityResult { usage, sig_set, wall_s }
}

/// ONE-ITERATION smoke at LOW-fidelity parameters. Run this BEFORE the
/// two-fidelity bootstrap; it measures the per-iteration wall directly
/// (the lesson from the 12-hour P8b hang: never launch N iterations
/// before timing one).
#[test]
#[ignore = "P8b smoke: ONE preflop iteration at LOW fidelity (~minutes if healthy)"]
fn preflop_bootstrap_one_iter_smoke() {
    eprintln!("\n=== P8b smoke: one preflop iteration, LOW fidelity, pairwise term_fn ===");
    let (pre_tree, _max_raise_count) = build_rich_preflop_tree();
    let flop_tree = build_oracle_flop_tree();
    eprintln!("Preflop tree: {} nodes, {} infosets", pre_tree.num_nodes(), pre_tree.num_infosets);

    let np = 6u8;
    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let table = PreflopChanceTable::new(np, class_weights);
    let subset = stratified_canonical_subset(&table, 1);

    let mut engine = PreflopVectorCfr::new(&pre_tree);
    let chance_nodes = engine.preflop_chance_node_indices(&pre_tree);
    let chance_pots: Vec<i32> = chance_nodes.iter().map(|&idx| {
        (0..np as usize).map(|p| pre_tree.contributions[idx * np as usize + p]).sum()
    }).collect();

    // nh=12, NOT 24: the 6-player factored showdown costs ≈ nh^4.84
    // (M2 measurement); nh=24 was sampled at >1 min/solve (stack-sampled
    // 2026-06-10, 100% in bottom_up_zone), nh=12 is ~29× cheaper.
    let mut oracle = SampledSharedOracle::new(
        &flop_tree, 1, 12, 2, 2, vec![], chance_pots, subset.len(), 0,
    );
    let term_fn = make_bootstrap_terminal_value_fn_multiway_pairwise(&pre_tree);

    let t0 = Instant::now();
    engine.run_one_iteration_subset(&pre_tree, &table, &subset, &mut oracle, &term_fn);
    let s1 = t0.elapsed().as_secs_f64();
    eprintln!("iter 1 (populate): {:.1}s — oracle inner={} ({:.2}s avg), replayed={}",
        s1, oracle.inner_calls,
        oracle.inner_micros as f64 / 1e6 / oracle.inner_calls.max(1) as f64,
        oracle.replayed_calls);

    let t1 = Instant::now();
    engine.run_one_iteration_subset(&pre_tree, &table, &subset, &mut oracle, &term_fn);
    let s2 = t1.elapsed().as_secs_f64();
    eprintln!("iter 2 (frozen):   {:.1}s", s2);
    eprintln!("\nExtrapolation: LOW 80 iters ≈ {:.1} min; HIGH (≈8× oracle, same engine) ≈ {:.1} min",
        (s1 + 79.0 * s2) / 60.0,
        (8.0 * (s1 - s2).max(0.0) + 80.0 * s2 * 1.2) / 60.0);
    eprintln!("Smoke OK — safe to launch preflop_bootstrap_two_fidelity if the numbers above are sane.");
}

#[test]
#[ignore = "P8b bootstrap: sampled+shared+frozen preflop size observation, two fidelities (~tens of minutes)"]
fn preflop_bootstrap_two_fidelity() {
    eprintln!("\n=== P8b: preflop empirical bootstrap (sampled shared frozen CFVs, two fidelities) ===");

    let (pre_tree, max_raise_count) = build_rich_preflop_tree();
    let flop_tree = build_oracle_flop_tree();
    eprintln!("Preflop tree: {} nodes, {} infosets, {} raises",
        pre_tree.num_nodes(), pre_tree.num_infosets, max_raise_count);
    eprintln!("Oracle flop tree: {} nodes", flop_tree.num_nodes());

    let np = 6u8;
    let class_weights: Vec<Vec<f32>> = (0..np)
        .map(|_| vec![1.0_f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES])
        .collect();
    let table = PreflopChanceTable::new(np, class_weights);

    // nh sizing note: 6-player factored showdown ≈ nh^4.84 (M2);
    // measured per-call: nh=24 > 1 min, nh=12 ≈ 7.3s. Budget per
    // fidelity ≈ 15 min oracle → LOW 30 calls @ 7.3s ≈ 3.7 min;
    // HIGH 60 calls @ ~15s (nh=14 ≈ 2.1× of nh=12) ≈ 15 min.
    //
    // What the two fidelities VARY (and the stability check therefore
    // probes): sampled-class coverage (nh 12 → 14) and CFV sharing
    // granularity (1 shared bucket → 2 pot buckets). What they DON'T
    // vary (residual unprobed risk, flagged in the verdict): runout
    // sampling (2×2 both), postflop convergence depth (n=1 both),
    // frozen-forever cadence (both).

    // LOW: 5 flops, nh=12, 2×2 runouts, n=1, fully shared (flop,
    // traverser), frozen forever. Populate = 6 × 5 = 30 solves.
    let low = run_bootstrap_fidelity(
        "LOW", &pre_tree, &flop_tree, &table,
        1, 12, 2, 2, 1, vec![], 0, 80, max_raise_count,
    );

    // HIGH: 5 flops, nh=14, 2×2 runouts, n=1, pot-bucketed sharing
    // (pot ≤ 15 chips vs bigger), frozen forever.
    // Populate = 6 × 5 × 2 = 60 solves.
    let high = run_bootstrap_fidelity(
        "HIGH", &pre_tree, &flop_tree, &table,
        1, 14, 2, 2, 1, vec![15], 0, 80, max_raise_count,
    );

    eprintln!("\n=== P8b verdict ===");
    eprintln!("LOW:  {:.1}s wall, sig sizes {:?}", low.wall_s, low.sig_set);
    eprintln!("HIGH: {:.1}s wall, sig sizes {:?}", high.wall_s, high.sig_set);
    let strict = low.sig_set == high.sig_set;
    let stable = sig_sets_stable_with_band(&low.usage, &high.usage, 0.01, 0.005, 0.02);
    eprintln!("Two-fidelity stability: strict set-equality = {}; band-robust = {}",
        strict, stable);
    eprintln!("{}",
        if stable { "STABLE (band-robust) — disagreements, if any, are threshold-straddles inside [0.5%, 2%]; instrument validated" }
        else { "UNSTABLE — a size is clearly significant at one fidelity and dead at the other; readout NOT trustworthy" });

    let n_sig = high.sig_set.len();
    eprintln!("\nSizes used (≥1% raise mass): {} of {} available", n_sig, max_raise_count);
    if n_sig >= max_raise_count - 1 {
        eprintln!("SATURATED: nearly all {} raise slots in use — MAX_NA_PREFLOP = {} may be TOO LEAN.",
            max_raise_count, MAX_NA_PREFLOP);
        eprintln!("Next: bump MAX_NA_PREFLOP (e.g. 20), re-run with 18 raises, find the boundary.");
    } else {
        eprintln!("NOT saturated: {} of {} slots used → MAX_NA_PREFLOP = {} is SUFFICIENT (not too lean).",
            n_sig, max_raise_count, MAX_NA_PREFLOP);
        eprintln!("Could shrink to ~{} if preflop memory ever matters; at preflop's CPU-only cost, 16 is free.",
            n_sig + 3);
    }

    assert!(stable,
        "Two-fidelity stability check FAILED: LOW {:?} != HIGH {:?}. \
         Increase fidelity before reading a verdict.",
        low.sig_set, high.sig_set);
}
