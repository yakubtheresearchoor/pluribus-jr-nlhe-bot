//! Blueprint production runner — cell A baseline: 1755 canonical flops
//! × 1×1 seeded runout, quantile B=8 maps, Design1Collapsed, 34 DCFR
//! iters (the 1%-pot target), banked to disk per flop.
//!
//! Priced by the B4 ladder at ~10.5h (16-way). RESUMABLE: per-flop
//! output files, written to .tmp then renamed; existing files skipped,
//! so the run survives interruption and re-launch.
//!
//! Banking format (version 1, fidelity-agnostic — the GPU blueprint
//! drops into the same slot): per flop `flop_NNNN.bp`:
//!   magic "SSBP1\n"
//!   one JSON header line (flop cards, B per street, runout cards,
//!     iters, tree config summary, nh, solve seconds)
//!   then named sections: name '\n' u64-LE byte length, raw bytes.
//!   Sections: cum_flop / cum_turn / cum_river (f32-LE), map_flop
//!   (u16-LE), map_turn (u16-LE, concatenated per ti), map_river
//!   (u16-LE, concatenated per (ti, ri)).
//!
//! Config notes (named, deliberate):
//!   - Tree: the preflop-bootstrap ORACLE shape (1 bet 1.0×pot, no
//!     raises, stacks 94, pot 12, contribs 0) — the seam-consistent
//!     single-pot-context family the frozen oracle keys on. The lean
//!     MAX_NA=4 action set re-prices the ladder (~terminal-count
//!     linear) and gets its own row before anyone banks it.
//!   - Runout: 1×1 per cell A, drawn per-flop by seeded splitmix
//!     (seed = flop index) so the ensemble isn't position-biased and
//!     the artifact is regenerable from pinned seeds.
//!   - Maps: quantile B=8 (the shipped family; GS14 is the in-tree
//!     challenger for the head-to-head A/B).
//!
//! Env: BP_OUT (default blueprint_out/), BP_THREADS (default 14 CPU /
//! 2 GPU), BP_ITERS (34), BP_B (8), BP_START / BP_END (flop index
//! range), BP_RUNOUTS ("1x1" default; e.g. "4x4" = 4 seeded turns ×
//! 4 seeded rivers each), BP_GPU=1 (solve on the G5 native GPU path —
//! bit-exact-gated kernels, striped production config; the artifact
//! self-describes gpu:true), BP_ROLE (header role string).

use solver_core::card::Card;
#[cfg(feature = "metal")]
use solver_core::gpu_metal::bucketed_native::BucketedNativeGpu;
#[cfg(feature = "metal")]
use solver_core::gpu_metal::context::MetalContext;
use solver_core::solver::bucketed_flop_cfr::{
    BucketedFlopCfr, FlopBucketing, TerminalDesign, NO_BUCKET,
};
use solver_core::solver::flop_start_game::{FlopChanceTable, FlopStartGame};
use solver_core::solver::preflop_start_game::PreflopChanceTable;
use solver_core::tree::action::{BetSize, BetSizeOptions, BoardState, TreeConfig};
use solver_core::tree::builder::build_tree;
use solver_core::tree::flat::FlatTree;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const N_CLASSES: usize = 169;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn build_oracle_tree() -> FlatTree {
    let cfg = TreeConfig {
        num_players: 6,
        initial_state: BoardState::Flop,
        starting_pot: 12,
        starting_stacks: vec![94; 6],
        initial_contributions: vec![0; 6],
        rake_rate: 0.0,
        rake_cap: 0.0,
        bet_sizes: BetSizeOptions { bet: vec![BetSize::PotRelative(1.0)], raise: vec![] },
        add_allin_threshold: 1.0,
        force_allin_threshold: 1.0,
        merging_threshold: 0.0,
        button_player: None,
    };
    build_tree(&cfg).expect("oracle tree builds")
}

fn quantile_maps(
    table: &FlopChanceTable,
    nb: usize,
) -> (Vec<u16>, Vec<Vec<u16>>, Vec<Vec<Vec<u16>>>) {
    let nh = table.num_valid;
    let conflicts = |h: usize, cards: &[u8]| -> bool {
        let c1 = table.hand_cards[h * 2];
        let c2 = table.hand_cards[h * 2 + 1];
        cards.iter().any(|&bc| bc == c1 || bc == c2)
    };
    let map_for = |pl_idx: &[u16], dead: &[u8]| -> Vec<u16> {
        let alive: Vec<usize> = pl_idx[..nh]
            .iter()
            .map(|&i| i as usize)
            .filter(|&h| !conflicts(h, dead))
            .collect();
        let n = alive.len();
        assert!(n >= nb);
        let mut map = vec![NO_BUCKET; nh];
        for (pos, &h) in alive.iter().enumerate() {
            map[h] = ((pos * nb) / n) as u16;
        }
        map
    };
    let (_, _, _, base_pi, _) = table.sorted_opp_arrays_base();
    let flop_map = map_for(&base_pi, &[]);
    let mut turn_maps = Vec::new();
    let mut river_maps = Vec::new();
    for &tc_card in &table.remaining_deck {
        let (_, _, _, pi) = table.turn_sorted_arrays(tc_card);
        turn_maps.push(map_for(pi, &[tc_card]));
        let mut rms = Vec::new();
        for &rc_card in &table.river_decks[tc_card as usize] {
            let (_, _, _, pi) = table.river_sorted_arrays(tc_card, rc_card);
            rms.push(map_for(pi, &[tc_card, rc_card]));
        }
        river_maps.push(rms);
    }
    (flop_map, turn_maps, river_maps)
}

/// Normalized flop-zone sigma_avg rows, flat — the convergence-stat
/// basis. `cum` is the flop cum-strategy slice (CPU solver buffer or
/// the GPU-native readback — identical layout).
fn normalized_flop_sigma(
    cum: &[f32],
    solver: &BucketedFlopCfr,
    tree: &FlatTree,
    bk: &FlopBucketing,
) -> Vec<f32> {
    use solver_core::tree::flat::MAX_NA_POSTFLOP;
    let nb = bk.nb_flop;
    let mut out = Vec::new();
    for &nid in &tree.decision_node_ids {
        let idx = nid as usize;
        let Some(local) = solver.flop_local_offset_at(idx) else { continue };
        let na = tree.nodes[idx].num_children as usize;
        let off = local * MAX_NA_POSTFLOP * nb;
        for b in 0..nb {
            let mut sum = 0.0f32;
            for a in 0..na {
                sum += cum[off + a * nb + b];
            }
            for a in 0..na {
                out.push(if sum > 0.0 { cum[off + a * nb + b] / sum } else { 1.0 / na as f32 });
            }
        }
    }
    out
}

fn write_section(out: &mut impl Write, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    writeln!(out, "{name}")?;
    out.write_all(&(bytes.len() as u64).to_le_bytes())?;
    out.write_all(bytes)
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u16s(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// CPU solve arm: iters−1 + 1 with the convergence stat in between.
fn solve_cpu(
    solver: &mut BucketedFlopCfr,
    tree: &FlatTree,
    game: &FlopStartGame,
    bk: &FlopBucketing,
    iters: u32,
    dsigma: &dyn Fn(&[f32], &[f32]) -> f64,
) -> (f64, Vec<f32>, Vec<f32>, Vec<f32>) {
    solver.run(tree, game, bk, iters - 1);
    let before = normalized_flop_sigma(solver.cum_strategy_flop(), solver, tree, bk);
    solver.run(tree, game, bk, 1);
    let after = normalized_flop_sigma(solver.cum_strategy_flop(), solver, tree, bk);
    (
        dsigma(&before, &after),
        solver.cum_strategy_flop().to_vec(),
        solver.cum_strategy_turn().to_vec(),
        solver.cum_strategy_river().to_vec(),
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_and_bank(
    fi: usize,
    flop: [Card; 3],
    tree: &FlatTree,
    nb: usize,
    iters: u32,
    nt: usize,
    nr: usize,
    out_dir: &str,
    role: &str,
    #[cfg(feature = "metal")] ctx: Option<&MetalContext>,
) -> std::io::Result<f64> {
    // Seeded nt×nr runout draw (chained splitmix without replacement;
    // nt=1/nr=1 first draws match the banked 1×1 baseline recipe).
    let board_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let deck: Vec<u8> = (0..52u8).filter(|c| board_mask & (1u64 << c) == 0).collect();
    let mut turn_cards: Vec<u8> = Vec::with_capacity(nt);
    {
        let mut pool = deck.clone();
        let mut x = fi as u64;
        for _ in 0..nt {
            x = splitmix64(x);
            let i = (x % pool.len() as u64) as usize;
            turn_cards.push(pool.remove(i));
        }
    }
    let mut river_decks: Vec<Vec<u8>> = vec![vec![]; 52];
    for &tc in &turn_cards {
        let mut pool: Vec<u8> = deck.iter().copied().filter(|&c| c != tc).collect();
        let mut x = (fi as u64) ^ 0xDEAD_BEEF ^ ((tc as u64) << 40);
        for _ in 0..nr {
            x = splitmix64(x);
            let i = (x % pool.len() as u64) as usize;
            river_decks[tc as usize].push(pool.remove(i));
        }
    }

    let t0 = Instant::now();
    let table = FlopChanceTable::build_full_nh_sampled(flop, 6, &turn_cards, &river_decks);
    let (fm, tm, rm) = quantile_maps(&table, nb);
    let game = FlopStartGame::new(table);
    let bk = FlopBucketing::from_maps(game.table(), nb, nb, nb, fm, tm, rm);
    let mut solver = BucketedFlopCfr::new(tree, game.table(), &bk);
    solver.set_terminal_design(TerminalDesign::Design1Collapsed);
    // Convergence stat: mean |delta sigma_flop| between the last two
    // iterations (normalized rows) — the per-flop "convergence reached"
    // number the harness reads without re-solving.
    assert!(iters >= 2);

    // Solve on CPU or the G5 native GPU path; either way the cum
    // buffers come back in the SAME ZoneDims-concatenated layout.
    let dsigma = |before: &[f32], after: &[f32]| -> f64 {
        let n = before.len().max(1);
        before.iter().zip(after).map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>()
            / n as f64
    };
    #[cfg(feature = "metal")]
    let gpu_flag = ctx.is_some();
    #[cfg(not(feature = "metal"))]
    let gpu_flag = false;

    let (conv, cum_flop, cum_turn, cum_river): (f64, Vec<f32>, Vec<f32>, Vec<f32>);
    #[cfg(feature = "metal")]
    if let Some(ctx) = ctx {
        let mut n = BucketedNativeGpu::new(ctx, tree, game.table(), &bk, &solver, (32 / nb) as u32)
            .map_err(|e| std::io::Error::other(format!("native gpu: {e}")))?;
        n.run(iters - 1);
        let before = normalized_flop_sigma(n.cum_strategy_flop(), &solver, tree, &bk);
        n.run(1);
        let after = normalized_flop_sigma(n.cum_strategy_flop(), &solver, tree, &bk);
        conv = dsigma(&before, &after);
        cum_flop = n.cum_strategy_flop().to_vec();
        cum_turn = n.cum_strategy_turn().to_vec();
        cum_river = n.cum_strategy_river().to_vec();
    } else {
        (conv, cum_flop, cum_turn, cum_river) = solve_cpu(&mut solver, tree, &game, &bk, iters, &dsigma);
    }
    #[cfg(not(feature = "metal"))]
    {
        (conv, cum_flop, cum_turn, cum_river) = solve_cpu(&mut solver, tree, &game, &bk, iters, &dsigma);
    }
    let secs = t0.elapsed().as_secs_f64();

    let path = format!("{out_dir}/flop_{fi:04}.bp");
    let tmp = format!("{path}.tmp");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(b"SSBP1\n")?;
        let code = std::env::var("BP_GIT").unwrap_or_else(|_| "unknown".into());
        let turns_json =
            turn_cards.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
        let rivers_json = turn_cards
            .iter()
            .map(|&tc| {
                format!(
                    "[{}]",
                    river_decks[tc as usize]
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            f,
            "{{\"role\":\"{role}\",\
             \"format\":1,\"fi\":{fi},\"flop\":[{},{},{}],\"b\":[{nb},{nb},{nb}],\
             \"turn\":[{turns_json}],\"rivers\":[{rivers_json}],\
             \"runout_seed\":\"chained splitmix64 w/o replacement; turn x0=fi; river x0=fi^0xDEADBEEF^(tc<<40)\",\
             \"iters\":{iters},\"conv_mean_dsigma_last_iter\":{conv:.6},\
             \"tree\":{{\"np\":6,\"pot\":12,\"stacks\":94,\"contribs\":0,\"bets\":[1.0],\"raises\":[]}},\
             \"nh\":{},\"terminal\":\"design1_collapsed\",\"maps\":\"quantile-strength\",\
             \"gpu\":{gpu_flag},\"code\":\"{code}\",\"secs\":{secs:.1}}}",
            flop[0], flop[1], flop[2],
            game.table().num_valid,
        )?;
        write_section(&mut f, "cum_flop", &f32s(&cum_flop))?;
        write_section(&mut f, "cum_turn", &f32s(&cum_turn))?;
        write_section(&mut f, "cum_river", &f32s(&cum_river))?;
        write_section(&mut f, "map_flop", &u16s(&bk.flop_map))?;
        let turn_cat: Vec<u16> = bk.turn_map.iter().flatten().copied().collect();
        write_section(&mut f, "map_turn", &u16s(&turn_cat))?;
        let river_cat: Vec<u16> =
            bk.river_map.iter().flatten().flatten().copied().collect();
        write_section(&mut f, "map_river", &u16s(&river_cat))?;
        f.flush()?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(secs)
}

fn main() {
    let out_dir = std::env::var("BP_OUT").unwrap_or_else(|_| "blueprint_out".into());
    let gpu: bool = std::env::var("BP_GPU").map(|s| s == "1").unwrap_or(false);
    if gpu && !cfg!(feature = "metal") {
        panic!("BP_GPU=1 requires building with --features metal");
    }
    let threads: usize = std::env::var("BP_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if gpu { 2 } else { 14 });
    let iters: u32 = std::env::var("BP_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(34);
    let nb: usize = std::env::var("BP_B").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let start: usize = std::env::var("BP_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let runouts = std::env::var("BP_RUNOUTS").unwrap_or_else(|_| "1x1".into());
    let (nt, nr): (usize, usize) = {
        let mut it = runouts.split('x');
        (
            it.next().and_then(|s| s.parse().ok()).expect("BP_RUNOUTS NTxNR"),
            it.next().and_then(|s| s.parse().ok()).expect("BP_RUNOUTS NTxNR"),
        )
    };
    let role = std::env::var("BP_ROLE").unwrap_or_else(|_| {
        "bootstrap-config-baseline (oracle-shape tree; NOT the production blueprint)".into()
    });
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    // Run-level manifest: the artifact set is self-describing as a whole.
    {
        let code = std::env::var("BP_GIT").unwrap_or_else(|_| "unknown".into());
        let mut mf = std::fs::File::create(format!("{out_dir}/manifest.json")).expect("manifest");
        writeln!(
            mf,
            "{{\"role\":\"{role}\",\
             \"cell\":\"1755 x seeded {nt}x{nr}\",\"b\":{nb},\"iters\":{iters},\"gpu\":{gpu},\
             \"tree\":{{\"np\":6,\"pot\":12,\"stacks\":94,\"contribs\":0,\"bets\":[1.0],\"raises\":[]}},\
             \"terminal\":\"design1_collapsed\",\"maps\":\"quantile-strength\",\
             \"runout_seed\":\"chained splitmix64 w/o replacement; turn x0=fi; river x0=fi^0xDEADBEEF^(tc<<40)\",\
             \"code\":\"{code}\",\"format\":1}}"
        )
        .expect("manifest write");
    }

    eprintln!(
        "blueprint_runner: B={nb}, {iters} iters, {nt}×{nr} seeded runouts, \
         {} arm",
        if gpu { "GPU-native" } else { "CPU" }
    );
    let ranges: Vec<Vec<f32>> =
        (0..6).map(|_| vec![1.0 / N_CLASSES as f32; N_CLASSES]).collect();
    let ptable = PreflopChanceTable::new(6, ranges);
    let canon = ptable.canonical_flops.clone();
    let end: usize =
        std::env::var("BP_END").ok().and_then(|s| s.parse().ok()).unwrap_or(canon.len());
    let tree = build_oracle_tree();
    eprintln!(
        "{} canonical flops [{start}..{end}), tree {} nodes, {} threads, out={out_dir}",
        canon.len(),
        tree.num_nodes(),
        threads
    );

    let todo: Vec<usize> = (start..end)
        .filter(|fi| !std::path::Path::new(&format!("{out_dir}/flop_{fi:04}.bp")).exists())
        .collect();
    eprintln!("{} flops to solve ({} already banked)", todo.len(), (end - start) - todo.len());

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t_all = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads.min(todo.len()) {
            s.spawn(|| {
                // One Metal context (own queue) per worker thread.
                #[cfg(feature = "metal")]
                let ctx: Option<MetalContext> =
                    if gpu { Some(MetalContext::new().expect("Metal")) } else { None };
                loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= todo.len() {
                    break;
                }
                let fi = todo[k];
                match solve_and_bank(
                    fi,
                    canon[fi],
                    &tree,
                    nb,
                    iters,
                    nt,
                    nr,
                    &out_dir,
                    &role,
                    #[cfg(feature = "metal")]
                    ctx.as_ref(),
                ) {
                    Ok(secs) => {
                        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if d % 25 == 0 || d == todo.len() {
                            let el = t_all.elapsed().as_secs_f64();
                            eprintln!(
                                "[{d}/{}] flop {fi} done in {secs:.0}s | elapsed {:.1}h | ETA {:.1}h",
                                todo.len(),
                                el / 3600.0,
                                el / d as f64 * (todo.len() - d) as f64 / 3600.0
                            );
                        }
                    }
                    Err(e) => eprintln!("flop {fi} FAILED: {e}"),
                }
                }
            });
        }
    });
    eprintln!("all done in {:.1}h", t_all.elapsed().as_secs_f64() / 3600.0);
}
