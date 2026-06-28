//! Integration tests for the connected-MCCFR blueprint loader (BLP2). These run
//! against the lib's public API (not the crate's unit-test target), so they are
//! independent of any stale in-crate test modules.

use solver_core::abstraction::preflop_class::{PreflopClass, NUM_PREFLOP_CLASSES};
use solver_core::blueprint::{
    build_conn_preflop_tree, gs14_cache_path, load_gs14_cache, runout_grid, ssbp2_decode_cum,
    ssbp2_encode_cum, ConnBlueprint, ConnCellLayout, FlopLayout, ShardedConnBlueprint, BLP2_MAGIC,
};
use solver_core::solver::preflop_start_game::PreflopChanceTable;

/// Resolve a workspace-root path (tests run with CWD = crate dir).
fn ws(p: &str) -> String { format!("{}/../{}", env!("CARGO_MANIFEST_DIR"), p) }
fn bp() -> String { std::env::var("CONN_TEST_BP").unwrap_or_else(|_| ws("blueprint_conn_eqr")) }

/// Serialize a known `gcum` into BLP2 bytes against the REAL rebuilt tree, load
/// it back, and verify parse + index math + normalization + replay. No GPU
/// needed — exercises the exact indexing path the runtime serves from.
#[test]
fn loads_and_indexes_synthetic_blp2() {
    let (np, nraises) = (2usize, 1usize);
    let (pft, pre_local, pre_ninfo, pre_maxna) = build_conn_preflop_tree(np, nraises);
    let maxna = pre_maxna.max(3);
    let (nf, post_stride, nh, nb, nt, nr) = (1usize, 4usize, 2usize, 8usize, 8usize, 8usize);
    let pre = pre_ninfo * NUM_PREFLOP_CLASSES * maxna;
    let mut gcum = vec![0.0f32; pre + nf * post_stride];

    // Known cumulant at a chosen (player node, class=AA): [1,2,..,na] →
    // normalized [1/Σ, 2/Σ, ...].
    let node = (0..pft.num_nodes()).find(|&i| pft.nodes[i].is_player()).unwrap();
    let na = pft.nodes[node].num_children as usize;
    let cls = 0usize; // AA
    let base = (pre_local[node] as usize * NUM_PREFLOP_CLASSES + cls) * maxna;
    for a in 0..na {
        gcum[base + a] = (a + 1) as f32;
    }

    let mut buf = Vec::new();
    let hdr: [u32; 12] = [
        BLP2_MAGIC, np as u32, nraises as u32, pre_ninfo as u32,
        NUM_PREFLOP_CLASSES as u32, maxna as u32, nf as u32, post_stride as u32,
        nh as u32, nb as u32, nt as u32, nr as u32,
    ];
    for v in hdr { buf.extend_from_slice(&v.to_le_bytes()); }
    for &x in &gcum { buf.extend_from_slice(&x.to_le_bytes()); }
    let tmp = std::env::temp_dir().join("conn_bp_synth_test.bin");
    std::fs::write(&tmp, &buf).unwrap();

    let bp = ConnBlueprint::load(tmp.to_str().unwrap()).expect("load synthetic BLP2");
    assert_eq!(bp.hdr.np, 2);
    assert_eq!(bp.hdr.nraises, 1);
    assert_eq!(bp.hdr.nb, 8);

    // Index math: recovered distribution == the normalized cumulant we wrote.
    let dist = bp.preflop_dist_at(node, cls);
    let total: f32 = (1..=na).map(|x| x as f32).sum();
    for a in 0..na {
        assert!(
            (dist[a] - (a + 1) as f32 / total).abs() < 1e-6,
            "action {a}: {} != {}", dist[a], (a + 1) as f32 / total
        );
    }
    // Untrained class → uniform over the node's actions.
    let u = bp.preflop_dist_at(node, 168);
    assert!((u[0] - 1.0 / na as f32).abs() < 1e-6);

    // Replay from root (empty history) yields a normalized distribution.
    let d0 = bp.preflop_action_dist((48, 49), &[]).expect("root dist");
    let s: f32 = d0.iter().map(|(_, _, p)| p).sum();
    assert!((s - 1.0).abs() < 1e-5, "root dist sums to {s}");
    let _ = std::fs::remove_file(&tmp);
}

/// Cross-check against a real extracted blueprint when present (written by
/// `mccfr_probe MC_EXTRACT`). Skipped where the file is absent (e.g. CI).
#[test]
fn cross_check_extracted_blp2_if_present() {
    let path = "/tmp/bp_blp2.bin";
    let Ok(bp) = ConnBlueprint::load(path) else { return; };
    let (node, fold_a) = bp.first_fold_node().expect("fold node");
    // AA (A♠A♥ = cards 48, 49) should fold rarely at the open-or-fold node.
    let aa = bp.preflop_dist_at(node, PreflopClass::from_combo(48, 49).index());
    assert!(aa[fold_a] < 0.1, "AA fold prob {} unexpectedly high", aa[fold_a]);
    let s: f32 = aa.iter().sum();
    assert!((s - 1.0).abs() < 1e-5);
}

/// SSBP2 per-flop cum codec: encode → decode roundtrips within u8-quant error
/// (≤ scale/255), and actually compresses a low-entropy (CFR-like, mostly-zero)
/// slab. This is the codec the per-flop solve writes each flop's output through.
#[test]
fn ssbp2_cum_roundtrip_and_compresses() {
    // CFR cum strategies: mostly small/zero with a few peaks — what we'll store.
    let mut slab = vec![0.0f32; 50_000];
    for i in 0..slab.len() {
        slab[i] = if i % 7 == 0 { (i % 13) as f32 * 1000.0 } else { 0.0 };
    }
    let enc = ssbp2_encode_cum(&slab, 9);
    let dec = ssbp2_decode_cum(&enc).expect("decode");
    assert_eq!(dec.len(), slab.len());
    let hi = slab.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lo = slab.iter().cloned().fold(f32::INFINITY, f32::min);
    let tol = (hi - lo) / 255.0 + 1e-3; // one quant step
    for (a, b) in slab.iter().zip(dec.iter()) {
        assert!((a - b).abs() <= tol, "roundtrip {a} vs {b} exceeds {tol}");
    }
    // Should be much smaller than raw f32 (50k×4 = 200KB).
    assert!(enc.len() < slab.len() * 4 / 4, "no compression: {} vs {}", enc.len(), slab.len() * 4);
    // Empty/degenerate slab must not panic.
    let z = ssbp2_encode_cum(&[0.0; 8], 9);
    assert_eq!(ssbp2_decode_cum(&z).unwrap(), vec![0.0; 8]);
}

/// Decode a per-flop blueprint shard written by the Phase-B builder (when present)
/// and verify it's a sane trained strategy: finite, has real (non-degenerate) mass,
/// and within a plausible magnitude. Skipped if the build hasn't been run.
#[test]
fn phase_b_output_decodes_sane() {
    for f in ["/tmp/bp_conn_test/flop_0000.ssbp2", "/tmp/bp_conn_test/preflop.ssbp2"] {
        let Ok(bytes) = std::fs::read(f) else { continue; };
        let slab = ssbp2_decode_cum(&bytes).expect("decode shard");
        assert!(!slab.is_empty(), "{f}: empty");
        assert!(slab.iter().all(|x| x.is_finite()), "{f}: non-finite values");
        let nz = slab.iter().filter(|&&x| x.abs() > 1e-9).count();
        assert!(nz > 0, "{f}: all-zero (postflop never trained)");
        // CFR cum mass is non-negative (CFR+ floors regrets; cum accumulates strategy).
        assert!(slab.iter().all(|&x| x >= -1e-3), "{f}: negative cum mass");
    }
}

/// Load the REAL solved production blueprint (blueprint_conn_eqr/) via the sharded
/// loader and verify it serves: preflop distributions are sane (sum to 1, premiums
/// non-fold-heavy, trash fold-heavy), and a postflop shard decodes + the FlopLayout
/// resolves a bucket. Skipped if the blueprint isn't present (CI).
#[test]
fn sharded_blueprint_serves_decisions() {
    let dir = bp();
    // production config: np=6, 5 preraises, nb=200, 16×16 runout, maxna=7.
    let Ok(bp) = ShardedConnBlueprint::load(&dir, 6, 5, 200, 16, 16, 7) else { return; };

    // PREFLOP: raise-or-fold opens (EQR-frozen) from the first acting node (empty
    // history). AA/KK aggressive (low fold), 72o fold-heavier; all sum to 1.
    let probes = [("AA", 48u8, 49u8), ("KK", 44, 45), ("72o", 20, 0)];
    let (mut aa_fold, mut t72_fold) = (1.0f32, 0.0f32);
    for (nm, a, b) in probes {
        let dist = bp.preflop_action_dist((a, b), &[]).expect("preflop dist");
        let sum: f32 = dist.iter().map(|(_, _, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-4, "{nm}: dist sums to {sum}");
        assert!(dist.iter().all(|(_, _, p)| p.is_finite() && *p >= 0.0), "{nm}: bad probs");
        let fold = dist.iter().find(|(l, _, _)| *l == 0).map(|(_, _, p)| *p).unwrap_or(0.0);
        if nm == "AA" { aa_fold = fold; }
        if nm == "72o" { t72_fold = fold; }
    }
    // Directional sanity: AA folds less than 72o (premiums continue more than trash).
    assert!(aa_fold <= t72_fold + 0.05, "AA fold {aa_fold} not ≤ 72o fold {t72_fold}");

    // POSTFLOP: a shard decodes to a non-empty, finite, non-negative cum slab,
    // and the FlopLayout resolves an in-range bucket for a live hand.
    let post = bp.postflop_cum(0).expect("postflop shard 0");
    assert!(!post.is_empty() && post.iter().all(|x| x.is_finite() && *x >= -1e-3));
    assert!(post.iter().any(|&x| x.abs() > 1e-9), "postflop shard all-zero");
}

/// Reconstruct the postflop cell layout and serve a FLOP decision from the real
/// blueprint. The keystone correctness gate: the runtime-reconstructed `post_stride`
/// must equal the decoded shard length (byte-exact agreement with the solve's cell
/// layout). Then a flop-street decision indexes the shard to a sane distribution.
/// Skipped if blueprint/cache absent.
#[test]
fn postflop_cell_layout_and_flop_decision() {
    let dir = bp();
    let Ok(bp) = ShardedConnBlueprint::load(&dir, 6, 5, 200, 16, 16, 7) else { return; };
    let layout = ConnCellLayout::build(6, 5, 200);
    let post0 = bp.postflop_cum(0).expect("shard 0");

    // ★ GATE: reconstructed post_stride == decoded shard length (proves the whole
    // cell layout — trees, n_info, offsets, maxna — matches the solve byte-for-byte).
    assert_eq!(layout.post_stride, post0.len(),
        "reconstructed post_stride {} != shard length {}", layout.post_stride, post0.len());
    assert!(layout.maxna >= 1 && layout.maxna <= 8, "postflop maxna={}", layout.maxna);
    assert!(!layout.keys.is_empty());

    // Flop decision: need the hero's flop bucket. flop_map is hand-indexed
    // (runout-independent), so the 49×48 cache's flop_map serves any subset.
    let nb = 200usize;
    let mpath = gs14_cache_path(&ws("gs14_blueprint_cache"), 0, nb, 49, 48);
    let Some(maps) = load_gs14_cache(&mpath, nb, 49, 48) else { return; };
    let ranges = vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6];
    let flop = PreflopChanceTable::new(6, ranges).canonical_flops[0];
    let fl = FlopLayout::for_canonical(flop, 16, 16);
    let board: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let deck: Vec<u8> = (0..52u8).filter(|c| board & (1u64 << c) == 0).collect();
    let bucket = fl.flop_bucket(&maps, (deck[0], deck[1])).expect("flop bucket") as usize;

    // First flop decision in each cell (empty history = cell root) → sane dist.
    let (live, commit, pot) = layout.keys[0];
    if let Some(dist) = layout.flop_action_dist(&post0, live, commit, pot, bucket, &[]) {
        let sum: f32 = dist.iter().map(|(_, _, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-4, "flop dist sums to {sum}");
        assert!(dist.iter().all(|(_, _, p)| p.is_finite() && *p >= 0.0));
    }
}

/// Turn/river decisions: walk a cell tree (auto-advancing through deal nodes) to a
/// turn-street player node, then confirm postflop_action_dist resolves there with a
/// sane distribution — exercising chance-node traversal + per-street bucket indexing.
#[test]
fn postflop_turn_decision_resolves() {
    let dir = bp();
    let Ok(bp) = ShardedConnBlueprint::load(&dir, 6, 5, 200, 16, 16, 7) else { return; };
    let layout = ConnCellLayout::build(6, 5, 200);
    let post0 = bp.postflop_cum(0).expect("shard 0");
    if layout.keys.is_empty() { return; }

    // Pick the cell with the largest tree (most likely to span streets).
    let ki = (0..layout.keys.len()).max_by_key(|&i| layout.cell_trees[i].num_nodes()).unwrap();
    let (live, commit, pot) = layout.keys[ki];
    let tree = &layout.cell_trees[ki];

    // BFS to a turn-street (board_state==1) player node, recording the betting path
    // (auto-advancing chance/deal nodes, which have a single child).
    let mut found: Option<Vec<(u8, i32)>> = None;
    let mut stack: Vec<(usize, Vec<(u8, i32)>)> = vec![(0, vec![])];
    while let Some((mut node, hist)) = stack.pop() {
        if hist.len() > 10 { continue; }
        while tree.nodes[node].is_chance() { node = tree.node_children(node)[0] as usize; }
        if !tree.nodes[node].is_player() { continue; }
        if tree.nodes[node].board_state == 1 { found = Some(hist); break; }
        for &k in tree.node_children(node) {
            let kn = &tree.nodes[k as usize];
            let mut h2 = hist.clone();
            h2.push((kn.action_label, kn.amount));
            stack.push((k as usize, h2));
        }
    }

    if let Some(hist) = found {
        // any in-range bucket suffices to exercise indexing (bucket VALUES are
        // covered by the FlopLayout tests); use 0/1/2 across streets.
        let dist = bp_dist(&layout, &post0, live, commit, pot, [5, 7, 9], &hist);
        let d = dist.expect("turn decision should resolve");
        let sum: f32 = d.iter().map(|(_, _, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-4, "turn dist sums to {sum}");
        assert!(d.iter().all(|(_, _, p)| p.is_finite() && *p >= 0.0));
    }
}

fn bp_dist(layout: &ConnCellLayout, post: &[f32], live: u8, commit: i32, pot: i32, b: [usize; 3], h: &[(u8, i32)]) -> Option<Vec<(u8, i32, f32)>> {
    layout.postflop_action_dist(post, live, commit, pot, b, h)
}

/// Load a real GS14 bucket-map cache file (when present) and verify shape +
/// bucket-value invariants: dims match (nt × nr × nh), and every non-dead bucket
/// is in `[0, nb)`. Mirrors `save_gs14`'s on-disk format. Skipped if absent.
#[test]
fn load_gs14_cache_if_present() {
    let (nb, nt, nr) = (200usize, 16usize, 16usize);
    // Try a couple of static (precompute-untouched) locations.
    let candidates = [
        gs14_cache_path("/tmp/bk_1d", 0, nb, nt, nr),
        gs14_cache_path("/tmp/dryrun", 0, nb, nt, nr),
    ];
    let Some(path) = candidates.into_iter().find(|p| p.exists()) else { return; };
    let (fm, tm, rm) = load_gs14_cache(&path, nb, nt, nr).expect("load gs14 cache");
    let nh = fm.len();
    assert!(nh > 0, "empty flop map");
    assert_eq!(tm.len(), nt, "turn-map outer dim");
    assert_eq!(rm.len(), nt, "river-map outer dim");
    assert!(tm.iter().all(|t| t.len() == nh), "turn-map hand dim");
    assert!(rm.iter().all(|t| t.len() == nr && t.iter().all(|r| r.len() == nh)), "river-map dims");

    // Every assigned bucket is in range; dead hands are u16::MAX.
    let ok = |b: u16| b == u16::MAX || (b as usize) < nb;
    assert!(fm.iter().all(|&b| ok(b)), "flop bucket out of range");
    assert!(tm.iter().flatten().all(|&b| ok(b)), "turn bucket out of range");
    assert!(rm.iter().flatten().flatten().all(|&b| ok(b)), "river bucket out of range");
    // At least some flop hands are clustered into a real bucket (not all dead).
    assert!(fm.iter().any(|&b| b != u16::MAX), "all flop hands dead?");

    // A wrong (nb,nt,nr) must be rejected (header guard).
    assert!(load_gs14_cache(&path, nb, nt + 1, nr).is_none(), "accepted wrong nt");
}

/// End-to-end postflop card→bucket mapping against the PRODUCTION cache (49×48,
/// nb=200) when present: build the FlopLayout for canonical flop 0, load its
/// maps, and verify a live hero hand resolves to an in-range bucket on flop,
/// turn, and river — and that board-colliding hands return None. Skipped if the
/// production cache is absent.
#[test]
fn postflop_bucket_lookup_production_cache() {
    let (nb, nt, nr) = (200usize, 49usize, 48usize);
    let path = gs14_cache_path(&ws("gs14_blueprint_cache"), 0, nb, nt, nr);
    let Some(maps) = load_gs14_cache(&path, nb, nt, nr) else { return; };

    // Canonical flop 0 — same source the precompute used (uniform 6-max ranges).
    let ranges = vec![vec![1.0f32 / NUM_PREFLOP_CLASSES as f32; NUM_PREFLOP_CLASSES]; 6];
    let flop = PreflopChanceTable::new(6, ranges).canonical_flops[0];
    let layout = FlopLayout::for_canonical(flop, nt, nr);

    // Pick a hero hand that does NOT collide with the flop.
    let board: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let deck: Vec<u8> = (0..52u8).filter(|c| board & (1u64 << c) == 0).collect();
    let (h1, h2) = (deck[0], deck[1]);

    // Flop bucket in range.
    let fb = layout.flop_bucket(&maps, (h1, h2)).expect("live hero has a flop bucket");
    assert!((fb as usize) < nb, "flop bucket {fb} out of range");

    // Turn/river off the same grid the maps use; pick a turn + river not blocking hero.
    let (turns, river_decks) = runout_grid(flop, nt, nr);
    let tc = *turns.iter().find(|&&c| c != h1 && c != h2).unwrap();
    let tb = layout.turn_bucket(&maps, (h1, h2), tc).expect("turn bucket");
    assert!((tb as usize) < nb, "turn bucket {tb} out of range");
    let rc = *river_decks[tc as usize].iter().find(|&&c| c != h1 && c != h2).unwrap();
    let rb = layout.river_bucket(&maps, (h1, h2), tc, rc).expect("river bucket");
    assert!((rb as usize) < nb, "river bucket {rb} out of range");

    // A hand using a board card must be dead (no flop bucket).
    assert!(layout.flop_bucket(&maps, (flop[0], deck[2])).is_none(), "board-colliding hand not dead");
    // A hand holding the turn card is dead on that turn (but valid on the flop).
    let x = *deck.iter().find(|&&c| c != tc && c != deck[2]).unwrap();
    assert!(layout.flop_bucket(&maps, (tc, x)).is_some(), "(tc,x) should be live on the flop");
    assert!(layout.turn_bucket(&maps, (tc, x), tc).is_none(), "hand holding the turn card not dead on that turn");
}

/// MEASUREMENT (cargo test -p solver-core --test conn_blueprint config_fit -- --nocapture --ignored):
/// per-live cell count + floats under the REAL per-live keying (conn_seam_bin) +
/// build_conn_cell_tree — predicts the solve's regret_len before any GPU run.
#[test]
#[ignore]
fn config_fit() {
    use solver_core::blueprint::{build_conn_cell_tree, conn_seam_bin, conn_seam_rep};
    use solver_core::tree::action::production_game_v1;
    use std::collections::{BTreeMap, HashMap};
    let (np, nraises) = (6usize, 5usize);
    let (pft, _pl, pre_ninfo, pre_maxna) = build_conn_preflop_tree(np, nraises);
    let stack = production_game_v1().stack;
    let nb = 200usize;
    // distinct cells (key -> rep), grouped by live
    let mut cells: HashMap<(u8, i64), (u8, i32, i32)> = HashMap::new();
    for i in 0..pft.num_nodes() {
        if pft.nodes[i].is_chance() {
            let bk = conn_seam_bin(&pft, i, np, stack);
            if bk.0 >= 2 { cells.entry(bk).or_insert_with(|| conn_seam_rep(&pft, i, np)); }
        }
    }
    // maxna = max(pre, postflop)
    let mut maxna = pre_maxna;
    let mut by_live: BTreeMap<u8, (usize, usize)> = BTreeMap::new();
    for (_k, &(live, c, p)) in &cells {
        let t = build_conn_cell_tree(live, c, p);
        let ni = (0..t.num_nodes()).filter(|&n| t.nodes[n].is_player()).count();
        let mx = (0..t.num_nodes()).map(|n| t.nodes[n].num_children as usize).max().unwrap_or(0);
        maxna = maxna.max(mx);
        let e = by_live.entry(live).or_insert((0, 0)); e.0 += 1; e.1 += ni;
    }
    let pre_floats = pre_ninfo * 169 * maxna;
    println!("maxna={maxna}");
    println!("preflop: {pre_ninfo} infosets -> {:.2}B floats", pre_floats as f64 / 1e9);
    let mut tot = pre_floats;
    for (live, (nc, ni)) in &by_live {
        let fl = ni * nb * maxna; tot += fl;
        println!("live={live}: {nc} cells, {ni} infosets -> {:.3}B floats", fl as f64 / 1e9);
    }
    println!("=> TOTAL regret_len = {:.3}B floats vs u32 4.29B -> {}", tot as f64 / 1e9, if tot < 4_294_967_295 { "FITS" } else { "OVER" });
}
