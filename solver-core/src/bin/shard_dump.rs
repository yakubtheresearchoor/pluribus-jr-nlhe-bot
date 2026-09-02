// row-occupancy dump for a preflop.ssbp2 shard
fn main() {
    if std::env::var("FIND_ACE_FLOP").is_ok() {
        let canon = solver_core::abstraction::flop_isomorphism::enumerate_canonical_flops();
        for (i, f) in canon.iter().enumerate() {
            if f.iter().any(|&c| c >> 2 == 12) {
                println!("ace flop idx {i}: {:?}", f);
                if i > 5 { break; }
            }
        }
        std::process::exit(0);
    }
    let dir = std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(format!("{dir}/preflop.ssbp2")).unwrap();
    let cum = solver_core::blueprint::ssbp2_decode_cum(&bytes).unwrap();
    let np: usize = std::env::var("DUMP_NP").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let (pft, pre_local, pre_ninfo, _maxna) = solver_core::blueprint::build_conn_preflop_tree(np, std::env::var("DUMP_NRAISES").ok().and_then(|s| s.parse().ok()).unwrap_or(5));
    let nc = 169usize;
    let maxna = cum.len() / (pre_ninfo * nc);
    println!("len={} pre_ninfo={pre_ninfo} maxna={maxna}", cum.len());
    // CONVERT_F32=<out>: write the shard as an MC_LOAD_PREFLOP-compatible .f32
    // (PreflopVectorCfr layout [local][a][c], stride MAX_NA_PREFLOP) so Phase B
    // can freeze THIS preflop. Rows are l1-normalized per (local, class) — the
    // loader treats them as frozen regrets, so regret-matching reproduces the
    // average strategy exactly.
    if let Ok(out) = std::env::var("CONVERT_F32") {
        let stride = solver_core::tree::flat::MAX_NA_PREFLOP;
        let mut v = vec![0f32; pre_ninfo * stride * nc];
        for l in 0..pre_ninfo {
            for c in 0..nc {
                let row: Vec<f32> = (0..maxna).map(|a| cum[(l*nc + c)*maxna + a].max(0.0)).collect();
                let z: f32 = row.iter().sum();
                if z <= 0.0 { continue; }
                for a in 0..maxna {
                    v[l*stride*nc + a*nc + c] = row[a] / z;
                }
            }
        }
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&out, &bytes).unwrap();
        println!("wrote {} ({} floats, stride {stride})", out, v.len());
        std::process::exit(0);
    }
    // per-local: total mass
    let mut nonzero_locals = 0;
    let mut per_local: Vec<(usize, f32)> = (0..pre_ninfo).map(|l| {
        let s: f32 = cum[l*nc*maxna .. (l+1)*nc*maxna].iter().map(|v| v.max(0.0)).sum();
        (l, s)
    }).collect();
    for &(_, s) in &per_local { if s > 0.0 { nonzero_locals += 1; } }
    per_local.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("nonzero locals: {nonzero_locals}/{pre_ninfo}");
    // trained mainline: follow max-cum actions from the root, printing each
    // node's action masses (which sizes the solve actually plays)
    if std::env::var("DUMP_MAINLINE").is_ok() {
        let mut node = 0usize;
        for step in 0..8 {
            if !pft.nodes[node].is_player() { println!("step {step}: node {node} not player"); break; }
            let l = pre_local[node];
            if l < 0 { break; }
            let l = l as usize;
            let na = pft.nodes[node].num_children as usize;
            let acts: Vec<(usize, u8, i32, f32)> = pft.node_children(node).iter().enumerate().map(|(a, &k)| {
                let m: f32 = (0..nc).map(|c| cum[(l*nc + c)*maxna + a].max(0.0)).sum();
                (a, pft.nodes[k as usize].action_label, pft.get_contribution(k as usize, pft.nodes[node].player_id), m)
            }).collect();
            let s: Vec<String> = acts.iter().map(|(_, lb, ct, m)| format!("{lb}@{ct}:{m:.0}")).collect();
            println!("step {step} node {node} (seat {}): [{}]", pft.nodes[node].player_id, s.join(", "));
            // follow the max-mass AGGRESSIVE action if present, else max-mass
            let best = acts.iter().filter(|(_, lb, _, _)| *lb >= 3).max_by(|x, y| x.3.partial_cmp(&y.3).unwrap())
                .or_else(|| acts.iter().max_by(|x, y| x.3.partial_cmp(&y.3).unwrap())).unwrap();
            node = pft.node_children(node)[best.0] as usize;
        }
        std::process::exit(0);
    }
    // depth histogram of trained locals: hard cutoff vs reach starvation
    if std::env::var("DUMP_DEPTH").is_ok() {
        let mut depth = vec![u32::MAX; pft.num_nodes()];
        depth[0] = 0;
        for i in 0..pft.num_nodes() {
            if depth[i] == u32::MAX { continue; }
            for &k in pft.node_children(i) {
                let k = k as usize;
                if depth[k] > depth[i] + 1 { depth[k] = depth[i] + 1; }
            }
        }
        let mut per: std::collections::BTreeMap<u32, (u32, u32)> = Default::default(); // depth -> (trained, total)
        for i in 0..pft.num_nodes() {
            if pre_local[i] < 0 || depth[i] == u32::MAX { continue; }
            let l = pre_local[i] as usize;
            let m: f32 = cum[l*nc*maxna .. (l+1)*nc*maxna].iter().map(|v| v.max(0.0)).sum();
            let e = per.entry(depth[i]).or_insert((0, 0));
            e.1 += 1;
            if m > 0.0 { e.0 += 1; }
        }
        for (d, (t, n)) in &per {
            println!("depth {d}: trained {t}/{n}");
        }
        std::process::exit(0);
    }
    println!("top mass locals: {:?}", &per_local[..8.min(per_local.len())]);
    if std::env::var("DUMP_CLASSES").is_ok() {
        use solver_core::abstraction::preflop_class::PreflopClass;
        let n0 = 0usize;
        let l0 = pre_local[n0] as usize;
        let na = pft.nodes[n0].num_children as usize;
        let labels: Vec<u8> = pft.node_children(n0).iter().map(|&k| pft.nodes[k as usize].action_label).collect();
        let mut rows: Vec<(usize, f32, f32)> = Vec::new(); // (cls, raise%, mass)
        for cls in 0..nc {
            let row: Vec<f32> = (0..na).map(|a| cum[(l0*nc + cls)*maxna + a].max(0.0)).collect();
            let z: f32 = row.iter().sum();
            if z <= 0.0 { rows.push((cls, -1.0, 0.0)); continue; }
            // "aggressive-or-continue" share: raises if present, else call/check
            let has_raise = labels.iter().any(|&l| l >= 3);
            let r: f32 = (0..na).filter(|&a| if has_raise { labels[a] >= 3 } else { labels[a] == 2 || labels[a] == 1 }).map(|a| row[a]).sum();
            rows.push((cls, r / z * 100.0, z));
        }
        // full root rows for AA and 22 (fold/call/raise split — raise% alone
        // cannot distinguish slow-play from folding)
        for cls in [0usize, 12] {
            let row: Vec<f32> = (0..na).map(|a| cum[(l0*nc + cls)*maxna + a].max(0.0)).collect();
            let z: f32 = row.iter().sum::<f32>().max(1e-9);
            let s: Vec<String> = (0..na).map(|a| format!("{}@{:.2}", labels[a], row[a]/z)).collect();
            println!("FULL cls {cls}: [{}]", s.join(", "));
        }
        println!("cls raise% mass  (name)");
        for &(cls, r, m) in &rows {
            let combos = solver_core::abstraction::preflop_class::class_combos(PreflopClass(cls as u8));
            let (c1, c2) = combos[0];
            let rc = |c: u8| b"23456789TJQKA"[(c >> 2) as usize] as char;
            let cl = PreflopClass(cls as u8);
            let tag = if cl.is_pair() { format!("{}{}", rc(c1), rc(c2)) }
                else { format!("{}{}{}", rc(c1.max(c2)), rc(c1.min(c2)), if cl.is_suited() { "s" } else { "o" }) };
            println!("{:>3} {:>6.1} {:>12.0}  {}", cls, r, m, tag);
        }
        std::process::exit(0);
    }
    // what local does the battery's SB-vs-open node have?
    let mut node = 0usize;
    let hist: Vec<(u8,i32)> = vec![(0,0),(0,0),(0,0),(4,5)];
    for &(label, _) in &hist {
        let kids: Vec<usize> = pft.node_children(node).iter().map(|&k| k as usize).collect();
        let next = kids.iter().find(|&&k| {
            let l = pft.nodes[k].action_label;
            if label >= 3 { l >= 3 } else { l == label }
        });
        node = *next.unwrap();
    }
    // targeted-node probe: mass at the BB-defense node behind (fff, open, 3bet)
    if std::env::var("DUMP_TARGET").is_ok() {
        // walk: 3 folds, BTN raise (each size), SB raise (each size) -> BB node
        let mut n = 0usize;
        for _ in 0..3 {
            n = pft.node_children(n).iter().map(|&k| k as usize)
                .find(|&k| pft.nodes[k].action_label == 0).unwrap();
        }
        let btn = n;
        for &ko in pft.node_children(btn) {
            let ko = ko as usize;
            if pft.nodes[ko].action_label < 3 { continue; }
            let osz = pft.get_contribution(ko, pft.nodes[btn].player_id);
            for &k3 in pft.node_children(ko) {
                let k3 = k3 as usize;
                if pft.nodes[k3].action_label < 3 { continue; }
                let ssz = pft.get_contribution(k3, pft.nodes[ko].player_id);
                println!("chain: btn={btn} ko={ko}(seat {} ty {}) k3={k3}(seat {} ty {}) local {}",
                    pft.nodes[ko].player_id, pft.nodes[ko].node_type,
                    pft.nodes[k3].player_id, pft.nodes[k3].node_type, pre_local[k3]);
                if pft.nodes[k3].is_player() && pre_local[k3] >= 0 {
                    let l = pre_local[k3] as usize;
                    let m: f32 = cum[l*nc*maxna .. (l+1)*nc*maxna].iter().map(|v| v.max(0.0)).sum();
                    if m > 0.0 {
                        // 94s row (cls of 9h4h) + AA row
                        use solver_core::abstraction::preflop_class::PreflopClass;
                        let c94 = PreflopClass::from_combo(29, 9).index();
                        let caa = PreflopClass::from_combo(48, 49).index();
                        let na = pft.nodes[k3].num_children as usize;
                        let row = |cls: usize| -> Vec<f32> {
                            let r: Vec<f32> = (0..na).map(|a| cum[(l*nc + cls)*maxna + a].max(0.0)).collect();
                            let z: f32 = r.iter().sum::<f32>().max(1e-9);
                            r.iter().map(|v| v/z).collect()
                        };
                        let labels: Vec<u8> = pft.node_children(k3).iter().map(|&k| pft.nodes[k as usize].action_label).collect();
                        println!("open@{osz} 3bet@{ssz} BB node {k3} mass {m:.1} labels {labels:?}");
                        println!("   94s: {:?}", row(c94).iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>());
                        println!("   AA:  {:?}", row(caa).iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>());
                    }
                }
            }
        }
        std::process::exit(0);
    }
    // ALL-CLASS root open raise%: the broken-set pattern discriminates
    // indexing bugs (contiguous cls ranges) from eval bugs (card-structural).
    // contribution audit: BTN open children sizes + which branches carry mass
    {
        let mut n = 0usize;
        for _ in 0..3 { // three folds
            n = pft.node_children(n).iter().map(|&k| k as usize)
                .find(|&k| pft.nodes[k].action_label == 0).unwrap();
        }
        let actor = pft.nodes[n].player_id;
        println!("BTN node {n} (seat {actor}) children:");
        for &k in pft.node_children(n) {
            let k = k as usize;
            let l = pft.nodes[k].action_label;
            let contrib = pft.get_contribution(k, actor);
            // mass at the FOLLOWING node (hero SB decision behind this branch)
            let m: f32 = if pft.nodes[k].is_player() && pre_local[k] >= 0 {
                let l0 = pre_local[k] as usize;
                cum[l0*nc*maxna .. (l0+1)*nc*maxna].iter().map(|v| v.max(0.0)).sum()
            } else { -1.0 };
            println!("  child {k}: label {l} contrib {contrib} next-node-mass {m}");
        }
    }
    // BTN's own node: mass + strategy for AA (cls of AcAd) and K7o-ish
    {
        let n = 11usize;
        let l0 = pre_local[n] as usize;
        let m: f32 = cum[l0*nc*maxna .. (l0+1)*nc*maxna].iter().map(|v| v.max(0.0)).sum();
        println!("BTN node 11 local {l0} mass {m}");
        let na = pft.nodes[n].num_children as usize;
        for cls in [0usize, 80, 168] {
            let row: Vec<f32> = (0..na).map(|a| cum[(l0*nc + cls)*maxna + a].max(0.0)).collect();
            let z: f32 = row.iter().sum::<f32>().max(1e-9);
            let p: Vec<String> = row.iter().map(|v| format!("{:.2}", v/z)).collect();
            println!("  cls {cls}: [{}]", p.join(", "));
        }
        // also: locals of nodes 1..24 mass map
        for i in 0..24 {
            if pre_local[i] >= 0 {
                let l = pre_local[i] as usize;
                let m: f32 = cum[l*nc*maxna .. (l+1)*nc*maxna].iter().map(|v| v.max(0.0)).sum();
                println!("  node {i} local {l} mass {m:.0}");
            }
        }
    }
    println!("SB-vs-open node={} local={} mass={}", node, pre_local[node],
        if pre_local[node] >= 0 { per_local.iter().find(|(l,_)| *l == pre_local[node] as usize).unwrap().1 } else { -1.0 });
}
