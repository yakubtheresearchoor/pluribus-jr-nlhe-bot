use crate::tree::flat::{FlatTree, NODE_TYPE_PLAYER};
use super::game::GameSpec;

const UNUSED: usize = usize::MAX;

pub struct CpuMccfr {
    num_players: u8,
    num_hands: Vec<usize>,
    regrets: Vec<f32>,
    cum_strategy: Vec<f32>,
    node_data_offset: Vec<usize>,
    iteration: u32,
    regret_floor: f32,
    // QRE generalization (S0, 2026-06-13): per-seat inverse-temperature
    // λ. `None` = pure Nash (regret matching) — bit-identical to the
    // pre-QRE solver, so existing gates are untouched. `Some(λ)` =
    // QUANTAL response: the acting player's strategy is a LOGIT over the
    // action counterfactual values, σ_a ∝ exp(λ_seat · cfv_a) (per hand).
    // As λ→∞ the logit → argmax(cfv) = best-response-to-last-iterate, so
    // the averaged play is fictitious play → Nash (the reduction the S0
    // gate proves). At finite λ it is a genuine quantal response (λ
    // controls rationality), which S2 uses to model rough opponents.
    // `last_cfv` holds the previous iteration's per-(node,action,hand)
    // counterfactual value the logit responds to.
    lambda: Option<Vec<f32>>,
    last_cfv: Vec<f32>,
    // DEPTH-LIMITED SEARCH (S1, 2026-06-13): nodes flagged `frozen` play
    // a FIXED strategy (the blueprint's) and are NOT updated — the
    // searcher optimizes only the un-frozen (subgame) nodes against the
    // frozen continuation. Freezing the post-boundary street = the
    // depth-limited leaf-continuation valuation (leaf value = continue
    // with the blueprint).
    frozen: Vec<bool>,
    // PLURIBUS MULTI-CONTINUATION (Modicum 2018 / Pluribus 2019): instead of
    // a SINGLE frozen continuation, hold `num_variants` continuation
    // strategies — variant 0 = blueprint, 1..k = action-biased (toward
    // fold/call/raise). At a depth-limit BOUNDARY node the acting player
    // SELECTS among the k variants (a regret-matched decision inside the
    // solved subgame), so the searcher must be robust to the opponent
    // deviating from the exact blueprint at the leaf. `num_variants == 1`
    // ⇒ pure single-continuation, bit-identical to the pre-Pluribus path.
    // `frozen_variants[c]` is one full [a*nh+h] strategy array over all
    // frozen nodes; `boundary[n]` = a frozen node reached from the searched
    // region (its player selects the variant); `sel_*` hold the per-boundary
    // k-action selection regret/cum, indexed by `sel_offset`.
    num_variants: usize,
    frozen_variants: Vec<Vec<f32>>,
    boundary: Vec<bool>,
    sel_offset: Vec<usize>,
    sel_regret: Vec<f32>,
    sel_cum: Vec<f32>,
    // DEPTH-LIMIT BOUNDARY (Pluribus-faithful multiway): at these nodes the
    // searcher stops and values the leaf via `game.evaluate_continuation`
    // (the bucketed + MC-sampled blueprint continuation) INSTEAD of recursing
    // into the (frozen) next street. This is what keeps multiway flat in
    // players — the O(nh^np) showdown never enters the searched tree, it lives
    // in the sampled continuation. Empty ⇒ no depth limit (recurse as before).
    depth_limit: Vec<bool>,
    // PARALLELIZATION (snapshot CFR): the per-iteration strategy, computed ONCE
    // from the regrets/last_cfv at iter start. The walk READS this (not the live
    // regrets), so parallel per-traverser walks never read a value another
    // thread is writing — the race-free basis for multi-threading (preflop's
    // proven pattern). Empty ⇒ legacy on-the-fly path (sequential).
    snapshot: Vec<f32>,
    use_snapshot: bool,
}

impl CpuMccfr {
    pub fn new(tree: &FlatTree, num_hands: Vec<usize>) -> Self {
        assert_eq!(num_hands.len(), tree.num_players as usize);

        let mut offsets = Vec::with_capacity(tree.num_nodes());
        let mut total = 0usize;

        for node in &tree.nodes {
            if node.node_type == NODE_TYPE_PLAYER {
                offsets.push(total);
                total += node.num_children as usize * num_hands[node.player_id as usize];
            } else {
                offsets.push(UNUSED);
            }
        }

        CpuMccfr {
            num_players: tree.num_players,
            num_hands,
            regrets: vec![0.0; total],
            cum_strategy: vec![0.0; total],
            node_data_offset: offsets,
            iteration: 0,
            regret_floor: -1e30,
            lambda: None,
            last_cfv: vec![0.0; total],
            frozen: vec![false; tree.num_nodes()],
            num_variants: 1,
            frozen_variants: vec![vec![0.0; total]],
            boundary: vec![false; tree.num_nodes()],
            sel_offset: vec![UNUSED; tree.num_nodes()],
            sel_regret: Vec::new(),
            sel_cum: Vec::new(),
            depth_limit: vec![false; tree.num_nodes()],
            snapshot: Vec::new(),
            use_snapshot: false,
        }
    }

    /// Enable SNAPSHOT-CFR mode (prerequisite for the multi-threaded walk): the
    /// per-iter strategy is frozen at iter start and the walk reads it, so
    /// parallel traversers never read a regret another thread is writing. This
    /// is a valid CFR variant (all traversers use the iteration's strategy) but
    /// NOT bit-identical to the legacy sequential-update path, hence opt-in (the
    /// bit-exact leduc gates keep the default off).
    pub fn enable_parallel(&mut self) {
        self.use_snapshot = true;
        self.snapshot = vec![0.0; self.regrets.len()];
    }

    /// Freeze the per-iteration strategy for every un-frozen player node into
    /// `self.snapshot` (read by the walk in snapshot mode). Computes each node's
    /// strategy from the current regrets/last_cfv (immutable), then writes — so
    /// the walk's reads are race-free even when traversers run in parallel.
    fn compute_snapshot(&mut self, tree: &FlatTree) {
        let updates: Vec<(usize, Vec<f32>)> = (0..tree.num_nodes())
            .filter(|&n| tree.nodes[n].is_player() && !self.frozen[n])
            .map(|n| {
                let na = tree.nodes[n].num_children as usize;
                let player = tree.nodes[n].player_id;
                let nh = self.num_hands[player as usize];
                (self.node_data_offset[n], self.compute_strategy(n, na, nh, player))
            })
            .collect();
        for (off, strat) in updates {
            self.snapshot[off..off + strat.len()].copy_from_slice(&strat);
        }
    }

    /// DEPTH-LIMITED SEARCH boundary: mark nodes where the searcher values the
    /// leaf via `game.evaluate_continuation` instead of recursing. For a flop
    /// search these are the turn-entry nodes (the first node of the next
    /// street). The subtree past a depth-limit node is never walked, so it need
    /// not be frozen (though freezing is harmless). This is the lever that
    /// makes multiway tractable: the continuation is sampled, not enumerated.
    pub fn set_depth_limit(&mut self, nodes: &[usize]) {
        for b in self.depth_limit.iter_mut() {
            *b = false;
        }
        for &n in nodes {
            self.depth_limit[n] = true;
        }
    }

    /// DEPTH-LIMITED SEARCH: freeze a node to a fixed (blueprint)
    /// strategy. `strat` is `num_actions × nh` in [a*nh + h] order. The
    /// node plays this strategy, is never regret/cum-updated, and its
    /// cum_strategy is set to it so the scored profile reads the
    /// blueprint there. The un-frozen nodes are searched against it.
    pub fn freeze_node(&mut self, node_idx: usize, strat: &[f32]) {
        let off = self.node_data_offset[node_idx];
        assert_ne!(off, UNUSED, "freeze_node on a non-player node {node_idx}");
        assert!(off + strat.len() <= self.frozen_variants[0].len(), "frozen strat overruns node block");
        self.frozen[node_idx] = true;
        self.frozen_variants[0][off..off + strat.len()].copy_from_slice(strat);
        // Seed cum_strategy so StrategyProfile reads the blueprint here.
        self.cum_strategy[off..off + strat.len()].copy_from_slice(strat);
    }

    /// PLURIBUS multi-continuation setup. After all blueprint continuations
    /// are frozen (variant 0), generate `k-1` action-biased variants and the
    /// boundary/selection bookkeeping. `bias` multiplies the targeted action
    /// class's probability then renormalizes (per hand, per frozen node):
    ///   variant 1 = fold-biased, 2 = call/check-biased, 3 = raise/bet/allin-biased.
    /// k=1 is a no-op (single-continuation). The acting player at each
    /// boundary then regret-matches over the k variants during `run`.
    pub fn setup_pluribus_continuations(&mut self, tree: &FlatTree, k: usize, bias: f32) {
        assert!(k >= 1, "k>=1");
        self.num_variants = k;
        if k == 1 {
            return;
        }
        let total = self.frozen_variants[0].len();
        // Action-class targets per variant (by builder action_label): fold=0;
        // check=1,call=2; bet=3,raise=4,allin=5. Variant 0 = blueprint (no bias).
        let targets: Vec<&[u8]> = vec![&[], &[0], &[1, 2], &[3, 4, 5]];
        self.frozen_variants.resize(k, vec![0.0; total]);
        for c in 1..k {
            self.frozen_variants[c] = self.frozen_variants[0].clone();
        }
        for node_idx in 0..tree.num_nodes() {
            if !self.frozen[node_idx] {
                continue;
            }
            let off = self.node_data_offset[node_idx];
            if off == UNUSED {
                continue;
            }
            let na = tree.nodes[node_idx].num_children as usize;
            let nh = self.num_hands[tree.nodes[node_idx].player_id as usize];
            let labels: Vec<u8> = tree
                .node_children(node_idx)
                .iter()
                .map(|&c| tree.nodes[c as usize].action_label)
                .collect();
            for c in 1..k {
                let tgt = if c < targets.len() { targets[c] } else { &[] };
                for h in 0..nh {
                    let mut sum = 0.0f32;
                    for a in 0..na {
                        let mut p = self.frozen_variants[c][off + a * nh + h];
                        if tgt.contains(&labels[a]) {
                            p *= bias;
                        }
                        self.frozen_variants[c][off + a * nh + h] = p;
                        sum += p;
                    }
                    if sum > 0.0 {
                        for a in 0..na {
                            self.frozen_variants[c][off + a * nh + h] /= sum;
                        }
                    }
                }
            }
        }
        // Selection nodes = each player's FIRST frozen node on each path (the
        // owner selects its OWN continuation variant there; deeper frozen nodes
        // of that player just play the selected variant). DFS from the root
        // carrying a bitmask of frozen-region players already encountered.
        let mut sel_total = 0usize;
        let mut stack: Vec<(usize, u32)> = vec![(0usize, 0u32)];
        while let Some((n, seen)) = stack.pop() {
            let mut new_seen = seen;
            if self.frozen[n]
                && tree.nodes[n].node_type == NODE_TYPE_PLAYER
                && self.node_data_offset[n] != UNUSED
            {
                let p = tree.nodes[n].player_id as u32;
                if seen & (1 << p) == 0 {
                    self.boundary[n] = true;
                    self.sel_offset[n] = sel_total;
                    sel_total += k * self.num_hands[p as usize];
                    new_seen = seen | (1 << p);
                }
            }
            for &ch in tree.node_children(n) {
                stack.push((ch as usize, new_seen));
            }
        }
        self.sel_regret = vec![0.0; sel_total];
        self.sel_cum = vec![0.0; sel_total];
        if std::env::var("PLURIBUS_DEBUG").is_ok() {
            let nb = self.boundary.iter().filter(|&&b| b).count();
            let nfr = self.frozen.iter().filter(|&&f| f).count();
            let vdiff: f32 = self.frozen_variants[1]
                .iter()
                .zip(&self.frozen_variants[0])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            eprintln!("[pluribus] frozen={nfr} boundaries={nb} k={k} max|var1-var0|={vdiff:.4}");
        }
    }

    /// Enable QUANTAL-response (QRE) mode with per-seat inverse-
    /// temperature λ (length = num_players). High λ → Nash limit; the
    /// S0 reduction gate proves QRE-at-high-λ reproduces the Nash solve.
    pub fn set_lambda(&mut self, lambda: Vec<f32>) {
        assert_eq!(lambda.len(), self.num_players as usize, "lambda per seat");
        self.lambda = Some(lambda);
    }

    pub fn run(
        &mut self,
        tree: &FlatTree,
        game: &dyn GameSpec,
        num_iterations: u32,
    ) -> Vec<f32> {
        let nh0 = self.num_hands[0];
        let np = self.num_players as usize;

        let mut root_cfv_sum = vec![0.0f32; nh0];
        let mut p0_count = 0u32;

        for _ in 0..num_iterations {
            self.iteration += 1;
            let weight = self.iteration as f32;

            // Snapshot mode: freeze this iter's strategy so the walk's reads are
            // race-free (prerequisite for the parallel traverser loop).
            if self.use_snapshot {
                self.compute_snapshot(tree);
            }

            let mut cfreach: Vec<Vec<f32>> = (0..np)
                .map(|p| game.initial_weight(p as u8))
                .collect();

            for traverser in 0..np {
                let mut traverser_reach = game.initial_weight(traverser as u8);
                let mut var = vec![None; np];
                let cfv = self.walk_cfr(tree, game, traverser as u8, 0, &mut cfreach, &mut traverser_reach, weight, &mut var);

                if traverser == 0 {
                    for h in 0..nh0 {
                        root_cfv_sum[h] += cfv[h];
                    }
                    p0_count += 1;
                }
            }
        }

        for h in 0..nh0 {
            root_cfv_sum[h] /= p0_count as f32;
        }

        root_cfv_sum
    }

    fn walk_cfr(
        &mut self,
        tree: &FlatTree,
        game: &dyn GameSpec,
        traverser: u8,
        node_idx: usize,
        cfreach: &mut [Vec<f32>],
        traverser_reach: &mut [f32],
        weight: f32,
        // PER-PLAYER continuation variant (Pluribus multi-continuation, v2).
        // var[p] = None until p crosses into the frozen region; then Some(c)
        // = p plays continuation variant c at ITS frozen nodes. Each player
        // selects its OWN continuation independently, so the traverser faces
        // opponents picking their continuations adversarially (the searcher's
        // own continuation is whatever it best-responds to — no joint bias).
        var: &mut Vec<Option<usize>>,
    ) -> Vec<f32> {
        let node = &tree.nodes[node_idx];

        // DEPTH LIMIT: value the leaf via the (sampled) blueprint continuation
        // and stop — do NOT recurse into the next street. Checked first so it
        // overrides chance/player handling at the boundary node.
        if self.depth_limit[node_idx] {
            return game.evaluate_continuation(traverser, node_idx, tree, cfreach);
        }

        if node.is_terminal() {
            return game.evaluate_terminal(traverser, node_idx, tree, cfreach);
        }

        if node.is_chance() {
            let children = tree.node_children(node_idx);
            if children.is_empty() {
                return game.evaluate_terminal(traverser, node_idx, tree, cfreach);
            }
            let nh_t = self.num_hands[traverser as usize];
            let mut cfv = vec![0.0f32; nh_t];

            let n_outcomes = game.num_chance_outcomes();
            if n_outcomes > 0 && children.len() == 1 {
                let child = children[0] as usize;
                for outcome in 0..n_outcomes {
                    if outcome > 0 { game.clear_chance_outcome(); }
                    let probs: Vec<f32> = (0..nh_t).map(|h| game.chance_probability(outcome, h)).collect();
                    game.set_chance_outcome(outcome);
                    let child_cfv = self.walk_cfr(tree, game, traverser, child, cfreach, traverser_reach, weight, var);
                    for h in 0..nh_t {
                        cfv[h] += probs[h] * child_cfv[h];
                    }
                }
                game.clear_chance_outcome();
            } else {
                for (outcome, &child) in children.iter().enumerate() {
                    if outcome > 0 { game.clear_chance_outcome(); }
                    let probs: Vec<f32> = (0..nh_t).map(|h| game.chance_probability(outcome, h)).collect();
                    game.set_chance_outcome(outcome);
                    let child_cfv = self.walk_cfr(tree, game, traverser, child as usize, cfreach, traverser_reach, weight, var);
                    for h in 0..nh_t {
                        cfv[h] += probs[h] * child_cfv[h];
                    }
                }
                game.clear_chance_outcome();
            }
            return cfv;
        }

        let player = node.player_id;
        let num_actions = node.num_children as usize;
        let nh = self.num_hands[player as usize];
        let children = tree.node_children(node_idx);

        // PLURIBUS SELECTION (v2): the OWNER's first frozen node — it selects
        // its own continuation variant. `boundary[]` marks each player's first
        // frozen node (set only when num_variants>1); `var[player].is_none()`
        // confirms the owner hasn't selected yet on this path. k=1 leaves
        // boundary[] empty ⇒ never taken ⇒ bit-identical to single-continuation.
        let frozen_here = self.frozen[node_idx];
        if frozen_here && self.boundary[node_idx] && var[player as usize].is_none() {
            return self.boundary_select(tree, game, traverser, node_idx, cfreach, traverser_reach, weight, var);
        }

        // Frozen node: play the OWNER's selected variant (var[owner]); 0 if
        // unset (e.g. k=1). Searched node: regret-match / QRE.
        let strategy: Vec<f32> = if frozen_here {
            let off = self.node_data_offset[node_idx];
            let c = var[player as usize].unwrap_or(0);
            self.frozen_variants[c][off..off + num_actions * nh].to_vec()
        } else if self.use_snapshot {
            let off = self.node_data_offset[node_idx];
            self.snapshot[off..off + num_actions * nh].to_vec()
        } else {
            self.compute_strategy(node_idx, num_actions, nh, player)
        };

        let mut cfv_all: Vec<Vec<f32>> = Vec::with_capacity(num_actions);

        for (a, &child) in children.iter().enumerate() {
            let saved_reach = if player == traverser {
                let saved = traverser_reach.to_vec();
                for h in 0..nh {
                    traverser_reach[h] *= strategy[a * nh + h];
                }
                Some(saved)
            } else {
                let saved = cfreach[player as usize].clone();
                for h in 0..nh {
                    cfreach[player as usize][h] *= strategy[a * nh + h];
                }
                Some(saved)
            };

            cfv_all.push(self.walk_cfr(tree, game, traverser, child as usize, cfreach, traverser_reach, weight, var));

            if player == traverser {
                traverser_reach.copy_from_slice(saved_reach.as_ref().unwrap());
            } else {
                cfreach[player as usize] = saved_reach.unwrap();
            }
        }

        let nh_traverser = self.num_hands[traverser as usize];
        let mut cfv_avg = vec![0.0f32; nh_traverser];
        if player == traverser {
            for h in 0..nh_traverser {
                for a in 0..num_actions {
                    cfv_avg[h] += strategy[a * nh + h] * cfv_all[a][h];
                }
            }
        } else {
            for h in 0..nh_traverser {
                for a in 0..num_actions {
                    cfv_avg[h] += cfv_all[a][h];
                }
            }
        }

        if player == traverser && !self.frozen[node_idx] {
            let offset = self.node_data_offset[node_idx];
            for h in 0..nh {
                for a in 0..num_actions {
                    let idx = offset + a * nh + h;
                    let regret = cfv_all[a][h] - cfv_avg[h];
                    self.regrets[idx] += weight * regret;
                    if self.regrets[idx] < self.regret_floor {
                        self.regrets[idx] = self.regret_floor;
                    }
                }
            }

            for h in 0..nh {
                for a in 0..num_actions {
                    let idx = offset + a * nh + h;
                    self.cum_strategy[idx] += weight * traverser_reach[h] * strategy[a * nh + h];
                }
            }

            // QRE: ACCUMULATE action values so the logit responds to the
            // value against the TIME-AVERAGE opponent (smooth fictitious
            // play — convergent in zero-sum), NOT the last iterate
            // (Cournot best-response dynamics — oscillates, does not
            // converge; the S0 v1 bug). cfv is linear in opponent reach,
            // so the average cfv IS the cfv against the average opponent.
            // No-op for `lambda == None` (Nash path ignores last_cfv).
            if self.lambda.is_some() {
                for h in 0..nh {
                    for a in 0..num_actions {
                        self.last_cfv[offset + a * nh + h] += cfv_all[a][h];
                    }
                }
            }
        }

        cfv_avg
    }

    /// PLURIBUS boundary: the acting player selects among the k continuation
    /// variants (a regret-matched decision inside the solved subgame). For
    /// each variant c we play this node onward as a frozen continuation
    /// (in_frozen = Some(c)); the player regret-matches over the resulting
    /// values, robustifying the searched strategy against the opponent
    /// deviating from the exact blueprint at the depth limit.
    fn boundary_select(
        &mut self,
        tree: &FlatTree,
        game: &dyn GameSpec,
        traverser: u8,
        node_idx: usize,
        cfreach: &mut [Vec<f32>],
        traverser_reach: &mut [f32],
        weight: f32,
        var: &mut Vec<Option<usize>>,
    ) -> Vec<f32> {
        let player = tree.nodes[node_idx].player_id;
        let k = self.num_variants;
        let nh = self.num_hands[player as usize];
        let sel = self.compute_sel_strategy(node_idx, k, nh);

        let mut cfv_c: Vec<Vec<f32>> = Vec::with_capacity(k);
        for c in 0..k {
            let saved = if player == traverser {
                let s = traverser_reach.to_vec();
                for h in 0..nh {
                    traverser_reach[h] *= sel[c * nh + h];
                }
                s
            } else {
                let s = cfreach[player as usize].clone();
                for h in 0..nh {
                    cfreach[player as usize][h] *= sel[c * nh + h];
                }
                s
            };
            // Set ONLY this player's continuation variant; other players keep
            // their own (selected at their own first frozen node). Re-entering
            // this node now finds var[player]=Some(c) ⇒ plays variant c, not
            // re-select.
            var[player as usize] = Some(c);
            cfv_c.push(self.walk_cfr(
                tree, game, traverser, node_idx, cfreach, traverser_reach, weight, var,
            ));
            var[player as usize] = None;
            if player == traverser {
                traverser_reach.copy_from_slice(&saved);
            } else {
                cfreach[player as usize] = saved;
            }
        }

        let nh_t = self.num_hands[traverser as usize];
        let mut cfv_avg = vec![0.0f32; nh_t];
        if player == traverser {
            for h in 0..nh_t {
                for c in 0..k {
                    cfv_avg[h] += sel[c * nh + h] * cfv_c[c][h];
                }
            }
            let off = self.sel_offset[node_idx];
            for h in 0..nh {
                for c in 0..k {
                    let idx = off + c * nh + h;
                    self.sel_regret[idx] += weight * (cfv_c[c][h] - cfv_avg[h]);
                    if self.sel_regret[idx] < self.regret_floor {
                        self.sel_regret[idx] = self.regret_floor;
                    }
                }
            }
            for h in 0..nh {
                for c in 0..k {
                    self.sel_cum[off + c * nh + h] += weight * traverser_reach[h] * sel[c * nh + h];
                }
            }
        } else {
            for h in 0..nh_t {
                for c in 0..k {
                    cfv_avg[h] += cfv_c[c][h];
                }
            }
        }
        cfv_avg
    }

    /// Regret-match the per-boundary continuation selection (k variants).
    /// The opponent picks the worst-for-searcher continuation in the limit —
    /// the Modicum robustification.
    fn compute_sel_strategy(&self, node_idx: usize, k: usize, nh: usize) -> Vec<f32> {
        let off = self.sel_offset[node_idx];
        let mut s = vec![0.0f32; k * nh];
        for h in 0..nh {
            let mut pos = 0.0f32;
            for c in 0..k {
                let r = self.sel_regret[off + c * nh + h];
                if r > 0.0 {
                    pos += r;
                }
            }
            if pos > 0.0 {
                for c in 0..k {
                    let r = self.sel_regret[off + c * nh + h];
                    s[c * nh + h] = if r > 0.0 { r / pos } else { 0.0 };
                }
            } else {
                let u = 1.0 / k as f32;
                for c in 0..k {
                    s[c * nh + h] = u;
                }
            }
        }
        s
    }

    fn compute_strategy(&self, node_idx: usize, num_actions: usize, nh: usize, player: u8) -> Vec<f32> {
        let offset = self.node_data_offset[node_idx];

        // Depth-limited search: a frozen node plays its fixed blueprint
        // strategy (variant 0). The walk plays the ACTIVE variant directly;
        // this branch only serves the informational accessor.
        if self.frozen[node_idx] {
            return self.frozen_variants[0][offset..offset + num_actions * nh].to_vec();
        }

        let mut strategy = vec![0.0f32; num_actions * nh];

        // QRE (quantal) mode: logit over action counterfactual values at
        // the acting seat's inverse-temperature λ. σ_a ∝ exp(λ·cfv_a),
        // numerically stabilized by subtracting the per-hand max. First
        // iterate (last_cfv all zero) → uniform, the natural QRE start.
        if let Some(lam) = &self.lambda {
            let l = lam[player as usize];
            // Respond to the AVERAGE action value (accumulated cfv ÷ count
            // of accumulation iterations = value vs the time-average
            // opponent). Iteration was incremented at run-start, and
            // last_cfv accumulates from prior iterations, so the average
            // denominator is (iteration − 1). First iterate: 0 → uniform.
            let denom = (self.iteration as f32 - 1.0).max(1.0);
            for h in 0..nh {
                let mut mx = f32::NEG_INFINITY;
                for a in 0..num_actions {
                    mx = mx.max(self.last_cfv[offset + a * nh + h] / denom);
                }
                let mut z = 0.0f32;
                for a in 0..num_actions {
                    let avg = self.last_cfv[offset + a * nh + h] / denom;
                    let w = (l * (avg - mx)).exp();
                    strategy[a * nh + h] = w;
                    z += w;
                }
                let z = if z > 0.0 { z } else { 1.0 };
                for a in 0..num_actions {
                    strategy[a * nh + h] /= z;
                }
            }
            return strategy;
        }

        for h in 0..nh {
            let mut pos_sum = 0.0f32;
            for a in 0..num_actions {
                let r = self.regrets[offset + a * nh + h];
                if r > 0.0 {
                    pos_sum += r;
                }
            }

            if pos_sum > 0.0 {
                for a in 0..num_actions {
                    let r = self.regrets[offset + a * nh + h];
                    strategy[a * nh + h] = if r > 0.0 { r / pos_sum } else { 0.0 };
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    strategy[a * nh + h] = uniform;
                }
            }
        }

        strategy
    }

    pub fn get_regrets(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let offset = self.node_data_offset[node_idx];
        if offset == UNUSED {
            return vec![];
        }
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for a in 0..num_actions {
            for h in 0..nh {
                result[a][h] = self.regrets[offset + a * nh + h];
            }
        }
        result
    }

    pub fn get_cum_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let offset = self.node_data_offset[node_idx];
        if offset == UNUSED {
            return vec![];
        }
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for a in 0..num_actions {
            for h in 0..nh {
                result[a][h] = self.cum_strategy[offset + a * nh + h];
            }
        }
        result
    }

    pub fn get_current_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let player = 0u8; // informational accessor; player unknown here, Nash-mode uses regrets only
        let raw = self.compute_strategy(node_idx, num_actions, nh, player);
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for a in 0..num_actions {
            for h in 0..nh {
                result[a][h] = raw[a * nh + h];
            }
        }
        result
    }

    pub fn get_average_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let offset = self.node_data_offset[node_idx];
        if offset == UNUSED {
            return vec![];
        }

        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for h in 0..nh {
            let mut total = 0.0f32;
            for a in 0..num_actions {
                total += self.cum_strategy[offset + a * nh + h];
            }
            if total > 0.0 {
                for a in 0..num_actions {
                    result[a][h] = self.cum_strategy[offset + a * nh + h] / total;
                }
            } else {
                let uniform = 1.0 / num_actions as f32;
                for a in 0..num_actions {
                    result[a][h] = uniform;
                }
            }
        }
        result
    }

    pub fn node_offsets(&self) -> &[usize] {
        &self.node_data_offset
    }

    pub fn iteration_count(&self) -> u32 {
        self.iteration
    }

    pub fn cum_strategy_slice(&self) -> &[f32] {
        &self.cum_strategy
    }

    pub fn regrets_slice(&self) -> &[f32] {
        &self.regrets
    }
}
