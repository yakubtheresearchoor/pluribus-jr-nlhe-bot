use crate::tree::action::{Action, BetSize, BoardState, TreeConfig};
use crate::tree::flat::{FlatNode, FlatTree, NODE_TYPE_PLAYER, NODE_TYPE_TERMINAL};

const MAX_DEPTH: usize = 64;

pub fn build_tree(config: &TreeConfig) -> Result<FlatTree, String> {
    let num_players = config.num_players as usize;
    if num_players < 2 || num_players > 10 {
        return Err(format!("num_players must be 2-10, got {}", num_players));
    }
    if config.starting_stacks.len() != num_players {
        return Err(format!(
            "starting_stacks.len() must be {}, got {}",
            num_players,
            config.starting_stacks.len()
        ));
    }
    if config.initial_contributions.len() != num_players {
        return Err(format!(
            "initial_contributions.len() must be {}, got {}",
            num_players,
            config.initial_contributions.len()
        ));
    }
    if config.starting_pot <= 0 {
        return Err(format!("starting_pot must be > 0, got {}", config.starting_pot));
    }
    for (i, &s) in config.starting_stacks.iter().enumerate() {
        if s < 0 {
            return Err(format!("starting_stacks[{}] = {} is negative", i, s));
        }
    }

    let mut builder = TreeBuilder {
        config,
        tree: FlatTree::new(
            config.num_players,
            config.starting_pot,
            config.starting_stacks.clone(),
            config.rake_rate,
            config.rake_cap,
        ),
    };

    let initial_contributions: Vec<i32> = config.initial_contributions.clone();
    let active_players: Vec<bool> = vec![true; num_players];
    let stacks: Vec<i32> = config
        .starting_stacks
        .iter()
        .zip(config.initial_contributions.iter())
        .map(|(s, b)| s - b)
        .collect();

    let first_player = builder.first_postflop_player(&active_players);

    let root = FlatNode::player(first_player, config.initial_state, 0);
    let root_idx = builder.tree.alloc_node(root);
    for p in 0..num_players {
        builder
            .tree
            .set_contribution(root_idx, p as u8, initial_contributions[p]);
    }

    let info = BuildInfo {
        stacks,
        active: active_players,
        folded: vec![false; num_players],
        has_acted_this_round: vec![false; num_players],
        round_starter: first_player as usize,
        prev_action: Action::None,
        num_bets: 0,
        allin_flag: false,
        board_state: config.initial_state,
        depth: 0,
    };

    builder.build_recursive(root_idx, info);

    Ok(builder.tree)
}

struct TreeBuilder<'a> {
    config: &'a TreeConfig,
    tree: FlatTree,
}

struct BuildInfo {
    stacks: Vec<i32>,
    active: Vec<bool>,
    folded: Vec<bool>,
    has_acted_this_round: Vec<bool>,
    round_starter: usize,
    prev_action: Action,
    num_bets: i32,
    allin_flag: bool,
    board_state: BoardState,
    depth: usize,
}

impl BuildInfo {
    fn folded_mask(&self) -> u16 {
        let mut mask: u16 = 0;
        for (p, &f) in self.folded.iter().enumerate() {
            if f {
                mask |= 1u16 << p;
            }
        }
        mask
    }

    fn clone_for_child(&self) -> Self {
        BuildInfo {
            stacks: self.stacks.clone(),
            active: self.active.clone(),
            folded: self.folded.clone(),
            has_acted_this_round: self.has_acted_this_round.clone(),
            round_starter: self.round_starter,
            prev_action: self.prev_action,
            num_bets: self.num_bets,
            allin_flag: self.allin_flag,
            board_state: self.board_state,
            depth: self.depth + 1,
        }
    }

    fn num_active(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    fn contributions_sum(&self, base_contributions: &[i32]) -> i32 {
        base_contributions.iter().sum()
    }
}

impl<'a> TreeBuilder<'a> {
    fn first_postflop_player(&self, active: &[bool]) -> u8 {
        let num_players = self.config.num_players as usize;
        for i in 0..num_players {
            if active[i] {
                return i as u8;
            }
        }
        0
    }

    fn next_active_player(&self, current: usize, active: &[bool]) -> Option<usize> {
        let num_players = self.config.num_players as usize;
        for offset in 1..=num_players {
            let next = (current + offset) % num_players;
            if active[next] {
                return Some(next);
            }
        }
        None
    }

    fn is_round_complete(&self, info: &BuildInfo) -> bool {
        let num_players = self.config.num_players as usize;
        for p in 0..num_players {
            if info.active[p] && !info.has_acted_this_round[p] {
                return false;
            }
        }

        let active_stacks: Vec<i32> = (0..num_players)
            .filter(|&p| info.active[p])
            .map(|p| info.stacks[p])
            .collect();
        if active_stacks.is_empty() {
            return true;
        }
        let first = active_stacks[0];
        active_stacks.iter().all(|&s| s == first)
    }

    fn only_one_active(&self, info: &BuildInfo) -> bool {
        info.num_active() <= 1
    }

    fn get_pot(&self, node_idx: usize) -> i32 {
        let num_players = self.config.num_players as usize;
        (0..num_players)
            .map(|p| self.tree.get_contribution(node_idx, p as u8))
            .sum()
    }

    fn build_recursive(&mut self, node_idx: usize, info: BuildInfo) {
        if info.depth > MAX_DEPTH {
            return;
        }

        if self.only_one_active(&info) {
            self.tree.nodes[node_idx].node_type = NODE_TYPE_TERMINAL;
            self.tree.set_folded_mask(node_idx, info.folded_mask());
            return;
        }

        if info.allin_flag && info.board_state != BoardState::River {
            let all_allin = (0..self.config.num_players as usize)
                .all(|p| !info.active[p] || info.stacks[p] == info.stacks[info.active.iter().position(|&a| a).unwrap_or(0)]);
            if all_allin {
                let mut child_info = info.clone_for_child();
                match info.board_state {
                    BoardState::Flop => {
                        child_info.board_state = BoardState::Turn;
                        self.add_chance_child(node_idx, child_info);
                    }
                    BoardState::Turn => {
                        child_info.board_state = BoardState::River;
                        self.add_chance_child(node_idx, child_info);
                    }
                    BoardState::River => {
                        self.tree.nodes[node_idx].node_type = NODE_TYPE_TERMINAL;
                        self.tree.set_folded_mask(node_idx, info.folded_mask());
                    }
                }
                return;
            }
        }

        if info.prev_action == Action::Call || self.is_round_complete_after_action(&info) {
            if info.board_state == BoardState::River {
                self.tree.nodes[node_idx].node_type = NODE_TYPE_TERMINAL;
                self.tree.set_folded_mask(node_idx, info.folded_mask());
                return;
            }
            let mut child_info = info.clone_for_child();
            match info.board_state {
                BoardState::Flop => child_info.board_state = BoardState::Turn,
                BoardState::Turn => child_info.board_state = BoardState::River,
                BoardState::River => unreachable!(),
            }
            self.add_chance_child(node_idx, child_info);
            return;
        }

        let player = self.tree.nodes[node_idx].player_id as usize;
        if !info.active[player] {
            if let Some(next) = self.next_active_player(player, &info.active) {
                self.tree.nodes[node_idx].player_id = next as u8;
                self.build_recursive(node_idx, info);
            }
            return;
        }

        let mut child_infos_and_actions = Vec::new();
        self.compute_actions(node_idx, &info, &mut child_infos_and_actions);

        if child_infos_and_actions.is_empty() {
            self.tree.nodes[node_idx].node_type = NODE_TYPE_TERMINAL;
            self.tree.set_folded_mask(node_idx, info.folded_mask());
            return;
        }

        let mut child_indices = Vec::with_capacity(child_infos_and_actions.len());
        for (action, child_info) in &child_infos_and_actions {
            let child_idx = self.make_child_node(node_idx, action, child_info);
            child_indices.push(child_idx as u32);
        }
        self.tree.set_children(node_idx, child_indices);

        for (i, (_, child_info)) in child_infos_and_actions.into_iter().enumerate() {
            let child_idx = self.tree.nodes[node_idx].children_start as usize + i;
            let child_idx = self.tree.children[child_idx] as usize;
            self.build_recursive(child_idx, child_info);
        }
    }

    fn is_round_complete_after_action(&self, info: &BuildInfo) -> bool {
        if !self.is_round_complete(info) {
            return false;
        }
        !matches!(info.prev_action, Action::None | Action::Chance(_))
    }

    fn add_chance_child(&mut self, parent_idx: usize, mut info: BuildInfo) {
        let chance_node = FlatNode::chance(info.board_state);
        let chance_idx = self.tree.alloc_node(chance_node);
        for p in 0..self.config.num_players as usize {
            let contrib = self.tree.get_contribution(parent_idx, p as u8);
            self.tree.set_contribution(chance_idx, p as u8, contrib);
        }

        let child_node = {
            let first = self.first_postflop_player(&info.active);
            FlatNode::player(first, info.board_state, self.tree.nodes[parent_idx].amount)
        };
        let child_idx = self.tree.alloc_node(child_node);
        for p in 0..self.config.num_players as usize {
            let contrib = self.tree.get_contribution(parent_idx, p as u8);
            self.tree.set_contribution(child_idx, p as u8, contrib);
        }

        info.has_acted_this_round = vec![false; self.config.num_players as usize];
        info.num_bets = 0;
        info.prev_action = Action::Chance(0);
        info.round_starter = self.first_postflop_player(&info.active) as usize;

        self.tree.set_children(chance_idx, vec![child_idx as u32]);
        self.tree.set_children(parent_idx, vec![chance_idx as u32]);

        self.build_recursive(child_idx, info);
    }

    fn compute_actions(
        &self,
        node_idx: usize,
        info: &BuildInfo,
        out: &mut Vec<(Action, BuildInfo)>,
    ) {
        let player = self.tree.nodes[node_idx].player_id as usize;
        let num_players = self.config.num_players as usize;

        let mut max_other_stack = 0i32;
        for p in 0..num_players {
            if p != player && info.active[p] {
                max_other_stack = max_other_stack.max(info.stacks[p]);
            }
        }

        let player_stack = info.stacks[player];
        let to_call = (max_other_stack - player_stack).max(0);
        let pot = self.get_pot(node_idx);
        let prev_amount = max_other_stack;
        let max_amount = player_stack + prev_amount;
        let min_amount = (prev_amount + to_call).clamp(1, max_amount);

        if player_stack <= 0 {
            let mut child_info = info.clone_for_child();
            child_info.has_acted_this_round[player] = true;
            child_info.prev_action = Action::Check;
            out.push((Action::Check, child_info));
            return;
        }

        let spr_after_call =
            (player_stack - to_call) as f64 / (pot + to_call * (info.num_active() as i32)) as f64;

        let num_remaining_streets = match info.board_state {
            BoardState::Flop => 3,
            BoardState::Turn => 2,
            BoardState::River => 1,
        };

        let bet_options = &self.config.bet_sizes;

        let mut actions = Vec::new();

        if matches!(
            info.prev_action,
            Action::None | Action::Check | Action::Chance(_)
        ) {
            actions.push(Action::Check);

            for &bet_size in &bet_options.bet {
                self.add_bet_size_action(
                    &bet_size,
                    pot,
                    0,
                    max_amount,
                    min_amount,
                    num_remaining_streets,
                    0,
                    spr_after_call,
                    &mut actions,
                );
            }

            if max_amount <= (pot as f64 * self.config.add_allin_threshold).round() as i32 {
                actions.push(Action::AllIn(max_amount));
            }
        } else {
            actions.push(Action::Fold);

            actions.push(Action::Call);

            if !info.allin_flag {
                for &bet_size in &bet_options.raise {
                    self.add_raise_size_action(
                        &bet_size,
                        pot,
                        prev_amount,
                        max_amount,
                        min_amount,
                        num_remaining_streets,
                        info.num_bets,
                        spr_after_call,
                        &mut actions,
                    );
                }

                let allin_threshold = pot as f64 * self.config.add_allin_threshold;
                if max_amount <= prev_amount + allin_threshold.round() as i32 {
                    actions.push(Action::AllIn(max_amount));
                }
            }
        }

        self.clamp_and_force_allin(&mut actions, pot, prev_amount, max_amount, min_amount);

        actions.sort();
        actions.dedup();

        actions = merge_bet_actions(actions, pot, prev_amount, self.config.merging_threshold);

        for action in actions {
            let mut child_info = info.clone_for_child();
            child_info.prev_action = action;

            match action {
                Action::Fold => {
                    child_info.active[player] = false;
                    child_info.folded[player] = true;
                    child_info.has_acted_this_round[player] = true;
                }
                Action::Check => {
                    child_info.has_acted_this_round[player] = true;
                }
                Action::Call => {
                    child_info.stacks[player] = max_other_stack;
                    child_info.has_acted_this_round[player] = true;
                }
                Action::Bet(amount) | Action::Raise(amount) => {
                    child_info.stacks[player] = amount;
                    child_info.num_bets += 1;
                    child_info.has_acted_this_round = vec![false; num_players];
                    child_info.has_acted_this_round[player] = true;
                    child_info.round_starter = player;
                }
                Action::AllIn(amount) => {
                    child_info.stacks[player] = amount;
                    child_info.allin_flag = true;
                    child_info.has_acted_this_round = vec![false; num_players];
                    child_info.has_acted_this_round[player] = true;
                    child_info.round_starter = player;
                }
                _ => {}
            }

            out.push((action, child_info));
        }
    }

    fn add_bet_size_action(
        &self,
        bet_size: &BetSize,
        pot: i32,
        _prev_amount: i32,
        _max_amount: i32,
        _min_amount: i32,
        num_remaining_streets: i32,
        _num_bets: i32,
        spr_after_call: f64,
        actions: &mut Vec<Action>,
    ) {
        let compute_geometric = |n: i32, max_ratio: f64| -> i32 {
            let ratio =
                ((2.0 * spr_after_call + 1.0).powf(1.0 / n as f64) - 1.0) / 2.0;
            (pot as f64 * ratio.min(max_ratio)).round() as i32
        };

        match bet_size {
            BetSize::PotRelative(ratio) => {
                let amount = (pot as f64 * ratio).round() as i32;
                actions.push(Action::Bet(amount));
            }
            BetSize::PrevBetRelative(_) => {}
            BetSize::Additive(adder, _) => {
                actions.push(Action::Bet(*adder));
            }
            BetSize::Geometric(n, max_ratio) => {
                let n = if *n == 0 { num_remaining_streets } else { *n };
                let amount = compute_geometric(n, *max_ratio);
                actions.push(Action::Bet(amount));
            }
            BetSize::AllIn => {
                actions.push(Action::AllIn(actions.iter().map(|a| match a {
                    Action::Bet(v) | Action::Raise(v) | Action::AllIn(v) => *v,
                    _ => 0,
                }).max().unwrap_or(0)));
            }
        }
    }

    fn add_raise_size_action(
        &self,
        bet_size: &BetSize,
        pot: i32,
        prev_amount: i32,
        _max_amount: i32,
        _min_amount: i32,
        num_remaining_streets: i32,
        num_bets: i32,
        spr_after_call: f64,
        actions: &mut Vec<Action>,
    ) {
        let compute_geometric = |n: i32, max_ratio: f64| -> i32 {
            let ratio =
                ((2.0 * spr_after_call + 1.0).powf(1.0 / n as f64) - 1.0) / 2.0;
            (pot as f64 * ratio.min(max_ratio)).round() as i32
        };

        match bet_size {
            BetSize::PotRelative(ratio) => {
                let amount = prev_amount + (pot as f64 * ratio).round() as i32;
                actions.push(Action::Raise(amount));
            }
            BetSize::PrevBetRelative(ratio) => {
                let amount = (prev_amount as f64 * ratio).round() as i32;
                actions.push(Action::Raise(amount));
            }
            BetSize::Additive(adder, raise_cap) => {
                if *raise_cap == 0 || num_bets <= *raise_cap {
                    actions.push(Action::Raise(prev_amount + adder));
                }
            }
            BetSize::Geometric(n, max_ratio) => {
                let n = if *n == 0 {
                    (num_remaining_streets - num_bets + 1).max(1)
                } else {
                    (*n - num_bets + 1).max(1)
                };
                let amount = compute_geometric(n, *max_ratio);
                actions.push(Action::Raise(prev_amount + amount));
            }
            BetSize::AllIn => {}
        }
    }

    fn clamp_and_force_allin(
        &self,
        actions: &mut Vec<Action>,
        pot: i32,
        prev_amount: i32,
        max_amount: i32,
        min_amount: i32,
    ) {
        for action in actions.iter_mut() {
            match *action {
                Action::Bet(amount) => {
                    let clamped = amount.clamp(min_amount, max_amount);
                    let new_amount_diff = clamped - prev_amount;
                    let new_pot = pot + 2 * new_amount_diff;
                    let threshold =
                        (new_pot as f64 * self.config.force_allin_threshold).round() as i32;
                    if max_amount <= clamped + threshold {
                        *action = Action::AllIn(max_amount);
                    } else if clamped != amount {
                        *action = Action::Bet(clamped);
                    }
                }
                Action::Raise(amount) => {
                    let clamped = amount.clamp(min_amount, max_amount);
                    let new_amount_diff = clamped - prev_amount;
                    let new_pot = pot + 2 * new_amount_diff;
                    let threshold =
                        (new_pot as f64 * self.config.force_allin_threshold).round() as i32;
                    if max_amount <= clamped + threshold {
                        *action = Action::AllIn(max_amount);
                    } else if clamped != amount {
                        *action = Action::Raise(clamped);
                    }
                }
                _ => {}
            }
        }
    }

    fn make_child_node(&mut self, parent_idx: usize, action: &Action, info: &BuildInfo) -> usize {
        let parent = &self.tree.nodes[parent_idx];
        let player = parent.player_id;

        let child_type = match action {
            Action::Fold => NODE_TYPE_TERMINAL,
            _ => {
                if self.only_one_active(info) {
                    NODE_TYPE_TERMINAL
                } else {
                    NODE_TYPE_PLAYER
                }
            }
        };

        let next_player = if child_type == NODE_TYPE_TERMINAL {
            0
        } else {
            self.determine_next_player(player as usize, action, info)
        };

        let child = FlatNode {
            node_type: child_type,
            player_id: next_player as u8,
            board_state: parent.board_state,
            num_children: 0,
            children_start: 0,
            amount: parent.amount,
            action_label: action_to_label(action),
        };

        let child_idx = self.tree.alloc_node(child);
        for p in 0..self.config.num_players as usize {
            let contrib = self.tree.get_contribution(parent_idx, p as u8);
            self.tree.set_contribution(child_idx, p as u8, contrib);
        }

        match action {
            Action::Bet(a) | Action::Raise(a) | Action::AllIn(a) => {
                let prev_contrib = self.tree.get_contribution(parent_idx, player);
                self.tree
                    .set_contribution(child_idx, player, prev_contrib + (a - prev_contrib));
            }
            Action::Call => {
                let max_other = (0..self.config.num_players as usize)
                    .filter(|&p| p != player as usize && info.active[p])
                    .map(|p| info.stacks[p])
                    .max()
                    .unwrap_or(0);
                let prev_contrib = self.tree.get_contribution(parent_idx, player);
                let call_amount = max_other - info.stacks[player as usize];
                self.tree
                    .set_contribution(child_idx, player, prev_contrib + call_amount.max(0));
            }
            _ => {}
        }

        self.tree.set_folded_mask(child_idx, info.folded_mask());

        child_idx
    }

    fn determine_next_player(
        &self,
        current: usize,
        action: &Action,
        info: &BuildInfo,
    ) -> usize {
        if matches!(action, Action::Fold) {
            return self
                .next_active_player(current, &info.active)
                .unwrap_or(0);
        }

        if matches!(action, Action::Check) {
            if self.is_round_complete(info) {
                return self.first_postflop_player(&info.active) as usize;
            }
            return self
                .next_active_player(current, &info.active)
                .unwrap_or(0);
        }

        self.next_active_player(current, &info.active)
            .unwrap_or(0)
    }
}

fn action_to_label(action: &Action) -> u8 {
    match action {
        Action::Fold => 0,
        Action::Check => 1,
        Action::Call => 2,
        Action::Bet(_) => 3,
        Action::Raise(_) => 4,
        Action::AllIn(_) => 5,
        Action::Chance(_) => 6,
        _ => 0,
    }
}

fn merge_bet_actions(actions: Vec<Action>, pot: i32, offset: i32, param: f64) -> Vec<Action> {
    if param <= 0.0 {
        return actions;
    }
    const EPS: f64 = 1e-12;

    let get_amount = |action: Action| match action {
        Action::Bet(amount) | Action::Raise(amount) | Action::AllIn(amount) => amount,
        _ => -1,
    };

    let mut cur_amount = i32::MAX;
    let mut ret = Vec::new();

    for &action in actions.iter().rev() {
        let amount = get_amount(action);
        if amount > 0 {
            let ratio = (amount - offset) as f64 / pot as f64;
            let cur_ratio = (cur_amount - offset) as f64 / pot as f64;
            let threshold_ratio = (cur_ratio - param) / (1.0 + param);
            if ratio < threshold_ratio * (1.0 - EPS) {
                ret.push(action);
                cur_amount = amount;
            }
        } else {
            ret.push(action);
        }
    }

    ret.reverse();
    ret
}
