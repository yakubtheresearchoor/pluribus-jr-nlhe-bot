use crate::tree::flat::FlatTree;

pub trait GameSpec {
    fn num_hands(&self, player: u8) -> usize;
    fn initial_weight(&self, player: u8) -> Vec<f32>;
    fn evaluate_terminal(
        &self,
        traverser: u8,
        node_idx: usize,
        tree: &FlatTree,
        cfreach: &[Vec<f32>],
    ) -> Vec<f32>;
    /// Depth-limit leaf value for real-time search. At a depth-limit boundary
    /// node the searcher does NOT recurse into the (frozen) continuation; it
    /// values the leaf here. Pluribus-faithful default for the flop search:
    /// the bucketed + MC-sampled blueprint continuation (flat in players),
    /// lossless on the current round (per-hand output). The default falls back
    /// to `evaluate_terminal` so games without a depth-limit are unchanged.
    fn evaluate_continuation(
        &self,
        traverser: u8,
        node_idx: usize,
        tree: &FlatTree,
        cfreach: &[Vec<f32>],
    ) -> Vec<f32> {
        self.evaluate_terminal(traverser, node_idx, tree, cfreach)
    }
    fn chance_probability(&self, outcome: usize, hand: usize) -> f32;
    fn num_combinations(&self) -> f64 { 1.0 } // default: no normalization
    fn num_chance_outcomes(&self) -> usize {
        0
    }
    fn set_chance_outcome(&self, _outcome: usize) {}
    fn clear_chance_outcome(&self) {}
}
