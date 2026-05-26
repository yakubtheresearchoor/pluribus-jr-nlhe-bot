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
    fn chance_probability(&self, outcome: usize, hand: usize) -> f32;
    fn num_chance_outcomes(&self) -> usize {
        0
    }
    fn set_chance_outcome(&self, _outcome: usize) {}
    fn clear_chance_outcome(&self) {}
}
