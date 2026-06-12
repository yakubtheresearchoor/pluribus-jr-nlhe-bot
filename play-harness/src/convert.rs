//! The card-encoding boundary between solver-core and clean-rules.
//! Both happen to use `4·rank + suit` (rank 0..=12 = 2..=A; suit
//! c,d,h,s = 0..=3), so conversion is the identity — but that is a
//! COINCIDENCE OF FORMATS, not shared code, and it is pinned by the
//! test below: if either side ever changes encoding, the pin fires
//! instead of every showdown silently mis-scoring.

/// Solver card → clean-rules card.
pub fn to_rules(c: u8) -> u8 {
    c
}
/// Clean-rules card → solver card.
pub fn to_solver(c: u8) -> u8 {
    c
}

/// Chip units → big blinds, THE reporting boundary (2026-06-12 bb-units
/// pin). All harness-internal money stays in integer chip units; every
/// human-facing number goes through here. The conversion constant lives
/// in solver-core (`UNITS_PER_BB`) — never redefine it locally.
pub fn units_to_bb(units: i64) -> f64 {
    units as f64 / solver_core::tree::action::UNITS_PER_BB as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the identity across the full deck via both crates' own
    /// semantics: solver's string parser vs clean-rules' rank/suit
    /// accessors.
    #[test]
    fn encoding_pin_full_deck() {
        let ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"];
        let suits = ["c", "d", "h", "s"];
        for (ri, r) in ranks.iter().enumerate() {
            for (si, s) in suits.iter().enumerate() {
                let solver_card =
                    solver_core::card::card_from_str(&format!("{r}{s}")).unwrap();
                let rules_card = to_rules(solver_card);
                assert_eq!(clean_rules::eval::rank_of(rules_card), ri as u32);
                assert_eq!(clean_rules::eval::suit_of(rules_card), si as u32);
                assert_eq!(to_solver(rules_card), solver_card);
            }
        }
    }
}
