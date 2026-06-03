// Suit-isomorphism reduction measurement on real flop textures.
//
// The "assumed 4x" projection is the average of within-flop hand-level
// isomorphism across all flop textures. The actual per-flop reduction
// varies sharply by texture: rainbow ~1x (no benefit), two-tone ~2x,
// monotone up to ~6x. This test measures the actual class-count
// reduction for our test flop and representative alternates, so the
// baseline can be planned against the right number.
//
// Method: for a given flop, enumerate all 1326 starting hands. Two
// hands are isomorphic IF there exists a suit permutation π such that
//   1) π applied to hand maps to the other hand, AND
//   2) π fixes the flop board (maps board cards to themselves).
// Group hands into orbits under this equivalence relation. Count
// orbits = number of distinct CFVs to compute = reduction factor.

#![cfg(feature = "metal")]

use solver_core::card::{card_from_str, index_to_card_pair, Card, NUM_POSSIBLE_HANDS};

fn suit_of(c: Card) -> u8 { c & 3 }
fn rank_of(c: Card) -> u8 { c >> 2 }

fn apply_perm(c: Card, perm: &[u8; 4]) -> Card {
    (rank_of(c) << 2) | perm[suit_of(c) as usize]
}

fn fixes_flop(perm: &[u8; 4], flop: &[Card; 3]) -> bool {
    // Permutation fixes the flop iff each board card maps to a board card
    // (i.e., the set of board cards is invariant under the suit permutation).
    let flop_set: std::collections::HashSet<Card> = flop.iter().copied().collect();
    flop.iter().all(|&c| flop_set.contains(&apply_perm(c, perm)))
}

fn all_suit_perms() -> Vec<[u8; 4]> {
    let mut perms = vec![];
    for a in 0..4 {
        for b in 0..4 {
            if b == a { continue; }
            for c in 0..4 {
                if c == a || c == b { continue; }
                let d = (0..4).find(|&x| x != a && x != b && x != c).unwrap();
                perms.push([a as u8, b as u8, c as u8, d as u8]);
            }
        }
    }
    perms
}

fn flop_stabilizer(flop: &[Card; 3]) -> Vec<[u8; 4]> {
    all_suit_perms().into_iter().filter(|p| fixes_flop(p, flop)).collect()
}

fn hand_canonical(hand_idx: usize, stab: &[[u8; 4]]) -> usize {
    let (c1, c2) = index_to_card_pair(hand_idx);
    let mut best = hand_idx;
    for perm in stab {
        let c1p = apply_perm(c1, perm);
        let c2p = apply_perm(c2, perm);
        // Hand idx is (max, min) ordering — find canonical pair index
        let (lo, hi) = if c1p < c2p { (c1p, c2p) } else { (c2p, c1p) };
        // Re-derive canonical idx from (lo, hi). Match `index_to_card_pair`
        // inverse: hand_idx = lo * 52 - lo*(lo+1)/2 + (hi - lo - 1) for hi>lo
        // Simpler: scan NUM_POSSIBLE_HANDS for matching pair
        for h in 0..NUM_POSSIBLE_HANDS {
            let (a, b) = index_to_card_pair(h);
            if (a == lo && b == hi) || (a == hi && b == lo) {
                if h < best { best = h; }
                break;
            }
        }
    }
    best
}

fn measure_for_flop(name: &str, flop_strs: &[&str; 3]) {
    let flop: [Card; 3] = [
        card_from_str(flop_strs[0]).unwrap(),
        card_from_str(flop_strs[1]).unwrap(),
        card_from_str(flop_strs[2]).unwrap(),
    ];
    let flop_mask: u64 = flop.iter().fold(0u64, |m, &c| m | (1u64 << (c as u8)));
    let stab = flop_stabilizer(&flop);

    let mut valid_hands: Vec<usize> = Vec::new();
    for h in 0..NUM_POSSIBLE_HANDS {
        let (c1, c2) = index_to_card_pair(h);
        if flop_mask & (1u64 << c1) != 0 || flop_mask & (1u64 << c2) != 0 { continue; }
        valid_hands.push(h);
    }
    let n_valid = valid_hands.len();

    let mut canonical_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for &h in &valid_hands {
        canonical_set.insert(hand_canonical(h, &stab));
    }
    let n_classes = canonical_set.len();
    let reduction = n_valid as f64 / n_classes as f64;

    // Texture classification
    let suit_counts: Vec<u8> = (0..4).map(|s|
        flop.iter().filter(|&&c| suit_of(c) == s as u8).count() as u8
    ).collect();
    let mut sorted = suit_counts.clone(); sorted.sort();
    let texture = match (sorted[3], sorted[2]) {
        (3, _) => "monotone",
        (2, 1) => "two-tone",
        (1, 1) => "rainbow",
        _ => "unknown",
    };
    let stab_size = stab.len();

    eprintln!("{:>14} | flop {} {} {} | tex={:<8} | stab={} | hands {} → {} classes ({:.2}x reduction)",
        name, flop_strs[0], flop_strs[1], flop_strs[2], texture, stab_size,
        n_valid, n_classes, reduction);
}

#[test]
fn suit_isomorphism_real_textures() {
    eprintln!("\n=== Suit-isomorphism hand-level orbit reduction on real flop textures ===");
    eprintln!("(stab = size of suit-permutation group that fixes the flop; identity = 1, no symmetry)");
    eprintln!();
    eprintln!("           name | flop          | texture  | stab | hands → classes (reduction)");
    eprintln!("{}", "-".repeat(95));

    // Rainbow (3 different suits, ~55% of all flops)
    measure_for_flop("test-board", &["2h", "7d", "Ks"]);
    measure_for_flop("rainbow-low", &["2h", "5d", "8s"]);
    measure_for_flop("rainbow-broadway", &["Th", "Jd", "Qs"]);

    // Two-tone (2+1, ~40%)
    measure_for_flop("two-tone-hd", &["2h", "7h", "Ks"]);
    measure_for_flop("two-tone-paired", &["2h", "2d", "Ks"]);
    measure_for_flop("two-tone-conn", &["8h", "9h", "Td"]);

    // Monotone (3 same suit, ~5%)
    measure_for_flop("monotone", &["2h", "7h", "Kh"]);
    measure_for_flop("monotone-conn", &["8h", "9h", "Th"]);

    eprintln!();
    eprintln!("Expected: rainbow stab=1 → 1x reduction (no benefit)");
    eprintln!("          two-tone stab=2 → ~2x reduction");
    eprintln!("          monotone stab=6 → up to 6x reduction");
    eprintln!();
    eprintln!("Flop-texture distribution (ATC, all 22,100 flops):");
    eprintln!("  rainbow  : ~55%");
    eprintln!("  two-tone : ~40%");
    eprintln!("  monotone : ~5%");
    eprintln!("Weighted average within-flop reduction: ~0.55*1 + 0.40*2 + 0.05*6 = 1.65x");
}
