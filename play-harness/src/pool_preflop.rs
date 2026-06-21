//! NL10 pool PREFLOP model (production-baseline piece 2). The loose-passive
//! field: VPIP 42.9 / PFR 18.3 / 3bet 8.8 / F3B 23.1 / steal 35.3. The bot's
//! EQR tree is raise-or-fold, but the pool LIMPS and FLATS — so this model emits
//! the full action set {Fold, Limp, Call, Raise}, calibrated by a Chen-formula
//! strength rank of the 169 classes + position. Limps/flats resolve to flops in
//! the orchestrator.

use solver_core::abstraction::preflop_class::{class_combos, PreflopClass, NUM_PREFLOP_CLASSES};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PreAction {
    Fold,
    Limp, // open-limp (no raise yet)
    Call, // flat a raise
    Raise,
}

/// Position index along the fold-chain: 0=UTG 1=HJ 2=CO 3=BTN 4=SB 5=BB.
pub const POS_UTG: usize = 0;
pub const POS_BTN: usize = 3;
pub const POS_SB: usize = 4;
pub const POS_BB: usize = 5;

/// Chen strength score for a class (higher = stronger).
fn chen_score(class: usize) -> f32 {
    let (c1, c2) = class_combos(PreflopClass::new(class as u8))[0];
    let (r1, r2) = (c1 >> 2, c2 >> 2); // 0=2 .. 12=A
    let suited = (c1 & 3) == (c2 & 3);
    let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    let pts = |r: u8| -> f32 {
        match r {
            12 => 10.0, // A
            11 => 8.0,  // K
            10 => 7.0,  // Q
            9 => 6.0,   // J
            8 => 5.0,   // T
            _ => (r as f32 + 2.0) / 2.0,
        }
    };
    if hi == lo {
        return (pts(hi) * 2.0).max(5.0); // pair
    }
    let mut s = pts(hi);
    if suited {
        s += 2.0;
    }
    let gap = hi - lo - 1;
    s -= match gap {
        0 => 0.0,
        1 => 1.0,
        2 => 2.0,
        3 => 4.0,
        _ => 5.0,
    };
    if gap <= 1 && hi < 10 {
        s += 1.0; // connector bonus (both below Q)
    }
    s.ceil()
}

/// Pool preflop model: per-class strength PERCENTILE (combos-weighted, 0=weakest
/// .. 1=strongest), so thresholds map directly to VPIP/PFR-style frequencies.
pub struct PoolPreflop {
    pct: Vec<f32>, // pct[class] = fraction of the 1326-combo deck WEAKER than this class
}

impl PoolPreflop {
    pub fn new() -> Self {
        let mut order: Vec<usize> = (0..NUM_PREFLOP_CLASSES).collect();
        order.sort_by(|&a, &b| chen_score(a).partial_cmp(&chen_score(b)).unwrap());
        // cumulative combo fraction from weakest → strongest
        let total: f32 = (0..NUM_PREFLOP_CLASSES)
            .map(|c| PreflopClass::new(c as u8).num_combos() as f32)
            .sum();
        let mut pct = vec![0.0f32; NUM_PREFLOP_CLASSES];
        let mut cum = 0.0f32;
        for &c in &order {
            let w = PreflopClass::new(c as u8).num_combos() as f32;
            pct[c] = (cum + w * 0.5) / total; // mid-rank percentile
            cum += w;
        }
        PoolPreflop { pct }
    }

    /// Open threshold (raise if pct ≥ this) by position — tighter early, wider on
    /// the steal seats. Combos-weighted these average ≈ PFR 18.3 (UTG ~11 →
    /// BTN/SB ~33, matching steal 35.3).
    fn open_raise_cut(pos: usize) -> f32 {
        match pos {
            POS_UTG => 0.89, // top ~11%
            1 => 0.86,       // HJ ~14%
            2 => 0.82,       // CO ~18%
            POS_BTN => 0.67, // BTN ~33% (steal)
            POS_SB => 0.68,  // SB ~32% (steal)
            _ => 0.82,       // BB rare as opener (limped pots handled elsewhere)
        }
    }

    /// First-in action (no raise yet): raise the top band, LIMP the loose-passive
    /// gap below it (VPIP−PFR ≈ 25%), fold the rest.
    pub fn first_in(&self, pos: usize, class: usize, rng: &mut u64) -> PreAction {
        let p = self.pct[class];
        let raise_cut = Self::open_raise_cut(pos);
        if p >= raise_cut {
            return PreAction::Raise;
        }
        // Limp band: ~25% of hands just below the raise band (the passive gap),
        // a bit wider in late position. Some randomization at the edge.
        let limp_width = 0.22 + 0.05 * (pos as f32 / 5.0);
        let limp_cut = (raise_cut - limp_width).max(0.0);
        if p >= limp_cut {
            // mostly limp, occasionally raise (the loose-aggro tail)
            let r = (super::preflop_player::splitmix64(rng) % 1000) as f32 / 1000.0;
            return if r < 0.07 { PreAction::Raise } else { PreAction::Limp };
        }
        PreAction::Fold
    }

    /// Facing a raise: 3bet the very top (≈8.8%), FLAT a medium band (loose
    /// calling), fold the rest. `n_raises_faced` ≥1.
    pub fn facing_raise(&self, _pos: usize, class: usize, rng: &mut u64) -> PreAction {
        let p = self.pct[class];
        if p >= 0.945 {
            return PreAction::Raise; // 3bet top always
        }
        if p >= 0.89 {
            // next band: mix 3bet / flat
            let r = (super::preflop_player::splitmix64(rng) % 1000) as f32 / 1000.0;
            return if r < 0.5 { PreAction::Raise } else { PreAction::Call };
        }
        // loose flatting band (calling station) — ~20% of hands flat
        if p >= 0.70 {
            return PreAction::Call;
        }
        PreAction::Fold
    }

    pub fn pct(&self, class: usize) -> f32 {
        self.pct[class]
    }
}
