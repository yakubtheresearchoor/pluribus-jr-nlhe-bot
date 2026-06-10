//! Per-zone buffer dimensions — the SINGLE SOURCE for stride/offset math
//! on both the CPU and GPU sides (Phase B3, step "dims threading").
//!
//! Until bucketing, one `nh` drove every zone's strides on both sides
//! (CPU: flop_start_vector_cfr.rs strides; GPU: flop_solver.rs strides +
//! the Phase-2 offset helper). Per-canonical-flop bucketing makes the
//! effective hand count DIVERGE PER STREET for the first time
//! (nh_flop = B_f, nh_turn = B_t, nh_river = B_r), which is the same
//! silent-corruption class as the per-stage MAX_NA work: a stride
//! mismatch between Rust and Metal, or between two Rust call sites,
//! produces plausible-but-wrong numbers rather than crashes.
//!
//! Discipline (mirrors what worked in Phase 2): this struct is the only
//! place the formulas live; CPU solver, GPU solver, and the kernel dims
//! struct all consume it; the unit tests below pin hand-computed offsets
//! for DIVERGENT per-zone nh, so the gates that run later (identity at
//! B = nh is uniform-width and structurally cannot exercise divergence)
//! sit on an already-validated stride layer. An identity-gate failure
//! can then only mean terminal/walk logic, not layout.
//!
//! Layout (identical to the pre-bucketing GPU concatenated layout, with
//! nh generalized per zone):
//!
//!   Flop:  base 0,                    slice = flop_infosets  × MAX_NA × nh_flop
//!   Turn:  base flop_stride,          per-ti slice = turn_infosets  × MAX_NA × nh_turn
//!   River: base flop_stride+turn_tot, per-(ti,ri) slice = river_infosets × MAX_NA × nh_river
//!
//! Inner indexing within a zone slice stays `local_infoset × MAX_NA ×
//! nh_zone + a × nh_zone + h`, with nh_zone from `nh_for_zone`.

/// Zone + outcome reference, metal-agnostic (the GPU's `BufferZone`
/// converts to/from this 1:1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoneRef {
    Flop,
    Turn { ti: usize },
    River { outcome_idx: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZoneDims {
    pub max_na: usize,
    pub nh_flop: usize,
    pub nh_turn: usize,
    pub nh_river: usize,
    pub flop_infosets: usize,
    pub turn_infosets: usize,
    pub river_infosets: usize,
    pub n_turn: usize,
    pub max_river: usize,
}

impl ZoneDims {
    /// Pre-bucketing world: one nh everywhere. Existing solvers build
    /// through this constructor so their strides are now sourced here.
    #[allow(clippy::too_many_arguments)]
    pub fn uniform(
        max_na: usize,
        nh: usize,
        flop_infosets: usize,
        turn_infosets: usize,
        river_infosets: usize,
        n_turn: usize,
        max_river: usize,
    ) -> Self {
        Self {
            max_na,
            nh_flop: nh,
            nh_turn: nh,
            nh_river: nh,
            flop_infosets,
            turn_infosets,
            river_infosets,
            n_turn,
            max_river,
        }
    }

    pub fn nh_for_zone(&self, zone: ZoneRef) -> usize {
        match zone {
            ZoneRef::Flop => self.nh_flop,
            ZoneRef::Turn { .. } => self.nh_turn,
            ZoneRef::River { .. } => self.nh_river,
        }
    }

    pub fn flop_stride(&self) -> usize {
        self.flop_infosets * self.max_na * self.nh_flop
    }
    pub fn turn_stride(&self) -> usize {
        self.turn_infosets * self.max_na * self.nh_turn
    }
    pub fn river_stride(&self) -> usize {
        self.river_infosets * self.max_na * self.nh_river
    }
    pub fn turn_total(&self) -> usize {
        self.n_turn * self.turn_stride()
    }
    pub fn river_total(&self) -> usize {
        self.n_turn * self.max_river * self.river_stride()
    }
    pub fn total_floats(&self) -> usize {
        self.flop_stride() + self.turn_total() + self.river_total()
    }

    pub fn flop_offset(&self) -> usize { 0 }
    pub fn turn_offset(&self) -> usize { self.flop_stride() }
    pub fn river_offset(&self) -> usize { self.flop_stride() + self.turn_total() }

    /// Float offset of a zone outcome's slice in the concatenated buffer
    /// — the generalization of the Phase-2 offset helper.
    pub fn zone_float_offset(&self, zone: ZoneRef) -> usize {
        match zone {
            ZoneRef::Flop => self.flop_offset(),
            ZoneRef::Turn { ti } => {
                debug_assert!(ti < self.n_turn);
                self.turn_offset() + ti * self.turn_stride()
            }
            ZoneRef::River { outcome_idx } => {
                debug_assert!(outcome_idx < self.n_turn * self.max_river);
                self.river_offset() + outcome_idx * self.river_stride()
            }
        }
    }

    pub fn zone_byte_offset(&self, zone: ZoneRef) -> u64 {
        (self.zone_float_offset(zone) * std::mem::size_of::<f32>()) as u64
    }

    /// Offset WITHIN a zone slice for (local_infoset, action, hand/bucket).
    pub fn intra_zone_offset(
        &self,
        zone: ZoneRef,
        local_infoset: usize,
        action: usize,
        h: usize,
    ) -> usize {
        let nh = self.nh_for_zone(zone);
        debug_assert!(action < self.max_na);
        debug_assert!(h < nh);
        local_infoset * self.max_na * nh + action * nh + h
    }

    /// Kernel-facing dims (repr(C), u32) — passed to Metal from this one
    /// source so Rust-side and shader-side index math cannot diverge.
    pub fn to_kernel_dims(&self) -> KernelZoneDims {
        KernelZoneDims {
            max_na: self.max_na as u32,
            nh_flop: self.nh_flop as u32,
            nh_turn: self.nh_turn as u32,
            nh_river: self.nh_river as u32,
            flop_stride: self.flop_stride() as u32,
            turn_stride: self.turn_stride() as u32,
            river_stride: self.river_stride() as u32,
            turn_offset: self.turn_offset() as u32,
            river_offset: self.river_offset() as u32,
        }
    }
}

/// Mirrors a Metal-side struct field-for-field. Consumed by bucketed
/// kernels when the GPU port lands; produced ONLY by
/// `ZoneDims::to_kernel_dims`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KernelZoneDims {
    pub max_na: u32,
    pub nh_flop: u32,
    pub nh_turn: u32,
    pub nh_river: u32,
    pub flop_stride: u32,
    pub turn_stride: u32,
    pub river_stride: u32,
    pub turn_offset: u32,
    pub river_offset: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed offsets for DIVERGENT per-zone nh — the case the
    /// identity gate (uniform width at B = nh) structurally cannot
    /// exercise. Every number below is computed by hand in the comments.
    #[test]
    fn divergent_nh_offsets_hand_computed() {
        // max_na=4; nh: flop=3, turn=2, river=5
        // infosets: flop=2, turn=3, river=1; n_turn=2, max_river=2
        let d = ZoneDims {
            max_na: 4,
            nh_flop: 3,
            nh_turn: 2,
            nh_river: 5,
            flop_infosets: 2,
            turn_infosets: 3,
            river_infosets: 1,
            n_turn: 2,
            max_river: 2,
        };
        // flop_stride = 2 × 4 × 3 = 24
        assert_eq!(d.flop_stride(), 24);
        // turn_stride = 3 × 4 × 2 = 24  (different nh, same value — a
        // deliberate trap: code that confuses the two strides will pass
        // a stride-equality check here but fail the offset checks below)
        assert_eq!(d.turn_stride(), 24);
        // river_stride = 1 × 4 × 5 = 20
        assert_eq!(d.river_stride(), 20);
        // turn_total = 2 × 24 = 48; river_total = 2 × 2 × 20 = 80
        assert_eq!(d.turn_total(), 48);
        assert_eq!(d.river_total(), 80);
        // total = 24 + 48 + 80 = 152
        assert_eq!(d.total_floats(), 152);

        // Zone bases: flop 0; turn 24; river 24 + 48 = 72.
        assert_eq!(d.zone_float_offset(ZoneRef::Flop), 0);
        assert_eq!(d.zone_float_offset(ZoneRef::Turn { ti: 0 }), 24);
        assert_eq!(d.zone_float_offset(ZoneRef::Turn { ti: 1 }), 48);
        assert_eq!(d.zone_float_offset(ZoneRef::River { outcome_idx: 0 }), 72);
        assert_eq!(d.zone_float_offset(ZoneRef::River { outcome_idx: 1 }), 92);
        assert_eq!(d.zone_float_offset(ZoneRef::River { outcome_idx: 3 }), 132);
        // Last river slice must end exactly at total_floats:
        // 132 + 20 = 152 ✓
        assert_eq!(
            d.zone_float_offset(ZoneRef::River { outcome_idx: 3 }) + d.river_stride(),
            d.total_floats()
        );

        // Byte offsets are ×4.
        assert_eq!(d.zone_byte_offset(ZoneRef::Turn { ti: 1 }), 48 * 4);

        // Intra-zone: flop infoset 1, action 2, h 1 →
        //   1 × 4 × 3 + 2 × 3 + 1 = 12 + 6 + 1 = 19
        assert_eq!(d.intra_zone_offset(ZoneRef::Flop, 1, 2, 1), 19);
        // Turn (nh=2): infoset 2, action 3, h 1 →
        //   2 × 4 × 2 + 3 × 2 + 1 = 16 + 6 + 1 = 23
        assert_eq!(d.intra_zone_offset(ZoneRef::Turn { ti: 0 }, 2, 3, 1), 23);
        // River (nh=5): infoset 0, action 1, h 4 → 0 + 5 + 4 = 9
        assert_eq!(d.intra_zone_offset(ZoneRef::River { outcome_idx: 0 }, 0, 1, 4), 9);

        // nh_for_zone dispatch.
        assert_eq!(d.nh_for_zone(ZoneRef::Flop), 3);
        assert_eq!(d.nh_for_zone(ZoneRef::Turn { ti: 1 }), 2);
        assert_eq!(d.nh_for_zone(ZoneRef::River { outcome_idx: 2 }), 5);

        // Kernel dims mirror the same numbers.
        let k = d.to_kernel_dims();
        assert_eq!(k.flop_stride, 24);
        assert_eq!(k.turn_stride, 24);
        assert_eq!(k.river_stride, 20);
        assert_eq!(k.turn_offset, 24);
        assert_eq!(k.river_offset, 72);
        assert_eq!((k.nh_flop, k.nh_turn, k.nh_river), (3, 2, 5));
    }

    /// Uniform constructor must reproduce the pre-bucketing formulas
    /// exactly (stride = infosets × MAX_NA × nh for every zone, offsets
    /// concatenated flop | turn | river).
    #[test]
    fn uniform_matches_prebucketing_formulas() {
        let (max_na, nh) = (4usize, 7usize);
        let (fi, ti_, ri) = (11usize, 13usize, 5usize);
        let (n_turn, max_river) = (3usize, 2usize);
        let d = ZoneDims::uniform(max_na, nh, fi, ti_, ri, n_turn, max_river);
        assert_eq!(d.flop_stride(), fi * max_na * nh);
        assert_eq!(d.turn_stride(), ti_ * max_na * nh);
        assert_eq!(d.river_stride(), ri * max_na * nh);
        assert_eq!(d.turn_offset(), d.flop_stride());
        assert_eq!(d.river_offset(), d.flop_stride() + n_turn * d.turn_stride());
        assert_eq!(
            d.total_floats(),
            d.flop_stride() + n_turn * d.turn_stride()
                + n_turn * max_river * d.river_stride()
        );
        for ti in 0..n_turn {
            assert_eq!(
                d.zone_float_offset(ZoneRef::Turn { ti }),
                d.turn_offset() + ti * d.turn_stride()
            );
        }
        for o in 0..n_turn * max_river {
            assert_eq!(
                d.zone_float_offset(ZoneRef::River { outcome_idx: o }),
                d.river_offset() + o * d.river_stride()
            );
        }
    }
}
