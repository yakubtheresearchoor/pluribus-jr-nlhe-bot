//! Host side of the bucketed Design-1-collapsed terminal offload
//! (G2 step 3). One `BucketedTerminalGpu` per (flop, bucketing);
//! installs as `BucketedFlopCfr`'s terminal offload hook. The CPU
//! keeps the (measured-nearly-free, bucket-granular) walk; the GPU
//! takes filter→reduce→enumerate→expand for every terminal of a zone
//! walk in ONE dispatch (one sync per zone walk — Fix-B at the
//! offload boundary; Fix-A tg_size=32; outcome batching across flops
//! is G4's multi-flop concurrency axis).
//!
//! Dispatch configs: `stripes = 1` is the unstriped CPU-order
//! reference (the G3 bit-exact middle link); `stripes > 1` is
//! production (f64-reference + trajectory standard).

use crate::solver::bucketed_flop_cfr::{BucketedFlopCfr, FlopBucketing, TerminalOffloadHook};
use crate::solver::bucketed_showdown::MAX_BUCKETS_GPU;
use crate::solver::flop_start_game::FlopChanceTable;
use crate::solver::flop_start_vector_cfr::Zone;
use crate::tree::flat::FlatTree;
use metal::{Buffer, CommandQueue, ComputePipelineState, MTLResourceOptions, MTLSize};

use super::context::{MetalContext, MetalResult};

/// Field-for-field mirror of the Metal-side `BucketedTermParams`.
/// SINGLE Rust definition; any layout drift produces loudly-garbage
/// values caught by the G3 functional gates (and the struct is ≤ the
/// 4 KB set_bytes limit).
#[repr(C)]
#[derive(Clone, Copy)]
struct BucketedTermParams {
    num_terminals: u32,
    np: u32,
    nh: u32,
    nb: u32,
    num_opp: u32,
    stripes: u32,
    traverser: u32,
    board0: u32,
    board1: u32,
    table_off: u32,
    map_off: u32,
    nc: f32,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    _pad: u32,
}

struct ZoneTerminals {
    node_ids: Buffer,
    contribs: Buffer,
    fold_masks: Buffer,
    count: u32,
}

pub struct BucketedTerminalGpu {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    fractions: Buffer,
    maps: Buffer,
    hand_cards: Buffer,
    reach_buf: Buffer,
    cfv_buf: Buffer,
    flop_terms: ZoneTerminals,
    turn_terms: ZoneTerminals,
    river_terms: ZoneTerminals,
    /// Terminal node ids per zone (host copy, for cfv readback).
    flop_term_ids: Vec<u32>,
    turn_term_ids: Vec<u32>,
    river_term_ids: Vec<u32>,
    nh: usize,
    np: usize,
    nb_flop: usize,
    nb_turn: usize,
    nb_river: usize,
    /// Per-runout float offsets into `fractions` (runout-indexed).
    table_offs: Vec<u32>,
    n_turn: usize,
    max_river: usize,
    remaining_deck: Vec<u8>,
    river_decks: Vec<Vec<u8>>,
    nc: f32,
    starting_pot: i32,
    rake_rate: f32,
    rake_cap: f32,
    stripes: u32,
}

fn buf_from<T: Copy>(ctx: &MetalContext, data: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(4);
    let buf = ctx
        .device()
        .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}

impl BucketedTerminalGpu {
    /// Runout index in the concatenated fraction/map buffers:
    /// flop = 0, turn ti = 1 + ti, river (ti, ri) = 1 + n_turn +
    /// ti·max_river + ri. (The same outcome flattening ZoneDims uses.)
    fn runout_index(&self, zone: Zone, tc: Option<usize>, rc: Option<usize>) -> usize {
        match zone {
            Zone::Flop => 0,
            Zone::Turn => 1 + tc.unwrap(),
            Zone::River => 1 + self.n_turn + tc.unwrap() * self.max_river + rc.unwrap(),
            Zone::Preflop => unreachable!(),
        }
    }

    pub fn new(
        ctx: &MetalContext,
        tree: &FlatTree,
        table: &FlopChanceTable,
        bk: &FlopBucketing,
        solver: &BucketedFlopCfr,
        stripes: u32,
    ) -> MetalResult<Self> {
        assert!(tree.num_players >= 4, "bucketed terminal: np ≥ 4 only");
        for nb in [bk.nb_flop, bk.nb_turn, bk.nb_river] {
            assert!(nb <= MAX_BUCKETS_GPU, "B > MAX_BUCKETS_GPU stays CPU-only");
        }
        let nb_max = bk.nb_flop.max(bk.nb_turn).max(bk.nb_river);
        assert!(
            stripes >= 1 && (32 / nb_max as u32) >= stripes,
            "stripes × max(nb) ≤ 32"
        );

        let nh = table.num_valid;
        let np = tree.num_players as usize;
        let nn = tree.num_nodes();
        let n_turn = table.remaining_deck.len();
        let max_river = table
            .remaining_deck
            .iter()
            .map(|&tc| table.river_decks[tc as usize].len())
            .max()
            .unwrap_or(0);

        let pipeline = ctx.create_pipeline("bucketed_terminal_collapsed")?;

        // ── Concatenated fraction tables + maps, runout-indexed.
        //    Per-runout table size is 4·nb_zone² (divergent per-street
        //    B is the layout's reason for existing); offsets accumulate. ──
        let n_runouts = 1 + n_turn + n_turn * max_river;
        let mut fractions: Vec<f32> = Vec::new();
        let mut table_offs = vec![0u32; n_runouts];
        let mut maps = vec![u16::MAX; n_runouts * nh];
        {
            let mut put = |ridx: usize,
                           nb: usize,
                           t: &crate::solver::bucketed_showdown::BucketedRunoutTables,
                           m: &[u16],
                           fractions: &mut Vec<f32>,
                           table_offs: &mut Vec<u32>,
                           maps: &mut Vec<u16>| {
                assert_eq!(t.nb, nb);
                table_offs[ridx] = fractions.len() as u32;
                fractions.extend_from_slice(&t.f_w);
                fractions.extend_from_slice(&t.f_t);
                fractions.extend_from_slice(&t.f_l);
                fractions.extend_from_slice(&t.f_n);
                maps[ridx * nh..(ridx + 1) * nh].copy_from_slice(m);
            };
            put(0, bk.nb_flop, &bk.flop_tables, &bk.flop_map, &mut fractions, &mut table_offs, &mut maps);
            for ti in 0..n_turn {
                put(1 + ti, bk.nb_turn, &bk.turn_tables[ti], &bk.turn_map[ti], &mut fractions, &mut table_offs, &mut maps);
                for ri in 0..bk.river_map[ti].len() {
                    put(
                        1 + n_turn + ti * max_river + ri,
                        bk.nb_river,
                        &bk.river_tables[ti][ri],
                        &bk.river_map[ti][ri],
                        &mut fractions,
                        &mut table_offs,
                        &mut maps,
                    );
                }
            }
        }

        // ── Per-zone terminal metadata (zone via the gated layout) ──
        let layout = solver.gpu_layout(bk);
        let mut zone_terms =
            |zone: Zone| -> (Vec<u32>, Vec<i32>, Vec<u32>) {
                let mut ids = Vec::new();
                let mut contribs = Vec::new();
                let mut folds = Vec::new();
                for idx in 0..nn {
                    if layout.zone_of(idx) != zone || !tree.nodes[idx].is_terminal() {
                        continue;
                    }
                    ids.push(idx as u32);
                    for p in 0..np {
                        contribs.push(tree.get_contribution(idx, p as u8));
                    }
                    folds.push(tree.get_folded_mask(idx) as u32);
                }
                (ids, contribs, folds)
            };
        let (f_ids, f_c, f_m) = zone_terms(Zone::Flop);
        let (t_ids, t_c, t_m) = zone_terms(Zone::Turn);
        let (r_ids, r_c, r_m) = zone_terms(Zone::River);
        let mk = |ids: &Vec<u32>, c: &Vec<i32>, m: &Vec<u32>| ZoneTerminals {
            node_ids: buf_from(ctx, ids),
            contribs: buf_from(ctx, c),
            fold_masks: buf_from(ctx, m),
            count: ids.len() as u32,
        };

        let reach_buf = ctx.device().new_buffer(
            (nn * np * nh * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cfv_buf = ctx
            .device()
            .new_buffer((nn * nh * 4) as u64, MTLResourceOptions::StorageModeShared);

        Ok(Self {
            queue: ctx.queue().to_owned(),
            pipeline,
            fractions: buf_from(ctx, &fractions),
            maps: buf_from(ctx, &maps),
            hand_cards: buf_from(ctx, &table.hand_cards),
            reach_buf,
            cfv_buf,
            flop_terms: mk(&f_ids, &f_c, &f_m),
            turn_terms: mk(&t_ids, &t_c, &t_m),
            river_terms: mk(&r_ids, &r_c, &r_m),
            flop_term_ids: f_ids,
            turn_term_ids: t_ids,
            river_term_ids: r_ids,
            nh,
            np,
            nb_flop: bk.nb_flop,
            nb_turn: bk.nb_turn,
            nb_river: bk.nb_river,
            table_offs,
            n_turn,
            max_river,
            remaining_deck: table.remaining_deck.clone(),
            river_decks: table.river_decks.clone(),
            nc: table.num_combinations as f32,
            starting_pot: tree.starting_pot,
            rake_rate: tree.rake_rate as f32,
            rake_cap: tree.rake_cap as f32,
            stripes,
        })
    }

    /// Fill every terminal cfv slice for one zone walk. Returns true
    /// (handled) — the walk skips its terminal branch.
    pub fn fill_terminals(
        &mut self,
        zone: Zone,
        tc: Option<usize>,
        rc: Option<usize>,
        traverser: u8,
        reach: &[f32],
        cfv: &mut [f32],
    ) -> bool {
        let (terms, ids): (&ZoneTerminals, &[u32]) = match zone {
            Zone::Flop => (&self.flop_terms, &self.flop_term_ids),
            Zone::Turn => (&self.turn_terms, &self.turn_term_ids),
            Zone::River => (&self.river_terms, &self.river_term_ids),
            Zone::Preflop => unreachable!(),
        };
        if terms.count == 0 {
            return true;
        }
        // Board cards for the filter (verbatim CPU board_cards logic).
        let (board0, board1) = match zone {
            Zone::Flop => (0xFFFFu32, 0xFFFFu32),
            Zone::Turn => (self.remaining_deck[tc.unwrap()] as u32, 0xFFFF),
            Zone::River => {
                let tcc = self.remaining_deck[tc.unwrap()];
                let rcc = self.river_decks[tcc as usize][rc.unwrap()];
                (tcc as u32, rcc as u32)
            }
            Zone::Preflop => unreachable!(),
        };
        let ridx = self.runout_index(zone, tc, rc);
        let nb_zone = match zone {
            Zone::Flop => self.nb_flop,
            Zone::Turn => self.nb_turn,
            Zone::River => self.nb_river,
            Zone::Preflop => unreachable!(),
        };
        // Per-zone stripes: as wide as the lane budget allows, capped
        // at the requested production setting (1 = unstriped reference
        // everywhere — the gate config).
        let stripes_zone = self.stripes.min(32 / nb_zone as u32).max(1);
        let params = BucketedTermParams {
            num_terminals: terms.count,
            np: self.np as u32,
            nh: self.nh as u32,
            nb: nb_zone as u32,
            num_opp: (self.np - 1) as u32,
            stripes: stripes_zone,
            traverser: traverser as u32,
            board0,
            board1,
            table_off: self.table_offs[ridx],
            map_off: (ridx * self.nh) as u32,
            nc: self.nc,
            starting_pot: self.starting_pot,
            rake_rate: self.rake_rate,
            rake_cap: self.rake_cap,
            _pad: 0,
        };

        // Stage reach (unified memory; v1 copies, the no-copy variant
        // writes the walk's reach directly into this buffer later).
        unsafe {
            std::ptr::copy_nonoverlapping(
                reach.as_ptr(),
                self.reach_buf.contents() as *mut f32,
                reach.len(),
            );
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<BucketedTermParams>() as u64,
            &params as *const _ as *const std::ffi::c_void,
        );
        enc.set_buffer(1, Some(&terms.node_ids), 0);
        enc.set_buffer(2, Some(&terms.contribs), 0);
        enc.set_buffer(3, Some(&terms.fold_masks), 0);
        enc.set_buffer(4, Some(&self.reach_buf), 0);
        enc.set_buffer(5, Some(&self.fractions), 0);
        enc.set_buffer(6, Some(&self.maps), 0);
        enc.set_buffer(7, Some(&self.hand_cards), 0);
        enc.set_buffer(8, Some(&self.cfv_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize { width: terms.count as u64, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Read back only the terminal slices.
        let out = unsafe {
            std::slice::from_raw_parts(self.cfv_buf.contents() as *const f32, cfv.len())
        };
        for &nid in ids {
            let base = nid as usize * self.nh;
            cfv[base..base + self.nh].copy_from_slice(&out[base..base + self.nh]);
        }
        true
    }

    /// Consume self into a `BucketedFlopCfr` terminal offload hook.
    pub fn into_hook(mut self) -> TerminalOffloadHook {
        Box::new(move |zone, tc, rc, traverser, reach, cfv| {
            self.fill_terminals(zone, tc, rc, traverser, reach, cfv)
        })
    }
}
