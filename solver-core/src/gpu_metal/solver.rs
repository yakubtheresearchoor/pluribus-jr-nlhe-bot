use crate::gpu_metal::{MetalBuffer, MetalContext};
use crate::tree::flat::{FlatNode, FlatTree, MAX_NA_POSTFLOP};

use metal::{MTLSize, ComputePipelineState};

const UNUSED: u32 = u32::MAX;

struct DcfrParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DcfrParams {
    fn new(current_iteration: u32) -> Self {
        let nearest_lower_power_of_4 = match current_iteration {
            0 => 0u32,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };
        let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
        let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;
        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);
        Self {
            alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pow_gamma as f32,
        }
    }
}

/// Depth-limited bucketed continuation leaf (HU). Installed via
/// `set_continuation`; when present, `encode_bottom_up` fills the cfv of the
/// childless-chance depth-limit leaves with the closed-form HU Arm-1 showdown
/// (see vcfr.metal `vcfr_continuation_*` + hu_continuation_closed_form.rs)
/// BEFORE the bottom-up level loop reads them.
struct Continuation {
    // HU exact (np=2) pipelines
    reduce_pipeline: ComputePipelineState,
    fill_pipeline: ComputePipelineState,
    // Multiway sampled (np≥3) pipelines
    reduce_mw_pipeline: ComputePipelineState,
    cdf_mw_pipeline: ComputePipelineState,
    showdown_mw_pipeline: ComputePipelineState,
    expand_pipeline: ComputePipelineState,
    d_map: MetalBuffer<u16>,        // [nh] hand→flop bucket
    d_leaf_nodes: MetalBuffer<u32>, // [n_leaf] childless-chance leaf node ids
    d_f_w: MetalBuffer<f32>,        // [nb*nb] runout win fractions
    d_f_t: MetalBuffer<f32>,
    d_f_l: MetalBuffer<f32>,
    d_f_n: MetalBuffer<f32>,
    d_bucket_reach: MetalBuffer<f32>, // [n_leaf*(num_opp)*nb] (HU: num_opp=1)
    d_cdf: MetalBuffer<f32>,          // [n_leaf*num_opp*nb] (mw only)
    d_zsum: MetalBuffer<f32>,         // [n_leaf*num_opp]     (mw only)
    d_cfv_bucket: MetalBuffer<f32>,   // [n_leaf*nb]          (mw only)
    nb: usize,
    n_leaf: usize,
    num_opp: usize,
    sample_m: u32,
    rng_seed: u64,
    rake_rate: f32,
    rake_cap: f32,
}

/// Fast lone-survivor terminal offload (np=3). When installed, the listed
/// lone-survivor terminals (num_active<=1) are valued by `vcfr_lone_terminal_par`
/// (parallel over hands, bit-exact to the slow path) before the bottom-up level
/// loop, which then skips them (BuParams.skip_lone_terminals=1).
struct LoneTerminals {
    pipeline: ComputePipelineState,           // brute g0×g1 (bit-exact)
    factored_pipeline: ComputePipelineState,  // O(nh) inclusion-exclusion
    d_term_nodes: MetalBuffer<u32>,
    d_pair2hand: MetalBuffer<i32>,            // [52*52] card-pair → hand idx (-1)
    n_term: usize,
    factored: bool,
}

/// Metal-backed vector CFR solver.
/// Mirrors the CPU `VectorCfr` but runs all kernels on the GPU via Metal.
pub struct MetalVectorCfr {
    // Pipelines
    strategies_pipeline: ComputePipelineState,
    qre_strategies_pipeline: ComputePipelineState,
    init_reach_pipeline: ComputePipelineState,
    top_down_pipeline: ComputePipelineState,
    bottom_up_pipeline: ComputePipelineState,
    zero_pipeline: ComputePipelineState,

    // GPU buffers — tree structure (read-only after init)
    d_nodes: MetalBuffer<FlatNode>,
    d_children: MetalBuffer<u32>,
    d_contributions: MetalBuffer<i32>,
    d_folded_masks: MetalBuffer<u16>,
    d_infoset_offsets: MetalBuffer<u32>,
    d_decision_node_ids: MetalBuffer<u32>,

    // GPU buffers — solver state (read-write)
    d_regrets: MetalBuffer<f32>,
    d_strategy: MetalBuffer<f32>,
    d_cum_strategy: MetalBuffer<f32>,
    // QRE: accumulated per-(infoset,action,hand) cfv. Only used when lambda>0.
    d_last_cfv: MetalBuffer<f32>,
    d_reach: MetalBuffer<f32>,
    d_cfv: MetalBuffer<f32>,

    // GPU buffers — game data (read-only)
    d_initial_weight: MetalBuffer<f32>,
    d_sorted_opp_strength: MetalBuffer<u16>,
    d_sorted_opp_indices: MetalBuffer<u16>,
    d_sorted_pl_strength: MetalBuffer<u16>,
    d_sorted_pl_indices: MetalBuffer<u16>,
    d_hand_cards: MetalBuffer<u8>,

    // Level node buffers (one per depth level)
    d_level_nodes: Vec<MetalBuffer<u32>>,
    level_counts: Vec<i32>,

    // Solver parameters
    num_players: u8,
    num_hands: usize,
    num_infosets: usize,
    max_depth: usize,
    iteration: u32,
    regret_floor: f32,
    starting_pot: i32,
    num_combinations: f32,

    // Scratch buffer for params (legacy sync path; the async path passes
    // params inline via set_bytes so there is no shared-buffer clobber).
    d_params_buf: MetalBuffer<u8>,

    // When true, per-dispatch `wait_until_completed` is skipped — command
    // buffers commit on the serial queue (execution = commit order) and the
    // caller flushes ONCE at the end (run_batched). This is the latency path;
    // the convergence/parity `run()` keeps it false (per-dispatch wait).
    async_mode: std::cell::Cell<bool>,

    // Optional depth-limited bucketed continuation leaf (HU). None ⇒ leaves
    // value normally (showdown/childless-chance-zero); Some ⇒ filled by the
    // closed-form HU continuation before the bottom-up level loop.
    continuation: Option<Continuation>,

    // Optional fast lone-survivor terminal offload (np=3 search path).
    lone_terminals: Option<LoneTerminals>,

    // QRE inverse-temperature λ. 0 = off (regret-matching / DCFR). > 0 ⇒ the
    // strategy is a logit over time-averaged action cfv (matches CPU set_lambda).
    lambda: f32,

    // PER-SOLVE command queue (created from the shared device). Every command
    // buffer this solver creates goes through its OWN queue, so concurrent
    // solvers never build command buffers on the same queue — that was the
    // shared-context deadlock (Metal command-buffer creation is not safe across
    // threads on ONE queue). The device + metallib stay shared (the leak fix);
    // queues are cheap and per-instance. flush() waits on THIS queue.
    queue: metal::CommandQueue,
}

impl MetalVectorCfr {
    /// Create a new Metal vector CFR solver for the given tree and game data.
    pub fn new(
        ctx: &MetalContext,
        tree: &FlatTree,
        num_hands: usize,
        initial_weight: &[Vec<f32>],  // [np][nh]
        sorted_opp_strength: &[u16],   // [num_opp * nh]
        sorted_opp_indices: &[u16],    // [num_opp * nh]
        sorted_pl_strength: &[u16],    // [nh]
        sorted_pl_indices: &[u16],     // [nh]
        hand_cards: &[u8],             // [nh * 2]
        num_combinations: f64,
    ) -> Self {
        let np = tree.num_players as usize;
        let nn = tree.num_nodes();

        // Determine num_hands
        let nh = num_hands;
        let num_infosets = tree.num_infosets as usize;
        let max_depth = tree.max_depth as usize;
        let infoset_data_size = num_infosets * MAX_NA_POSTFLOP * nh;

        // Create pipelines
        let strategies_pipeline = ctx.create_pipeline("vcfr_compute_strategies")
            .expect("Failed to create strategies pipeline");
        let qre_strategies_pipeline = ctx.create_pipeline("vcfr_compute_strategies_qre")
            .expect("Failed to create qre strategies pipeline");
        let init_reach_pipeline = ctx.create_pipeline("vcfr_init_reach")
            .expect("Failed to create init_reach pipeline");
        let top_down_pipeline = ctx.create_pipeline("vcfr_top_down_reach")
            .expect("Failed to create top_down pipeline");
        let bottom_up_pipeline = ctx.create_pipeline("vcfr_bottom_up")
            .expect("Failed to create bottom_up pipeline");
        let zero_pipeline = ctx.create_pipeline("vcfr_zero_buffer")
            .expect("Failed to create zero_buffer pipeline");

        // Upload tree structure
        let d_nodes = ctx.upload(&tree.nodes);
        let d_children = ctx.upload(&tree.children);
        let d_contributions = ctx.upload(&tree.contributions);
        let d_folded_masks = ctx.upload(&tree.folded_masks);
        let d_infoset_offsets = ctx.upload(&tree.infoset_offsets);
        let d_decision_node_ids = ctx.upload(&tree.decision_node_ids);

        // Allocate solver state buffers
        let d_regrets = ctx.alloc_zeros(infoset_data_size);
        let d_strategy = ctx.alloc_zeros(infoset_data_size);
        let d_cum_strategy = ctx.alloc_zeros(infoset_data_size);
        let d_last_cfv = ctx.alloc_zeros(infoset_data_size);
        let d_reach = ctx.alloc_zeros(nn * np * nh);
        let d_cfv = ctx.alloc_zeros(nn * nh);

        // Flatten initial weights: [p0_nh, p1_nh, ..., pN_nh]
        let initial_weight_flat: Vec<f32> = initial_weight.iter().flat_map(|w| w.iter().copied()).collect();
        let d_initial_weight = ctx.upload(&initial_weight_flat);

        // Upload sorted arrays
        let d_sorted_opp_strength = ctx.upload(sorted_opp_strength);
        let d_sorted_opp_indices = ctx.upload(sorted_opp_indices);
        let d_sorted_pl_strength = ctx.upload(sorted_pl_strength);
        let d_sorted_pl_indices = ctx.upload(sorted_pl_indices);
        let d_hand_cards = ctx.upload(hand_cards);

        // Level node buffers
        let mut d_level_nodes = Vec::with_capacity(max_depth + 1);
        let mut level_counts = Vec::with_capacity(max_depth + 1);
        for level in 0..=max_depth {
            let nodes = tree.nodes_at_level(level as u32).to_vec();
            level_counts.push(nodes.len() as i32);
            d_level_nodes.push(ctx.upload(&nodes));
        }

        // Params scratch buffer (large enough for all param structs)
        let d_params_buf = ctx.alloc_zeros(64);

        // Per-solve command queue from the shared device (see struct field doc).
        let queue = ctx.device().new_command_queue();

        MetalVectorCfr {
            queue,
            strategies_pipeline,
            qre_strategies_pipeline,
            init_reach_pipeline,
            top_down_pipeline,
            bottom_up_pipeline,
            zero_pipeline,
            d_nodes,
            d_children,
            d_contributions,
            d_folded_masks,
            d_infoset_offsets,
            d_decision_node_ids,
            d_regrets,
            d_strategy,
            d_cum_strategy,
            d_last_cfv,
            d_reach,
            d_cfv,
            d_initial_weight,
            d_sorted_opp_strength,
            d_sorted_opp_indices,
            d_sorted_pl_strength,
            d_sorted_pl_indices,
            d_hand_cards,
            d_level_nodes,
            level_counts,
            num_players: tree.num_players,
            num_hands: nh,
            num_infosets,
            max_depth,
            iteration: 0,
            regret_floor: -1e30,
            starting_pot: tree.starting_pot,
            num_combinations: num_combinations as f32,
            d_params_buf,
            async_mode: std::cell::Cell::new(false),
            continuation: None,
            lone_terminals: None,
            lambda: 0.0,
        }
    }

    /// Enable QRE (quantal-response) mode at inverse-temperature `lambda` (0 =
    /// off). Matches the CPU `CpuMccfr::set_lambda` — the strategy becomes a
    /// logit over time-averaged action cfv instead of regret-matching.
    pub fn set_lambda(&mut self, lambda: f32) {
        self.lambda = lambda;
    }

    /// Install the fast lone-survivor terminal offload (np=3). `term_nodes` =
    /// the terminal node ids with num_active<=1 (every terminal of a no-all-in
    /// depth-limited tree). They are then valued by `vcfr_lone_terminal_par`
    /// (parallel over hands) and skipped in the bottom-up level loop.
    pub fn set_fast_lone_terminals(&mut self, ctx: &MetalContext, term_nodes: &[u32]) {
        self.set_fast_lone_terminals_ex(ctx, term_nodes, false);
    }

    /// As `set_fast_lone_terminals`; `factored=true` uses the O(nh)
    /// inclusion-exclusion kernel (faster at large nh, within-tolerance not
    /// bit-exact) instead of the bit-exact brute g0×g1 kernel.
    pub fn set_fast_lone_terminals_ex(&mut self, ctx: &MetalContext, term_nodes: &[u32], factored: bool) {
        let pipeline = ctx.create_pipeline("vcfr_lone_terminal_par")
            .expect("vcfr_lone_terminal_par pipeline");
        let factored_pipeline = ctx.create_pipeline("vcfr_lone_terminal_factored")
            .expect("vcfr_lone_terminal_factored pipeline");
        // pair2hand[c1*52+c2] = hand index of {c1,c2} (symmetric), else -1.
        let nh = self.num_hands;
        let mut pair2hand = vec![-1i32; 52 * 52];
        let hc = self.d_hand_cards.as_slice();
        for h in 0..nh {
            let (a, b) = (hc[h * 2] as usize, hc[h * 2 + 1] as usize);
            pair2hand[a * 52 + b] = h as i32;
            pair2hand[b * 52 + a] = h as i32;
        }
        self.lone_terminals = Some(LoneTerminals {
            pipeline,
            factored_pipeline,
            d_term_nodes: ctx.upload(term_nodes),
            d_pair2hand: ctx.upload(&pair2hand),
            n_term: term_nodes.len(),
            factored,
        });
    }

    /// Install the depth-limited HU bucketed continuation leaf. `leaf_nodes` =
    /// the childless-chance depth-limit node ids; `map` = hand→flop bucket
    /// ([nh], u16, NO_BUCKET=0xFFFF for dead hands); `tables` = the flop runout
    /// win/tie/lose/compat fractions ([nb*nb] each). After this, `run_batched`
    /// fills those leaves' cfv with the closed-form HU Arm-1 showdown each pass.
    #[allow(clippy::too_many_arguments)]
    pub fn set_continuation(
        &mut self,
        ctx: &MetalContext,
        leaf_nodes: &[u32],
        map: &[u16],
        nb: usize,
        f_w: &[f32],
        f_t: &[f32],
        f_l: &[f32],
        f_n: &[f32],
        rake_rate: f32,
        rake_cap: f32,
        sample_m: u32,
        rng_seed: u64,
    ) {
        let p = |name: &str| ctx.create_pipeline(name).unwrap_or_else(|_| panic!("{name} pipeline"));
        let n_leaf = leaf_nodes.len();
        let num_opp = (self.num_players - 1) as usize;
        self.continuation = Some(Continuation {
            reduce_pipeline: p("vcfr_continuation_reduce"),
            fill_pipeline: p("vcfr_continuation_fill"),
            reduce_mw_pipeline: p("vcfr_continuation_reduce_mw"),
            cdf_mw_pipeline: p("vcfr_continuation_cdf_mw"),
            showdown_mw_pipeline: p("vcfr_continuation_showdown_mw"),
            expand_pipeline: p("vcfr_continuation_expand"),
            d_map: ctx.upload(map),
            d_leaf_nodes: ctx.upload(leaf_nodes),
            d_f_w: ctx.upload(f_w),
            d_f_t: ctx.upload(f_t),
            d_f_l: ctx.upload(f_l),
            d_f_n: ctx.upload(f_n),
            d_bucket_reach: ctx.alloc_zeros(n_leaf.max(1) * num_opp.max(1) * nb),
            d_cdf: ctx.alloc_zeros(n_leaf.max(1) * num_opp.max(1) * nb),
            d_zsum: ctx.alloc_zeros(n_leaf.max(1) * num_opp.max(1)),
            d_cfv_bucket: ctx.alloc_zeros(n_leaf.max(1) * nb),
            nb,
            n_leaf,
            num_opp,
            sample_m,
            rng_seed,
            rake_rate,
            rake_cap,
        });
    }

    /// Conditional per-dispatch wait. Sync mode (default) waits after every
    /// commit — the byte-exact convergence/parity path. Async mode skips the
    /// wait; the serial queue preserves order and the caller flushes once.
    #[inline]
    fn maybe_wait(&self, buf: &metal::CommandBufferRef) {
        if !self.async_mode.get() {
            buf.wait_until_completed();
        }
    }

    /// Drain queued GPU work: commit an empty command buffer and wait. Metal
    /// serializes command buffers within a queue, so waiting on a LATER buffer
    /// guarantees all prior committed buffers completed. Terminates async runs.
    fn flush(&self, ctx: &MetalContext) {
        let final_buf = self.queue.new_command_buffer();
        final_buf.commit();
        final_buf.wait_until_completed();
    }

    /// Run the solver for the specified number of iterations.
    /// Returns root CFV for player 0.
    pub fn run(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        num_iterations: u32,
    ) -> Vec<f32> {
        let np = self.num_players as usize;
        let nh = self.num_hands;
        let ni = self.num_infosets;
        let nn = tree.num_nodes();

        let mut root_cfv_sum = vec![0.0f32; nh];
        let mut count = 0u32;

        for _ in 0..num_iterations {
            // Drain each iteration's autoreleased command buffers immediately (sync
            // mode waits per dispatch, so they're already complete) — otherwise they
            // pile up on the worker thread's never-drained pool, pinning the GPU
            // buffers they reference (the wired-memory leak). See run_batched.
            objc::rc::autoreleasepool(|| {
                let params = DcfrParams::new(self.iteration);
                self.iteration += 1;

                // Sequential (alternating) updates: recompute strategies and reach
                // before each traverser's bottom-up pass. This matches the CUDA
                // GPU solver and Tammelin CFR+ — each traverser sees the most
                // up-to-date regret state, yielding ~8x faster convergence than
                // computing strategies once per iteration.
                for traverser in 0..np {
                    self.launch_compute_strategies(ctx, ni, nh);
                    self.launch_init_reach(ctx, np, nh);
                    self.launch_top_down(ctx, nh, np as u32);

                    self.launch_bottom_up(
                        ctx, tree, traverser as u32,
                        params.alpha_t, params.beta_t, params.gamma_t,
                        nh, np as u32,
                    );

                    if traverser == 0 {
                        let cfv = self.d_cfv.as_slice();
                        for h in 0..nh {
                            root_cfv_sum[h] += cfv[h];
                        }
                        count += 1;
                    }
                }
            });
        }

        for h in 0..nh {
            root_cfv_sum[h] /= count as f32;
        }
        root_cfv_sum
    }

    /// Latency-grade run: identical CFR math to `run()` but async — no
    /// per-dispatch `wait_until_completed` and NO per-iter `d_cfv` CPU
    /// readback (the readback in `run()` forces a full GPU sync every iter).
    /// Command buffers commit on the serial queue (execution = commit order);
    /// hazard tracking on the shared state buffers (d_strategy/d_reach/
    /// d_regrets/d_cfv) orders the stages; a single `flush()` at the end
    /// drains the queue. Use this for the real-time depth-limited search.
    pub fn run_batched(&mut self, ctx: &MetalContext, tree: &FlatTree, num_iterations: u32) {
        // Drain Metal's autoreleased objects (the per-iter command buffers, which
        // RETAIN the large per-solve GPU buffers) when this solve finishes. Without
        // this, on a long-lived worker thread the thread's autorelease pool never
        // drains → command buffers accumulate → the buffers they hold are never
        // freed → wired GPU memory grows unbounded across requests (the OOM). The
        // pool wraps the whole solve incl. flush(), so it drains only after the GPU
        // has completed every committed buffer — safe.
        objc::rc::autoreleasepool(|| {
            let np = self.num_players as usize;
            let nh = self.num_hands;
            let ni = self.num_infosets;

            self.async_mode.set(true);
            for _ in 0..num_iterations {
                let params = DcfrParams::new(self.iteration);
                self.iteration += 1;
                for traverser in 0..np {
                    // ONE command buffer per traverser pass: all stage dispatches
                    // (strategies → init_reach → top_down → bottom_up) encode into
                    // it as separate encoders; Metal hazard-tracks the shared state
                    // buffers to order them. Collapses ~23 command-buffer creations
                    // (the measured `_MTLCommandBuffer init` hot path) into 1.
                    let cmd = self.queue.new_command_buffer();
                    self.encode_compute_strategies(ctx, cmd, ni, nh);
                    self.encode_init_reach(ctx, cmd, np, nh);
                    self.encode_top_down(ctx, cmd, nh, np as u32);
                    self.encode_bottom_up(
                        ctx, cmd, tree, traverser as u32,
                        params.alpha_t, params.beta_t, params.gamma_t,
                        nh, np as u32,
                    );
                    cmd.commit(); // no wait — serial queue preserves order
                }
            }
            self.flush(ctx);
            self.async_mode.set(false);
        });
    }

    /// `run_batched` with a HARD WALL-CLOCK BUDGET — the runaway guard for the
    /// real-time path. The unbudgeted loop commits ALL iterations up front, so a
    /// big tree (deep multiway) can pin the GPU for minutes with no way to stop —
    /// and if the caller vanishes (client disconnect), the work grinds on orphaned.
    /// Here iterations are committed in CHUNKS with a flush (GPU sync) between
    /// them; once `budget_ms` is spent the loop stops committing. Worst-case
    /// overrun is one chunk. Returns the number of iterations actually run.
    /// The per-chunk flush costs one sync per chunk (negligible vs chunk runtime).
    pub fn run_batched_budget(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        num_iterations: u32,
        budget_ms: u64,
    ) -> u32 {
        let start = std::time::Instant::now();
        let mut done = 0u32;
        // ADAPTIVE first probe: run ONE iteration and time it before committing
        // any batch. A fixed first chunk is unbounded in wall-clock — a deep
        // multiway tree at ~seconds/iter turned "25 iters then check" into a
        // multi-minute GPU pin (the fleet incident). One iter bounds the initial
        // exposure to a single iteration's cost, then chunks are sized from the
        // measured rate so each stays ~1s of GPU work (frequent budget checks,
        // small overrun).
        if num_iterations == 0 {
            return 0;
        }
        self.run_batched(ctx, tree, 1);
        done += 1;
        let per_iter_ms = (start.elapsed().as_millis() as u64).max(1);
        while done < num_iterations {
            let remaining_ms = budget_ms.saturating_sub(start.elapsed().as_millis() as u64);
            if remaining_ms < per_iter_ms {
                break;
            }
            // ~1s of GPU work per chunk (min 1, max 25 iters), bounded by budget.
            let by_time = (1_000 / per_iter_ms).clamp(1, 25) as u32;
            let by_budget = (remaining_ms / per_iter_ms).max(1) as u32;
            let n = by_time.min(by_budget).min(num_iterations - done);
            self.run_batched(ctx, tree, n);
            done += n;
        }
        done
    }

    fn launch_compute_strategies(&mut self, ctx: &MetalContext, num_infosets: usize, nh: usize) {
        let cmd = self.queue.new_command_buffer();
        self.encode_compute_strategies(ctx, cmd, num_infosets, nh);
        cmd.commit();
        self.maybe_wait(cmd);
    }

    fn encode_compute_strategies(&self, ctx: &MetalContext, cmd: &metal::CommandBufferRef, num_infosets: usize, nh: usize) {
        let enc = cmd.new_compute_command_encoder();
        if self.lambda > 0.0 {
            // QRE: logit over time-averaged action cfv (last_cfv / denom).
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct QreParams { num_infosets: i32, nh: i32, lambda: f32, denom: f32 }
            let denom = (self.iteration as f32 - 1.0).max(1.0);
            let p = QreParams { num_infosets: num_infosets as i32, nh: nh as i32, lambda: self.lambda, denom };
            enc.set_compute_pipeline_state(&self.qre_strategies_pipeline);
            enc.set_buffer(0, Some(self.d_last_cfv.as_ref()), 0);
            enc.set_buffer(1, Some(self.d_strategy.as_ref()), 0);
            enc.set_buffer(2, Some(self.d_decision_node_ids.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_bytes(5, std::mem::size_of::<QreParams>() as u64, &p as *const _ as *const std::ffi::c_void);
        } else {
            let params_data: [i32; 2] = [num_infosets as i32, nh as i32];
            enc.set_compute_pipeline_state(&self.strategies_pipeline);
            enc.set_buffer(0, Some(self.d_regrets.as_ref()), 0);
            enc.set_buffer(1, Some(self.d_strategy.as_ref()), 0);
            enc.set_buffer(2, Some(self.d_decision_node_ids.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_bytes(5, 8, params_data.as_ptr() as *const std::ffi::c_void);
        }
        let max_tpg = self.strategies_pipeline.max_total_threads_per_threadgroup() as usize;
        let (grid, tg) = ctx.dispatch_2d(num_infosets, nh, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    fn launch_init_reach(&mut self, ctx: &MetalContext, np: usize, nh: usize) {
        let cmd = self.queue.new_command_buffer();
        self.encode_init_reach(ctx, cmd, np, nh);
        cmd.commit();
        self.maybe_wait(cmd);
    }

    fn encode_init_reach(&self, ctx: &MetalContext, cmd: &metal::CommandBufferRef, np: usize, nh: usize) {
        let nn = self.d_nodes.len();
        let total_reach = nn * np * nh;
        let np_nh = np * nh;
        let params_data: [i32; 2] = [total_reach as i32, np_nh as i32];

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.init_reach_pipeline);
        enc.set_buffer(0, Some(self.d_reach.as_ref()), 0);
        enc.set_buffer(1, Some(self.d_initial_weight.as_ref()), 0);
        enc.set_bytes(2, 8, params_data.as_ptr() as *const std::ffi::c_void);

        let (grid, tg) = ctx.dispatch_1d(total_reach, 256);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    fn launch_top_down(&mut self, ctx: &MetalContext, nh: usize, np: u32) {
        let cmd = self.queue.new_command_buffer();
        self.encode_top_down(ctx, cmd, nh, np);
        cmd.commit();
        self.maybe_wait(cmd);
    }

    fn encode_top_down(&self, ctx: &MetalContext, cmd: &metal::CommandBufferRef, nh: usize, np: u32) {
        let nh_i32 = nh as i32;

        for level in 0..=self.max_depth {
            let count = self.level_counts[level];
            if count == 0 { continue; }

            let params_data: [i32; 3] = [count, np as i32, nh_i32];

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.top_down_pipeline);
            enc.set_buffer(0, Some(self.d_level_nodes[level].as_ref()), 0);
            enc.set_bytes(1, 12, params_data.as_ptr() as *const std::ffi::c_void);
            enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_strategy.as_ref()), 0);
            enc.set_buffer(5, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_buffer(6, Some(self.d_reach.as_ref()), 0);

            let max_tpg = self.top_down_pipeline.max_total_threads_per_threadgroup() as usize;
            let (grid, tg) = ctx.dispatch_2d(count as usize, nh, max_tpg);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_bottom_up(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        traverser: u32,
        alpha_t: f32,
        beta_t: f32,
        gamma_t: f32,
        nh: usize,
        np: u32,
    ) {
        let cmd = self.queue.new_command_buffer();
        self.encode_bottom_up(ctx, cmd, tree, traverser, alpha_t, beta_t, gamma_t, nh, np);
        cmd.commit();
        self.maybe_wait(cmd);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_bottom_up(
        &self,
        ctx: &MetalContext,
        cmd: &metal::CommandBufferRef,
        _tree: &FlatTree,
        traverser: u32,
        alpha_t: f32,
        beta_t: f32,
        gamma_t: f32,
        nh: usize,
        np: u32,
    ) {
        let nh_i32 = nh as i32;

        // Zero CFV buffer before bottom-up pass
        {
            let params_data: [i32; 1] = [self.d_cfv.len() as i32];

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.zero_pipeline);
            enc.set_buffer(0, Some(self.d_cfv.as_ref()), 0);
            enc.set_bytes(1, 4, params_data.as_ptr() as *const std::ffi::c_void);

            let (grid, tg) = ctx.dispatch_1d(self.d_cfv.len(), 256);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
        }

        // Depth-limited continuation leaves: fill their cfv with the closed-form
        // HU showdown AFTER the zero (so it isn't wiped) and BEFORE the level
        // loop (so the player parents read it). No-op when not installed.
        self.encode_continuation(ctx, cmd, traverser, nh, np);

        // Fast lone-survivor terminals (np=3): fill cfv parallel over hands,
        // before the level loop (which skips them). No-op when not installed.
        self.encode_lone_terminals(ctx, cmd, traverser, nh, np);

        // Bottom-up: process levels from deepest to shallowest
        for level in (0..=self.max_depth).rev() {
            let count = self.level_counts[level];
            if count == 0 { continue; }

            // Pack params struct: [level_count, num_players, nh, traverser, alpha_t, beta_t, gamma_t, regret_floor, starting_pot, num_combinations]
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct BuParams {
                level_count: i32,
                num_players: i32,
                nh: i32,
                traverser: u32,
                alpha_t: f32,
                beta_t: f32,
                gamma_t: f32,
                regret_floor: f32,
                starting_pot: i32,
                num_combinations: f32,
                skip_lone_terminals: i32,
                lambda_active: i32,
            }

            let bu_params = BuParams {
                level_count: count,
                num_players: np as i32,
                nh: nh_i32,
                traverser,
                alpha_t,
                beta_t,
                gamma_t,
                regret_floor: self.regret_floor,
                starting_pot: self.starting_pot,
                num_combinations: self.num_combinations,
                skip_lone_terminals: if self.lone_terminals.is_some() { 1 } else { 0 },
                lambda_active: if self.lambda > 0.0 { 1 } else { 0 },
            };

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.bottom_up_pipeline);
            enc.set_buffer(0, Some(self.d_level_nodes[level].as_ref()), 0);
            enc.set_bytes(1, std::mem::size_of::<BuParams>() as u64,
                          &bu_params as *const _ as *const std::ffi::c_void);
            enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
            enc.set_buffer(3, Some(self.d_children.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_contributions.as_ref()), 0);
            enc.set_buffer(5, Some(self.d_folded_masks.as_ref()), 0);
            enc.set_buffer(6, Some(self.d_strategy.as_ref()), 0);
            enc.set_buffer(7, Some(self.d_infoset_offsets.as_ref()), 0);
            enc.set_buffer(8, Some(self.d_reach.as_ref()), 0);
            enc.set_buffer(9, Some(self.d_cfv.as_ref()), 0);
            enc.set_buffer(10, Some(self.d_regrets.as_ref()), 0);
            enc.set_buffer(11, Some(self.d_cum_strategy.as_ref()), 0);
            enc.set_buffer(12, Some(self.d_initial_weight.as_ref()), 0);
            enc.set_buffer(13, Some(self.d_sorted_opp_strength.as_ref()), 0);
            enc.set_buffer(14, Some(self.d_sorted_opp_indices.as_ref()), 0);
            enc.set_buffer(15, Some(self.d_sorted_pl_strength.as_ref()), 0);
            enc.set_buffer(16, Some(self.d_sorted_pl_indices.as_ref()), 0);
            enc.set_buffer(17, Some(self.d_hand_cards.as_ref()), 0);
            // buffer 18 (rake_marker) intentionally unbound (debug-only, as before).
            enc.set_buffer(19, Some(self.d_last_cfv.as_ref()), 0); // QRE accumulator

            let (grid, tg) = ctx.dispatch_1d(count as usize, 1);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
        }
    }

    /// Encode the fast lone-survivor terminal fill (parallel over hands).
    fn encode_lone_terminals(&self, ctx: &MetalContext, cmd: &metal::CommandBufferRef, traverser: u32, nh: usize, np: u32) {
        let Some(lt) = &self.lone_terminals else { return; };
        if lt.n_term == 0 { return; }
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct LoneTermParams {
            nh: i32, np: i32, traverser: i32, n_term: i32,
            starting_pot: i32, rake_rate: f32, rake_cap: f32, num_combinations: f32,
        }
        let params = LoneTermParams {
            nh: nh as i32, np: np as i32, traverser: traverser as i32, n_term: lt.n_term as i32,
            starting_pot: self.starting_pot, rake_rate: 0.0, rake_cap: 0.0,
            num_combinations: self.num_combinations,
        };
        let pipeline = if lt.factored { &lt.factored_pipeline } else { &lt.pipeline };
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(self.d_cfv.as_ref()), 0);
        enc.set_buffer(1, Some(lt.d_term_nodes.as_ref()), 0);
        enc.set_buffer(2, Some(self.d_nodes.as_ref()), 0);
        enc.set_buffer(3, Some(self.d_contributions.as_ref()), 0);
        enc.set_buffer(4, Some(self.d_folded_masks.as_ref()), 0);
        enc.set_buffer(5, Some(self.d_reach.as_ref()), 0);
        enc.set_buffer(6, Some(self.d_hand_cards.as_ref()), 0);
        if lt.factored {
            enc.set_buffer(7, Some(lt.d_pair2hand.as_ref()), 0);
            enc.set_bytes(8, std::mem::size_of::<LoneTermParams>() as u64, &params as *const _ as *const std::ffi::c_void);
        } else {
            enc.set_bytes(7, std::mem::size_of::<LoneTermParams>() as u64, &params as *const _ as *const std::ffi::c_void);
        }
        let max_tpg = pipeline.max_total_threads_per_threadgroup() as usize;
        let (grid, tg) = ctx.dispatch_2d(lt.n_term, nh, max_tpg);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    /// Encode the HU continuation-leaf fill (reduce reach→bucket, then closed-
    /// form showdown+expand → cfv at the leaf nodes). No-op if not installed.
    fn encode_continuation(&self, ctx: &MetalContext, cmd: &metal::CommandBufferRef, traverser: u32, nh: usize, np: u32) {
        let Some(c) = &self.continuation else { return; };
        if c.n_leaf == 0 { return; }
        if np == 2 {
            self.encode_continuation_hu(ctx, cmd, c, traverser, nh, np);
        } else {
            self.encode_continuation_mw(ctx, cmd, c, traverser, nh, np);
        }
    }

    /// HU exact closed-form continuation (reduce → fill).
    fn encode_continuation_hu(&self, ctx: &MetalContext, cmd: &metal::CommandBufferRef, c: &Continuation, traverser: u32, nh: usize, np: u32) {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct ContParams {
            nb: i32, nh: i32, np: i32, traverser: i32, n_leaf: i32,
            starting_pot: i32, rake_rate: f32, rake_cap: f32, num_combinations: f32,
        }
        let params = ContParams {
            nb: c.nb as i32, nh: nh as i32, np: np as i32, traverser: traverser as i32,
            n_leaf: c.n_leaf as i32, starting_pot: self.starting_pot,
            rake_rate: c.rake_rate, rake_cap: c.rake_cap, num_combinations: self.num_combinations,
        };
        let pbytes = std::mem::size_of::<ContParams>() as u64;
        let pptr = &params as *const _ as *const std::ffi::c_void;
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.reduce_pipeline);
            enc.set_buffer(0, Some(c.d_bucket_reach.as_ref()), 0);
            enc.set_buffer(1, Some(self.d_reach.as_ref()), 0);
            enc.set_buffer(2, Some(c.d_map.as_ref()), 0);
            enc.set_buffer(3, Some(c.d_leaf_nodes.as_ref()), 0);
            enc.set_bytes(4, pbytes, pptr);
            let max_tpg = c.reduce_pipeline.max_total_threads_per_threadgroup() as usize;
            let (grid, tg) = ctx.dispatch_2d(c.n_leaf, c.nb, max_tpg);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.fill_pipeline);
            enc.set_buffer(0, Some(self.d_cfv.as_ref()), 0);
            enc.set_buffer(1, Some(c.d_bucket_reach.as_ref()), 0);
            enc.set_buffer(2, Some(c.d_map.as_ref()), 0);
            enc.set_buffer(3, Some(c.d_leaf_nodes.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_contributions.as_ref()), 0);
            enc.set_buffer(5, Some(c.d_f_w.as_ref()), 0);
            enc.set_buffer(6, Some(c.d_f_t.as_ref()), 0);
            enc.set_buffer(7, Some(c.d_f_l.as_ref()), 0);
            enc.set_buffer(8, Some(c.d_f_n.as_ref()), 0);
            enc.set_bytes(9, pbytes, pptr);
            let max_tpg = c.fill_pipeline.max_total_threads_per_threadgroup() as usize;
            let (grid, tg) = ctx.dispatch_2d(c.n_leaf, nh, max_tpg);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
        }
    }

    /// Multiway MC-sampled continuation (reduce_mw → cdf → showdown_mw → expand).
    fn encode_continuation_mw(&self, ctx: &MetalContext, cmd: &metal::CommandBufferRef, c: &Continuation, traverser: u32, nh: usize, np: u32) {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct ContMwParams {
            nb: i32, nh: i32, np: i32, traverser: i32, n_leaf: i32, num_opp: i32,
            starting_pot: i32, sample_m: u32, rake_rate: f32, rake_cap: f32,
            num_combinations: f32, rng_seed: u64,
        }
        let params = ContMwParams {
            nb: c.nb as i32, nh: nh as i32, np: np as i32, traverser: traverser as i32,
            n_leaf: c.n_leaf as i32, num_opp: c.num_opp as i32, starting_pot: self.starting_pot,
            sample_m: c.sample_m, rake_rate: c.rake_rate, rake_cap: c.rake_cap,
            num_combinations: self.num_combinations, rng_seed: c.rng_seed,
        };
        let pbytes = std::mem::size_of::<ContMwParams>() as u64;
        let pptr = &params as *const _ as *const std::ffi::c_void;
        // reduce_mw: grid (n_leaf*num_opp, nb)
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.reduce_mw_pipeline);
            enc.set_buffer(0, Some(c.d_bucket_reach.as_ref()), 0);
            enc.set_buffer(1, Some(self.d_reach.as_ref()), 0);
            enc.set_buffer(2, Some(c.d_map.as_ref()), 0);
            enc.set_buffer(3, Some(c.d_leaf_nodes.as_ref()), 0);
            enc.set_bytes(4, pbytes, pptr);
            let max_tpg = c.reduce_mw_pipeline.max_total_threads_per_threadgroup() as usize;
            let (grid, tg) = ctx.dispatch_2d(c.n_leaf * c.num_opp, c.nb, max_tpg);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
        }
        // cdf_mw: grid (n_leaf, num_opp)
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.cdf_mw_pipeline);
            enc.set_buffer(0, Some(c.d_cdf.as_ref()), 0);
            enc.set_buffer(1, Some(c.d_zsum.as_ref()), 0);
            enc.set_buffer(2, Some(c.d_bucket_reach.as_ref()), 0);
            enc.set_bytes(3, pbytes, pptr);
            let max_tpg = c.cdf_mw_pipeline.max_total_threads_per_threadgroup() as usize;
            let (grid, tg) = ctx.dispatch_2d(c.n_leaf, c.num_opp, max_tpg);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
        }
        // showdown_mw: grid (n_leaf, nb)
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.showdown_mw_pipeline);
            enc.set_buffer(0, Some(c.d_cfv_bucket.as_ref()), 0);
            enc.set_buffer(1, Some(c.d_cdf.as_ref()), 0);
            enc.set_buffer(2, Some(c.d_zsum.as_ref()), 0);
            enc.set_buffer(3, Some(c.d_leaf_nodes.as_ref()), 0);
            enc.set_buffer(4, Some(self.d_contributions.as_ref()), 0);
            enc.set_buffer(5, Some(c.d_f_w.as_ref()), 0);
            enc.set_buffer(6, Some(c.d_f_t.as_ref()), 0);
            enc.set_buffer(7, Some(c.d_f_l.as_ref()), 0);
            enc.set_buffer(8, Some(c.d_f_n.as_ref()), 0);
            enc.set_buffer(9, Some(self.d_folded_masks.as_ref()), 0);
            enc.set_bytes(10, pbytes, pptr);
            let max_tpg = c.showdown_mw_pipeline.max_total_threads_per_threadgroup() as usize;
            let (grid, tg) = ctx.dispatch_2d(c.n_leaf, c.nb, max_tpg);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
        }
        // expand: grid (n_leaf, nh)
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.expand_pipeline);
            enc.set_buffer(0, Some(self.d_cfv.as_ref()), 0);
            enc.set_buffer(1, Some(c.d_cfv_bucket.as_ref()), 0);
            enc.set_buffer(2, Some(c.d_map.as_ref()), 0);
            enc.set_buffer(3, Some(c.d_leaf_nodes.as_ref()), 0);
            enc.set_bytes(4, pbytes, pptr);
            let max_tpg = c.expand_pipeline.max_total_threads_per_threadgroup() as usize;
            let (grid, tg) = ctx.dispatch_2d(c.n_leaf, nh, max_tpg);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
        }
    }

    /// Get average strategy at a node.
    pub fn get_average_strategy(&self, node_idx: usize, num_actions: usize, nh: usize) -> Vec<Vec<f32>> {
        let offset = self.node_data_offset(node_idx);
        if offset == UNUSED as usize {
            return vec![];
        }
        let cum = self.d_cum_strategy.as_slice();
        let mut result = vec![vec![0.0f32; nh]; num_actions];
        for h in 0..nh {
            let mut total = 0.0f32;
            for a in 0..num_actions {
                total += cum[offset + a * nh + h];
            }
            if total > 0.0 {
                for a in 0..num_actions {
                    result[a][h] = cum[offset + a * nh + h] / total;
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

    pub fn cum_strategy_slice(&self) -> Vec<f32> {
        self.d_cum_strategy.as_slice().to_vec()
    }

    pub fn node_offsets(&self) -> Vec<usize> {
        let nh = self.num_hands;
        self.d_infoset_offsets.as_slice()
            .iter()
            .map(|&off| {
                if off == UNUSED { usize::MAX }
                else { off as usize * MAX_NA_POSTFLOP * nh }
            })
            .collect()
    }

    pub fn iteration_count(&self) -> u32 {
        self.iteration
    }

    fn node_data_offset(&self, node_idx: usize) -> usize {
        let off = self.d_infoset_offsets.as_slice()[node_idx];
        if off == UNUSED { usize::MAX }
        else { off as usize * MAX_NA_POSTFLOP * self.num_hands }
    }

    // ---- Diagnostic methods for debugging kernel divergence ----

    pub fn strategy_slice(&self) -> Vec<f32> {
        self.d_strategy.as_slice().to_vec()
    }

    pub fn regrets_slice(&self) -> Vec<f32> {
        self.d_regrets.as_slice().to_vec()
    }

    pub fn reach_slice(&self) -> Vec<f32> {
        self.d_reach.as_slice().to_vec()
    }

    pub fn cfv_slice(&self) -> Vec<f32> {
        self.d_cfv.as_slice().to_vec()
    }

    /// Expose internal state for single-iteration diagnostics.
    /// Runs one full iteration: strategies -> init_reach -> top_down -> bottom_up.
    /// After each stage, the current state of the named buffer is returned.
    pub fn run_one_iteration_diagnostic(
        &mut self,
        ctx: &MetalContext,
        tree: &FlatTree,
        traverser: u32,
    ) -> DiagnosticSnapshot {
        let np = self.num_players as usize;
        let nh = self.num_hands;
        let ni = self.num_infosets;
        let nn = tree.num_nodes();
        let params = DcfrParams::new(self.iteration);
        self.iteration += 1;

        // Step 1: compute strategies
        self.launch_compute_strategies(ctx, ni, nh);
        let strategies = self.strategy_slice();

        // Step 2: init reach
        self.launch_init_reach(ctx, np, nh);
        let reach_after_init = self.reach_slice();

        // Step 3: top-down reach
        self.launch_top_down(ctx, nh, np as u32);
        let reach_after_topdown = self.reach_slice();

        // Step 4: bottom-up for this traverser
        self.launch_bottom_up(
            ctx, tree, traverser,
            params.alpha_t, params.beta_t, params.gamma_t,
            nh, np as u32,
        );
        let cfv = self.cfv_slice();
        let regrets = self.regrets_slice();
        let cum_strategy = self.cum_strategy_slice();

        DiagnosticSnapshot {
            strategies,
            reach_after_init,
            reach_after_topdown,
            cfv,
            regrets,
            cum_strategy,
            traverser,
            alpha_t: params.alpha_t,
            beta_t: params.beta_t,
            gamma_t: params.gamma_t,
        }
    }
}

pub struct DiagnosticSnapshot {
    pub strategies: Vec<f32>,
    pub reach_after_init: Vec<f32>,
    pub reach_after_topdown: Vec<f32>,
    pub cfv: Vec<f32>,
    pub regrets: Vec<f32>,
    pub cum_strategy: Vec<f32>,
    pub traverser: u32,
    pub alpha_t: f32,
    pub beta_t: f32,
    pub gamma_t: f32,
}
