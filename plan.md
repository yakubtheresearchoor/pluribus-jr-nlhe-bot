Implementation Plan: pluribus-junior GPU MCCFR Solver
Project Structure
C:\pluribus-junior\
├── NORTH_STAR.md                    # (existing)
├── research papers/                 # (existing)
├── Cargo.toml                       # workspace root
├── solver-core/                     # Core solver library (GPU MCCFR engine)
│   ├── Cargo.toml
│   ├── build.rs                     # nvcc → PTX compilation
│   └── src/
│       ├── lib.rs
│       ├── tree/
│       │   ├── mod.rs               # Flat tree types (FlatNode, FlatTree)
│       │   ├── builder.rs           # N-player action tree builder
│       │   └── action.rs            # Action enum, BoardState, TreeConfig
│       ├── hand/
│       │   ├── mod.rs               # Hand struct + evaluate()
│       │   └── table.rs             # HAND_TABLE 4824-entry constant
│       ├── range.rs                 # Range struct (1326 f32), FromStr parser
│       ├── card.rs                  # Card type, deck indexing, dead cards
│       ├── solver/
│       │   ├── mod.rs               # Solver public API (solve() entry point)
│       │   ├── state.rs             # SolverState: regrets, strategy SoA buffers
│       │   └── config.rs            # SolverConfig (time budget, batch size, etc.)
│       └── gpu/
│           ├── mod.rs               # CudaContext singleton (adapt from cuda/mod.rs)
│           ├── kernel.rs            # PTX loading, kernel launch helpers
│           └── kernels/
│               └── mccfr.cu         # MCCFR traversal + hand eval (single kernel)
├── solver-cli/                      # CLI binary (surstromming-solver-cli replacement)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                  # CLI entry (adapt from surstromming-solver-cli)
│       ├── config.rs                # SolveRequest JSON schema (adapt)
│       ├── solver.rs                # Solve dispatch, action history, strategy extraction
│       ├── cache.rs                 # SQLite cache with solver_version
│       └── range.rs                 # Opponent range construction from pool stats
└── tests/
    ├── kuhn.rs                      # Kuhn poker convergence test
    └── leduc.rs                     # Leduc poker convergence test
Four Milestones (from user)
#	Milestone	What it proves	Failure mode
M1	N-player flat action tree	Architecture holds for N>2, side pots, round completion	Flat representation gets messy, need DAG
M2	First MCCFR iteration on Kuhn/Leduc	GPU state + atomic updates + trajectory parallelism compose correctly	Kernel produces garbage, race conditions dominate
M3	HU equivalence (SC1)	Algorithm implemented correctly on real poker	Exploitability >5% off, convergence pattern wrong
M4	Speedup measurement (SC3)	Architecture delivers on its promise	GPU underutilized, bottleneck elsewhere
Phases
Phase 0: Project Scaffold
Goal: Compiling Rust workspace with CUDA build support. No solver logic yet.
Steps:
1. 
Create workspace Cargo.toml with two members: solver-core, solver-cli
2. 
Create solver-core/Cargo.toml with dependencies:
- 
cudarc = "0.19" with cuda-13020 and nvrtc features (same as existing)
- 
serde/serde_json for JSON
- 
regex for range parsing
- 
rayon for parallel CPU ops during tree construction
3. 
Create solver-cli/Cargo.toml with dependencies:
- 
solver-core = { path = "../solver-core" }
- 
clap, rusqlite, sha2, chrono, anyhow (same as existing)
4. 
Port build.rs from existing project (nvcc → PTX, gpu_arch(), find_nvcc(), find_msvc_bindir()) — this is proven infrastructure that handles CUDA compilation on Windows
5. 
Create empty lib.rs, main.rs, verify cargo build succeeds (without CUDA feature)
6. 
Verify cargo build --features cuda succeeds (compiles empty .cu stub)
Carry-over: build.rs structure (adapt directly — it's build infrastructure, not solver logic)
Duration estimate: Small
Checkpoint: cargo build --features cuda succeeds, PTX file generated
Phase 1: Core Data Types (CPU)
Goal: All CPU-side types defined. No solver logic, no GPU code. These are the types that everything else builds on.
Steps:
1. 
tree/action.rs — Adapt from action_tree.rs lines 19-55:
- 
Action enum: Fold, Check, Call, Bet(i32), Raise(i32), AllIn(i32), Chance(u8) (card index, not Card struct — flat representation)
- 
BoardState enum: Flop=0, Turn=1, River=2
- 
TreeConfig struct — key change from 2-player: replace [BetSizeOptions; 2] with BetSizeOptions (same options for all players — Pluribus uses position-independent bet sizes). Add num_players: u8, positions: Vec<Position> (acting order), starting_stacks: Vec<i32> (one per player), blinds: Vec<i32> (one per player, 0 for non-blind positions)
- 
BetSizeOptions, DonkSizeOptions — adapt directly from existing (these are player-count-agnostic)
2. 
tree/mod.rs — Flat tree types:
- 
FlatNode struct (repr(C) for GPU compatibility):
struct FlatNode {
    node_type: u8,      // 0=terminal, 1=chance, 2=player_action
    player_id: u8,      // acting player (for player_action nodes)
    board_state: u8,    // BoardState as u8
    num_children: u16,  // number of children
    children_start: u32, // index into children array
    amount: i32,        // bet amount / pot amount
    action_label: u8,   // Action variant as u8 (for strategy extraction)
}
- 
FlatTree struct:
struct FlatTree {
    nodes: Vec<FlatNode>,
    children: Vec<u32>,      // flat array of child node indices
    num_players: u8,
    starting_pot: i32,
    starting_stacks: Vec<i32>,
    // Precomputed for GPU upload:
    terminal_offsets: Vec<u32>,  // node indices of terminals
    chance_offsets: Vec<u32>,    // node indices of chance nodes
    player_offsets: Vec<u32>,    // node indices of player decision nodes
}
- 
Methods: node_children(i) -> &[u32], num_nodes(), gpu_bytes() (total bytes for upload)
3. 
hand/table.rs — Transfer directly from hand_table.rs:
- 
HAND_TABLE: [i32; 4824] — copy the entire constant as-is
- 
No changes needed, this is player-count-agnostic
4. 
hand/mod.rs — Transfer directly from hand.rs:
- 
Hand struct (cards: [usize; 7], num_cards: usize)
- 
Hand::evaluate() -> u16 — binary search into HAND_TABLE
- 
Hand::evaluate_internal() -> i32 — raw evaluation
- 
Helper functions: keep_n_msb, find_straight
- 
No changes needed
5. 
range.rs — Adapt from range.rs:
- 
Range struct: data: [f32; 1326] — same layout
- 
Range::new(), Range::ones(), Range::from_raw_data()
- 
FromStr parser for PioSOLVER notation
- 
Weight accessors: get_weight_by_cards(), set_weight_by_cards()
- 
No 2-player assumptions exist in Range — transfer directly
- 
Consumer uses Vec<Range> for N players
6. 
card.rs — New file, N-player from scratch:
- 
Card type (u8 index 0-51, or explicit struct with suit/rank)
- 
Board representation: BoardState + dealt cards
- 
Dead card masking from hero's hole cards + board
- 
Valid hand index computation: for each player, which of the 1326 two-card combos don't conflict with board cards and other players' blockers
- 
Hand strength lookup: given board + player hand, return strength rank
- 
Key difference from existing: replace [Vec<u16>; 2] patterns with Vec<Vec<u16>> indexed by player
Carry-over: Action enum (adapt), BoardState (adapt), BetSizeOptions (adapt), HAND_TABLE (direct), Hand (direct), Range (direct)
Reject: [T; 2] arrays, PLAYER_OOP/PLAYER_IP constants, player ^ 1 XOR
Duration estimate: Medium
Checkpoint: All types defined, cargo check passes, no solver logic yet
Phase 2: N-Player Action Tree Builder — MILESTONE 1
Goal: Build correct N-player postflop action trees, output as FlatTree. This is the first real test — if the flat representation works with N players, the foundation is solid.
Steps:
1. 
tree/builder.rs — Adapt from action_tree.rs:
Core logic to carry over (bet-size-to-action enumeration, all-in capping, round transitions):
- 
push_actions() logic (lines 526-730) — the bet size → action enumeration
- 
Round completion detection (has everyone acted with equalized bets?)
- 
All-in threshold logic (add_allin_threshold, force_allin_threshold)
- 
Merging threshold logic
Core logic to rewrite for N-player:
- 
Acting order: Replace player ^ 1 with next_to_act(current_player, round, has_folded, num_players). In N-player poker: preflop starts left of BB, postflop starts left of dealer. Action proceeds clockwise through non-folded players.
- 
Blind posting: Replace OOP/IP with position-based blind posting. SB posts small blind, BB posts big blind. Positions come from TreeConfig::positions.
- 
Pot tracking: Replace stack: [i32; 2] with stacks: Vec<i32> (one per player). Track each player's contribution separately.
- 
Side pots: When a player goes all-in for less than the current bet, create side pots. For the initial build: track the all-in amount per player and compute side pots at terminal nodes during evaluation. The tree builder marks all-in actions but does not split the tree — side pot computation happens at payoff time.
- 
Round completion: A betting round ends when all non-folded players have acted and have equal bets in the current round. Replace the simple oop_call_flag with round_actions: Vec<u8> tracking who has acted this round.
- 
After-check flow: Replace the match player { PLAYER_OOP => opponent, _ => player_after_call } pattern with: after the last-to-act player checks, transition to chance node (next street) or terminal (showdown if river). The "last to act" is determined by position order.
- 
After-call flow: A call that equalizes all bets ends the round → chance node or terminal.
- 
Fold handling: When a player folds, they're marked as folded for the rest of the hand. If only one player remains, terminal node (fold win).
Output: FlatTree with all nodes numbered, children stored as index ranges.
2. 
Validation tests:
- 
Build a 2-player tree (same config as existing solver), compare node count and topology
- 
Build a 3-player tree, verify:
- 
SB and BB post correct blinds
- 
Action order is correct (UTG/SB/BB or BTN/SB/BB depending on positions)
- 
Round transitions happen when all non-folded players have equalized
- 
All-in creates correct terminal/side-pot structures
- 
Build a 6-player tree (min bluffs, single bet size), verify it completes without error
- 
Test action history navigation: given a postflop_action_sequence, find the correct subtree root in the flat tree
Key risk: The N-player betting round logic is the most complex piece. The existing 2-player logic has implicit assumptions everywhere (OOP always acts first on flop, IP always acts second). N-player has variable acting order, variable number of active players, and side pots.
Checkpoint: cargo test passes for tree builder. 2-player tree topology matches old solver. 3-player tree builds without error and has correct action order.
Phase 3: GPU MCCFR Kernel — MILESTONE 2
Goal: Run one MCCFR iteration on a Kuhn or Leduc test game and verify output is meaningful. This is where we find out if GPU-resident state, atomic regret updates, and trajectory parallelism compose correctly.
Why Kuhn/Leduc first: These games have known exact equilibria. Kuhn poker has ~5-10 nodes. Leduc has ~1,000 nodes. Both fit in a single GPU warp. If the kernel produces correct results on these, the algorithm is right; scaling to full poker is an engineering problem, not an algorithmic one.
Steps:
1. 
solver/state.rs — Solver state buffers:
struct SolverState {
    num_infosets: usize,         // number of player decision nodes × (1 per infoset)
    max_actions: usize,          // max actions per infoset
    // SoA GPU buffers:
    regrets: GpuBuffer<f32>,     // [num_infosets * max_actions]
    avg_strategy: GpuBuffer<f32>, // [num_infosets * max_actions]
    // CPU mirror for initialization:
    initial_ranges: Vec<Range>,  // one per player
}
- 
SolverState::new(tree, ranges) — allocate GPU buffers, upload initial regrets (zeros)
- 
SolverState::read_strategy() — download avg_strategy from GPU, normalize
2. 
gpu/mod.rs — Adapt from cuda/mod.rs:
- 
CudaContext singleton (Arc<CudaStream>)
- 
CudaContext::init() — same pattern as existing
3. 
gpu/kernel.rs — PTX loading:
- 
Load mccfr.ptx from OUT_DIR
- 
Get kernel function by name
- 
Launch config: (B + 255) / 256 blocks, 256 threads per block
4. 
gpu/kernels/mccfr.cu — The core MCCFR kernel:
// Single kernel: one thread = one MCCFR trajectory
__global__ void k_mccfr_iter(
    // Tree structure (read-only)
    const FlatNode* nodes,
    const uint32_t* children,
    const uint32_t* terminal_indices,
    // Solver state (read-write, atomic)
    float* regrets,           // [num_infosets * max_actions]
    float* avg_strategy,      // [num_infosets * max_actions]
    // Game data (read-only)
    const float* ranges,      // [num_players * 1326]
    const int32_t* hand_table, // [4824]
    const uint8_t* board_cards, // [5]
    // Parameters
    int num_nodes,
    int num_players,
    int traverser,            // which player is traversing
    int max_actions,
    float regret_floor,
    int iteration,            // for linear weighting
    // RNG state
    uint64_t seed
);
Per-thread logic:
- 
Initialize curand from seed + thread_id
- 
Walk the tree from root:
- 
Terminal node: Evaluate hand strengths for all players, compute payoffs. For Kuhn/Leduc: direct card comparison. For full poker: binary search into HAND_TABLE.
- 
Chance node: Sample one board card (for full poker). For Kuhn/Leduc: sample one opponent card from remaining deck.
- 
Player decision node (traverser): Explore ALL actions. For each child, recurse. Compute CFV per action. Compute regret = CFV(action) - CFV(average). atomicAdd(®rets[infoset * max_actions + a], regret). Also update avg_strategy.
- 
Player decision node (opponent): Compute strategy from regret matching on positive regrets. Sample ONE action. Recurse into sampled child. Multiply reach probability.
- 
The walk is recursive in logic but implemented iteratively on GPU (stack-allocated traversal state, max depth = tree depth ≈ 20-40 nodes)
5. 
solver/mod.rs — Solver public API:
pub fn solve(tree: &FlatTree, ranges: &[Range], config: &SolverConfig) -> SolveResult {
    let state = SolverState::new(tree, ranges);
    let ctx = CudaContext::init();
    let start = Instant::now();
    let mut iteration = 0;
    while start.elapsed() < config.time_budget {
        for traverser in 0..tree.num_players {
            launch_mccfr_kernel(&ctx, &state, tree, traverser, iteration);
            iteration += 1;
        }
    }
    let strategy = state.read_strategy();
    SolveResult { strategy, iterations: iteration, elapsed: start.elapsed() }
}
6. 
Kuhn test (tests/kuhn.rs):
- 
Build Kuhn tree manually (3 cards, 2 players, check/bet structure)
- 
Convert to FlatTree
- 
Run solver for N iterations
- 
Compare strategy at each decision point to known Nash equilibrium:
- 
Player 1 (OOP): bet with King always, bet with Jack 1/3 of time, check with Queen
- 
Player 2 (IP) facing bet: call with King always, call with Queen 1/3 of time, fold Jack
- 
This is the first correctness gate: if the GPU kernel converges to known equilibrium, the MCCFR implementation is correct.
7. 
Leduc test (tests/leduc.rs):
- 
Build Leduc tree (6-card deck, 2 suits × 3 ranks, 2 betting rounds)
- 
Convert to FlatTree
- 
Run solver, compare to known Leduc equilibrium values from literature
- 
Test both 2-player and 3-player Leduc (3-player Leduc equilibrium is known)
Key risk: The GPU kernel is the highest-risk component. It must:
- 
Handle tree traversal without recursion (GPU stack is limited)
- 
Do correct regret matching (positive regret / sum, with uniform fallback)
- 
Do correct CFV propagation (reach probabilities multiplied correctly)
- 
Handle atomic adds without data corruption
- 
Use curand for sampling (opponent actions, chance outcomes)
Mitigation: Build a CPU reference implementation first (same logic, no atomics). Verify CPU version converges on Kuhn. Then port to GPU. If GPU diverges from CPU, the diff is isolated to the GPU-specific parts (atomics, RNG, memory layout).
Checkpoint: Kuhn poker converges to within 1% of known equilibrium. Leduc converges to within 5% of known equilibrium. Both on GPU.
Phase 4: Full Poker Hand Evaluation
Goal: Extend the kernel to handle full Texas hold'em hand evaluation (7-card hands, 1326 possible holdings per player, showdowns against multiple opponents).
Steps:
1. 
Hand evaluation in the kernel:
- 
Upload HAND_TABLE to GPU constant memory (4824 × 4 bytes = ~19 KB — fits in constant memory)
- 
Hand::evaluate() ported to CUDA: build the 7-card hand encoding, binary search into HAND_TABLE
- 
Terminal node evaluation: for each non-folded player's possible holdings, evaluate hand strength, compare across all active players, compute payoffs with side pot logic
2. 
Multiplayer showdown:
- 
Replace the existing 2-player binary win/lose comparison with N-player ranking
- 
For each terminal node: identify which players are folded, compute main pot and side pots, distribute winnings
- 
This is the most complex payoff logic — side pots with multiple all-in players require careful math
3. 
Valid hand indices:
- 
For each board state (flop, turn, river), precompute which of the 1326 holdings are valid (no conflict with board cards)
- 
Upload as GPU buffer per board state
- 
Kernel uses these to iterate only valid holdings during showdown evaluation
4. 
Integration test:
- 
Build a simple 2-player flop-only tree (SPR=2, one bet size)
- 
Run solver for 100 iterations
- 
Verify output is not degenerate (not all-fold or all-check)
- 
Compare to old solver's output at 100 iterations on same config
Checkpoint: 2-player flop subgame produces non-degenerate strategy. Hand evaluation matches CPU implementation.
Phase 5: CLI Integration + Action History — MILESTONE 3 PREP
Goal: Wire up the solver to the JSON API so it can be called from the bot. Handle action history, strategy extraction, and JSON response.
Steps:
1. 
solver-cli/src/config.rs — Adapt SolveRequest:
- 
Same schema as existing (copy + modify)
- 
Remove synthesis_mode requirement for opponents.len() > 1
- 
Add solver_version field for cache compatibility
2. 
solver-cli/src/solver.rs — Solve dispatch:
- 
Parse SolveRequest → extract ranges for all players (hero + all opponents)
- 
Build TreeConfig from solve_config bet sizes
- 
Parse preflop_action_sequence to determine starting ranges and pot/stack state
- 
Parse postflop_action_sequence to determine subtree root
- 
Build FlatTree from TreeConfig
- 
Call solver::solve(tree, ranges, config)
- 
Extract strategy for hero's specific hand from the avg_strategy buffer
- 
Build JSON response
3. 
solver-cli/src/cache.rs — SQLite cache:
- 
Adapt from existing cache logic
- 
Add solver_version to cache key (as per North Star §8)
- 
Old entries coexist, age out via eviction
4. 
solver-cli/src/range.rs — Opponent range construction:
- 
Copy from existing range.rs in surstromming-solver-cli
- 
This is the pool-stats → Range pipeline — independent of solver, transfers as-is
- 
Now called once per opponent (producing Vec<Range>)
5. 
Action history → subtree root:
- 
Walk the flat tree matching postflop_action_sequence entries to actions
- 
Return the node index of the subtree root
- 
The solver builds the tree from the flop forward, then we solve only the subtree from this root
6. 
Timeout handling:
- 
--timeout-seconds flag from CLI
- 
Solver checks elapsed time between iterations (already in solver/mod.rs)
- 
Return best strategy available when timeout hits
Checkpoint: solver-cli --input request.json --output result.json works end-to-end on a 2-player flop spot. JSON response contains action frequencies for hero's hand.
Phase 6: HU Equivalence Test — MILESTONE 3
Goal: Validate SC1 — new solver converges to within 5% exploitability of old solver on identical 2-player configs.
Steps:
1. 
Test matrix (3-5 configs spanning difficulty):
- 
HU SPR=2, 1 bet size (easy)
- 
HU SPR=5, 2 bet sizes (medium)
- 
HU SPR=10, 3 bet sizes (hard)
- 
Different board textures (connected, paired, monotone)
2. 
Method:
- 
Run old solver on each config for 200 iterations, record exploitability
- 
Run new solver on identical config (same tree, same ranges, same iteration count), record exploitability
- 
Compare: new solver exploitability must be ≤ 1.05 × old solver exploitability
3. 
Best-response computation:
- 
Implement a best-response exploiter for 2-player (walk tree, compute best response value for each player)
- 
This is needed to measure exploitability — it's a separate tool, not part of the solver
- 
Adapt from existing game/tests.rs best-response code (transfer as pattern)
4. 
If SC1 fails:
- 
Compare per-node strategies between old and new solver to find divergence
- 
Check regret values at divergent nodes
- 
Common causes: incorrect CFV propagation, incorrect reach probability calculation, RNG bias in sampling
Checkpoint: All test configs pass 5% threshold. SC1 met.
Phase 7: Multiway Validation
Goal: Validate SC2 — solver handles 3-6 player spots natively.
Steps:
1. 
3-player test:
- 
Build 3-player flop tree (BTN/SB/BB positions, 2 bet sizes)
- 
Run solver for time budget
- 
Verify all 3 players have meaningful strategies (not degenerate)
- 
Compare against old multiway synthesis approach: new solver should produce different (hopefully better) strategies since it accounts for multiway interactions
2. 
6-player test:
- 
Build 6-player tree with minimal bet sizes
- 
Verify it fits in 8GB VRAM (SC5)
- 
Run solver for time budget
- 
Verify solve time < 10 seconds (SC6)
3. 
Side pot tests:
- 
Create configs with short stacks (all-in scenarios)
- 
Verify payoff calculation handles side pots correctly
- 
Compare against manual calculation for simple cases
Checkpoint: 3-player and 6-player spots produce meaningful strategies. VRAM fits. Solve time within budget.
Phase 8: Performance Optimization — MILESTONE 4
Goal: Validate SC3 (5x speedup) and SC6 (solve time budgets). Profile and optimize.
Steps:
1. 
Benchmark suite:
- 
HU SPR=5, 2 bet sizes
- 
3-player SPR=5, 2 bet sizes
- 
3-player SPR=10, 3 bet sizes (hardest production config)
- 
Measure: wall-clock time, iterations/second, GPU utilization
2. 
Profiling:
- 
nvprof or nsys to measure kernel occupancy, memory throughput, warp divergence
- 
Identify bottlenecks: is it the tree traversal, hand evaluation, atomic updates, or memory bandwidth?
3. 
Likely optimizations (only if needed):
- 
Batch size tuning: Q5 says start at B=10,000. May need to increase for small trees or decrease for large trees.
- 
Shared memory for hand table: If constant memory becomes a bottleneck, copy HAND_TABLE to shared memory per block
- 
Warp-level reduction: For terminal node evaluation (summing over opponent holdings), use warp shuffle instructions instead of atomic adds
- 
Tree compression: If VRAM pressure, compress the FlatNode array (currently ~24 bytes/node; can reduce to 16 bytes by packing fields)
4. 
Regression testing: After each optimization, re-run Kuhn/Leduc tests to verify correctness is preserved
Checkpoint: SC3 met (5x speedup on HU). SC6 met (<10s for 3-player, <5s for HU). SC5 met (<8GB VRAM).
Summary: Phase → Milestone Mapping
Phase	Description	Milestone	Key Deliverable
0	Project scaffold	—	cargo build --features cuda works
1	Core data types (CPU)	—	FlatNode, FlatTree, Range, Hand, Card types
2	N-player tree builder	M1	Correct 3-player flat tree with side pots
3	GPU MCCFR kernel + Kuhn/Leduc	M2	Converges to known equilibria on GPU
4	Full poker hand evaluation	—	2-player flop subgame produces real strategy
5	CLI integration + action history	—	JSON-in/JSON-out works end-to-end
6	HU equivalence testing	M3	SC1: within 5% exploitability of old solver
7	Multiway validation	—	SC2: 3-6 player spots work natively
8	Performance optimization	M4	SC3: 5x speedup. SC6: time budgets met
Carry-over Summary
Component	Source	Action	Phase
Action enum	action_tree.rs:19-44	Adapt (add Chance variant)	1
BetSizeOptions	action_tree.rs	Adapt directly	1
HAND_TABLE	hand_table.rs	Transfer directly	1
Hand::evaluate()	hand.rs	Transfer directly	1
Range struct	range.rs	Transfer directly	1
Tree building logic	action_tree.rs:486-730	Adapt for N-player	2
build.rs	build.rs	Transfer directly	0
CudaContext singleton	cuda/mod.rs	Adapt	3
SolveRequest schema	config.rs	Adapt	5
Opponent range construction	range.rs (CLI)	Transfer directly	5
Cache logic	main.rs	Adapt + solver_version	5
Kuhn test pattern	tests/kuhn.rs	New N-player Kuhn game	3
Leduc test pattern	tests/leduc.rs	New N-player Leduc game	3
Best-response code	game/tests.rs	Transfer as pattern	6
Reject Summary
Component	Source	Reason
solve_recursive tree walk	game/mod.rs, cuda/solve_gpu.rs	Replaced by GPU trajectory parallelism
[T; 2] arrays	Throughout game/mod.rs, card.rs, solver_data.rs	N-player uses Vec indexed by player_id
PLAYER_OOP/PLAYER_IP constants	action_tree.rs:8-9	Replaced by position-based acting order
player ^ 1 XOR	action_tree.rs:528,792, solve_gpu.rs:46	Replaced by next_to_act()
DCFR alpha/beta/gamma	sliceops.cu:171-192	Replaced by Linear CFR weighting
GpuSolverContext/GpuStorage	cuda/gpu_context.rs, gpu_storage.rs	Replaced by single-kernel architecture
Per-node kernel dispatch	slice_dispatch.rs, eval_dispatch.rs	Replaced by one kernel per iteration
All existing CUDA kernels	sliceops.cu, evaluate.cu	Replaced by single mccfr.cu
Multiway synthesis	multiway.rs	Replaced by native N-player solving