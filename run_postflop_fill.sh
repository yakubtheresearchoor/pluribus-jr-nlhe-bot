#!/bin/zsh
# Production POSTFLOP BLUEPRINT FILL (v1, 2026-06-14): loop the per-family
# seam runner over the 72 width-0.25 SPR cells. Reach-weighted B (live-3/4 @
# B15, live-5 @ B8), 1×1 runout, 34 DCFR iters, Design1Collapsed, GPU-native.
# RESUMABLE: per-cell BP_OUT dirs, per-flop files skipped if banked — re-run
# to resume after interruption. Usage: ./run_postflop_fill.sh [BP_END]
#   BP_END=2 → 2-flop dry run (validate the loop); default 1755 = full fill.
set -u
ROOT=~/surstromming-solver-cuda
OUT_BASE=$ROOT/blueprint_out_v1
END=${1:-1755}
# STABLE copies (immune to dev rebuilds that clobber the metal binary).
export OUT_DIR=/tmp/bp_metal
RUNNER=/tmp/bp_metal/blueprint_runner
CELLS=$OUT_BASE/cells.txt
total=$(grep -c "^CELL live=" "$CELLS")
i=0
t_start=$(date +%s)
grep "^CELL live=" "$CELLS" | while read -r _ a b c d; do
  live=${a#live=}; commit=${b#commit=}; pot=${c#pot=}; bb=${d#b=}
  i=$((i+1))
  cell_dir="$OUT_BASE/live${live}_c${commit}_p${pot}_b${bb}"
  banked=$(ls "$cell_dir"/flop_*.bp 2>/dev/null | wc -l | tr -d ' ')
  el=$(( ($(date +%s) - t_start) / 60 ))
  echo "[$i/$total | ${el}min] live-$live c=$commit p=$pot B=$bb (banked=$banked/$END) → $cell_dir"
  if [ "$banked" -ge "$END" ]; then echo "  (complete, skip)"; continue; fi
  BP_OUT="$cell_dir" BP_GPU=1 BP_LIVE=$live BP_COMMIT=$commit BP_POT=$pot BP_B=$bb \
    BP_ITERS=34 BP_END=$END BP_THREADS=2 "$RUNNER" 2>&1 | grep -E "done in|all done|FAILED|panic" | tail -1
done
echo "FILL_COMPLETE ($(( ($(date +%s) - t_start) / 60 )) min total)"
