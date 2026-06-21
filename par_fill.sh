#!/bin/bash
# Parallel re-fill of REMAINING (incomplete) cells. Resume-safe: blueprint_runner
# skips already-banked flop_NNNN.bp, so re-running a partial cell finishes it.
set -u
ROOT=~/surstromming-solver-cuda
OUT_BASE=$ROOT/blueprint_out_v1
RUNNER=/tmp/bp_metal/blueprint_runner
PAR=${PAR:-6}
END=1755

# Build the to-do list: incomplete cells, SLOWEST (lowest commit) first so the
# CPU-bound deep-tree cells overlap maximally.
: > /tmp/todo.tsv
grep "^CELL live=" "$OUT_BASE/cells.txt" | while read -r _ a b c d; do
  live=${a#live=}; commit=${b#commit=}; pot=${c#pot=}; bb=${d#b=}
  cell_dir="$OUT_BASE/live${live}_c${commit}_p${pot}_b${bb}"
  banked=$(ls "$cell_dir"/flop_*.bp 2>/dev/null | wc -l | tr -d ' ')
  [ "$banked" -ge "$END" ] && continue
  printf "%s\t%s\t%s\t%s\t%s\n" "$commit" "$live" "$pot" "$bb" "$cell_dir"
done | sort -n > /tmp/todo.tsv
echo "PAR=$PAR | $(wc -l < /tmp/todo.tsv) incomplete cells to fill ($(date +%H:%M))"
cat /tmp/todo.tsv | sed 's#'"$OUT_BASE"'/##'

cat /tmp/todo.tsv | xargs -P "$PAR" -L 1 bash -c '
  commit=$1; live=$2; pot=$3; bb=$4; cell_dir=$5
  OUT_DIR=/tmp/bp_metal BP_OUT="$cell_dir" BP_GPU=1 BP_LIVE=$live BP_COMMIT=$commit \
    BP_POT=$pot BP_B=$bb BP_ITERS=34 BP_END=1755 BP_THREADS=2 \
    /tmp/bp_metal/blueprint_runner > "${cell_dir}.parlog" 2>&1
  echo "[done $(date +%H:%M)] live-$live c=$commit p=$pot"
' _
echo "PARALLEL_FILL_COMPLETE $(date +%H:%M)"
