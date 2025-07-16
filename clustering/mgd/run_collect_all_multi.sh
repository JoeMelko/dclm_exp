#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_collect_all_multi.sh  (persistent GPU workers)
# -----------------------------------------------------------------------------
# Variant of `run_collect_all.sh` that starts **one long-running Python process
# per GPU**, each of which sequentially processes a *contiguous slice* of the
# dataset directories.  This avoids the costly repeated model initialisation
# overhead when many sub-directories must be handled.
# -----------------------------------------------------------------------------
# Usage:
#   run_collect_all_multi.sh PARENT_DIR [flags forwarded to collect_cosine_sim_multi.py]
#
# Example:
#   ./run_collect_all_multi.sh /path/to/parent_dir \
#       --target-vector vec.npy \
#       --uuid 123e4567-e89b-12d3-a456-426614174000 \
#       --out-dir clustering/mgd \
#       --iter 0 --lora-rank 128 --num-blocks 8 --max-items 64
# -----------------------------------------------------------------------------
set -euo pipefail

# ----------------------------- configuration ---------------------------------
GPU_IDS=(0 1 2 3 4 5 6 7)   # adapt to your machine
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/collect_cosine_sim_multi.py"
# -----------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PARENT_DIR [extra flags for collect_cosine_sim_multi.py]" >&2
  exit 1
fi

PARENT_DIR="$1"; shift
[[ -d "$PARENT_DIR" ]] || { echo "ERROR: not a directory → $PARENT_DIR" >&2; exit 1; }

# Generate sorted list of sub-directories once so all workers agree on indices
mapfile -t SUBDIRS < <(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
TOTAL=${#SUBDIRS[@]}
if [[ $TOTAL -eq 0 ]]; then
  echo "No sub-directories found in $PARENT_DIR." >&2
  exit 0
fi

echo "Found $TOTAL sub-directories to process." >&2

NUM_GPUS=${#GPU_IDS[@]}
CHUNK_SIZE=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))  # ceil division

declare -a PIDS
for idx in "${!GPU_IDS[@]}"; do
  START=$(( idx * CHUNK_SIZE ))
  END=$(( (idx + 1) * CHUNK_SIZE - 1 ))
  [[ $START -ge $TOTAL ]] && continue  # nothing left for this GPU
  [[ $END -ge $TOTAL ]] && END=$(( TOTAL - 1 ))

  echo "[GPU ${GPU_IDS[$idx]}] handling indices $START – $END" >&2
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$idx]}" \
    python "$PY_SCRIPT" \
      --parent-dir "$PARENT_DIR" \
      --start-idx "$START" --end-idx "$END" \
      "$@" &
  PIDS[$idx]=$!
done

# Wait for all background jobs
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "All workers completed successfully." 