#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_collect_all.sh  (simple PID-based GPU queue)
# -----------------------------------------------------------------------------
# For every first-level subdirectory of PARENT_DIR, launch
#     collect_cosine_sim_dc.py --wds-dir <subdir> <extra flags>
# while keeping at most one job per GPU listed in GPU_IDS.
#
# Implementation strategy – keep it *simple*:
#   • Maintain one PID slot per GPU in GPU_PIDS[].
#   • In a loop, look for a GPU whose PID is unset or whose process has ended.
#   • Immediately dispatch the next dataset on that GPU.
#   • Repeat until every dataset has been started, then `wait` for remaining
#     PIDs to finish.
#
# Advantages over the nvidia-smi variant:
#   • No polling of external utilities; only builtin Bash + `kill -0` checks.
#   • Clear 1-to-1 mapping of job ↔ GPU throughout its lifetime.
# -----------------------------------------------------------------------------
# Usage:
#   run_collect_all.sh PARENT_DIR [options_for_collect_cosine_sim_dc.py]
#
# Arguments:
#   PARENT_DIR   Directory whose first-level sub-directories will each be
#                processed by collect_cosine_sim_dc.py.
#   options      Any extra command-line flags are forwarded verbatim to
#                collect_cosine_sim_dc.py for every dataset.
#
# Example:
#   ./run_collect_all.sh /path/to/parent_dir \
#       --target-vector vec.npy \
#       --ckpt EleutherAI/pythia-1b \
#       --iter 0 \
#       --lora-rank 128 --num-blocks 8
#
# Forwarded arguments for collect_cosine_sim_dc.py:
#   --uuid UUID | --ckpt CKPT  (required, exactly one) model checkpoint source
#   --target-vector PATH   (required) .npy file containing flattened target
#                          vector (already whitened & L2-normalised)
#   --iter INT             iteration index added to output filename [default: 0]
#   --lora-rank INT        LoRA rank for adapters [default: 128]
#   --num-blocks INT       number of transformer blocks logged [default: 8]
#   --max-items INT        maximum number of batches processed per dataset before early stop [default: 500]
#   --out-dir DIR          directory where outputs are written; results for each --iter value
#                          are stored in a sub-directory named "iter_<iter>"
# -----------------------------------------------------------------------------
set -euo pipefail

# ----------------------------- configuration ---------------------------------
GPU_IDS=(0 1 2 3 4 5 6 7)   # change if your machine has a different layout
SLEEP_SEC=5                 # seconds to wait before re-checking GPU status
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_cosine_sim_dc.py"
# -----------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PARENT_DIR [extra flags for collect_cosine_sim_dc.py]" >&2
  exit 1
fi

PARENT_DIR="$1"; shift
[[ -d "$PARENT_DIR" ]] || { echo "ERROR: not a directory → $PARENT_DIR" >&2; exit 1; }

mapfile -t SUBDIRS < <(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
[[ ${#SUBDIRS[@]} -gt 0 ]] || { echo "No sub-directories found in $PARENT_DIR." >&2; exit 0; }

echo "Found ${#SUBDIRS[@]} sub-directories to process." >&2

# One PID slot per GPU (indexed the same as GPU_IDS)
declare -a GPU_PIDS
for _ in "${GPU_IDS[@]}"; do GPU_PIDS+=(""); done

next_idx=0
total=${#SUBDIRS[@]}

launch_job() {
  local gpu_idx=$1
  local wds_dir=$2
  shift 2               # drop positional params so only extra flags remain
  echo "[GPU ${GPU_IDS[$gpu_idx]}] Launching $wds_dir" >&2
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$gpu_idx]}" \
    python "$COLLECT_SCRIPT" --wds-dir "$wds_dir" "$@" &
  GPU_PIDS[$gpu_idx]=$!
}

# Dispatch loop ---------------------------------------------------------------
while :; do
  # Break if all datasets have been dispatched *and* no running jobs remain.
  running=false
  for pid in "${GPU_PIDS[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      running=true; break
    fi
  done
  if [[ $next_idx -ge $total && $running == false ]]; then
    break  # all work done
  fi

  # Try to launch work on any available GPU.
  for idx in "${!GPU_IDS[@]}"; do
    pid=${GPU_PIDS[$idx]}
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      # GPU free
      if [[ $next_idx -lt $total ]]; then
        launch_job "$idx" "${SUBDIRS[$next_idx]}" "$@"
        echo "after1"
        ((next_idx+=1))
      else
        GPU_PIDS[$idx]=""  # no more work, clear slot
      fi
    fi
  done

  sleep "$SLEEP_SEC"
done

wait  # wait for any stragglers (no-ops if none)
echo "All $total jobs completed successfully." 