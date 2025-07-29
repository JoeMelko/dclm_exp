#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_full_workflow.sh
# -----------------------------------------------------------------------------
# End-to-end driver that executes the full MGD evaluation pipeline in five
# stages:
#   1) launch_collect_features.sh – extract raw gradient features to a shared memmap
#   2) hessian.py                – compute regularised Fisher whitening blocks
#   3) launch_get_target.sh      – aggregate unwhitened target direction vectors
#   4) condition.py              – whiten & L2-normalise the aggregate target
#   5) run_collect_all_multi.sh  – per-dataset cosine-similarity evaluation via persistent workers
#
# Each stage runs **synchronously** and the script aborts immediately on any
# error (set -e).  Minimal sanity checks ensure that every stage produced the
# artefacts required by subsequent steps before moving on.
#
# Usage (example – adjust the variables below as needed):
#   ./run_full_workflow.sh
#
# The configuration section at the top contains *all* paths & hyper-parameters
# you may want to customise for your environment.
# -----------------------------------------------------------------------------

set -euo pipefail

##################################
#  ⁕  USER CONFIGURATION START  ⁕  
##################################
# -------- General paths -------- #
PROJECT_DIR="$(dirname "$(realpath "$0")")"   # this script lives in clustering/mgd
DATA_PARENT_DIR="/home/jmelko/400m_tok"             # <- directory whose *sub-dirs* are processed by run_collect_all_multi.sh
WDS_DIR="/home/jmelko/mgd/four_3_dir/d_dot2"                     # <- tokenised WebDataset root used by later stages
OH_DIR="/home/jmelko/openhermes_tok"

# -------- Model checkpoint ----- #
# Exactly *one* of UUID or CKPT must be defined.  Leave the unused one empty.
MODEL_UUID="four_iter3_ckpt4-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=124-tokens=8232325120"         # Datacomp-LM / Open-LM run UUID (preferred)
MODEL_CKPT=""         # HuggingFace checkpoint path or hub ID (fallback)

# -------- LoRA dimensions ------ #
LORA_RANK=128
NUM_BLOCKS=8

# -------- Hessian computation ---#
GRAD_MEMMAP="/home/jmelko/mgd/four_4_dir/grads.mmap"     # mem-mapped gradient features (input for hessian.py)
COND_TARGET=1e4                        # target condition number after ridge regularisation
WHITENERS_PATH="/home/jmelko/mgd/four_4_dir/whiteners.npy"        # output file written by hessian.py

# -------- Target aggregation ---- #
NUM_GPUS=8                             # number of GPU workers launched by launch_get_target.sh
SHARDS_PER_GPU=15                      # chunk-size handed to each get_target worker
SHARD_SIZE=8192                        # samples per shard
TARGET_DIR="/home/jmelko/mgd/four_4_dir/targets"                  # directory that will contain sum_*.npy (created automatically)

# -------- Condition / whitening --#
WHITENED_TARGET="/home/jmelko/mgd/four_4_dir/hw_target.npy" # 1-D, unit-norm vector consumed by run_collect_all_multi.sh

# --- Feature collection / mmap -- #
FEATURE_MEMMAP="/home/jmelko/mgd/four_4_dir/grads.mmap"         # destination memmap initialised by create_mmap_features.py
CF_SHARDS_PER_GPU=2048                   # shards per GPU for collect_features
CF_SHARD_SIZE=64

# -------- Cosine-similarity eval ---- #
ITER=4                               # iteration index forwarded to collect_cosine_sim_multi.py
OUT_DIR="/home/jmelko/mgd/four_4_dir"            # base directory where similarity results will be written
MAX_ITEMS=64                        # maximum number of batches processed per dataset
HESSIAN_DTYPE="fp16"                # storage dtype of the gradient memmap
CLIP_PERCENTILE=99.9                # clipping threshold percentile
COND_DTYPE="fp32"                  # output dtype for condition.py
##################################
#  ⁕  USER  CONFIGURATION END   ⁕  
##################################

# Helper: ensure that exactly one of MODEL_UUID / MODEL_CKPT is set.
if { [[ -z "${MODEL_UUID}" ]] && [[ -z "${MODEL_CKPT}" ]]; } || { [[ -n "${MODEL_UUID}" ]] && [[ -n "${MODEL_CKPT}" ]]; }; then
  echo "ERROR: Exactly one of MODEL_UUID or MODEL_CKPT must be specified." >&2
  exit 1
fi

# Collect_cosine_sim_multi.py currently supports *only* --uuid.  Abort early if the user
# attempted to configure the workflow via MODEL_CKPT which would fail downstream.
if [[ -z "${MODEL_UUID}" ]]; then
  echo "ERROR: run_collect_all_multi.sh requires MODEL_UUID to be set (ckpt-based checkpoints are not supported)." >&2
  exit 1
fi

# Convenience array for model selection flags
MODEL_FLAGS=()
if [[ -n "${MODEL_UUID}" ]]; then
  MODEL_FLAGS+=(--uuid "${MODEL_UUID}")
else
  MODEL_FLAGS+=(--ckpt "${MODEL_CKPT}")
fi

# -----------------------------------------------------------------------------
# 1) Feature collection (launch_collect_features.sh)
# -----------------------------------------------------------------------------

echo "[workflow] Stage 1/5 – launch_collect_features.sh"

"${PROJECT_DIR}/launch_collect_features.sh" \
    --wds-dir "${WDS_DIR}" \
    --shards-per-gpu "${CF_SHARDS_PER_GPU}" \
    --shard-size "${CF_SHARD_SIZE}" \
    --out "${FEATURE_MEMMAP}" \
    --lora-rank "${LORA_RANK}" \
    --num-blocks "${NUM_BLOCKS}" \
    "${MODEL_FLAGS[@]}"

# Verify memmap exists after feature collection
if [[ ! -f "${FEATURE_MEMMAP}" ]]; then
  echo "ERROR: Expected memmap '${FEATURE_MEMMAP}' not found after feature collection." >&2
  exit 1
fi

echo "[workflow] Stage 1 complete – feature memmap stored at ${FEATURE_MEMMAP}."

# -----------------------------------------------------------------------------
# 2) Hessian computation (hessian.py)
# -----------------------------------------------------------------------------
# Update GRAD_MEMMAP to FEATURE_MEMMAP by default if user didn't override
# If GRAD_MEMMAP is still the placeholder path, fall back to the memmap from Stage 1
if [[ "${GRAD_MEMMAP}" == "/path/to/grads.mmap" ]]; then
  GRAD_MEMMAP_RESOLVED="${FEATURE_MEMMAP}"
else
  GRAD_MEMMAP_RESOLVED="${GRAD_MEMMAP}"
fi

echo "[workflow] Stage 2/5 – hessian.py"

python "${PROJECT_DIR}/hessian.py" \
  --mmap-path "${GRAD_MEMMAP_RESOLVED}" \
  --rank "${LORA_RANK}" \
  --num-blocks "${NUM_BLOCKS}" \
  --dtype "${HESSIAN_DTYPE}" \
  --cond "${COND_TARGET}" \
  --clip-percentile "${CLIP_PERCENTILE}" \
  --out-path "${WHITENERS_PATH}" \
  --verbose

# Sanity check: did the file appear?
if [[ ! -f "${WHITENERS_PATH}" ]]; then
  echo "ERROR: Hessian stage did not produce '${WHITENERS_PATH}'." >&2
  exit 1
fi

echo "[workflow] Stage 2 complete – whiteners saved to ${WHITENERS_PATH}."

# -----------------------------------------------------------------------------
# 3) Target aggregation (launch_get_target.sh)
# -----------------------------------------------------------------------------

echo "[workflow] Stage 3/5 – launch_get_target.sh"

mkdir -p "${TARGET_DIR}"

"${PROJECT_DIR}/launch_get_target.sh" \
    --wds-dir "${OH_DIR}" \
    --shards-per-gpu "${SHARDS_PER_GPU}" \
    --shard-size "${SHARD_SIZE}" \
    --out-dir "${TARGET_DIR}" \
    --lora-rank "${LORA_RANK}" \
    --num-blocks "${NUM_BLOCKS}" \
    "${MODEL_FLAGS[@]}"

# Verify that every worker produced its partial sum
for ((g=0; g<NUM_GPUS; g++)); do
  f="${TARGET_DIR}/sum_${g}.npy"
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Expected file '$f' not found after launch_get_target stage." >&2
    exit 1
  fi
done

echo "[workflow] Stage 3 complete – all sum_*.npy present in ${TARGET_DIR}."

# -----------------------------------------------------------------------------
# 4) Condition & whitening (condition.py)
# -----------------------------------------------------------------------------

echo "[workflow] Stage 4/5 – condition.py"

python "${PROJECT_DIR}/condition.py" \
  --num-gpus "${NUM_GPUS}" \
  --root-dir "${TARGET_DIR}" \
  --whiteners-path "${WHITENERS_PATH}" \
  --out-path "${WHITENED_TARGET}" \
  --dtype "${COND_DTYPE}"

if [[ ! -f "${WHITENED_TARGET}" ]]; then
  echo "ERROR: condition.py did not create '${WHITENED_TARGET}'." >&2
  exit 1
fi

echo "[workflow] Stage 4 complete – whitened target saved to ${WHITENED_TARGET}."

# -----------------------------------------------------------------------------
# 5) Cosine-similarity collection (run_collect_all_multi.sh)
# -----------------------------------------------------------------------------

echo "[workflow] Stage 5/5 – run_collect_all_multi.sh"

"${PROJECT_DIR}/run_collect_all_multi.sh" "${DATA_PARENT_DIR}" \
    --target-vector "${WHITENED_TARGET}" \
    "${MODEL_FLAGS[@]}" \
    --lora-rank "${LORA_RANK}" \
    --num-blocks "${NUM_BLOCKS}" \
    --iter "${ITER}" \
    --out-dir "${OUT_DIR}" \
    --max-items "${MAX_ITEMS}"

echo "[workflow] Stage 5 complete – cosine similarity collection finished."

echo "[workflow] All stages finished successfully. ✨" 