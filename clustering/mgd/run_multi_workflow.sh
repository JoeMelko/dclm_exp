#!/usr/bin/env bash
set -euo pipefail

# Fill these parameters
UUIDS=(
  "iter0_1"
  "iter0_2"
  "iter0_4"
  "iter0_8"
)
TAGS=(
  "ckpt1"
  "ckpt2"
  "ckpt4"
  "ckpt8"
)
BASE_OUT="/mnt/eu/home/jmelko/curric/mgd_multi"
ITER=0
COUNTS_JSONS=(
  /mnt/eu/home/jmelko/curric/counts_subset0.json
  /mnt/eu/home/jmelko/curric/counts_subset0.json
  /mnt/eu/home/jmelko/curric/counts_subset0.json
  /mnt/eu/home/jmelko/curric/counts_subset0.json
)

# Optional per-run global shard reader offsets (one per UUID). If left empty,
# a default of 0 will be used for all runs.
CF_GLOBAL_SHARD_OFFSETS=()

# Optional overrides (leave empty to use defaults in run_full_workflow.sh)
DATA_PARENT_DIR="/mnt/eu/home/jmelko/curric/400m_tok"
WDS_DIR="/mnt/eu/home/jmelko/curric/baseline0_chunked/ready_to_train"
OH_DIR="/mnt/eu/home/jmelko/curric/openhermes_tok_new"
LORA_RANK=128
NUM_BLOCKS=8
NUM_GPUS=8
SHARDS_PER_GPU=15
SHARD_SIZE=8192
CF_SHARDS_PER_GPU=2048
CF_SHARD_SIZE=64
MAX_ITEMS=64
HESSIAN_DTYPE="fp16"
CLIP_PERCENTILE=99.9
COND_DTYPE="fp32"
COND_TARGET=1e4
STEP6_SCORES_DIR=""
UPDATE_LR=0.1
UPDATE_MAX_Z=5

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
WF="${SCRIPT_DIR}/run_full_workflow.sh"

if [[ ${#UUIDS[@]} -ne ${#TAGS[@]} || ${#UUIDS[@]} -ne ${#COUNTS_JSONS[@]} ]]; then
  echo "UUIDS, TAGS, and COUNTS_JSONS must have the same length" >&2; exit 1
fi
for p in "${COUNTS_JSONS[@]}"; do
  if [[ -z "$p" ]] || [[ ! -f "$p" ]]; then
    echo "Missing COUNTS_JSON: $p" >&2; exit 1
  fi
done

# If CF_GLOBAL_SHARD_OFFSETS not provided, default to zeros matching UUID count.
if [[ ${#CF_GLOBAL_SHARD_OFFSETS[@]} -eq 0 ]]; then
  for _ in "${UUIDS[@]}"; do CF_GLOBAL_SHARD_OFFSETS+=(0); done
fi
# Validate optional offsets length if provided
if [[ ${#UUIDS[@]} -ne ${#CF_GLOBAL_SHARD_OFFSETS[@]} ]]; then
  echo "UUIDS and CF_GLOBAL_SHARD_OFFSETS must have the same length" >&2; exit 1
fi

for i in "${!UUIDS[@]}"; do
  UUID="${UUIDS[$i]}"
  TAG="${TAGS[$i]}"
  RUN_DIR="${BASE_OUT}/${TAG}"
  echo "[multi] Running ${TAG:-$UUID} -> ${RUN_DIR}"
  mkdir -p "${RUN_DIR}/targets"
  MODEL_UUID="${UUID}" \
  OUT_DIR="${RUN_DIR}" \
  FEATURE_MEMMAP="${RUN_DIR}/grads.mmap" \
  GRAD_MEMMAP="${RUN_DIR}/grads.mmap" \
  WHITENERS_PATH="${RUN_DIR}/whiteners.npy" \
  TARGET_DIR="${RUN_DIR}/targets" \
  WHITENED_TARGET="${RUN_DIR}/hw_target.npy" \
  ITER="${ITER}" \
  COUNTS_JSON="${COUNTS_JSONS[$i]}" \
  CF_GLOBAL_SHARD_OFFSET="${CF_GLOBAL_SHARD_OFFSETS[$i]}" \
  DATA_PARENT_DIR="${DATA_PARENT_DIR}" \
  WDS_DIR="${WDS_DIR}" \
  OH_DIR="${OH_DIR}" \
  LORA_RANK="${LORA_RANK}" \
  NUM_BLOCKS="${NUM_BLOCKS}" \
  NUM_GPUS="${NUM_GPUS}" \
  SHARDS_PER_GPU="${SHARDS_PER_GPU}" \
  SHARD_SIZE="${SHARD_SIZE}" \
  CF_SHARDS_PER_GPU="${CF_SHARDS_PER_GPU}" \
  CF_SHARD_SIZE="${CF_SHARD_SIZE}" \
  MAX_ITEMS="${MAX_ITEMS}" \
  HESSIAN_DTYPE="${HESSIAN_DTYPE}" \
  CLIP_PERCENTILE="${CLIP_PERCENTILE}" \
  COND_DTYPE="${COND_DTYPE}" \
  COND_TARGET="${COND_TARGET}" \
  STEP6_SCORES_DIR="${STEP6_SCORES_DIR}" \
  UPDATE_LR="${UPDATE_LR}" \
  UPDATE_MAX_Z="${UPDATE_MAX_Z}" \
  "${WF}"
done


