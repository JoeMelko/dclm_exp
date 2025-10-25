#!/usr/bin/env bash
set -euo pipefail

# Fill these parameters
UUIDS=(
  # "uuid_a"
  # "uuid_b"
)
TAGS=(
  # "run_a"
  # "run_b"
)
BASE_OUT="/mnt/eu/home/jmelko/curric/mgd/multi"
ITER=2

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
WF="${SCRIPT_DIR}/run_full_workflow.sh"

if [[ ${#UUIDS[@]} -ne ${#TAGS[@]} ]]; then
  echo "UUIDS and TAGS must have same length" >&2; exit 1
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
  "${WF}"
done


