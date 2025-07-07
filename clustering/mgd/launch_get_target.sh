#!/usr/bin/env bash
# launch_collect_grads.sh
# -----------------------
# One-stop helper to (1) allocate the shared mem-map once and (2) spawn
# eight parallel `collect_grads.py` instances – one per GPU.
#
# Example:
#   ./launch_collect_grads.sh \
#       --wds-dir /data/tokenised_wds \
#       --uuid 123e4567-e89b-12d3-a456-426614174000 \
#       --shards-per-gpu 15
#
# The script forwards all unknown flags to both the mmap-creation step and the
# worker processes so you can reuse `--out`, `--index`, `--lora-rank`, …
set -euo pipefail

# ------------- default parameters ------------- #
GPU_COUNT=8
SHARDS_PER_GPU=15       # aka chunk-size
SHARD_SIZE=1000         # samples per shard (placeholder default)
TOTAL_SHARDS=$((GPU_COUNT*SHARDS_PER_GPU))
EXTRA_ARGS=()
CUSTOM_TOTAL_SHARDS=0

# ------------- CLI parsing -------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards-per-gpu|--chunk-size)
            SHARDS_PER_GPU="$2"; shift 2;;
        --shard-size)
            SHARD_SIZE="$2"; shift 2;;
        --num-shards)
            TOTAL_SHARDS="$2"; CUSTOM_TOTAL_SHARDS=1; shift 2;;
        *)
            EXTRA_ARGS+=("$1"); shift;;
    esac
done

# recompute total shards only if the user did NOT supply --num-shards explicitly
if [[ "$CUSTOM_TOTAL_SHARDS" -eq 0 ]]; then
    TOTAL_SHARDS=$((GPU_COUNT*SHARDS_PER_GPU))
fi

echo "[launcher] Spawning $GPU_COUNT GPU workers (chunk $SHARDS_PER_GPU)"

# ------------- spawn workers ------------------ #
for GPU_ID in $(seq 0 $((GPU_COUNT-1))); do
    echo "[launcher] GPU $GPU_ID → shards chunk-size $SHARDS_PER_GPU"
    START_OFFSET=$((GPU_ID * SHARDS_PER_GPU * SHARD_SIZE))
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python get_target.py \
        --gpu-id "$GPU_ID" \
        --total-gpus "$GPU_COUNT" \
        --chunk-size "$SHARDS_PER_GPU" \
        --num-shards "$TOTAL_SHARDS" \
        --shard-size "$SHARD_SIZE" \
        --start-offset "$START_OFFSET" \
        "${EXTRA_ARGS[@]}" &
done

wait
echo "[launcher] All workers finished." 