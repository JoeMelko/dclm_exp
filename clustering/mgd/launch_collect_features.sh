#!/usr/bin/env bash
# launch_collect_features.sh
# --------------------------
# Convenience wrapper that
#   1) pre-allocates the shared memmap (create_mmap_features.py) and then
#   2) spawns *eight* parallel `collect_features_dc.py` instances – one per GPU –
#      such that each process writes to a disjoint slice of the mmap without
#      overlap.
#
# The CLI closely mirrors `launch_collect_grads.sh` so you can reuse most flags.
# Unknown flags are forwarded to **both** the mmap-creation step and the worker
# processes.
#
# Example
#   ./launch_collect_features.sh \
#       --wds-dir /home/jmelko/baseline/ \
#       --uuid testt5-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=1-seed=3210-tokens=8232325120 \
#       --shards-per-gpu 128 \
#       --shard-size 1024 \
#       --out /home/jmelko/four/features.fp16
#
# Any flag accepted by create_mmap_features.py / collect_features_dc.py may be
# specified here (e.g. --lora-rank, --num-blocks, --out, ...).

set -euo pipefail

# ------------- default parameters ------------- #
GPU_COUNT=8
SHARDS_PER_GPU=15      # aka chunk-size
SHARD_SIZE=1000        # samples per shard (placeholder default)
TOTAL_SHARDS=$((GPU_COUNT*SHARDS_PER_GPU))
EXTRA_ARGS=()
CUSTOM_TOTAL_SHARDS=0
GLOBAL_SHARD_OFFSET=0  # optional global reader offset

# ------------- CLI parsing -------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards-per-gpu|--chunk-size)
            SHARDS_PER_GPU="$2"; shift 2;;
        --shard-size)
            SHARD_SIZE="$2"; shift 2;;
        --num-shards)
            TOTAL_SHARDS="$2"; CUSTOM_TOTAL_SHARDS=1; shift 2;;
        --global-shard-offset)
            GLOBAL_SHARD_OFFSET="$2"; shift 2;;
        *)
            EXTRA_ARGS+=("$1"); shift;;
    esac
done

# recompute total shards only if the user did NOT supply --num-shards explicitly
if [[ "$CUSTOM_TOTAL_SHARDS" -eq 0 ]]; then
    TOTAL_SHARDS=$((GPU_COUNT*SHARDS_PER_GPU))
fi

# ------------- run create_mmap_features.py ------------- #
python create_mmap_features.py \
    --num-shards "$TOTAL_SHARDS" \
    --shard-size "$SHARD_SIZE" \
    "${EXTRA_ARGS[@]}"

echo "[launcher] Memmap initialised – spawning $GPU_COUNT GPU workers (chunk $SHARDS_PER_GPU)"

# ------------- spawn workers ------------------ #
for GPU_ID in $(seq 0 $((GPU_COUNT-1))); do
    echo "[launcher] GPU $GPU_ID → shards chunk-size $SHARDS_PER_GPU"
    START_OFFSET=$((GPU_ID * SHARDS_PER_GPU * SHARD_SIZE))

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python collect_features_dc.py \
        --gpu-id "$GPU_ID" \
        --total-gpus "$GPU_COUNT" \
        --chunk-size "$SHARDS_PER_GPU" \
        --global-shard-offset "$GLOBAL_SHARD_OFFSET" \
        --num-shards "$TOTAL_SHARDS" \
        --shard-size "$SHARD_SIZE" \
        --start-offset "$START_OFFSET" \
        --workers 0 \
        "${EXTRA_ARGS[@]}" &

done

wait
echo "[launcher] All workers finished." 