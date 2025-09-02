#!/usr/bin/env bash
set -euo pipefail

# parent_dir [cargo_dir] [batch_size]
BATCH_SIZE=${3:-50}
PARENT_DIR=${1:-"$HOME/full_ds_split_1000"}
CARGO_DIR=${2:-"$HOME/dclm_exp/rust_processing/tokshuf-rs"}
THREADS=${THREADS:-$(nproc)}
BATCH_SIZE=${3:-50}

TOKENIZER=EleutherAI/gpt-neox-20b
SEQLEN=2049
WDS_CHUNK_SIZE=8192
NUM_LOCAL_CELLS=512

OUT_ROOT="${PARENT_DIR%/}_tok"

pids=()
n=0

for SUBDIR in "$PARENT_DIR"/*; do
  [ -d "$SUBDIR" ] || continue
  OUT_DIR="$OUT_ROOT/$(basename "$SUBDIR")"
  LOCAL_CELL_DIR="$OUT_DIR/local_cells"
  (
    cd "$CARGO_DIR"
    mkdir -p "$OUT_DIR" "$LOCAL_CELL_DIR"
    cargo run --release -- \
      --input "$SUBDIR" \
      --local-cell-dir "$LOCAL_CELL_DIR" \
      --output "$OUT_DIR" \
      --tokenizer "$TOKENIZER" \
      --seqlen "$SEQLEN" \
      --threads "$THREADS" \
      --wds-chunk-size "$WDS_CHUNK_SIZE" \
      --num-local-cells "$NUM_LOCAL_CELLS" \
      --use-tiktoken
  ) &
  pids+=("$!")
  n=$((n + 1))
  if (( n % BATCH_SIZE == 0 )); then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done

for pid in "${pids[@]}"; do wait "$pid"; done

