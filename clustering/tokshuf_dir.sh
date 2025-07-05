#!/usr/bin/env bash
# tokshuf_dir.sh  –  run tokshuf‑rs on every immediate sub‑directory.
# -------------------------------------------------------------------
# Positional arguments (unchanged):
#   $1  input_dir         root whose *immediate* sub‑dirs are datasets
#   $2  output_dir_base   root where per‑dataset outputs are written
#   $3  [local_cell_dir]  (optional) root for local‑cells (default ./tmp/local_cells)
#   $4  [tokenizer]       (optional) tokenizer id (default EleutherAI/gpt-neox-20b)
#
# Environment variables (all optional):
#   EXECUTE_COMMAND   0 = dry‑run (default) | 1 = actually run commands
#   WORKERS           max concurrent jobs                (default 1)
#   THREADS           --threads passed to tokshuf‑rs     (default $(nproc))
#   NUM_LOCAL_CELLS   --num-local-cells                  (default 512)
#   WDS_CHUNK_SIZE    --wds-chunk-size                   (default 8192)
#   TOKSHUF_EXTRA     extra flags appended verbatim
#
set -euo pipefail

# -- positional args ----------------------------------------------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_dir> <output_dir_base> [local_cell_dir_base] [tokenizer]" >&2
  exit 1
fi

INPUT_DIR="$(realpath "$1")"
OUTPUT_BASE="$(realpath "$2")"
LOCAL_CELL_BASE="$(realpath "${3:-./tmp/local_cells}")"
TOKENIZER="${4:-EleutherAI/gpt-neox-20b}"

# -- env‑config ---------------------------------------------------------------
EXECUTE_COMMAND="${EXECUTE_COMMAND:-0}"
WORKERS="${WORKERS:-1}"
THREADS="${THREADS:-$(nproc)}"
NUM_LOCAL_CELLS="${NUM_LOCAL_CELLS:-512}"
WDS_CHUNK_SIZE="${WDS_CHUNK_SIZE:-8192}"
TOKSHUF_EXTRA="${TOKSHUF_EXTRA:-}"

echo "Input directory       : $INPUT_DIR"
echo "Output base directory : $OUTPUT_BASE"
echo "Local cell base dir   : $LOCAL_CELL_BASE"
echo "Tokenizer             : $TOKENIZER"
echo "Workers (concurrent)  : $WORKERS"
echo "Threads / job         : $THREADS"
echo "Local cells / job     : $NUM_LOCAL_CELLS"
echo "WDS chunk size        : $WDS_CHUNK_SIZE"
echo "Extra flags           : ${TOKSHUF_EXTRA:-<none>}"
echo "EXECUTE_COMMAND       : $EXECUTE_COMMAND  (0=dry run, 1=execute)"
echo

mkdir -p "$OUTPUT_BASE" "$LOCAL_CELL_BASE"

# ‑‑ function to run one dataset ---------------------------------------------
run_one() {
  local dataset_dir="$1"
  local dataset_name
  dataset_name="$(basename "$dataset_dir")"

  local output_path="$OUTPUT_BASE/$dataset_name"
  local local_cell_dir="$LOCAL_CELL_BASE/$dataset_name"

  mkdir -p "$output_path" "$local_cell_dir"

  echo "🔹  $dataset_name"
  if [[ "$EXECUTE_COMMAND" != "1" ]]; then
      echo " (dry‑run) Would process → $output_path"
      return
  fi

  cargo run --release -- \
      --input "$dataset_dir" \
      --local-cell-dir "$local_cell_dir" \
      --output "$output_path" \
      --tokenizer "$TOKENIZER" \
      --seqlen 2049 \
      --threads "$THREADS" \
      --num-local-cells "$NUM_LOCAL_CELLS" \
      --wds-chunk-size "$WDS_CHUNK_SIZE" \
      ${TOKSHUF_EXTRA}

  echo "✅  $dataset_name finished"
}

export -f run_one
export OUTPUT_BASE LOCAL_CELL_BASE TOKENIZER THREADS NUM_LOCAL_CELLS WDS_CHUNK_SIZE \
       EXECUTE_COMMAND TOKSHUF_EXTRA

# ‑‑ queue jobs with a simple semaphore --------------------------------------
shopt -s nullglob
DATASET_DIRS=("$INPUT_DIR"/*/)
total=${#DATASET_DIRS[@]}
echo "Discovered $total dataset(s)"
echo

running=0
for ds in "${DATASET_DIRS[@]}"; do
  [[ -d "$ds" ]] || continue
  echo "Running job before"
  run_one "$ds" &   # background job
  echo "Running job after"
  ((running+=1))
  echo "Running jobs: $running"

  if (( running >= WORKERS )); then
    echo "Waiting for job to finish"
    wait -n          # Bash 4.3+: wait for any job to finish
    ((running-=1))
  fi
done

wait   # wait for all remaining jobs
echo
echo "🏁  All datasets processed."
