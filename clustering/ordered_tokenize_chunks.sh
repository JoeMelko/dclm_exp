#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR N_CHUNKS"
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
NCHUNKS="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for (( i=0; i<NCHUNKS; i++ )); do
  python "$SCRIPT_DIR/ordered_tokenize.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-chunks "$NCHUNKS" \
    --chunk "$i" &
done

wait
