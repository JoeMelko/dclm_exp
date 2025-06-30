#!/usr/bin/env bash

# condense_dirs.sh
# ----------------
# Condense every shard set located in a *sub-directory* of an input parent
# directory using the Python helper `dclm_exp.clustering.condense_ds`.
# For each sub-directory `<input_parent>/<name>` a matching output directory
# `<output_parent>/<name>` is created and the combined file is written there.
#
# Usage:
#   ./condense_dirs.sh <INPUT_PARENT_DIR> <OUTPUT_PARENT_DIR> [SHUFFLE_LINES] [NAME] [PATTERN] [LEVEL]
#
# Positional arguments
#   INPUT_PARENT_DIR   Parent directory whose *immediate* sub-directories contain
#                      shard_XXXXXXXX_processed.jsonl.zstd files.
#   OUTPUT_PARENT_DIR  Parent directory where condensed datasets will be written.
#
# Optional arguments
#   SHUFFLE_LINES  Enable shuffling of all JSONL lines before writing (true/false, default: false)
#   NAME     File name for the condensed output (default: combined.jsonl.zstd)
#   PATTERN  Glob pattern for input shards      (default: *_processed.jsonl.zstd)
#   LEVEL    Zstandard compression level        (default: 3)
#
# Examples
# --------
#   ./condense_dirs.sh /data/shard_families /data_condensed
#   ./condense_dirs.sh /data/shard_families /data_condensed true
#   ./condense_dirs.sh /data/shard_families /data_condensed false full.jsonl.zstd '*_proc.zstd' 5

set -euo pipefail

print_usage() {
  sed -n '1,/# Examples/p' "$0" | grep -E '^(#|$)' | sed 's/^# //'
}

if [[ "$#" -lt 2 ]]; then
  echo "Error: Need at least INPUT_PARENT_DIR and OUTPUT_PARENT_DIR." >&2
  echo
  print_usage
  exit 1
fi

INPUT_PARENT=$(realpath "$1")
OUTPUT_PARENT=$(realpath "$2")

# Reordered positional arguments
# $3 now denotes shuffling flag; defaults to false if omitted or set to anything other than "true"
SHUFFLE_LINES=${3:-false}
NAME=${4:-combined.jsonl.zstd}
PATTERN=${5:-*_processed.jsonl.zstd}
LEVEL=${6:-3}

if [[ "$INPUT_PARENT" == "$OUTPUT_PARENT" ]]; then
  echo "Error: INPUT_PARENT_DIR and OUTPUT_PARENT_DIR must differ." >&2
  exit 1
fi

# Ensure output parent exists
mkdir -p "$OUTPUT_PARENT"

# Ensure dclm_exp is discoverable by Python regardless of where the script is
# invoked from (assumes this script lives inside the repo that contains
# the dclm_exp package).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

shopt -s nullglob
SUBDIRS=("$INPUT_PARENT"/*/)
shopt -u nullglob

if [[ "${#SUBDIRS[@]}" -eq 0 ]]; then
  echo "No sub-directories found in $INPUT_PARENT" >&2
  exit 1
fi

echo "Found ${#SUBDIRS[@]} sub-director(ies) in $INPUT_PARENT"

for IN_DIR in "${SUBDIRS[@]}"; do
  # Remove trailing slash and extract basename
  IN_DIR=${IN_DIR%/}
  SUB_NAME=$(basename "$IN_DIR")
  OUT_DIR="$OUTPUT_PARENT/$SUB_NAME"

  echo "➤ Condensing $IN_DIR -> $OUT_DIR"

  python -m dclm_exp.clustering.condense_ds \
    --input-dir "$IN_DIR" \
    --output-dir "$OUT_DIR" \
    --name "$NAME" \
    --pattern "$PATTERN" \
    --level "$LEVEL" \
    $( [[ "$SHUFFLE_LINES" == "true" ]] && echo "--shuffle-lines" )

done

echo "All done!" 