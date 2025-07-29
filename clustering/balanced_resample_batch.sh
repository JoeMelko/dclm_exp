#!/usr/bin/env bash
# balanced_resample_batch.sh
#
# Given
#   1) a parent directory containing multiple WebDataset sub-directories, and
#   2) a JSON file mapping sub-directory names to the desired number of samples,
# this script invokes balanced_resample.py on every entry and writes the output
# under a *new* parent directory.
#
# Example JSON (mapping.json):
#   {
#     "cc_en":  5000000,
#     "cc_fr":  2000000,
#     "wiki":   1000000
#   }
#
# Usage:
#   ./balanced_resample_batch.sh --parent-dir /data/wds-src \
#                                --json mapping.json \
#                                --out-parent-dir /data/wds-resampled \
#                                [--maxcount 8192] [--seed 0] [--workers 4]
#
# For each key/value pair (SUBDIR, N) in the JSON, the script will run:
#     python -m dclm_exp.clustering.balanced_resample \
#            --indir  /data/wds-src/SUBDIR \
#            --outdir /data/wds-resampled/SUBDIR \
#            --n      N
#
set -euo pipefail

function usage() {
  echo "Usage: $0 --parent-dir DIR --json FILE --out-parent-dir DIR [--maxcount N] [--seed S] [--workers W]" >&2
  exit 1
}

# Default parameters
PARENT=""
JSON=""
OUTPARENT=""
MAXCOUNT=8192
SEED=0
WORKERS=4

# Argument parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parent-dir)
      PARENT="$2"; shift 2;;
    --json)
      JSON="$2"; shift 2;;
    --out-parent-dir|--parent-out-dir|--output-parent|--parent-dir-out)
      OUTPARENT="$2"; shift 2;;
    --maxcount)
      MAXCOUNT="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    --workers)
      WORKERS="$2"; shift 2;;
    -h|--help)
      usage;;
    *)
      echo "Unknown option: $1" >&2
      usage;;
  esac
done

if [[ -z "$PARENT" || -z "$JSON" || -z "$OUTPARENT" ]]; then
  echo "Error: --parent-dir, --json, and --out-parent-dir are required." >&2
  usage
fi

# Make sure JSON exists
if [[ ! -f "$JSON" ]]; then
  echo "Error: JSON mapping file not found: $JSON" >&2
  exit 1
fi

mkdir -p "$OUTPARENT"

# --------------------------------------------------------------------------------
# Launch resampling jobs with simple bash concurrency control.
# --------------------------------------------------------------------------------

# Use Python to emit tab-separated lines:  SUBDIR<TAB>N_TARGET
mapfile -t JOBS < <(
  python - <<PY "$JSON"
import json, sys, os
with open(sys.argv[1]) as f:
    for k, v in json.load(f).items():
        print(f"{k}\t{v}")
PY
)

total_jobs=${#JOBS[@]}
echo "Found $total_jobs dataset(s). Launching up to $WORKERS concurrent jobs." >&2

running=0
for line in "${JOBS[@]}"; do
  subdir="${line%%$'\t'*}"
  n_target="${line##*$'\t'}"

  indir="${PARENT%/}/$subdir"
  outdir="${OUTPARENT%/}/$subdir"

  if [[ ! -d "$indir" ]]; then
    echo "[WARN] Input directory does not exist, skipping: $indir" >&2
    continue
  fi

  mkdir -p "$outdir"

  (
    echo "[INFO] Resampling $subdir → $outdir ($n_target samples)" >&2
    python -m dclm_exp.clustering.balanced_resample \
           --indir  "$indir" \
           --outdir "$outdir" \
           --n      "$n_target" \
           --maxcount "$MAXCOUNT" \
           --seed   "$SEED"
  ) &

  # Control concurrency
  while (( $(jobs -pr | wc -l) >= WORKERS )); do
    sleep 1
  done
done

# Wait for all background jobs to finish
wait

echo "All resampling jobs completed." 