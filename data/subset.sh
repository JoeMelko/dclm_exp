#!/usr/bin/env bash
# subset.sh  –  Create a smaller subset of tokenised WebDataset shards.
# -----------------------------------------------------------------------------
# Required arguments:
#   --source DIR   Directory that contains one or more *dataset* sub-directories
#                  (each with shard_XXXXX.tar files + manifest.jsonl) **or** a
#                  single dataset directory itself.
#   --dest   DIR   Destination parent directory that will receive the shrunken
#                  dataset(s).  Must *not* already exist.
#   --frac   FLOAT Fraction (0–1) of sequences to keep (e.g. 0.3 keeps 30 %).
#
# Example
#   ./subset.sh --source full_ds --dest subset_ds --frac 0.25
# -----------------------------------------------------------------------------
set -euo pipefail

# --------------------------- small CLI parser ---------------------------------
SOURCE=""
DEST=""
FRAC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2;;
    --dest)   DEST="$2";   shift 2;;
    --frac)   FRAC="$2";   shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

if [[ -z "${SOURCE}" || -z "${DEST}" || -z "${FRAC}" ]]; then
  echo "Usage: $0 --source DIR --dest DIR --frac FLOAT" >&2
  exit 1
fi

# Shell-out to an embedded Python helper that does the heavy lifting -------------
python3 - "$SOURCE" "$DEST" "$FRAC" <<'PYTHON'
import sys, os, json, math, tarfile, shutil, pathlib

source, dest, frac_str = sys.argv[1:4]
frac = float(frac_str)
if not (0.0 < frac <= 1.0):
    sys.exit("--frac must be in the range (0,1]")

source = os.path.abspath(source)
dest = os.path.abspath(dest)

if os.path.exists(dest):
    sys.exit(f"Destination directory already exists: {dest}")

os.makedirs(dest, exist_ok=False)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def is_dataset_dir(d: str) -> bool:
    """Return True iff `d` contains a manifest.jsonl file."""
    return os.path.isfile(os.path.join(d, "manifest.jsonl"))

def process_dataset(ds_path: str, out_path: str, fraction: float):
    manifest_path = os.path.join(ds_path, "manifest.jsonl")
    try:
        with open(manifest_path, "r", encoding="utf-8") as fp:
            manifest_entries = [json.loads(line) for line in fp if line.strip()]
    except FileNotFoundError:
        print(f"⚠️  Skipping {ds_path} – no manifest.jsonl found", file=sys.stderr)
        return

    total_seqs = sum(e["num_sequences"] for e in manifest_entries)
    target = max(1, int(math.floor(total_seqs * fraction)))

    os.makedirs(out_path, exist_ok=True)

    written = 0
    new_manifest = []

    for entry in manifest_entries:
        if written >= target:
            break

        shard_name = entry["shard"]
        num_seq = entry["num_sequences"]
        remaining = target - written

        src_tar = os.path.join(ds_path, f"{shard_name}.tar")
        dst_tar = os.path.join(out_path, f"{shard_name}.tar")

        if remaining >= num_seq:
            # Copy entire shard verbatim
            shutil.copy2(src_tar, dst_tar)
            new_manifest.append({"num_sequences": num_seq, "shard": shard_name})
            written += num_seq
        else:
            # Need only the first `remaining` sequences – re-pack a smaller tar.
            with tarfile.open(src_tar, "r") as src, tarfile.open(dst_tar, "w") as dst:
                for i, member in enumerate(src):
                    if i >= remaining:
                        break
                    fileobj = src.extractfile(member)
                    # TarInfo objects are mutable → copy to avoid side-effects
                    info = member
                    dst.addfile(info, fileobj)
            new_manifest.append({"num_sequences": remaining, "shard": shard_name})
            written += remaining
            # Target reached -> no need to consider further shards
            break

    # Write the new manifest
    with open(os.path.join(out_path, "manifest.jsonl"), "w", encoding="utf-8") as fp:
        for ent in new_manifest:
            fp.write(json.dumps(ent) + "\n")

    pct = written / total_seqs * 100
    print(f"✅  {os.path.basename(ds_path)}: kept {written}/{total_seqs} seqs ({pct:.1f} %)")

# ---------------------------------------------------------------------------
# Determine which directories to process
# ---------------------------------------------------------------------------

datasets = []
if is_dataset_dir(source):
    datasets = [source]
else:
    # Immediate sub-directories only
    for name in sorted(os.listdir(source)):
        sub = os.path.join(source, name)
        if os.path.isdir(sub) and is_dataset_dir(sub):
            datasets.append(sub)

if not datasets:
    sys.exit("No dataset directories found in source path")

for ds in datasets:
    out_ds = os.path.join(dest, os.path.basename(ds))
    process_dataset(ds, out_ds, frac)
PYTHON
