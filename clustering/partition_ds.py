"""Partition a set of WebDataset shards into two new datasets.

This script reads *processed* shards from
```
/home/jmelko/dclm_exp/rust_processing/tokshuf-rs/dclm_tokshuf
```
(train-00000.tar … train-00799.tar) and splits the **samples** according to the
indices stored in ``score/sorted_keep.npy`` (first column = shard id,
second column = position within that shard).  Samples listed in
``sorted_keep.npy`` are written to the *upper* subset, all other samples
are written to the *lower* subset.

The final result is **two new datasets** residing in sibling directories
``subset_upper`` and ``subset_lower``.  Each dataset will contain *up to*
400 shards because the :class:`webdataset.ShardWriter` is configured to
start a new output tar file after a calculated number of samples.  That
threshold is chosen such that, given the number of samples, the writer
will produce at most 400 files.

Usage
-----
Basic invocation (uses defaults):

    python partition_ds.py

Override any of the paths or sample-per-shard limit:

    python partition_ds.py \
        --source-dir /path/to/shards \
        --keep-path  /path/to/sorted_keep.npy \
        --output-dir /path/for/output \
        --max-shards 300

All flags are optional; omitted options fall back to the defaults shown in
the *Configuration* section below.
"""

from collections import defaultdict
from pathlib import Path
import math
import numpy as np
import webdataset as wds
from tqdm import tqdm
import argparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default values – can be overridden via CLI flags
DEFAULT_SOURCE_DIR = Path("/home/jmelko/dclm_exp/rust_processing/tokshuf-rs/dclm_tokshuf")
DEFAULT_KEEP_PATH = Path(__file__).parent / "score" / "sorted_keep.npy"
DEFAULT_OUTPUT_DIR = Path(__file__).parent
DEFAULT_MAX_SHARDS = 480

parser = argparse.ArgumentParser(
    description="Partition WebDataset shards into upper/lower subsets based on a keep-array."
)
parser.add_argument(
    "--source-dir",
    type=Path,
    default=DEFAULT_SOURCE_DIR,
    help="Directory containing the input shards (*.tar).",
)
parser.add_argument(
    "--keep-path",
    type=Path,
    default=DEFAULT_KEEP_PATH,
    help="Path to the sorted_keep.npy file listing kept samples.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Base directory where subset_upper and subset_lower will be created.",
)
parser.add_argument(
    "--max-shards",
    type=int,
    default=DEFAULT_MAX_SHARDS,
    help="Maximum number of output shards per dataset.",
)

args = parser.parse_args()

# Resolved configuration after CLI parsing
SOURCE_DIR: Path = args.source_dir
KEEP_PATH: Path = args.keep_path
DEST_UPPER: Path = args.output_dir / "subset_upper"
DEST_LOWER: Path = args.output_dir / "subset_lower"
MAX_SHARDS: int = args.max_shards

# Ensure destination directories exist
DEST_UPPER.mkdir(parents=True, exist_ok=True)
DEST_LOWER.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load and organise the "keep" list
# ---------------------------------------------------------------------------

if not KEEP_PATH.exists():
    raise FileNotFoundError(f"Could not find keep-array: {KEEP_PATH}")

# First col = shard id, second col = offset within that shard
keep_arr = np.load(KEEP_PATH)

# Build mapping  shard_id -> set(sample_offsets)
keep_dict: dict[int, set[int]] = defaultdict(set)
for shard_id, offset in keep_arr:
    keep_dict[int(shard_id)].add(int(offset))

# ---------------------------------------------------------------------------
# Create ShardWriters for the two output datasets
# ---------------------------------------------------------------------------

# We want (roughly) MAX_SHARDS shards per dataset; compute sample thresholds.
num_kept_samples = len(keep_arr)

# If there are fewer samples than MAX_SHARDS, fall back to 1 sample / shard.
upper_maxcount = max(1, math.ceil(num_kept_samples / MAX_SHARDS))

# For the lower partition we don't yet know the exact number of samples.
# Assume worst case: every kept sample has a counterpart, so lower partition
# has at most the same number of samples.
lower_maxcount = upper_maxcount

writer_upper = wds.ShardWriter(
    str(DEST_UPPER / "shard_%08d.tar"), maxcount=upper_maxcount, verbose=1, encoder=False
)
writer_lower = wds.ShardWriter(
    str(DEST_LOWER / "shard_%08d.tar"), maxcount=lower_maxcount, verbose=1, encoder=False
)

# ---------------------------------------------------------------------------
# Iterate over shards one by one and write samples to the appropriate writer
# ---------------------------------------------------------------------------

for shard_id in tqdm(range(960)):
    shard_path = SOURCE_DIR / f"shard_{shard_id:08d}.tar"
    if not shard_path.exists():
        print(f"[WARN] Expected shard {shard_path} not found – skipping")
        continue

    # Build dataset for this single tar (no decoding – we deal with raw bytes).
    dataset = wds.WebDataset(str(shard_path)).with_length(None)

    # Build a quick look-up set for this shard (may be empty)
    kept_offsets = keep_dict.get(shard_id, set())

    for offset, sample in enumerate(dataset):
        if offset in kept_offsets:
            writer_upper.write(sample)
        else:
            writer_lower.write(sample)

# ---------------------------------------------------------------------------
# Finalise writers
# ---------------------------------------------------------------------------

writer_upper.close()
writer_lower.close()

print("DONE – created upper and lower subsets.")
