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
"""

from collections import defaultdict
from pathlib import Path
import math
import numpy as np
import webdataset as wds
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Location of the 800 processed shards
SOURCE_DIR = Path(
    "/home/jmelko/dclm_exp/rust_processing/tokshuf-rs/dclm_tokshuf"
)

# Path to **sorted** keep-array (shape = [N_kept, 2])
KEEP_PATH = Path(__file__).parent / "score" / "sorted_keep.npy"

# Destination directories for the two new datasets
DEST_UPPER = Path(__file__).parent / "subset_upper"
DEST_LOWER = Path(__file__).parent / "subset_lower"

# Ensure destination directories exist
DEST_UPPER.mkdir(parents=True, exist_ok=True)
DEST_LOWER.mkdir(parents=True, exist_ok=True)

# Maximum number of output shards **per** dataset.  We want 400 each.
MAX_SHARDS = 400

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

for shard_id in tqdm(range(800)):
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
