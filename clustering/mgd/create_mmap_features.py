#!/usr/bin/env python
"""
create_mmap_features.py
-----------------------
Pre-allocate a NumPy memmap that will hold **raw** LoRA gradient feature
vectors** (no whitening) produced by `mgd/collect_features_dc.py`.

The required shape is::

    (num_shards * shard_size, 2 * num_blocks, lora_rank * lora_rank)

so every GPU worker knows its global slice in advance and can write its chunk
without race conditions.

Usage (identical core flags to `collect_features_dc.py` except for GPU-specific
ones):

    python create_mmap_features.py \
        --num-shards 120 \
        --shard-size 1024 \
        --lora-rank 128 \
        --num-blocks 8 \
        --out clustering/mgd/features.fp16

Note: Index file is no longer created; features are stored strictly in the
order in which samples appear in the dataset.

The script is *idempotent*: existing files are overwritten.
"""

import argparse
from pathlib import Path
import numpy as np

DTYPE_OUT = np.float16  # must match collect_features_dc.DTYPE_OUT


def compute_total_samples(num_shards: int, shard_size: int) -> int:
    """Return the expected total number of dataset samples."""
    return num_shards * shard_size


def main(args):
    n_total = compute_total_samples(args.num_shards, args.shard_size)
    blocks  = args.num_blocks * 2
    dim2    = args.lora_rank * args.lora_rank

    out_path   = Path(args.out)

    # ensure directories exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[create_mmap_features] Allocating memmap of shape ({n_total}, {blocks}, {dim2}) → {out_path}")
    feats = np.memmap(out_path, mode="w+", dtype=DTYPE_OUT, shape=(n_total, blocks, dim2))
    feats.flush()

    # no index file – sample order is deterministic; each worker writes sequentially
    print("[create_mmap_features] Index file disabled; data stored in natural order.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, required=True,
                    help="total number of dataset shards")
    ap.add_argument("--shard-size", type=int, required=True,
                    help="number of samples per shard")
    ap.add_argument("--out", default="clustering/mgd/features.fp16",
                    help="output memmap filename")
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=8)
    ap.add_argument("--wds-dir", help="(ignored) dataset directory")
    ap.add_argument("--uuid", help="(ignored) model UUID")
    ap.add_argument("--ckpt", help="(ignored) HuggingFace checkpoint path")
    args = ap.parse_args()

    main(args) 