#!/usr/bin/env python3
"""
Partition the dataset into “upper” (high-score) and “lower” (low-score) halves.

For every part i = 0 … 7:
    1.  load   scores_part_i.npy   → shape (N, 2)
    2.  take the first column, argsort, split into top / bottom N/2 indices
    3.  look up the corresponding key in  index.tsv_part_i
        key format:  00000016_00003693_<uuid>
                     └───────┬──────┘ └───┘
                     shard-id│item-id   ignored
    4.  collect pairs  (shard-id, item-id)  for the two halves

Finally we concatenate across all eight parts and save

    <out_prefix>_upper.npy   shape (M, 2)  dtype=int32
    <out_prefix>_lower.npy   shape (M, 2)  dtype=int32

where M = 8 × N/2  (≈3.28 M for the default 819 200 rows/part).
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def parse_key(key: str) -> tuple[int, int]:
    """
    Extract (shard_id, item_id) from a WebDataset key of the form
        00000016_00003693_2472a82f-…
    """
    shard_str, item_str, *_ = key.split("_")
    return int(shard_str), int(item_str)


def collect_pairs(
    part_idx: int,
    scores_dir: Path,
    index_dir: Path,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Return two lists of [shard, item] pairs for the given part:
    (upper_half_pairs, lower_half_pairs)
    """
    scores_path = scores_dir / f"scores_part_{part_idx}.npy"
    index_path = index_dir / f"index.tsv_part_{part_idx}"

    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    # --- load scores & select halves ------------------------------------------------
    scores = np.load(scores_path, mmap_mode="r")   # shape (N, 2)
    first_col = scores[:, 0]
    N = first_col.shape[0]
    half = N // 2

    order = np.argsort(first_col)                  # ascending
    upper_idx = order[-half:]                      # highest scores

    # --- load corresponding keys ----------------------------------------------------
    with open(index_path, "r") as fh:
        keys = [line.split()[1] for line in fh]    # ignore leading row-id

    # sanity check
    assert len(keys) == N, "index.tsv and scores file mismatch"

    # --- build (shard, item) lists --------------------------------------------------
    upper_pairs, lower_pairs = [], []

    for idx in tqdm(upper_idx, desc="upper"):
        upper_pairs.append(parse_key(keys[idx]))

    return upper_pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Partition dataset into high / low score halves.")
    p.add_argument("--scores-dir", type=Path, default=Path("."), help="Directory with scores_part_*.npy")
    p.add_argument("--index-dir",  type=Path, default=Path("../datacomp_feats"),
                   help="Directory with index.tsv_part_* files")
    p.add_argument("--out-prefix", type=Path, default=Path("partition"),
                   help="Output file prefix (creates <prefix>_upper.npy and <prefix>_lower.npy)")
    p.add_argument("--parts", type=int, default=8, help="Number of parts to process")
    args = p.parse_args()

    upper_all = []

    for i in tqdm(range(args.parts), desc="parts"):
        upper = collect_pairs(i, args.scores_dir, args.index_dir)
        upper_all.extend(upper)

    upper_arr = np.asarray(upper_all, dtype=np.int32)

    np.save(f"{args.out_prefix}_upper.npy", upper_arr)

    print(f"Saved {upper_arr.shape[0]} upper-half pairs to {args.out_prefix}_upper.npy")


if __name__ == "__main__":
    main()