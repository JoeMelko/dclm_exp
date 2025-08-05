#!/usr/bin/env python3
"""merge_two_ds.py  –  Concatenate *two* tokenised WebDataset directories
into one while preserving the original sample order.

• **Within** each input directory the samples appear exactly as on disk – no
  shuffling of shards or samples.
• All samples from ``--dir1`` are written **before** any sample from ``--dir2``.
• The output dataset is written with up to ``--shard-size`` samples per shard
  via :class:`webdataset.ShardWriter` and accompanied by a ``manifest.jsonl``.

Example
-------

    python merge_two_ds.py \
        --dir1       dataset_A \
        --dir2       dataset_B \
        --output-dir merged_AB \
        --shard-size 8192
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List

import webdataset as wds

__all__ = [
    "merge_two_datasets",
    "main",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iter_samples_sequential(shard_dir: Path):
    """Yield samples from *shard_dir* in the on-disk order (no shuffling)."""
    shards = sorted(p for p in shard_dir.glob("*.tar") if p.is_file())
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")
    ds = wds.WebDataset([str(p) for p in shards], shardshuffle=False, handler=wds.warn_and_continue)
    ds = ds.with_length(None)  # keep raw bytes, unknown length
    for sample in ds:
        yield sample

# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def merge_two_datasets(
    dir1: Path,
    dir2: Path,
    out_dir: Path,
    shard_size: int,
):
    """Sequentially write samples from *dir1* then *dir2* into *out_dir*.

    Parameters
    ----------
    dir1, dir2
        Tokenised WebDataset directories.  ``dir1`` precedes ``dir2`` in the
        output order.
    out_dir
        Destination directory for the merged dataset **(must not exist)**.
    shard_size
        Maximum number of samples per output shard.
    """
    if shard_size <= 0:
        raise ValueError("--shard-size must be a positive integer")
    if out_dir.exists():
        raise SystemExit(
            f"Output directory '{out_dir}' already exists – please choose a NEW directory."
        )
    out_dir.mkdir(parents=True)

    sink = wds.ShardWriter(str(out_dir / "shard_%08d.tar"), maxcount=shard_size, verbose=1, encoder=False)

    shard_counts: List[int] = []
    current_shard_count = 0
    total_samples = 0

    # Helper inner function to stream a directory
    def _copy_dir(src_dir: Path):
        nonlocal current_shard_count, total_samples, shard_counts
        print(f"📥  Copying {src_dir}")
        for sample in _iter_samples_sequential(src_dir):
            sink.write(sample)
            current_shard_count += 1
            total_samples += 1
            if current_shard_count == shard_size:
                shard_counts.append(current_shard_count)
                current_shard_count = 0

    _copy_dir(dir1)
    _copy_dir(dir2)

    # Finalise writer and counts
    sink.close()
    if current_shard_count:
        shard_counts.append(current_shard_count)

    # ─────────────────────────────────────────────────────────────────────────
    # manifest.jsonl
    # ─────────────────────────────────────────────────────────────────────────
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, count in enumerate(shard_counts):
            shard_name = f"shard_{i:08d}"
            json.dump({"shard": shard_name, "num_sequences": count}, mf)
            mf.write("\n")

    print(
        f"✅  Wrote {total_samples:,} samples to {out_dir} across {len(shard_counts)} shard(s); manifest.jsonl created"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser("merge two tokenised WebDataset directories")
    p.add_argument("--dir1", type=Path, required=True, help="First dataset directory (written first)")
    p.add_argument("--dir2", type=Path, required=True, help="Second dataset directory (written after dir1)")
    p.add_argument("--output-dir", type=Path, required=True, help="Destination directory for the merged dataset")
    p.add_argument(
        "--shard-size",
        "--maxcount",
        dest="shard_size",
        type=int,
        default=1024,
        help="Maximum #samples per output shard (ShardWriter). Alias: --shard-size",
    )

    args = p.parse_args(argv)

    merge_two_datasets(
        dir1=args.dir1,
        dir2=args.dir2,
        out_dir=args.output_dir,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main() 