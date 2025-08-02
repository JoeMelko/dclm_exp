#!/usr/bin/env python3
"""concat_ds.py  –  Sequentially concatenate multiple tokenised WebDataset
sub-directories into a *single* WebDataset while preserving the original order.

Given *n* sub-directories that each contain one or more ``*.tar`` WebDataset
shards (for instance produced by ``tokshuf_dir.sh`` or similar tooling), this
script streams **all** samples *in directory order* and writes them into a new
WebDataset whose shards contain at most ``--shard-size`` samples.  A
``manifest.jsonl`` describing the output shards is also generated.

Examples
--------

Explicit directory list::

    python concat_ds.py \
        --input-dirs  part0 part1 part2 \
        --output-dir  merged_ds          \
        --shard-size  8192

Parent directory (concatenate *all* immediate sub-directories)::

    python concat_ds.py \
        --input-root  parent_dir \
        --output-dir  merged_ds  \
        --shard-size  2048
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List

import webdataset as wds

__all__ = [
    "concatenate_datasets",
    "main",
]

# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def _iter_samples_sequential(shard_dir: Path):
    """Yield samples from *shard_dir* **without any shuffling**.

    Shards (``*.tar`` files) are iterated in **sorted** order and passed to
    :class:`webdataset.WebDataset` with ``shardshuffle=False`` to avoid any
    internal re-ordering.  The resulting dataset is streamed in its original
    order.
    """
    shards = sorted(p for p in shard_dir.glob("*.tar") if p.is_file())
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")

    ds = wds.WebDataset([str(p) for p in shards], shardshuffle=False, handler=wds.warn_and_continue)
    ds = ds.with_length(None)  # keep raw bytes, length unknown
    for sample in ds:
        yield sample


def concatenate_datasets(
    input_dirs: List[Path],
    out_dir: Path,
    shard_size: int,
):
    """Stream all samples from *input_dirs* sequentially into *out_dir*.

    Parameters
    ----------
    input_dirs
        Ordered list of tokenised WebDataset directories.
    out_dir
        Destination directory for the concatenated dataset.
    shard_size
        Maximum number of samples per output shard.
    """
    if shard_size <= 0:
        raise ValueError("--shard-size must be a positive integer")

    out_dir.mkdir(parents=True, exist_ok=True)

    sink = wds.ShardWriter(str(out_dir / "shard_%08d.tar"), maxcount=shard_size, verbose=1, encoder=False)

    shard_counts: List[int] = []
    current_shard_count = 0

    total_samples = 0
    for dir_idx, in_dir in enumerate(input_dirs):
        print(f"📥  Loading directory {dir_idx}: {in_dir}")
        for sample in _iter_samples_sequential(in_dir):
            sink.write(sample)
            current_shard_count += 1
            total_samples += 1

            if current_shard_count == shard_size:
                shard_counts.append(current_shard_count)
                current_shard_count = 0

    # Finish the last shard
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
        f"✅  Wrote {total_samples:,} samples to {out_dir} across {len(shard_counts)} shards; "
        "manifest.jsonl created"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser("concatenate tokenised WebDataset directories")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        help="Explicit list of tokenised WebDataset directories to concatenate (order preserved)",
    )
    group.add_argument(
        "--input-root",
        type=Path,
        help="Parent directory whose *immediate* sub-directories (sorted by name) are concatenated",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the merged dataset",
    )
    p.add_argument(
        "--shard-size",
        "--maxcount",
        dest="shard_size",
        type=int,
        default=1024,
        help="Maximum #samples per output shard (ShardWriter). Alias: --shard-size",
    )

    args = p.parse_args(argv)

    # Resolve input directories
    if args.input_dirs is not None:
        input_dirs: List[Path] = args.input_dirs
    else:
        if not args.input_root.exists():
            raise SystemExit(f"--input-root directory not found: {args.input_root}")
        input_dirs = sorted(p for p in args.input_root.iterdir() if p.is_dir())
        if not input_dirs:
            raise SystemExit(f"--input-root contains no sub-directories: {args.input_root}")

    concatenate_datasets(
        input_dirs=input_dirs,
        out_dir=args.output_dir,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main() 