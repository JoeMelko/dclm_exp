#!/usr/bin/env python3
"""subset_ds.py  –  Extract the *first* ``N`` samples from a tokenised
WebDataset directory **without any re-ordering** and write them into a new
WebDataset (``*.tar`` shards) of at most ``--shard-size`` samples per shard.

The script simply streams samples from the *input* shards in lexicographic
order, starting with ``shard_00000000.tar`` (or whichever file comes first) and
stops after exactly ``N`` examples.  Assuming all shards contain *exactly* ``S``
samples each, a total of ⌈N∕S⌉ shards is read – matching the informal
statement *“reads the first N∕S shards”* in the task description.

A ``manifest.jsonl`` describing the newly written shards is generated so the
resulting dataset can be consumed like any other WebDataset produced by the
Datacomp-LM tooling.

Example
-------
Extract the first 10 million samples from ``my_dataset`` into ``subset_10m``
using 8 192 samples per output shard::

    python subset_ds.py \
        --input-dir   my_dataset \
        --n-examples  10000000   \
        --output-dir  subset_10m \
        --shard-size  8192
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import List

import webdataset as wds

__all__ = [
    "subset_dataset",
    "main",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iter_samples_sequential(shard_dir: Path):
    """Yield samples from *shard_dir* **without any shuffling**.

    Shards (``*.tar`` files) are iterated in **sorted** order and passed to
    :class:`webdataset.WebDataset` with ``shardshuffle=False`` to avoid any
    internal re-ordering.  The resulting dataset is streamed in its *original*
    order.
    """
    shards = sorted(p for p in shard_dir.glob("*.tar") if p.is_file())
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")

    ds = wds.WebDataset([str(p) for p in shards], shardshuffle=False, handler=wds.warn_and_continue)
    ds = ds.with_length(None)  # keep raw bytes, length unknown
    for sample in ds:
        yield sample

# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def subset_dataset(
    input_dir: Path,
    n_examples: int,
    out_dir: Path,
    shard_size: int,
):
    """Copy the *first* ``n_examples`` from *input_dir* into *out_dir*.

    Parameters
    ----------
    input_dir
        Directory containing tokenised WebDataset shards (``*.tar``).
    n_examples
        Exact number of samples to copy.
    out_dir
        Destination directory – must **not** already exist.
    shard_size
        Maximum number of samples per output shard (``ShardWriter.maxcount``).
    """
    if n_examples <= 0:
        raise ValueError("--n-examples must be a positive integer")
    if shard_size <= 0:
        raise ValueError("--shard-size must be a positive integer")
    if out_dir.exists():
        raise SystemExit(
            f"Output directory '{out_dir}' already exists – please choose a NEW directory to guarantee the original dataset stays untouched."
        )

    out_dir.mkdir(parents=True)

    sink = wds.ShardWriter(str(out_dir / "shard_%08d.tar"), maxcount=shard_size, verbose=1, encoder=False)

    total_written = 0
    for sample in _iter_samples_sequential(input_dir):
        sink.write(sample)
        total_written += 1
        if total_written == n_examples:
            break
    else:
        # Loop exhausted → input dataset shorter than requested number of samples
        sink.close()
        raise SystemExit(
            f"Input dataset contains only {total_written:,} samples – cannot satisfy requested {n_examples:,}."
        )

    sink.close()

    # ─────────────────────────────────────────────────────────────────────────
    # manifest.jsonl
    # ─────────────────────────────────────────────────────────────────────────
    num_full, remainder = divmod(n_examples, shard_size)
    shard_counts: List[int] = [shard_size] * num_full
    if remainder:
        shard_counts.append(remainder)

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, count in enumerate(shard_counts):
            shard_name = f"shard_{i:08d}"
            json.dump({"shard": shard_name, "num_sequences": count}, mf)
            mf.write("\n")

    print(
        f"✅  Copied {n_examples:,} samples to {out_dir} across {len(shard_counts)} shard(s); manifest.jsonl created"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser("extract the first N samples from a tokenised WebDataset")
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Tokenised WebDataset directory (source)",
    )
    p.add_argument(
        "--n-examples",
        type=int,
        required=True,
        help="Number of samples to copy from the start of the dataset",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the subset dataset (must not exist)",
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

    subset_dataset(
        input_dir=args.input_dir,
        n_examples=args.n_examples,
        out_dir=args.output_dir,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main() 