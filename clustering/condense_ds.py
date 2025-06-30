"""
condense_ds.py

Combine a directory of Zstandard-compressed JSONL shards
(e.g. shard_00000000_processed.jsonl.zstd, …, shard_0000000N_processed.jsonl.zstd)
into ONE big JSONL file that is again compressed with Zstandard.

The original shards are **never modified** – they are only read.  The combined
file is written to a *brand-new* directory supplied by the user.

Example
-------
python -m dclm_exp.clustering.condense_ds \
       --input-dir ./my_dataset/shards \
       --output-dir ./my_dataset_condensed \
       --name full_dataset.jsonl.zstd
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable, Iterator
import random

import zstandard as zstd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def jsonl_bytes_from_zstd(fp: Path) -> Iterator[bytes]:
    """
    Yield raw JSONL lines (UTF-8 encoded bytes) from a .zstd-compressed file.
    """
    dctx = zstd.ZstdDecompressor()
    with fp.open("rb") as f, dctx.stream_reader(f) as reader:
        with io.TextIOWrapper(reader, encoding="utf-8") as text_reader:
            for line in text_reader:
                yield line.encode("utf-8")


def enumerate_shards(input_dir: Path, pattern: str) -> Iterable[Path]:
    """
    Return shard paths in *input_dir* matching *pattern*, sorted lexicographically.
    """
    shards = sorted(input_dir.glob(pattern))
    if not shards:
        raise SystemExit(f"No files matching '{pattern}' found in {input_dir}")
    return shards


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────
def condense_shards(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*_processed.jsonl.zstd",
    output_name: str = "combined.jsonl.zstd",
    c_level: int = 3,
    shuffle_shards: bool = False,
    shuffle_lines: bool = False,
) -> None:
    """
    Stream-decompress every shard in *input_dir* that matches *pattern* and write
    all lines, concatenated, to *output_dir/output_name* (compressed at
    Zstandard level *c_level*).
    """
    if output_dir.exists():
        raise SystemExit(
            f"Output directory '{output_dir}' already exists – please supply a "
            "NEW directory to guarantee the original shards stay untouched."
        )
    output_dir.mkdir(parents=True)

    shards = list(enumerate_shards(input_dir, pattern))
    if shuffle_shards:
        random.shuffle(shards)

    order_descr = "shuffled" if shuffle_shards else "sorted"
    print(f"📂  Found {len(shards):,} shard(s) in {input_dir} ({order_descr})")

    out_path = output_dir / output_name
    cctx = zstd.ZstdCompressor(level=c_level)

    # ── Gather lines (optionally) ──────────────────────────────────────────
    if shuffle_lines:
        all_lines: list[bytes] = []
        for shard in shards:
            print(f"🔄  Reading {shard.name}")
            all_lines.extend(jsonl_bytes_from_zstd(shard))

        print(f"🔀  Shuffling {len(all_lines):,} line(s)")
        random.shuffle(all_lines)

        with out_path.open("wb") as out_fh, cctx.stream_writer(out_fh) as writer:
            for line in all_lines:
                writer.write(line)
        lines_written = len(all_lines)
    else:
        lines_written = 0
        with out_path.open("wb") as out_fh, cctx.stream_writer(out_fh) as writer:
            for shard in shards:
                print(f"🔄  Processing {shard.name}")
                for line in jsonl_bytes_from_zstd(shard):
                    writer.write(line)
                    lines_written += 1

    print(f"✅  Wrote {lines_written:,} JSONL record(s) to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Condense *_processed.jsonl.zstd shards into one file, writing the "
            "result to a NEW output directory."
        )
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing shard_XXXXXXXX_processed.jsonl.zstd files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Brand-new directory where the combined file will be written",
    )
    p.add_argument(
        "--name",
        default="combined.jsonl.zstd",
        help="File name for the condensed output (default: combined.jsonl.zstd)",
    )
    p.add_argument(
        "--pattern",
        default="*_processed.jsonl.zstd",
        help="Glob pattern for input shards (default: *_processed.jsonl.zstd)",
    )
    p.add_argument(
        "--level",
        type=int,
        default=3,
        help="Zstandard compression level for the output (default 3)",
    )
    p.add_argument(
        "--shuffle-shards",
        action="store_true",
        help="Shuffle the order of shards before reading (default: disabled)",
    )
    p.add_argument(
        "--shuffle-lines",
        action="store_true",
        help=(
            "Shuffle all JSONL lines after reading them. Loads the entire dataset "
            "into memory, so use with care."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise SystemExit("--input-dir and --output-dir must be different")
    condense_shards(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        output_name=args.name,
        c_level=args.level,
        shuffle_shards=args.shuffle_shards,
        shuffle_lines=args.shuffle_lines,
    )


if __name__ == "__main__":
    main()

