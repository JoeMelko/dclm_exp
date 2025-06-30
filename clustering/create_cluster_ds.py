#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream‑split DCLM shards into N separate datasets in ONE pass.

Example
-------
python create_cluster_ds.py \
        --input-dir   ../data/gs01_ls1              \
        --assign-map  shard_map.json                \
        --num-datasets 10000                              \
        --output-dir  ../data/gs01_ls1_split_10000
"""
from __future__ import annotations
import argparse, json, os, sys, io, pathlib, typing as T
import zstandard as zstd
from collections import defaultdict
from tqdm.auto import tqdm

Path = pathlib.Path


# ───────────────────────────────────────── helpers ───────────────────────────

def load_assignments(fp: Path, n_ds: int) -> dict[str, list[int | None]]:
    """Return basename → assignment list; validate IDs < n_ds or ‑1/None."""
    with fp.open("r", encoding="utf‑8") as f:
        raw: dict[str, list[int | None]] = json.load(f)
    for shard, arr in raw.items():
        for idx, d in enumerate(arr, 1):
            if d is None or d == -1:
                continue
            if not (0 <= d < n_ds):
                raise ValueError(
                    f"{shard}: assignment[{idx}] = {d} outside range 0 … {n_ds-1}"
                )
    # normalise keys to basenames
    return {os.path.basename(k): v for k, v in raw.items()}


def open_writer(
    base_out_dir: Path,
    dataset_id: int,
    rel_path: Path,
    compressors: dict[int, zstd.ZstdCompressor],
) -> io.BufferedWriter:
    """
    Lazily open a zstd stream_writer for (dataset_id, rel_path).
    One compressor per dataset is reused for efficiency.
    """
    out_path = base_out_dir / f"dataset{dataset_id}" / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp = compressors.setdefault(dataset_id, zstd.ZstdCompressor(level=3))
    return comp.stream_writer(out_path.open("wb"))


def process_shard(
    in_path: Path,
    rel_path: Path,
    assignments: list[int | None],
    base_out_dir: Path,
    compressors: dict[int, zstd.ZstdCompressor],
) -> None:
    """
    Decompress *in_path*, send each line to the appropriate dataset writer.
    """
    dctx = zstd.ZstdDecompressor()

    # stream read + N writers (opened lazily)
    writers: dict[int, io.BufferedWriter] = {}

    with open(in_path, "rb") as fin, dctx.stream_reader(fin) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf‑8")
        for i, line in enumerate(text_stream, start=0):            # 0‑based idx
            try:
                dataset_id = assignments[i]
            except IndexError:
                raise RuntimeError(
                    f"{in_path.name}: assignments shorter than shard "
                    f"({i+1} lines seen)"
                )
            if dataset_id is None or dataset_id == -1:
                continue  # drop
            if dataset_id not in writers:
                writers[dataset_id] = open_writer(
                    base_out_dir, dataset_id, rel_path, compressors
                )
            writers[dataset_id].write(line.encode("utf‑8"))

    # validate length
    if len(assignments) != (i + 1):
        raise RuntimeError(
            f"{in_path.name}: shard has {i+1} lines but assignments "
            f"lists {len(assignments)}"
        )

    # close writers for this shard
    for w in writers.values():
        w.flush()
        w.close()


# ────────────────────────────────────────── main ─────────────────────────────

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser("split DCLM shards into N datasets in one pass")
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Directory containing original *.jsonl.zst(d) shards")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Root dir to create dataset{0…N-1} sub‑dirs in")
    p.add_argument("--assign-map", required=True, type=Path,
                   help="JSON: shard_basename → [dataset_id | -1 | null]")
    p.add_argument("--num-datasets", required=True, type=int,
                   help="Total number of datasets (N)")
    args = p.parse_args(argv)

    assign_map = load_assignments(args.assign_map, args.num_datasets)

    # find shards that appear in the assignment map
    shard_paths = sorted(
        p for p in args.input_dir.glob("**/*")
        if p.suffix in {".zst", ".zstd"} and p.name in assign_map
    )
    if not shard_paths:
        sys.exit("No matching shards found.")

    compressors: dict[int, zstd.ZstdCompressor] = {}

    print(f"🚀  Processing {len(shard_paths)} shard(s) → {args.num_datasets} datasets")
    for in_path in tqdm(shard_paths, unit="shard"):
        rel = in_path.relative_to(args.input_dir)
        process_shard(
            in_path,
            rel,
            assignments=assign_map[in_path.name],
            base_out_dir=args.output_dir,
            compressors=compressors,
        )

    # ensure all compressors flush
    # (writers already closed per shard; no action needed)

    print("✅  Done. Output root:", args.output_dir)


if __name__ == "__main__":
    main()
