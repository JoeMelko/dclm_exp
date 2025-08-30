#!/usr/bin/env python3
"""
split_by_clusters_npy.py – Stream original shards and split into per-cluster datasets
using a 1D labels array saved by kmeans.py (NumPy .npy).

Assumptions
-----------
• The labels array is 1-D, dtype integer, length = number of VALID lines
  embedded by embed2.py (i.e. lines with a JSON object that contains a 'text' field).
• The global order of labels matches lexicographic shard order and per-shard line
  order, skipping malformed lines – identical to how embed2.py advanced its index.

Behavior
--------
• Scans input directory for shards ("*.jsonl.zst" or "*.jsonl.zstd") in lexicographic order.
• Decompresses each shard as a text stream, parses each line as JSON and checks for
  the 'text' field. Lines lacking 'text' or invalid JSON are treated as INVALID and
  do not advance the labels pointer; they are not written to any dataset.
• For each VALID line, takes the next cluster id from the labels array and writes the
  raw line to a zstd-compressed output file located at:
      <output_dir>/dataset{cluster_id}/<relative_path_of_input_shard>

Example
-------
python split_by_clusters_npy.py \
  --input-dir  data/gs01_ls1 \
  --labels-npy clusters.npy \
  --output-dir data/gs01_ls1_split
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import zstandard as zstd
from tqdm.auto import tqdm


def find_shards(input_dir: Path) -> list[Path]:
    """Return shards in lexicographic order under input_dir.

    We consider files ending with '.jsonl.zst' or '.jsonl.zstd'.
    """
    shards = []
    for suffix in (".jsonl.zst", ".jsonl.zstd"):
        shards.extend(sorted(p for p in input_dir.glob(f"*{suffix}") if p.is_file()))
    return sorted(set(shards))


def open_writer(
    base_out_dir: Path,
    dataset_id: int,
    rel_path: Path,
    compressors: Dict[int, zstd.ZstdCompressor],
) -> io.BufferedWriter:
    """Open a zstd stream writer for (dataset_id, rel_path) lazily.

    One compressor per dataset is reused for efficiency.
    """
    out_path = base_out_dir / f"dataset{dataset_id}" / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp = compressors.setdefault(dataset_id, zstd.ZstdCompressor(level=3))
    return comp.stream_writer(out_path.open("wb"))


def stream_split(
    input_dir: Path,
    labels_npy: Path,
    output_dir: Path,
) -> None:
    labels = np.load(labels_npy)
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        raise ValueError("labels-npy must be a 1-D NumPy array of cluster ids")
    if labels.dtype.kind not in {"i", "u"}:
        labels = labels.astype(np.int64, copy=False)

    total_valid_expected = int(labels.shape[0])
    valid_seen = 0

    shards = find_shards(input_dir)
    if not shards:
        raise SystemExit(f"No shards found in {input_dir}")

    compressors: Dict[int, zstd.ZstdCompressor] = {}

    # Process each shard; writers are scoped to a single shard and closed after
    for in_path in tqdm(shards, desc="shards"):
        rel_path = in_path.relative_to(input_dir)
        dctx = zstd.ZstdDecompressor()
        writers: Dict[Tuple[int, Path], io.BufferedWriter] = {}

        with in_path.open("rb") as fin, dctx.stream_reader(fin) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                # Replicate embed2.py's validity criterion: JSON with 'text' key
                is_valid = False
                try:
                    obj = json.loads(line)
                    _ = obj["text"]
                    is_valid = True
                except Exception:
                    is_valid = False

                if not is_valid:
                    continue

                if valid_seen >= total_valid_expected:
                    raise RuntimeError(
                        "Encountered more VALID lines than labels entries — mismatch with embeddings/kmeans."
                    )

                cid = int(labels[valid_seen])
                valid_seen += 1

                key = (cid, rel_path)
                if key not in writers:
                    writers[key] = open_writer(output_dir, cid, rel_path, compressors)
                writers[key].write(line.encode("utf-8"))

        # Close writers for this shard
        for w in writers.values():
            w.flush()
            w.close()

    if valid_seen != total_valid_expected:
        raise RuntimeError(
            f"labels length = {total_valid_expected}, but saw only {valid_seen} VALID lines across shards"
        )

    print(
        f"✅ Done. Wrote per-cluster datasets to {output_dir} (clusters: 0..{int(labels.max())})."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Split original shards into per-cluster datasets using a 1D labels .npy"
    )
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory with original shards (*.jsonl.zst/.zstd) in lexicographic order")
    p.add_argument("--labels-npy", type=Path, required=True,
                   help="Path to 1D NumPy array of cluster assignments (aligned to valid rows)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output root; creates dataset{ID}/<relative shard path> files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stream_split(args.input_dir, args.labels_npy, args.output_dir)


if __name__ == "__main__":
    main()


