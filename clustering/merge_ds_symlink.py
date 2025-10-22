#!/usr/bin/env python3
"""merge_ds_symlink.py – Merge tokenised WebDataset directories via symlinks.

This tool concatenates multiple tokenised WebDataset directories by creating
symlinks to existing ``shard_*.tar`` files in a new output directory. A fresh
``manifest.jsonl`` is written for the output, remapping rows from input
manifests when available.

Behavior
--------
- Shards within each input directory are discovered in sorted order
  (matching ``shard_*.tar``) and appended to the output sequentially.
- New shard names in the output are sequential: ``shard_%08d.tar``.
- For the manifest: if an input directory contains ``manifest.jsonl``, the
  corresponding row (matched by source shard stem) is copied with its
  ``num_sequences`` and other fields, updating only the ``shard`` field to the
  new name. If a source manifest row is missing, a minimal row with just
  ``{"shard": new_name}`` is written.

Example
-------

    python merge_ds_symlink.py \
        --dirs dataset_A dataset_B dataset_C \
        --output-dir merged_ABC
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "merge_datasets_symlink",
    "main",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_shards(directory: Path) -> List[Path]:
    shards = sorted(p for p in directory.glob("shard_*.tar") if p.is_file())
    if not shards:
        raise FileNotFoundError(f"No shards matching 'shard_*.tar' found in {directory}")
    return shards


def load_manifest_optional(directory: Path) -> Optional[List[dict]]:
    path = directory / "manifest.jsonl"
    if not path.exists():
        return None
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def merge_datasets_symlink(
    dirs: List[Path],
    out_dir: Path,
):
    """Create a merged dataset under ``out_dir`` by symlinking shards.

    - Shards from each directory in ``dirs`` are appended in order.
    - Output shards are named sequentially (``shard_%08d.tar``).
    - A new ``manifest.jsonl`` is written by remapping rows from any available
      input ``manifest.jsonl`` files.
    """
    if not dirs:
        raise ValueError("At least one dataset directory must be specified")
    if out_dir.exists():
        raise SystemExit(
            f"Output directory '{out_dir}' already exists – please choose a NEW directory."
        )
    out_dir.mkdir(parents=True)

    # Load manifests per input directory and build stem->row maps
    dir_to_manifest_rows: Dict[Path, Optional[List[dict]]] = {}
    dir_to_row_map: Dict[Path, Dict[str, dict]] = {}
    for d in dirs:
        rows = load_manifest_optional(d)
        dir_to_manifest_rows[d] = rows
        if rows is None:
            dir_to_row_map[d] = {}
        else:
            dir_to_row_map[d] = {row.get("shard", ""): row for row in rows if "shard" in row}

    out_rows: List[dict] = []
    next_index = 0

    for src_dir in dirs:
        print(f"📥  Linking shards from {src_dir}")
        shards = list_shards(src_dir)
        row_map = dir_to_row_map[src_dir]
        for src in shards:
            new_name = f"shard_{next_index:08d}.tar"
            dst = out_dir / new_name
            if dst.exists() or dst.is_symlink():
                try:
                    dst.unlink()
                except FileNotFoundError:
                    pass
            os.symlink(src.resolve(), dst)

            # Build manifest row
            src_stem = src.stem
            base_row = row_map.get(src_stem)
            if base_row is None:
                row = {"shard": new_name[:-4]}  # store stem without .tar
            else:
                row = dict(base_row)
                row["shard"] = new_name[:-4]
            out_rows.append(row)
            next_index += 1

    # Write manifest.jsonl
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for row in out_rows:
            mf.write(json.dumps(row) + "\n")

    print(f"✅  Linked {len(out_rows)} shards into {out_dir}; manifest.jsonl created")


def _coerce_dirs(values: List[Path]) -> List[Path]:
    out: List[Path] = []
    for v in values:
        p = Path(v)
        if not p.exists() or not p.is_dir():
            raise SystemExit(f"Input directory not found: {p}")
        out.append(p)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser("merge tokenised WebDataset directories via symlinks")
    p.add_argument(
        "--dirs",
        type=Path,
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more dataset directories to concatenate in the given order",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the merged dataset (will be created)",
    )

    args = p.parse_args(argv)

    merge_datasets_symlink(
        dirs=_coerce_dirs(args.dirs),
        out_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main() 