#!/usr/bin/env python3
"""count_sequences.py – Summarise sequence counts of multiple WebDatasets.

Given a *parent* directory, this script scans all immediate sub-directories
(``dataset0``, ``dataset1`` …) that contain a ``manifest.jsonl``.  Each manifest
line is expected to be a JSON object with a key ``num_sequences`` that reports
how many samples the corresponding shard holds, e.g.::

    {"num_sequences": 512, "shard": "shard_00000000"}

The script sums ``num_sequences`` across all shards **per** dataset directory
and writes the result to a single JSON file mapping directory basename → count.

Example
-------
python count_sequences.py \
        --input-root   path/to/parent_dir \
        --output-json  counts.json

The resulting ``counts.json`` might look like::

    {
        "dataset0": 123456,
        "dataset1": 654321,
        "dataset2": 111111
    }
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_manifest_sequences(manifest_path: Path) -> int:
    """Return the sum of ``num_sequences`` in *manifest_path*."""
    total = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{manifest_path}: invalid JSON – {e}") from None
            if "num_sequences" not in obj:
                raise KeyError(f"{manifest_path}: line missing 'num_sequences' key")
            total += int(obj["num_sequences"])
    return total

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser("Aggregate num_sequences across dataset sub-directories")
    p.add_argument("--input-root", type=Path, required=True,
                   help="Parent directory whose immediate sub-dirs are datasets")
    p.add_argument("--output-json", type=Path, default="dataset_sizes.json",
                   help="Path to write the output JSON mapping (default: dataset_sizes.json)")
    args = p.parse_args(argv)

    if not args.input_root.is_dir():
        raise SystemExit(f"input-root not found or not a directory: {args.input_root}")

    results: Dict[str, int] = {}

    subdirs = [p for p in sorted(args.input_root.iterdir()) if p.is_dir()]
    if not subdirs:
        raise SystemExit(f"No sub-directories found in {args.input_root}")

    for ds_dir in tqdm(subdirs, desc="datasets"):
        manifest = ds_dir / "manifest.jsonl"
        if not manifest.exists():
            print(f"[WARN] {ds_dir.name}: manifest.jsonl not found – skipping")
            continue
        total = count_manifest_sequences(manifest)
        results[ds_dir.name] = total

    if not results:
        raise SystemExit("No dataset manifests processed – aborting")

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"✅  Wrote sizes for {len(results)} dataset(s) to {args.output_json}")


if __name__ == "__main__":
    main() 