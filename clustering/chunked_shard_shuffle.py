#!/usr/bin/env python3
"""
chunked_shard_shuffle.py
========================

Chunk-stable coarse shuffle for already-tokenized datasets at the shard level.

Behavior
--------
- Discover input token shards named like ``shard_*.tar`` in ``--input-tokens-dir``.
- Split the sorted shard list into ``--n-coarse-chunks`` contiguous chunks as evenly as possible.
- Sample a random permutation of these chunks with ``--seed`` and concatenate in that order.
- Materialize a new dataset under ``--output-dir`` by creating either symlinks (default) or copies
  of shards named sequentially as ``shard_%08d.tar`` (and optionally counts shards as well).
- Always read ``manifest.jsonl`` from the input tokens directory and write a reordered
  ``manifest.jsonl`` to the output tokens directory to align with the new shard order.

Notes
-----
- If ``--n-coarse-chunks <= 1`` or the number of shards < 2, the operation is a no-op order-wise.
- If ``--n-coarse-chunks > num_shards``, it is clamped to ``num_shards`` (no empty chunks).
- The permutation plan is written to ``coarse_shuffle.json`` in ``--output-dir``.
- Requires ``manifest.jsonl`` to be present in ``--input-tokens-dir``; it will always be rewritten
  in ``--output-dir/tokens``.

"""

from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np


def list_shards(directory: Path) -> List[Path]:
    shards = sorted(directory.glob('shard_*.tar'))
    if len(shards) == 0:
        raise FileNotFoundError(f"No shards matching 'shard_*.tar' found in {directory}")
    return shards


def balanced_chunk_sizes(n_items: int, n_chunks: int) -> List[int]:
    if n_chunks <= 1:
        return [n_items]
    if n_chunks > n_items:
        n_chunks = n_items
    base = n_items // n_chunks
    rem = n_items % n_chunks
    return [base + 1] * rem + [base] * (n_chunks - rem)


def compute_offsets(sizes: List[int]) -> List[int]:
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    return offsets


def permute_by_chunk(items: List[Path], sizes: List[int], seed: int) -> Tuple[List[Path], List[int]]:
    if len(sizes) == 1:
        return items, list(range(1))
    k = len(sizes)
    offsets = compute_offsets(sizes)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(k).tolist()
    # Concatenate chunks in permuted order
    out: List[Path] = []
    for i in perm:
        i_int = int(i)
        start = offsets[i_int]
        end = start + sizes[i_int]
        out.extend(items[start:end])
    return out, perm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def materialize_shards(src_order: List[Path], dst_dir: Path, copy_mode: bool) -> List[str]:
    ensure_dir(dst_dir)
    out_basenames: List[str] = []
    for i, src in enumerate(src_order):
        dst_name = f'shard_{i:08d}.tar'
        dst = dst_dir / dst_name
        if dst.exists():
            dst.unlink()
        if copy_mode:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)
        out_basenames.append(dst.stem)
    return out_basenames


def load_manifest(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: List[dict] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rewrite_manifest(in_rows: List[dict], src_shards: List[Path], permuted: List[Path], out_basenames: List[str], out_path: Path) -> None:
    # Map original shard stem -> manifest row(s) in source order
    # We assume one row per shard, but if there are multiple, preserve alignment by sorted src order
    # Build mapping by the sorted order of src_shards
    stem_to_row = {}
    for row in in_rows:
        if 'shard' not in row:
            continue
        stem_to_row[row['shard']] = row

    src_stems = [p.stem for p in src_shards]
    perm_stems = [p.stem for p in permuted]

    out_rows: List[dict] = []
    for new_stem, src_stem in zip(out_basenames, perm_stems):
        row = stem_to_row.get(src_stem)
        if row is None:
            # Fallback: create minimal row if missing
            row = {'shard': new_stem}
        else:
            row = dict(row)
            row['shard'] = new_stem
        out_rows.append(row)

    with open(out_path, 'w') as f:
        for r in out_rows:
            f.write(json.dumps(r) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-tokens-dir', type=str, required=True,
                    help='Directory containing token shards named shard_*.tar')
    ap.add_argument('--input-counts-dir', type=str, default=None,
                    help='Optional counts shards directory to reorder alongside tokens')
    ap.add_argument('--output-dir', type=str, required=True,
                    help='Output directory; will contain tokens/ and optionally counts/')
    ap.add_argument('--n-coarse-chunks', type=int, default=1,
                    help='Number of contiguous chunks to split shards into before shuffling')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for chunk permutation')
    ap.add_argument('--copy', action='store_true', help='Copy shards instead of creating symlinks')
    args = ap.parse_args()

    in_tokens = Path(args.input_tokens_dir)
    in_counts = Path(args.input_counts_dir) if args.input_counts_dir else None
    out_root = Path(args.output_dir)
    out_tokens = out_root / 'tokens'
    out_counts = out_root / 'counts' if in_counts is not None else None

    src_token_shards = list_shards(in_tokens)
    n = len(src_token_shards)

    k = int(args.n_coarse_chunks)
    if k < 1:
        k = 1
    if k > n:
        k = n
    sizes = balanced_chunk_sizes(n, k)
    permuted_tokens, perm = permute_by_chunk(src_token_shards, sizes, seed=int(args.seed))

    # Write shuffle plan
    ensure_dir(out_root)
    with open(out_root / 'coarse_shuffle.json', 'w') as f:
        json.dump({
            'num_shards': n,
            'k_chunks': k,
            'sizes': sizes,
            'permutation': perm,
        }, f)

    # Materialize tokens
    out_token_names = materialize_shards(permuted_tokens, out_tokens, copy_mode=bool(args.copy))

    # Optionally materialize counts in the same permuted order
    if in_counts is not None:
        src_count_shards = list_shards(in_counts)
        if len(src_count_shards) != n:
            raise RuntimeError(f"Counts shards ({len(src_count_shards)}) do not match token shards ({n})")
        # Apply the same chunk permutation
        permuted_counts, _ = permute_by_chunk(src_count_shards, sizes, seed=int(args.seed))
        _ = materialize_shards(permuted_counts, out_counts, copy_mode=bool(args.copy))

    # Always rewrite manifest from input tokens dir to output tokens dir
    manifest_path = in_tokens / 'manifest.jsonl'
    in_rows = load_manifest(manifest_path)
    if len(in_rows) != n:
        print(f"Warning: manifest rows ({len(in_rows)}) != token shards ({n}); attempting best-effort remap")
    rewrite_manifest(in_rows, src_token_shards, permuted_tokens, out_token_names, out_tokens / 'manifest.jsonl')

    print(f"Reordered {n} token shards into {k} coarse chunks and materialized at {out_tokens}")
    if out_counts is not None:
        print(f"Reordered counts shards at {out_counts}")


if __name__ == '__main__':
    main()


