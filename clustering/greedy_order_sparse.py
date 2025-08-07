#!/usr/bin/env python3
"""
greedy_order_gpu.py
===================

Multi-GPU greedy ordering for ≈10 M sequences × 10 k clusters.

Key ideas
---------
∘ Counts are stored **once** per GPU as int16 – memory ≈ 25 GB / GPU
∘ A Boolean *used* mask on each GPU guarantees every sequence is picked once.
∘ Each iteration:
   1. Current global token totals are broadcast to every GPU
   2. Every GPU computes the error for **all of its local sequences** in one
      vectorised kernel and returns the *k* best (k ≪ N)
   3. CPU merges the candidates, chooses the global best, updates state
   4. That row’s error is flushed to +∞ on its GPU (mask = True)

Dependencies
------------
`pip install torch numpy tqdm`

No Faiss / no scipy required.

Usage
-----
 Run on 8 H100 GPUs:
 
     python greedy_order.py \
         --input-dir   /path/to/tokenized_shards \
         --ratio-file  cluster_ratios.json \
         --out-dir     /path/to/ordered_tokens \  # tokens shards + manifest
         --shard-size 8192 \
         --gpus 8
 
This writes token shards into ``--out-dir`` and matching counts-only
shards into ``--out-dir/counts``.
"""

from __future__ import annotations
import argparse, json, gzip, tarfile, uuid, os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
import webdataset as wds


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class SequenceMeta:
    shard_name : str   # original tar file
    uuid_str   : str   # uuid inside that tar


def read_ratio_file(path: Path) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    ratios = np.asarray(data['ratios'] if isinstance(data, dict) else data, dtype=np.float32)
    ratios /= ratios.sum()
    return ratios


# --------------------------------------------------------------------------- #
# Fast streaming loader                                                      #
# --------------------------------------------------------------------------- #

def stream_shards_to_arrays(
    shard_paths: list[Path],
    n_cluster: int,
    workers: int,
    prefetch: int,
) -> tuple[torch.Tensor, list[bytes], list[SequenceMeta]]:
    """Stream shards into a contiguous counts matrix + tokens list.

    Returns
    -------
    counts_all : torch.Tensor (N, n_cluster) float32 (CPU)
    tokens_all : List[bytes]              original gzip payload per seq
    meta_all   : List[SequenceMeta]       one per sequence
    """

    # ------------------------------------------------------------------ load counts tensors (one per shard) – stay in torch
    counts_tensor_list: list[torch.Tensor] = []
    for p in tqdm(shard_paths, desc="Loading counts", unit="shard"):
        counts_path = p.parent / f"{p.stem}_counts.pt"
        if not counts_path.exists():
            raise FileNotFoundError(f"Expected counts file {counts_path} for shard {p}")
        tens: torch.Tensor = torch.load(counts_path, map_location="cpu")  # type: ignore
        if tens.ndim != 2 or tens.shape[1] != n_cluster:
            raise ValueError(
                f"Counts tensor shape mismatch in {counts_path}; expected (_, {n_cluster})"
            )
        counts_tensor_list.append(tens.to(dtype=torch.float32, copy=False))

    counts_all: torch.Tensor = torch.cat(counts_tensor_list, dim=0)

    # ------------------------------------------------------------------ stream tokens (only) with WebDataset
    dataset = (
        wds.WebDataset([str(p) for p in shard_paths], shardshuffle=False, handler=wds.warn_and_continue)
        .to_tuple("__key__", "tokens.json.gz")
        .with_length(None)
    )

    loader = wds.WebLoader(
        dataset,
        num_workers=workers,
        batch_size=None,
        prefetch_factor=prefetch,
    )

    tokens_all: list[bytes] = []
    meta_all  : list[SequenceMeta] = []

    for key, tok_bytes in tqdm(loader, desc="Streaming tokens", unit="seq"):
        tokens_all.append(tok_bytes)
        meta_all.append(SequenceMeta(key.split("@")[0], key))

    if len(tokens_all) != counts_all.shape[0]:
        raise RuntimeError(
            f"Mismatch between token sequences ({len(tokens_all)}) and counts ({counts_all.shape[0]})"
        )

    return counts_all, tokens_all, meta_all


# --------------------------------------------------------------------------- #
# CPU-sparse helper                                                            #
# --------------------------------------------------------------------------- #


def build_sparse_rep(counts_all: torch.Tensor):
    """Return counts in CSR format and per-row L2 norms (both on CPU)."""

    counts_sparse = counts_all.to_sparse_csr()
    norms2 = counts_all.pow(2).sum(dim=1)  # (N,)
    return counts_sparse, norms2


# --------------------------------------------------------------------------- #
# Greedy loop (CPU + sparse)                                                  #
# --------------------------------------------------------------------------- #


def greedy_cpu_sparse(counts_all: torch.Tensor, counts_sparse: torch.Tensor, norms2: torch.Tensor,
                      target: np.ndarray, k_return: int = 1, error_log_path: str = "") -> list[int]:
    """Greedy ordering using CPU sparse matrix math (single big matrix)."""

    target_t = torch.as_tensor(target, dtype=torch.float32)
    n_cluster = target.shape[0]

    n_total = counts_all.shape[0]
    seq_len_const = int(counts_all[0].sum().item())  # sequences are uniform length (2049)

    cum_counts = torch.zeros(n_cluster, dtype=torch.float32)
    used_mask = torch.zeros(n_total, dtype=torch.bool)
    order: list[int] = []

    pbar = tqdm(total=n_total, desc="Ordering sequences (CPU-sparse)")

    log_fh = open(error_log_path, "w") if error_log_path else None

    while len(order) < n_total:
        desired_remaining = target_t * (n_total * seq_len_const) - cum_counts
        d_norm2 = float(desired_remaining.pow(2).sum().item())

        # Sparse matmul for dot products: (N,d) @ (d,) -> (N,)
        dots = torch.sparse.mm(counts_sparse, desired_remaining.unsqueeze(1)).squeeze(1)

        errs = norms2 + d_norm2 - 2 * dots
        errs[used_mask] = float("inf")

        # k_return allows quick fallback; default 1 for simplicity
        topk_val, topk_idx = torch.topk(errs, k_return, largest=False)
        chosen = None
        for idx in topk_idx:
            if not used_mask[idx]:
                chosen = int(idx.item())
                break
        if chosen is None:
            # exhaustive argmin
            chosen = int(torch.argmin(errs).item())

        # Update state
        used_mask[chosen] = True
        cum_counts += counts_all[chosen]
        order.append(chosen)

        # --- error logging ---
        if log_fh is not None:
            total_tokens = len(order) * seq_len_const
            current_ratio = cum_counts / total_tokens
            l2_err = torch.norm(current_ratio - target_t, p=2).item()
            import json as _json
            log_fh.write(_json.dumps({"step": len(order), "l2": l2_err}) + "\n")

        pbar.update(1)

    pbar.close()
    if log_fh is not None:
        log_fh.close()
    return order


# --------------------------------------------------------------------------- #
# Tar writing & manifest                                                      #
# --------------------------------------------------------------------------- #
def write_output(order: list[int], tokens_all: List[bytes], counts_all: torch.Tensor,
                 out_sequences_dir: Path, counts_dir: Path, chunk: int) -> list[Dict]:
    out_sequences_dir.mkdir(parents=True, exist_ok=True)
    counts_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[Dict] = []
    shard_id = 0
    buf_tokens, buf_counts = [], []

    def flush():
        nonlocal shard_id, buf_tokens, buf_counts
        if not buf_tokens:
            return
        shard_path = out_sequences_dir / f'shard_{shard_id:08d}.tar'
        with tarfile.open(shard_path, 'w') as tar:
            for tok, cnt in zip(buf_tokens, buf_counts):
                uid = uuid.uuid4().hex
                tb = tok  # already gzipped bytes from source
                cb = gzip.compress(json.dumps([int(x) for x in cnt]).encode())
                ti = tarfile.TarInfo(f'{uid}.tokens.json.gz'); ti.size=len(tb)
                ci = tarfile.TarInfo(f'{uid}.counts.json.gz'); ci.size=len(cb)
                tar.addfile(ti, BytesIO(tb)); tar.addfile(ci, BytesIO(cb))
        manifest.append({'shard': shard_path.stem,
                         'num_sequences': len(buf_tokens)})
        shard_id += 1
        buf_tokens, buf_counts = [], []

    counts_writer = wds.ShardWriter(str(counts_dir / "shard_%08d.tar"), maxcount=chunk, encoder=False)

    for idx in order:
        tok_bytes = tokens_all[idx]
        cnt = counts_all[idx].to(dtype=torch.int32, device='cpu').tolist()
        buf_tokens.append(tok_bytes); buf_counts.append(cnt)

        # Write counts sample immediately via ShardWriter to avoid big memory
        uid = uuid.uuid4().hex
        counts_writer.write({"__key__": uid, "counts.json.gz": gzip.compress(json.dumps(cnt).encode())})

        if len(buf_tokens) == chunk:
            flush()
    flush()
    counts_writer.close()
    return manifest


def write_manifest(manifest: list[Dict], out_dir: Path,
                   target: np.ndarray, cum_counts: np.ndarray, tot_tokens: int):
    fout = out_dir / 'manifest.jsonl'
    final_ratio = (cum_counts / tot_tokens).tolist()
    with open(fout, 'w') as f:
        # First write shard entries, one per line, so the manifest starts with data samples.
        for m in manifest:
            f.write(json.dumps(m) + '\n')

        # Optionally compute summary stats for console output only (not written to file).
    summary = dict(
        num_shards            = len(manifest),
        num_sequences         = sum(m['num_sequences'] for m in manifest),
        num_tokens            = int(tot_tokens),
        target_cluster_ratios = target.tolist(),
        actual_cluster_ratios = final_ratio,
        total_squared_error   = float(np.square(np.asarray(final_ratio) - target).sum())
    )
    print("Manifest summary:", json.dumps(summary))
    print(f'Wrote manifest to {fout}')


# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--ratio-file', required=True)
    ap.add_argument('--out-dir', required=True,
                    help="Output directory. Will contain token shards + manifest.jsonl, a 'counts' subdir for counts shards, and error_log.jsonl")
    ap.add_argument('--shard-size', type=int, default=64,
                    help="Number of sequences per output shard")
    ap.add_argument('--gpus', type=int, default=torch.cuda.device_count(),
                    help='GPUs to use (default: all visible)')
    ap.add_argument('--loader-workers', type=int, default=8,
                    help='Worker processes for WebLoader streaming')
    ap.add_argument('--loader-prefetch', type=int, default=8,
                    help='Prefetch batches per worker (WebLoader)')
    # No separate error-log flag; it will be written to <out-dir>/error_log.jsonl automatically
    args = ap.parse_args()

    target = read_ratio_file(Path(args.ratio_file))
    n_cluster = len(target)
    print(f'Clusters: {n_cluster}')

    # ------------------------------------------------------------------ load tars
    shard_paths = sorted(Path(args.input_dir).glob('shard_*.tar'))
    print(f"Found {len(shard_paths)} shards; loading counts tensors and streaming token data …")

    counts_all, tokens_all, meta_all = stream_shards_to_arrays(
        shard_paths,
        n_cluster=n_cluster,
        workers=args.loader_workers,
        prefetch=args.loader_prefetch,
    )

    # ------------------------------------------------------------------ build sparse representation & greedy order (CPU)
    counts_sparse, norms2 = build_sparse_rep(counts_all)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_dir = out_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    counts_dir = out_dir / "counts"

    error_log_path = str(out_dir / "error_log.jsonl")

    order_indices = greedy_cpu_sparse(counts_all, counts_sparse, norms2, target, k_return=1, error_log_path=error_log_path)

    # ------------------------------------------------------------------ write output
    manifest = write_output(order_indices, tokens_all, counts_all, tokens_dir, counts_dir, args.shard_size)

    manifest_path = tokens_dir / 'manifest.jsonl'

    # final stats
    cum = counts_all[order_indices].sum(dim=0).cpu().numpy()
    tot = float(cum.sum())
    write_manifest(manifest, tokens_dir, target, cum, tot)
    print('Done.')

if __name__ == '__main__':
    main() 