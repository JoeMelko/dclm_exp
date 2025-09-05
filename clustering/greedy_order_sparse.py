#!/usr/bin/env python3
"""
greedy_order_sparse.py
======================

CPU-sparse greedy ordering for large sequence × cluster problems.

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
 Example run:
 
     python greedy_order_sparse.py \
         --input-dir   /path/to/tokenized_shards \
         --ratio-file  cluster_ratios.json \
         --out-dir     /path/to/ordered_tokens \  # tokens shards + manifest
         --shard-size 8192 \
         --gpus 8 \
         --reg-type l2sum_schedule \
         --reg-lambda 0.1
 
This writes token shards into ``--out-dir`` and matching counts-only
shards into ``--out-dir/counts``.
 
Regularization
--------------
- ``--reg-type``: ``none`` (default) or ``l2sum_schedule``.
  - ``l2sum_schedule`` adds a penalty equal to the squared deviation between
    the cumulative sum of per-sequence L2 norms (including the candidate) and
    the expected schedule (step/N * total_L2_sum).
- ``--reg-lambda`` scales the regularization term (0.0 disables).
 - ``histogram_schedule``: build buckets over L2 norms (``--n-buckets``; ``--bucket-method``
   quantile|uniform, default quantile) and penalize squared L2 discrepancy between expected
   and actual bucket counts at each step.
"""

from __future__ import annotations
import argparse, json, gzip, tarfile, uuid
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

    for _key, tok_bytes in tqdm(loader, desc="Streaming tokens", unit="seq"):
        tokens_all.append(tok_bytes)

    if len(tokens_all) != counts_all.shape[0]:
        raise RuntimeError(
            f"Mismatch between token sequences ({len(tokens_all)}) and counts ({counts_all.shape[0]})"
        )

    return counts_all, tokens_all, []


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


def make_regularizer(reg_type: str):
    """Return a vectorized regularization function.

    The returned function signature is:
        reg_fn(cum_so_far: float, candidate_norms: torch.Tensor, expected_next: float) -> torch.Tensor

    It returns a tensor of per-candidate penalties (without lambda applied).
    """
    if reg_type == "none":
        def _none(_cum_so_far: float, candidate_norms: torch.Tensor, _expected_next: float) -> torch.Tensor:
            return torch.zeros_like(candidate_norms)
        return _none

    if reg_type == "l2sum_schedule":
        # Penalty = ( (cum_so_far + l2norm(candidate)) - expected_next )^2
        def _l2sum(cum_so_far: float, candidate_norms: torch.Tensor, expected_next: float) -> torch.Tensor:
            return (candidate_norms + cum_so_far - expected_next).pow(2)
        return _l2sum

    raise ValueError(f"Unknown regularizer type: {reg_type}")


def greedy_cpu_sparse(counts_all: torch.Tensor, counts_sparse: torch.Tensor, norms2: torch.Tensor,
                      target: np.ndarray, k_return: int = 1, error_log_path: str = "", chunk_size: int = 512,
                      reg_type: str = "none", reg_lambda: float = 0.0,
                      l2norms: torch.Tensor | None = None, total_l2_sum: float | None = None,
                      bucket_ids: torch.Tensor | None = None, total_bucket_counts: torch.Tensor | None = None,
                      n_buckets: int | None = None) -> list[int]:
    """Greedy ordering using CPU sparse matrix math (single big matrix)."""

    target_t = torch.as_tensor(target, dtype=torch.float32)
    n_cluster = target.shape[0]

    n_total = counts_all.shape[0]
    seq_len_const = int(counts_all[0].sum().item())  # sequences are uniform length (2049)

    cum_counts = torch.zeros(n_cluster, dtype=torch.float32)
    used_mask = torch.zeros(n_total, dtype=torch.bool)
    order: list[int] = []

    # Chunk-wise logging (every chunk_size sequences)
    chunk_counts = torch.zeros(n_cluster, dtype=torch.float32)

    pbar = tqdm(total=n_total, desc="Ordering sequences (CPU-sparse)")

    log_fh = open(error_log_path, "w") if error_log_path else None

    # --- regularization setup ---
    reg_fn = make_regularizer(reg_type) if reg_type == "l2sum_schedule" else None
    use_reg = (reg_type != "none") and (reg_lambda != 0.0)
    if use_reg:
        if reg_type == "l2sum_schedule":
            if l2norms is None or total_l2_sum is None:
                # Fallback compute if not provided
                l2norms = torch.sqrt(norms2)
                total_l2_sum = float(l2norms.sum().item())
            # Ensure types
            l2norms = l2norms.to(dtype=torch.float32)
            cum_l2sum: float = 0.0
        elif reg_type == "histogram_schedule":
            if bucket_ids is None or total_bucket_counts is None or n_buckets is None:
                raise ValueError("Histogram regularizer requires bucket_ids, total_bucket_counts and n_buckets")
            bucket_ids = bucket_ids.to(dtype=torch.long)
            total_bucket_counts = total_bucket_counts.to(dtype=torch.float32)
            actual_bucket_counts = torch.zeros(n_buckets, dtype=torch.float32)
        else:
            pass
    else:
        cum_l2sum = 0.0  # maintained for logging consistency if ever needed

    while len(order) < n_total:
        tokens_after_step = (len(order) + 1) * seq_len_const
        desired_next_total = target_t * tokens_after_step
        residual_for_choice = desired_next_total - cum_counts
        d_norm2 = float(residual_for_choice.pow(2).sum().item())

        # Sparse matmul for dot products: (N,d) @ (d,) -> (N,)
        dots = torch.sparse.mm(counts_sparse, residual_for_choice.unsqueeze(1)).squeeze(1)

        errs = norms2 + d_norm2 - 2 * dots

        # --- regularization term (vector) ---
        if use_reg:
            if reg_type == "l2sum_schedule":
                expected_next_norm = (len(order) + 1) / n_total * float(total_l2_sum)  # type: ignore[arg-type]
                reg_vec = reg_fn(cum_l2sum, l2norms, expected_next_norm)  # type: ignore[operator]
                errs_total = errs + (reg_lambda * reg_vec)
            elif reg_type == "histogram_schedule":
                # expected counts for next step
                expected_next_counts = total_bucket_counts * ((len(order) + 1) / n_total)
                delta = actual_bucket_counts - expected_next_counts  # (B,)
                base_const = float((delta * delta).sum().item())
                # Precompute per-bucket penalties: ||delta + e_b||^2 = ||delta||^2 + 2*delta[b] + 1
                bucket_penalty = (2.0 * delta) + (1.0 + base_const)  # (B,)
                # Map each candidate to its bucket penalty
                reg_vec = bucket_penalty[bucket_ids]
                errs_total = errs + (reg_lambda * reg_vec)
            else:
                reg_vec = torch.zeros_like(errs)
                errs_total = errs
        else:
            # zero reg
            reg_vec = torch.zeros_like(errs)
            errs_total = errs

        errs_total[used_mask] = float("inf")

        # k_return allows quick fallback; default 1 for simplicity
        topk_val, topk_idx = torch.topk(errs_total, k_return, largest=False)
        chosen = None
        for idx in topk_idx:
            if not used_mask[idx]:
                chosen = int(idx.item())
                break
        if chosen is None:
            # exhaustive argmin
            chosen = int(torch.argmin(errs_total).item())

        # Update state
        used_mask[chosen] = True
        cum_counts += counts_all[chosen]
        chunk_counts += counts_all[chosen]
        order.append(chosen)
        if use_reg:
            if reg_type == "l2sum_schedule":
                cum_l2sum += float(l2norms[chosen].item())
            elif reg_type == "histogram_schedule":
                b = int(bucket_ids[chosen].item())
                actual_bucket_counts[b] += 1.0

        # --- error logging ---
        if log_fh is not None:
            total_tokens = len(order) * seq_len_const
            # 1D scalar distance from expected cumulative counts at this step
            expected_cum = target_t * total_tokens
            cum_residual = expected_cum - cum_counts
            cum_l2 = torch.norm(cum_residual, p=2).item()
            base_error = float(errs[chosen].item())
            reg_error = float(reg_vec[chosen].item()) if use_reg else 0.0
            import json as _json
            log_fh.write(_json.dumps({
                "step": len(order),
                "tokens": int(total_tokens),
                "cum_l2": cum_l2,
                "base_error": base_error,
                "reg_error": reg_error
            }) + "\n")

        pbar.update(1)

        # --- chunk logging every 512 sequences ---
        if log_fh is not None and (len(order) % chunk_size == 0):
            expected_chunk_tokens = chunk_size * seq_len_const
            expected_chunk_counts = target_t * expected_chunk_tokens
            chunk_residual = chunk_counts - expected_chunk_counts
            chunk_l2 = torch.norm(chunk_residual, p=2).item()
            import json as _json
            log_fh.write(_json.dumps({
                "step": len(order),
                "chunk_size": int(chunk_size),
                "chunk_tokens": int(expected_chunk_tokens),
                "chunk_l2": chunk_l2
            }) + "\n")
            # reset for next chunk
            chunk_counts.zero_()

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
    buf_tokens = []

    def flush():
        nonlocal shard_id, buf_tokens
        if not buf_tokens:
            return
        shard_path = out_sequences_dir / f'shard_{shard_id:08d}.tar'
        with tarfile.open(shard_path, 'w') as tar:
            for tok in buf_tokens:
                uid = uuid.uuid4().hex
                tb = tok  # already gzipped bytes from source
                ti = tarfile.TarInfo(f'{uid}.json.gz'); ti.size = len(tb)
                tar.addfile(ti, BytesIO(tb))
        manifest.append({'shard': shard_path.stem,
                         'num_sequences': len(buf_tokens)})
        shard_id += 1
        buf_tokens = []

    counts_writer = wds.ShardWriter(str(counts_dir / "shard_%08d.tar"), maxcount=chunk, encoder=False)

    for idx in order:
        tok_bytes = tokens_all[idx]
        cnt = counts_all[idx].to(dtype=torch.int32, device='cpu').tolist()
        buf_tokens.append(tok_bytes)

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
    ap.add_argument('--chunk-size', type=int, default=512,
                    help='Chunk size for per-chunk L2 logging (sequences)')
    ap.add_argument('--reg-type', type=str, default='none', choices=['none', 'l2sum_schedule', 'histogram_schedule'],
                    help='Regularization strategy to apply during selection')
    ap.add_argument('--reg-lambda', type=float, default=0.0,
                    help='Weight applied to the regularization term (0.0 disables)')
    ap.add_argument('--n-buckets', type=int, default=10,
                    help='Number of buckets for histogram_schedule regularizer')
    ap.add_argument('--bucket-method', type=str, default='quantile', choices=['uniform', 'quantile'],
                    help='Bucketization method for histogram_schedule regularizer')
    # No separate error-log flag; it will be written to <out-dir>/error_log.jsonl automatically
    args = ap.parse_args()

    target = read_ratio_file(Path(args.ratio_file))
    n_cluster = len(target)
    print(f'Clusters: {n_cluster}')

    # ------------------------------------------------------------------ load tars
    shard_paths = sorted(Path(args.input_dir).glob('shard_*.tar'))
    print(f"Found {len(shard_paths)} shards; loading counts tensors and streaming token data …")

    counts_all, tokens_all, _ = stream_shards_to_arrays(
        shard_paths,
        n_cluster=n_cluster,
        workers=args.loader_workers,
        prefetch=args.loader_prefetch,
    )

    # ------------------------------------------------------------------ build sparse representation & greedy order (CPU)
    counts_sparse, norms2 = build_sparse_rep(counts_all)
    # Precompute per-sequence L2 norms and their total sum for regularization
    l2norms = torch.sqrt(norms2)
    total_l2_sum = float(l2norms.sum().item())

    # Histogram regularizer precomputation if requested
    bucket_ids = None
    total_bucket_counts = None
    if args.reg_type == 'histogram_schedule':
        n_b = int(args.n_buckets)
        if args.bucket_method == 'quantile':
            quantiles = torch.linspace(0, 1, steps=n_b + 1)
            # Use numpy for quantile boundaries for numerical stability
            boundaries = torch.as_tensor(np.quantile(l2norms.numpy(), quantiles.numpy()), dtype=torch.float32)
            # Ensure strictly increasing boundaries to avoid edge issues
            boundaries[0] = float('-inf'); boundaries[-1] = float('inf')
            # torch.bucketize returns bin index such that boundaries[i-1] < x <= boundaries[i] for right=True
            bucket_ids = torch.bucketize(l2norms, boundaries[1:-1], right=True)
        else:
            lmin = float(l2norms.min().item()); lmax = float(l2norms.max().item())
            if lmax == lmin:
                bucket_ids = torch.zeros_like(l2norms, dtype=torch.long)
            else:
                # Map to [0, n_b-1]
                scaled = (l2norms - lmin) / max(1e-12, (lmax - lmin))
                bucket_ids = torch.clamp((scaled * n_b).floor().to(torch.long), 0, n_b - 1)
        n_buckets = n_b
        total_bucket_counts = torch.bincount(bucket_ids, minlength=n_buckets).to(dtype=torch.float32)
    else:
        n_buckets = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_dir = out_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    counts_dir = out_dir / "counts"

    error_log_path = str(out_dir / "error_log.jsonl")

    order_indices = greedy_cpu_sparse(
        counts_all,
        counts_sparse,
        norms2,
        target,
        k_return=1,
        error_log_path=error_log_path,
        chunk_size=args.chunk_size,
        reg_type=args.reg_type,
        reg_lambda=args.reg_lambda,
        l2norms=l2norms,
        total_l2_sum=total_l2_sum,
        bucket_ids=bucket_ids,
        total_bucket_counts=total_bucket_counts,
        n_buckets=n_buckets,
    )

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