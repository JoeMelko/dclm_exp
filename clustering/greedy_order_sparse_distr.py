#!/usr/bin/env python3
"""
Greedy ordering with energy-distribution tracking (CPU + sparse).

This script mirrors greedy_order_sparse.py but additionally:
- Computes per-sequence energy e_i = ||c_i / L||_2
- Builds an equal-width histogram (B bins) over energies to get corpus proportions p_b
- Tracks selected counts per bin n_b during the greedy loop
- Computes the L2 histogram penalty increment for each candidate at each step
  Δ_k = (n_k + 1 - (s+1) p_k)^2 - (n_k - (s+1) p_k)^2 = 2 (n_k - (s+1) p_k) + 1
- Uses a relative multiplicative score: min-max normalize base errors and penalties over unused candidates, then score = norm_err * norm_penalty

Dependencies: torch numpy tqdm webdataset
"""

from __future__ import annotations
import argparse, json, gzip, tarfile, uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple

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
# Energy histogram utilities                                                  #
# --------------------------------------------------------------------------- #

def compute_energies_and_histogram(
    norms2: torch.Tensor,
    seq_len_const: int,
    num_bins: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, float, float]:
    """Compute per-sequence energies, equal-width histogram, and proportions.

    Returns
    -------
    energies       : torch.Tensor (N,) of e_i = ||c_i / L||_2
    p_b            : np.ndarray (B,) corpus proportions per bin
    bin_indices    : np.ndarray (N,) bin index for each sequence [0..B-1]
    e_min, e_max   : float, the range used for equal-width bins (after percentile clipping)
    """
    with torch.no_grad():
        energies: torch.Tensor = torch.sqrt(norms2) / float(seq_len_const)

    # Handle degenerate case
    e_min_raw = float(energies.min().item())
    e_max_raw = float(energies.max().item())
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")

    # Percentile clipping for stability (1st to 99th percentiles)
    energies_np = energies.cpu().numpy().astype(np.float64)
    e_low = float(np.percentile(energies_np, 1.0))
    e_high = float(np.percentile(energies_np, 99.0))
    if not np.isfinite(e_low) or not np.isfinite(e_high) or e_high <= e_low:
        e_low, e_high = e_min_raw, e_max_raw

    if e_high <= e_low:
        # All energies identical; single-bin effective histogram
        p_b = np.zeros(num_bins, dtype=np.float64)
        p_b[0] = 1.0
        bin_indices = np.zeros(energies.numel(), dtype=np.int64)
        return energies, p_b, bin_indices, e_low, e_high

    # Equal-width bins on [e_low, e_high]
    B = num_bins
    width = (e_high - e_low) / B
    if width <= 0:
        p_b = np.zeros(num_bins, dtype=np.float64)
        p_b[0] = 1.0
        bin_indices = np.zeros(energies.numel(), dtype=np.int64)
        return energies, p_b, bin_indices, e_low, e_high

    # For numerical stability, map e == e_high to last bin; clamp outside to edges
    pos = torch.floor((energies - e_low) / width).to(torch.int64)
    pos = torch.clamp(pos, 0, B - 1)
    bin_indices = pos.cpu().numpy()

    # Corpus proportions p_b from full dataset histogram
    counts = np.bincount(bin_indices, minlength=B).astype(np.float64)
    if counts.sum() == 0:
        p_b = np.ones(B, dtype=np.float64) / float(B)
    else:
        p_b = counts / counts.sum()

    return energies, p_b, bin_indices, e_low, e_high


# --------------------------------------------------------------------------- #
# Greedy loop (CPU + sparse)                                                  #
# --------------------------------------------------------------------------- #


def greedy_cpu_sparse(
    counts_all: torch.Tensor,
    counts_sparse: torch.Tensor,
    norms2: torch.Tensor,
    target: np.ndarray,
    k_return: int = 1,
    error_log_path: str = "",
    energy_bins: int = 32,
) -> list[int]:
    """Greedy ordering using CPU sparse matrix math (single big matrix).

    Incorporates a relative multiplicative score between the min-max normalized
    base error and the min-max normalized histogram penalty over unused candidates.
    """

    target_t = torch.as_tensor(target, dtype=torch.float32)
    n_cluster = target.shape[0]

    n_total = counts_all.shape[0]
    seq_len_const = int(counts_all[0].sum().item())  # sequences are uniform length (e.g., 2049)

    cum_counts = torch.zeros(n_cluster, dtype=torch.float32)
    used_mask = torch.zeros(n_total, dtype=torch.bool)
    order: list[int] = []

    # --- energy histogram precomputation ---
    energies, p_b, bin_indices_all, e_min, e_max = compute_energies_and_histogram(
        norms2, seq_len_const, energy_bins
    )
    B = len(p_b)
    selected_bin_counts = np.zeros(B, dtype=np.int64)  # n_b

    pbar = tqdm(total=n_total, desc="Ordering sequences (CPU-sparse)")

    log_fh = open(error_log_path, "w") if error_log_path else None

    while len(order) < n_total:
        # --- base greedy scoring ---
        tokens_after_step = (len(order) + 1) * seq_len_const
        desired_next_total = target_t * tokens_after_step
        residual_for_choice = desired_next_total - cum_counts
        d_norm2 = float(residual_for_choice.pow(2).sum().item())

        # Sparse matmul for dot products: (N,d) @ (d,) -> (N,)
        dots = torch.sparse.mm(counts_sparse, residual_for_choice.unsqueeze(1)).squeeze(1)

        base_errs = norms2 + d_norm2 - 2 * dots
        base_errs[used_mask] = float("inf")

        # --- per-candidate histogram penalty Δ_k (vectorized by bin) ---
        if B > 0:
            s = len(order)
            delta_for_bin = 2.0 * (selected_bin_counts.astype(np.float64) - (s + 1) * p_b.astype(np.float64)) + 1.0
            delta_vec_np = delta_for_bin[bin_indices_all]
            penalty_vec = torch.as_tensor(delta_vec_np, dtype=torch.float32)
        else:
            penalty_vec = torch.zeros(n_total, dtype=torch.float32)
        penalty_vec[used_mask] = float("inf")

        # --- relative multiplicative score: min-max normalize on unused candidates and clamp to [0.1, 1] ---
        candidate_mask = ~used_mask
        be = base_errs[candidate_mask]
        pe = penalty_vec[candidate_mask]

        def minmax_norm_clamped(x: torch.Tensor) -> torch.Tensor:
            x_min = torch.min(x)
            x_max = torch.max(x)
            denom = x_max - x_min
            if float(denom) <= 0.0 or not torch.isfinite(denom):
                return torch.ones_like(x)  # maps to 1.0 after clamping
            x01 = (x - x_min) / denom
            return 0.1 + 0.9 * x01

        be_n = minmax_norm_clamped(be)
        pe_n = minmax_norm_clamped(pe)
        prod = be_n * pe_n

        # Build full score tensor with +inf for used
        scores = torch.full((n_total,), float("inf"), dtype=torch.float32)
        scores[candidate_mask] = prod

        # k_return allows quick fallback; default 1 for simplicity
        topk_val, topk_idx = torch.topk(scores, k_return, largest=False)
        chosen = None
        for idx in topk_idx:
            if not used_mask[idx]:
                chosen = int(idx.item())
                break
        if chosen is None:
            # exhaustive argmin
            chosen = int(torch.argmin(scores).item())

        # --- compute L2 histogram penalty increment for chosen candidate (for logging) ---
        if B > 0:
            k = int(bin_indices_all[chosen])
            delta_k = float(delta_for_bin[k])
            # current L2 deviation BEFORE update (for context)
            dev_before = selected_bin_counts.astype(np.float64) - s * p_b.astype(np.float64)
            l2_before = float(np.square(dev_before).sum())
        else:
            k = 0
            delta_k = 0.0
            l2_before = 0.0

        # Update state
        used_mask[chosen] = True
        cum_counts += counts_all[chosen]
        order.append(chosen)

        # Update selected bin counts AFTER selection
        if B > 0:
            selected_bin_counts[k] += 1
            # L2 deviation AFTER update
            s_after = len(order)
            dev_after = selected_bin_counts.astype(np.float64) - s_after * p_b.astype(np.float64)
            l2_after = float(np.square(dev_after).sum())
        else:
            l2_after = 0.0

        # --- error logging ---
        if log_fh is not None:
            total_tokens = len(order) * seq_len_const
            current_ratio = cum_counts / total_tokens
            l2_err = torch.norm(current_ratio - target_t, p=2).item()
            record = {
                "step": len(order),
                "l2_ratio": l2_err,
                "energy_penalty_delta": float(delta_k),
                "energy_hist_l2_before": float(l2_before),
                "energy_hist_l2_after": float(l2_after),
                "chosen_bin": int(k),
                "num_bins": int(B),
                "energy_min": float(e_min),
                "energy_max": float(e_max),
            }
            log_fh.write(json.dumps(record) + "\n")

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
    ap.add_argument('--energy-bins', type=int, default=32,
                    help='Number of equal-width bins for energy histogram (used for distribution-aware ranking)')
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

    order_indices = greedy_cpu_sparse(
        counts_all, counts_sparse, norms2, target, k_return=1,
        error_log_path=error_log_path, energy_bins=args.energy_bins
    )

    # ------------------------------------------------------------------ write output
    manifest = write_output(order_indices, tokens_all, counts_all, tokens_dir, counts_dir, args.shard_size)

    # final stats
    cum = counts_all[order_indices].sum(dim=0).cpu().numpy()
    tot = float(cum.sum())
    write_manifest(manifest, tokens_dir, target, cum, tot)
    print('Done.')

if __name__ == '__main__':
    main() 