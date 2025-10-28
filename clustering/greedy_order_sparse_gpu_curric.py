#!/usr/bin/env python3
"""
greedy_order_sparse_gpu_curric.py
=================================

GPU-sparse greedy ordering for large sequence × cluster problems with a
time-varying target schedule (curriculum) provided via --ratio-file. The
schedule is defined as piecewise ratios at token "knots" and is interpolated
in log-token space. The final row of the schedule is the end target.

This mirrors the CPU implementation but moves the hot path and state to CUDA:
 - counts_sparse: CSR on CUDA for fast SpMV per iteration
 - counts_all (dense): kept on CUDA for quick cumulative updates; CPU copy is
   also kept for output writing
 - regularizers: implemented with CUDA tensors (incl. doc-size token schedule)

Dependencies
------------
`pip install torch numpy tqdm webdataset`

Usage (time-varying curricula + doc-size regularizer)
----------------------------------------------------
  python greedy_order_sparse_gpu_curric.py \
    --input-dir /path/to/tokenized_shards \
    --ratio-file /path/to/schedule.json \
    --out-dir /path/to/ordered_tokens \
    --shard-size 8192 \
    --reg-type docsize_token_schedule \
    --doc-bins 10 \
    --reg-lambda 0.1 \
    --offset 0
Note: --ratio-file must be a dict-of-knots JSON where each per-knot dict uses
keys dataset{i} with i=0..n_cluster-1 (strict). You can use --offset to start
the schedule at an absolute token position.
"""

from __future__ import annotations
import argparse, json, gzip, tarfile, uuid, re
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


def _w2_total_cost_squared(source_counts: torch.Tensor, target_counts: torch.Tensor) -> float:
    """Compute W2^2 between two 1D histograms on equal-width bins with quadratic cost."""
    B = int(source_counts.shape[0])
    i = 0
    j = 0
    rem_i = float(source_counts[0].item()) if B > 0 else 0.0
    rem_j = float(target_counts[0].item()) if B > 0 else 0.0
    cost = 0.0
    while i < B and j < B:
        if rem_i == 0.0:
            i += 1
            if i >= B:
                break
            rem_i = float(source_counts[i].item())
            continue
        if rem_j == 0.0:
            j += 1
            if j >= B:
                break
            rem_j = float(target_counts[j].item())
            continue
        flow = rem_i if rem_i <= rem_j else rem_j
        cost += flow * float((i - j) * (i - j))
        rem_i -= flow
        rem_j -= flow
    return cost


def _w2_incremental_exact_costs_squared(actual_counts: torch.Tensor, expected_next_counts: torch.Tensor) -> torch.Tensor:
    """Exact per-bucket delta W2^2 for adding one unit to each bucket."""
    B = int(actual_counts.shape[0])
    base_cost2 = _w2_total_cost_squared(actual_counts, expected_next_counts)
    deltas = torch.empty(B, dtype=torch.float32)
    for r in range(B):
        tmp = actual_counts.clone()
        tmp[r] = tmp[r] + 1.0
        after_cost2 = _w2_total_cost_squared(tmp, expected_next_counts)
        deltas[r] = float(after_cost2 - base_cost2)
    return deltas



# --------------------------------------------------------------------------- #
# Fast streaming loader                                                      #
# --------------------------------------------------------------------------- #

def stream_shards_to_arrays(
    shard_paths: list[Path],
    n_cluster: int,
    workers: int,
    prefetch: int,
) -> tuple[torch.Tensor, list[bytes], list[SequenceMeta]]:
    """Stream shards into a contiguous counts matrix + tokens list (CPU)."""

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

    counts_all: torch.Tensor = torch.cat(counts_tensor_list, dim=0)  # CPU dense

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
# GPU helpers                                                                 #
# --------------------------------------------------------------------------- #

def make_regularizer(reg_type: str):
    if reg_type == "none":
        def _none(_cum_so_far: float, candidate_norms: torch.Tensor, _expected_next: float) -> torch.Tensor:
            return torch.zeros_like(candidate_norms)
        return _none

    if reg_type == "l2sum_schedule":
        def _l2sum(cum_so_far: float, candidate_norms: torch.Tensor, expected_next: float, seq_len_const: int) -> torch.Tensor:
            return ((candidate_norms + cum_so_far - expected_next) / seq_len_const).abs()
        return _l2sum

    raise ValueError(f"Unknown regularizer type: {reg_type}")


# --------------------------------------------------------------------------- #
# Greedy loop (GPU + sparse)                                                  #
# --------------------------------------------------------------------------- #

def greedy_gpu_sparse(counts_all_gpu: torch.Tensor, counts_sparse_gpu: torch.Tensor,
                     target: np.ndarray, k_return: int = 1, error_log_path: str = "", chunk_size: int = 512,
                     debug_log_path: str = "", addition_log_path: str = "",
                     doc_addition_log_path: str = "",
                     reg_type: str = "none", reg_lambda: float = 0.0,
                     l2norms: torch.Tensor | None = None, total_l2_sum: float | None = None,
                     bucket_ids: torch.Tensor | None = None, total_bucket_counts: torch.Tensor | None = None,
                     n_buckets: int | None = None, random_select: bool = False,
                     doc_hist_per_seq: torch.Tensor | None = None,
                     doc_total_tokens_per_bin: torch.Tensor | None = None,
                     doc_n_bins: int | None = None,
                     doc_row_norm2: torch.Tensor | None = None,
                     doc_cluster_bin_prob: torch.Tensor | None = None,
                     sched_P: torch.Tensor | None = None, sched_knots: torch.Tensor | None = None,
                     sched_U: torch.Tensor | None = None, sched_inv_du: torch.Tensor | None = None,
                     offset_tokens: int = 0,
                     max_sequences: int | None = None) -> list[int]:
    """Greedy ordering using CUDA sparse matrix math on a single GPU."""

    device = counts_all_gpu.device
    target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
    n_cluster = target.shape[0]

    n_total = counts_all_gpu.shape[0]
    seq_len_const = int(counts_all_gpu[0].sum().item())
    total_steps = int(min(n_total, max_sequences)) if (max_sequences is not None) else int(n_total)

    # Precompute per-sequence norms on device if not provided
    norms2 = (counts_all_gpu * counts_all_gpu).sum(dim=1)

    cum_counts = torch.zeros(n_cluster, dtype=torch.float32, device=device)
    used_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    order: list[int] = []

    chunk_counts = torch.zeros(n_cluster, dtype=torch.float32, device=device)

    pbar = tqdm(total=total_steps, desc="Ordering sequences (GPU-sparse)")

    log_fh = open(error_log_path, "w") if error_log_path else None
    debug_fh = open(debug_log_path, "w") if debug_log_path else None
    additions_fh = open(addition_log_path, "w") if addition_log_path else None
    doc_additions_fh = open(doc_addition_log_path, "w") if doc_addition_log_path else None
    _debug_every = 10000

    # --- regularization setup ---
    reg_fn = make_regularizer(reg_type) if reg_type == "l2sum_schedule" else None
    use_reg = (reg_type != "none") and (reg_lambda != 0.0)
    if use_reg:
        if reg_type == "l2sum_schedule":
            if l2norms is None or total_l2_sum is None:
                l2norms = torch.sqrt(norms2)
                total_l2_sum = float(l2norms.sum().item())
            l2norms = l2norms.to(device=device, dtype=torch.float32)
            cum_l2sum: float = 0.0
        elif reg_type == "histogram_schedule":
            if bucket_ids is None or total_bucket_counts is None or n_buckets is None:
                raise ValueError("Histogram regularizer requires bucket_ids, total_bucket_counts and n_buckets")
            bucket_ids = bucket_ids.to(device=device, dtype=torch.long)
            total_bucket_counts = total_bucket_counts.to(device=device, dtype=torch.float32)
            actual_bucket_counts = torch.zeros(n_buckets, dtype=torch.float32, device=device)
        elif reg_type == "w2_histogram_schedule":
            if bucket_ids is None or total_bucket_counts is None or n_buckets is None:
                raise ValueError("W2 histogram regularizer requires bucket_ids, total_bucket_counts and n_buckets")
            bucket_ids = bucket_ids.to(device=device, dtype=torch.long)
            total_bucket_counts = total_bucket_counts.to(device=device, dtype=torch.float32)
            actual_bucket_counts = torch.zeros(n_buckets, dtype=torch.float32, device=device)
        elif reg_type == "docsize_token_schedule":
            if doc_hist_per_seq is None or doc_total_tokens_per_bin is None or doc_n_bins is None or doc_row_norm2 is None:
                raise ValueError("docsize_token_schedule requires doc_hist_per_seq, doc_total_tokens_per_bin, doc_n_bins, doc_row_norm2")
            if doc_cluster_bin_prob is None:
                raise ValueError("docsize_token_schedule requires doc_cluster_bin_prob")
            doc_hist_per_seq = doc_hist_per_seq.to(device=device, dtype=torch.float32)
            doc_total_tokens_per_bin = doc_total_tokens_per_bin.to(device=device, dtype=torch.float32)
            doc_row_norm2 = doc_row_norm2.to(device=device, dtype=torch.float32)
            doc_cluster_bin_prob = doc_cluster_bin_prob.to(device=device, dtype=torch.float32)
            cum_doc_tokens_per_bin = torch.zeros(doc_n_bins, dtype=torch.int32, device=device)
        else:
            pass
    else:
        cum_l2sum = 0.0

    # --- schedule setup (optional) ---
    has_sched = (sched_P is not None)
    E_prev = torch.zeros(n_cluster, dtype=torch.float32, device=device)
    E_chunk_start = E_prev.clone()
    k_idx: int = 0  # active knot interval index

    def _p_at(N_tokens: float) -> torch.Tensor:
        nonlocal k_idx
        if not has_sched:
            return target_t
        # Clamp to endpoints
        first_k = float(sched_knots[0].item())  # type: ignore[index]
        last_k = float(sched_knots[-1].item())  # type: ignore[index]
        if N_tokens <= first_k:
            return sched_P[0]  # type: ignore[index]
        if N_tokens >= last_k:
            return sched_P[-1]  # type: ignore[index]
        # Advance interval if needed (N increases monotonically)
        K = int(sched_knots.shape[0])  # type: ignore[union-attr]
        while (k_idx + 1) < (K - 1) and N_tokens > float(sched_knots[k_idx + 1].item()):  # type: ignore[index]
            k_idx += 1
        # Interpolate linearly in log-tokens within [k_idx, k_idx+1]
        u_curr = torch.log(torch.tensor(N_tokens, dtype=torch.float32, device=device))
        t = (u_curr - sched_U[k_idx]) * sched_inv_du[k_idx]  # type: ignore[index]
        return (1.0 - t) * sched_P[k_idx] + t * sched_P[k_idx + 1]  # type: ignore[index]

    p_prev = _p_at(float(offset_tokens)) if has_sched else target_t

    while len(order) < total_steps:
        tokens_after_step = (len(order) + 1) * seq_len_const
        # Expected addition this step based on ratios/schedule
        if not has_sched:
            desired_add_step = target_t * float(seq_len_const)
            desired_next_total = target_t * tokens_after_step
        else:
            N_curr = float(offset_tokens + tokens_after_step)
            p_curr = _p_at(N_curr)
            dN = float(seq_len_const)
            desired_add_step = 0.5 * (p_prev + p_curr) * dN
            E_curr = E_prev + desired_add_step
            desired_next_total = E_curr
            E_prev = E_curr
            p_prev = p_curr
        residual_for_choice = desired_next_total - cum_counts
        residual_norm2 = (residual_for_choice * residual_for_choice).sum()

        # Match dtype for sparse.mm when counts are fp16/bf16/fp32
        residual_for_choice_mm = residual_for_choice.to(dtype=counts_sparse_gpu.dtype)

        # Sparse matmul for dot products: (N,d) @ (d,) -> (N,)
        dots = torch.sparse.mm(counts_sparse_gpu, residual_for_choice_mm.unsqueeze(1)).squeeze(1)

        errs2 = norms2 + residual_norm2 - 2 * dots
        errs = (errs2 / (seq_len_const * seq_len_const)).clamp_min(0).sqrt()

        if use_reg:
            if reg_type == "l2sum_schedule":
                expected_next_norm = (len(order) + 1) / n_total * float(total_l2_sum)  # type: ignore[arg-type]
                reg_vec = reg_fn(cum_l2sum, l2norms, expected_next_norm, seq_len_const)  # type: ignore[operator]
                errs_total = errs + (reg_lambda * reg_vec)
            elif reg_type == "histogram_schedule":
                expected_next_counts = total_bucket_counts * ((len(order) + 1) / n_total)
                delta = actual_bucket_counts - expected_next_counts
                base_const = float((delta * delta).sum().item())
                bucket_penalty = (base_const + (2.0 * delta) + 1.0).clamp_min(0.0).sqrt()
                reg_vec = bucket_penalty[bucket_ids]
                errs_total = errs + (reg_lambda * reg_vec)
            elif reg_type == "w2_histogram_schedule":
                expected_next_counts = total_bucket_counts * ((len(order) + 1) / n_total)
                bucket_w2_costs2 = _w2_incremental_exact_costs_squared(
                    actual_counts=actual_bucket_counts,
                    expected_next_counts=expected_next_counts
                )
                reg_vec = bucket_w2_costs2.clamp_min(0).sqrt()[bucket_ids]
                errs_total = errs + (reg_lambda * reg_vec)
            elif reg_type == "docsize_token_schedule":
                expected_next_tokens_per_bin = doc_cluster_bin_prob.t().matmul(desired_next_total)
                delta_tokens = cum_doc_tokens_per_bin.float() - expected_next_tokens_per_bin
                base_const = (delta_tokens * delta_tokens).sum()
                # Dense matmul: (N,B) @ (B,) -> (N,)
                dot_term = torch.matmul(doc_hist_per_seq, delta_tokens)
                reg_vec = (base_const + (2.0 * dot_term) + doc_row_norm2).clamp_min(0.0).sqrt()
                reg_vec = reg_vec / float(seq_len_const)
                errs_total = errs + (reg_lambda * reg_vec)
            else:
                reg_vec = torch.zeros_like(errs)
                errs_total = errs
        else:
            reg_vec = torch.zeros_like(errs)
            errs_total = errs

        errs_total[used_mask] = float("inf")

        if random_select:
            available = (~used_mask).nonzero(as_tuple=False).squeeze(1)
            ridx = torch.randint(0, available.numel(), (1,), device=device).item()
            chosen = int(available[ridx].item())
        else:
            topk_idx = torch.topk(errs_total, k_return, largest=False).indices
            chosen = None
            for idx in topk_idx:
                if not used_mask[idx]:
                    chosen = int(idx.item())
                    break
            if chosen is None:
                chosen = int(torch.argmin(errs_total).item())

        # Update state
        used_mask[chosen] = True
        cum_counts += counts_all_gpu[chosen]
        chunk_counts += counts_all_gpu[chosen]
        order.append(chosen)
        if use_reg:
            if reg_type == "l2sum_schedule":
                cum_l2sum += float(l2norms[chosen].item())
            elif reg_type == "histogram_schedule":
                b = int(bucket_ids[chosen].item())
                actual_bucket_counts[b] += 1.0
            elif reg_type == "w2_histogram_schedule":
                b = int(bucket_ids[chosen].item())
                actual_bucket_counts[b] += 1.0
            elif reg_type == "docsize_token_schedule":
                # Dense row add into cumulative bin counts
                cum_doc_tokens_per_bin += doc_hist_per_seq[chosen].to(dtype=torch.int32)

        # --- error logging ---
        if log_fh is not None:
            total_tokens = len(order) * seq_len_const
            reg_error = float(reg_vec[chosen].item()) if use_reg else 0.0
            import json as _json
            log_fh.write(_json.dumps({
                "step": len(order),
                "tokens": int(total_tokens),
                "cum_l2": float(errs[chosen].item()),
                "reg_error": reg_error,
                "err_total": float(errs_total[chosen].item())
            }) + "\n")

        # --- target addition logging (sampled every 1,000 steps) ---
        if additions_fh is not None and (len(order) % 1000 == 0):
            import json as _json
            additions_fh.write(_json.dumps({
                str(len(order)): desired_add_step.to(dtype=torch.float32).tolist()
            }) + "\n")

        # --- docsize target addition logging (every _debug_every steps) ---
        if (
            doc_additions_fh is not None
            and len(order) > 0
            and (len(order) % _debug_every == 0)
            and use_reg
            and reg_type == "docsize_token_schedule"
            and 'doc_cluster_bin_prob' in locals()
        ):
            import json as _json
            doc_add_step = doc_cluster_bin_prob.t().matmul(desired_add_step)
            doc_additions_fh.write(_json.dumps({
                str(len(order)): doc_add_step.to(dtype=torch.float32).tolist()
            }) + "\n")

        pbar.update(1)

        # --- periodic debug logging every 10,000 sequences ---
        # CPU version also performs a sanity check every 10k steps to ensure
        # token accounting aligns with sequence length multiples.
        if use_reg and reg_type == "docsize_token_schedule" and (len(order) % _debug_every == 0):
            # Match CPU behavior: check modulo of total tokens tracked in the last bin.
            # Note: cum_doc_tokens_per_bin is int32 on device.
            try:
                if (cum_doc_tokens_per_bin[-1] % seq_len_const).item() != 0:
                    import builtins as _bi
                    _bi.print("Warning: cum_doc_tokens_per_bin[-1] % seq_len_const != 0")
                else:
                    import builtins as _bi
                    _bi.print("sum(cum_doc_tokens_per_bin) % seq_len_const == 0")
            except Exception:
                pass

        if debug_fh is not None and len(order) > 0 and (len(order) % _debug_every == 0):
            total_tokens = len(order) * seq_len_const
            cluster_actual = cum_counts.tolist()
            if has_sched:
                cluster_expected = desired_next_total.to(dtype=torch.float32).tolist()
            else:
                cluster_expected = (target_t * total_tokens).to(dtype=torch.float32).tolist()
            reg_actual: list[float] | list[int] = []
            reg_expected: list[float] | list[int] = []
            if use_reg:
                if reg_type == "l2sum_schedule":
                    exp_norm = (len(order) / n_total) * float(total_l2_sum)  # type: ignore[arg-type]
                    reg_actual = [float(cum_l2sum)]
                    reg_expected = [float(exp_norm)]
                elif reg_type == "histogram_schedule" and 'actual_bucket_counts' in locals():
                    reg_actual = actual_bucket_counts.tolist()
                    reg_expected = (total_bucket_counts * (len(order) / n_total)).to(dtype=torch.float32).tolist()  # type: ignore[union-attr]
                elif reg_type == "w2_histogram_schedule" and 'actual_bucket_counts' in locals():
                    reg_actual = actual_bucket_counts.tolist()
                    reg_expected = (total_bucket_counts * (len(order) / n_total)).to(dtype=torch.float32).tolist()  # type: ignore[union-attr]
                elif reg_type == "docsize_token_schedule" and 'cum_doc_tokens_per_bin' in locals():
                    reg_actual = cum_doc_tokens_per_bin.tolist()
                    reg_expected = (doc_cluster_bin_prob.t().matmul(desired_next_total)).to(dtype=torch.float32).tolist()  # type: ignore[operator]
            import json as _json
            debug_fh.write(_json.dumps({
                "step": len(order),
                "tokens": int(total_tokens),
                "cluster_actual_counts": cluster_actual,
                "cluster_expected_counts": cluster_expected,
                "reg_actual_counts": reg_actual,
                "reg_expected_counts": reg_expected,
                "reg_type": reg_type,
            }) + "\n")

        # --- chunk logging every `chunk_size` sequences ---
        if log_fh is not None and (len(order) % chunk_size == 0):
            expected_chunk_tokens = chunk_size * seq_len_const
            if has_sched:
                expected_chunk_counts = (desired_next_total - E_chunk_start)
            else:
                expected_chunk_counts = target_t * expected_chunk_tokens
            chunk_residual = chunk_counts - expected_chunk_counts
            chunk_l2 = torch.norm(chunk_residual, p=2).item()
            chunk_reg_error = 0.0
            if use_reg:
                if reg_type == "l2sum_schedule":
                    chunk_indices = order[-chunk_size:]
                    idx_t = torch.tensor(chunk_indices, device=device, dtype=torch.long)
                    chunk_l2sum = float(l2norms[idx_t].sum().item())
                    expected_chunk_norm = (chunk_size / n_total) * float(total_l2_sum)
                    chunk_reg_error = ((chunk_l2sum - expected_chunk_norm) / (seq_len_const * seq_len_const)) ** 2
                elif reg_type == "histogram_schedule":
                    chunk_indices = order[-chunk_size:]
                    idx_t = torch.tensor(chunk_indices, device=device, dtype=torch.long)
                    b_ids = bucket_ids[idx_t]
                    actual_chunk_bucket_counts = torch.bincount(b_ids, minlength=n_buckets).to(dtype=torch.float32)
                    expected_chunk_bucket_counts = total_bucket_counts * (chunk_size / n_total)
                    delta_chunk = actual_chunk_bucket_counts - expected_chunk_bucket_counts
                    chunk_reg_error = float(torch.norm(delta_chunk, p=2).item())
                elif reg_type == "w2_histogram_schedule":
                    chunk_indices = order[-chunk_size:]
                    idx_t = torch.tensor(chunk_indices, device=device, dtype=torch.long)
                    b_ids = bucket_ids[idx_t]
                    actual_chunk_bucket_counts = torch.bincount(b_ids, minlength=n_buckets).to(dtype=torch.float32)
                    expected_chunk_bucket_counts = total_bucket_counts * (chunk_size / n_total)
                    # Move to CPU for W2 cost; small vector
                    chunk_reg_error = float(np.sqrt(max(0.0, _w2_total_cost_squared(actual_chunk_bucket_counts.cpu(), expected_chunk_bucket_counts.cpu()))))
                elif reg_type == "docsize_token_schedule":
                    # Sum dense rows for this chunk into a (B,) vector
                    chunk_indices = order[-chunk_size:]
                    idx_t = torch.tensor(chunk_indices, device=device, dtype=torch.long)
                    chunk_token_hist = doc_hist_per_seq.index_select(0, idx_t).sum(dim=0)
                    expected_chunk_tokens_per_bin = doc_cluster_bin_prob.t().matmul(expected_chunk_counts)
                    delta_chunk = chunk_token_hist - expected_chunk_tokens_per_bin
                    chunk_reg_error = float(torch.norm(delta_chunk, p=2).item())
            import json as _json
            log_fh.write(_json.dumps({
                "step": len(order),
                "chunk_size": int(chunk_size),
                "chunk_tokens": int(expected_chunk_tokens),
                "chunk_l2": chunk_l2,
                "chunk_reg_error": float(chunk_reg_error)
            }) + "\n")
            chunk_counts.zero_()
            if has_sched:
                E_chunk_start = desired_next_total.clone()

    pbar.close()
    if debug_fh is not None:
        total_tokens = len(order) * seq_len_const
        cluster_actual = cum_counts.tolist()
        if has_sched:
            cluster_expected = E_prev.to(dtype=torch.float32).tolist()
        else:
            cluster_expected = (target_t * total_tokens).to(dtype=torch.float32).tolist()
        reg_actual: list[float] | list[int] = []
        reg_expected: list[float] | list[int] = []
        if use_reg:
            if reg_type == "l2sum_schedule":
                exp_norm = float(total_l2_sum)  # type: ignore[arg-type]
                reg_actual = [float(cum_l2sum)]
                reg_expected = [float(exp_norm)]
            elif reg_type == "histogram_schedule" and 'actual_bucket_counts' in locals():
                reg_actual = actual_bucket_counts.tolist()
                reg_expected = total_bucket_counts.to(dtype=torch.float32).tolist()  # type: ignore[union-attr]
            elif reg_type == "w2_histogram_schedule" and 'actual_bucket_counts' in locals():
                reg_actual = actual_bucket_counts.tolist()
                reg_expected = total_bucket_counts.to(dtype=torch.float32).tolist()  # type: ignore[union-attr]
            elif reg_type == "docsize_token_schedule" and 'cum_doc_tokens_per_bin' in locals():
                reg_actual = cum_doc_tokens_per_bin.tolist()
                reg_expected = (doc_cluster_bin_prob.t().matmul(E_prev)).to(dtype=torch.float32).tolist()  # type: ignore[operator]
        import json as _json
        debug_fh.write(_json.dumps({
            "step": n_total,
            "tokens": int(total_tokens),
            "cluster_actual_counts": cluster_actual,
            "cluster_expected_counts": cluster_expected,
            "reg_actual_counts": reg_actual,
            "reg_expected_counts": reg_expected,
            "reg_type": reg_type,
        }) + "\n")
        debug_fh.close()
    if log_fh is not None:
        log_fh.close()
    if additions_fh is not None:
        additions_fh.close()
    if doc_additions_fh is not None:
        doc_additions_fh.close()
    return order


# --------------------------------------------------------------------------- #
# Tar writing & manifest                                                      #
# --------------------------------------------------------------------------- #
def write_output(order: list[int], tokens_all: List[bytes], counts_all_cpu: torch.Tensor,
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
        cnt = counts_all_cpu[idx].to(dtype=torch.int32, device='cpu').tolist()
        buf_tokens.append(tok_bytes)

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
    if tot_tokens > 0:
        final_ratio = (cum_counts / tot_tokens).tolist()
    else:
        final_ratio = [0.0 for _ in range(len(target))]
    with open(fout, 'w') as f:
        for m in manifest:
            f.write(json.dumps(m) + '\n')
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


def chunked_to_sparse_csr(tensor, chunk_size=1024 * 64):
    chunks = [tensor[i:i+chunk_size].to_sparse_csr() 
              for i in range(0, tensor.size(0), chunk_size)]
    
    values = torch.cat([c.values() for c in chunks])
    col_indices = torch.cat([c.col_indices() for c in chunks])
    
    crow_indices = [chunks[0].crow_indices()]
    for c in chunks[1:]:
        crow_indices.append(c.crow_indices()[1:] + crow_indices[-1][-1])
    crow_indices = torch.cat(crow_indices)
    
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=tensor.size())


# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--ratio-file', required=False, help='JSON dict of {knot_tokens: {dataset{i}: ratio, ...}}; Keys must be dataset0..dataset{n_cluster-1}; no other key format is accepted. If omitted, ratios are inferred from data')
    ap.add_argument('--out-dir', required=True,
                    help="Output directory. Will contain token shards + manifest.jsonl, a 'counts' subdir for counts shards, and error_log.jsonl")
    ap.add_argument('--shard-size', type=int, default=64,
                    help="Number of sequences per output shard")
    ap.add_argument('--loader-workers', type=int, default=8,
                    help='Worker processes for WebLoader streaming')
    ap.add_argument('--loader-prefetch', type=int, default=8,
                    help='Prefetch batches per worker (WebLoader)')
    ap.add_argument('--chunk-size', type=int, default=512,
                    help='Chunk size for per-chunk L2 logging (sequences)')
    ap.add_argument('--reg-type', type=str, default='docsize_token_schedule', choices=['none', 'l2sum_schedule', 'histogram_schedule', 'w2_histogram_schedule', 'docsize_token_schedule'],
                    help='Regularization strategy to apply during selection')
    ap.add_argument('--reg-lambda', type=float, default=1.0,
                    help='Weight applied to the regularization term (0.0 disables)')
    ap.add_argument('--n-buckets', type=int, default=10,
                    help='Number of buckets for histogram_schedule regularizer')
    ap.add_argument('--bucket-method', type=str, default='quantile', choices=['uniform', 'quantile'],
                    help='Bucketization method for histogram_schedule regularizer')
    ap.add_argument('--doc-bins', type=int, default=100,
                    help='Number of size bins for docsize_token_schedule (token-weighted quantiles)')
    ap.add_argument('--rand', action='store_true', help='Choose sequences uniformly at random instead of greedy error minimization')
    ap.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'],
                    help='GPU storage dtype for counts (affects counts_all on device only)')
    ap.add_argument('--truncate-mod', type=int, default=0,
                    help='If > 0, discard the final len(order) % value sequences so total is a multiple of this')
    ap.add_argument('--total-tokens', type=int, default=0,
                    help='If > 0, schedule only this many tokens (rounded down to full sequences) and stop')
    ap.add_argument('--offset', type=int, default=0,
                    help='Absolute token offset to start the schedule at. Only used when a schedule is provided.')
    args = ap.parse_args()

    if int(getattr(args, 'offset', 0)) < 0:
        raise ValueError('--offset must be >= 0')

    # ------------------------------------------------------------------ load tars
    shard_paths = sorted(Path(args.input_dir).glob('shard_*.tar'))
    if len(shard_paths) == 0:
        raise FileNotFoundError(f"No shards found in {args.input_dir}")

    # Determine number of clusters (n_cluster) from the first counts tensor
    first_counts_path = shard_paths[0].parent / f"{shard_paths[0].stem}_counts.pt"
    if not first_counts_path.exists():
        raise FileNotFoundError(f"Expected counts file {first_counts_path} for shard {shard_paths[0]}")
    _first_counts: torch.Tensor = torch.load(first_counts_path, map_location="cpu")  # type: ignore
    if _first_counts.ndim != 2:
        raise ValueError(f"Counts tensor in {first_counts_path} must be 2D")
    n_cluster = int(_first_counts.shape[1])
    print(f'Clusters: {n_cluster}')

    print(f"Found {len(shard_paths)} shards; loading counts tensors and streaming token data …")

    counts_all_cpu, tokens_all, _ = stream_shards_to_arrays(
        shard_paths,
        n_cluster=n_cluster,
        workers=args.loader_workers,
        prefetch=args.loader_prefetch,
    )

    # If ratio file is provided, read it and validate; else infer from data
    sched_info = None
    if args.ratio_file:
        ratio_path = Path(args.ratio_file)
        raw = json.loads(ratio_path.read_text())
        # Enforce dict-of-knots format only
        if not isinstance(raw, dict) or len(raw) == 0:
            raise ValueError("--ratio-file must be a non-empty JSON dict of {knot_tokens: {dataset{i}: ratio, ...}}")

        # Convert keys to floats, ensure finite and > 0, and sort strictly increasing
        knots_list = []
        per_knot_dicts = []
        for k, v in raw.items():
            if not isinstance(v, dict) or len(v) == 0:
                raise ValueError("Each knot must map to a non-empty dict of dataset ratios")
            try:
                k_float = float(k)
            except Exception:
                raise ValueError(f"Knot key '{k}' is not numeric")
            if not np.isfinite(k_float) or k_float <= 0.0:
                raise ValueError(f"Knot value {k} must be finite and > 0")
            knots_list.append(k_float)
            per_knot_dicts.append(v)

        # Create sorted order by knot value
        order = np.argsort(np.asarray(knots_list, dtype=np.float64))
        knots_sorted = [knots_list[i] for i in order]
        dicts_sorted = [per_knot_dicts[i] for i in order]
        # Check strictly increasing
        for i in range(1, len(knots_sorted)):
            if not (knots_sorted[i] > knots_sorted[i-1]):
                raise ValueError("Knot tokens must be strictly increasing after sorting and > 0")

        # Determine dataset column order from first knot
        # Keys must be dataset{i}; dataset{i} maps to cluster i (no fallback).
        first_keys = list(dicts_sorted[0].keys())
        pat = re.compile(r'^dataset(\d+)$')
        matches = [pat.fullmatch(k) for k in first_keys]
        if not all(m is not None for m in matches):
            raise ValueError("Per-knot ratio keys must be of the form dataset{i} with integer suffixes")
        suffixes = sorted({int(m.group(1)) for m in matches if m is not None})
        expected = list(range(n_cluster))
        if suffixes != expected:
            raise ValueError(f"Dataset keys must be contiguous dataset0..dataset{n_cluster-1}; got {suffixes}")
        cols = [f"dataset{i}" for i in expected]

        # Ensure all knots share exactly the same dataset key set
        first_key_set = set(first_keys)
        for idx, d in enumerate(dicts_sorted[1:], start=1):
            if set(d.keys()) != first_key_set:
                raise ValueError(f"Per-knot dataset key sets must match; mismatch at knot index {idx}")

        # Build ratio matrix P with established column order
        K = len(knots_sorted)
        C = len(cols)
        P = np.empty((K, C), dtype=np.float32)
        for i in range(K):
            d = dicts_sorted[i]
            row = []
            for name in cols:
                val = d.get(name, None)
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Ratio for dataset '{name}' at knot {knots_sorted[i]} must be a number")
                if not np.isfinite(val):
                    raise ValueError(f"Ratio for dataset '{name}' at knot {knots_sorted[i]} must be finite")
                row.append(float(val))
            P[i, :] = np.asarray(row, dtype=np.float32)

        # Validate number of datasets equals n_cluster
        if C != n_cluster:
            raise ValueError(f"Schedule has {C} datasets but data has {n_cluster} clusters")

        sched_knots_np = np.asarray(knots_sorted, dtype=np.float64)
        sched_P_np = P
        target = sched_P_np[-1].astype(np.float32)
        sched_info = { 'P': sched_P_np, 'knots': sched_knots_np }
    else:
        totals = counts_all_cpu.sum(dim=0).to(dtype=torch.float64)
        total_sum = float(totals.sum().item())
        if total_sum == 0.0:
            raise ValueError("Total token count across all clusters is zero; cannot infer ratios")
        target = (totals.cpu().numpy() / total_sum).astype(np.float32)

    # ------------------------------------------------------------------ move to GPU and build sparse representation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA device not available; this script requires a GPU")

    if args.dtype == 'fp16':
        dty = torch.float16
    elif args.dtype == 'bf16':
        dty = torch.bfloat16
    else:
        dty = torch.float32

    counts_all_gpu = counts_all_cpu.to(device=device, dtype=dty)
    # Build CSR directly on GPU to avoid duplicating CSR on CPU
    #counts_sparse_gpu = counts_all_gpu.to_sparse_csr()
    counts_sparse_gpu = chunked_to_sparse_csr(counts_all_gpu)

    # Precompute per-sequence L2 norms and their total sum for regularization (on GPU)
    l2norms = torch.sqrt((counts_all_gpu * counts_all_gpu).sum(dim=1).to(dtype=torch.float32))
    total_l2_sum = float(l2norms.sum().item())

    # Build schedule tensors on CUDA if present
    sched_P_t = None
    sched_knots_t = None
    sched_U_t = None
    sched_inv_du_t = None
    if sched_info is not None:
        sched_P_t = torch.tensor(sched_info['P'], dtype=torch.float32, device=device)
        sched_knots_t = torch.tensor(sched_info['knots'], dtype=torch.float32, device=device)
        # Log-domain interpolation support
        sched_U_t = torch.log(sched_knots_t)
        du = sched_U_t[1:] - sched_U_t[:-1]
        sched_inv_du_t = 1.0 / du

    # Histogram regularizer precomputation if requested (on GPU)
    bucket_ids = None
    total_bucket_counts = None
    if args.reg_type in ('histogram_schedule', 'w2_histogram_schedule'):
        n_b = int(args.n_buckets)
        if args.bucket_method == 'quantile':
            quantiles = torch.linspace(0, 1, steps=n_b + 1, device='cpu')
            boundaries = torch.as_tensor(np.quantile(l2norms.cpu().numpy(), quantiles.cpu().numpy()), dtype=torch.float32, device='cpu')
            boundaries[0] = float('-inf'); boundaries[-1] = float('inf')
            print(f"Quantile histogram boundaries (including endpoints): {boundaries.tolist()}")
            print(f"Quantile histogram inner edges ({n_b - 1}): {boundaries[1:-1].tolist()}")
            # bucketize on GPU using inner edges
            inner_edges = boundaries[1:-1].to(device=device)
            bucket_ids = torch.bucketize(l2norms, inner_edges, right=True)
        else:
            lmin = float(l2norms.min().item()); lmax = float(l2norms.max().item())
            if lmax == lmin:
                bucket_ids = torch.zeros_like(l2norms, dtype=torch.long)
            else:
                scaled = (l2norms - lmin) / max(1e-12, (lmax - lmin))
                bucket_ids = torch.clamp((scaled * n_b).floor().to(torch.long), 0, n_b - 1)
        n_buckets = n_b
        total_bucket_counts = torch.bincount(bucket_ids, minlength=n_buckets).to(dtype=torch.float32)
    else:
        n_buckets = None

    # Docsize-token regularizer precomputation if requested (dense on GPU)
    doc_hist_per_seq = None
    doc_total_tokens_per_bin = None
    doc_n_bins = None
    doc_row_norm2 = None
    doc_cluster_bin_prob = None
    if args.reg_type == 'docsize_token_schedule':
        B = int(args.doc_bins)
        crow = counts_sparse_gpu.crow_indices()
        vals = counts_sparse_gpu.values()
        N = int(counts_all_cpu.shape[0])
        seq_len_const = int(counts_all_cpu[0].sum().item())
        sizes = vals.round().to(dtype=torch.int64).clamp_min(1).clamp_max(seq_len_const)
        doc_count_by_size = torch.bincount(sizes, minlength=seq_len_const + 1).to(dtype=torch.int64)
        token_mass_by_size = (doc_count_by_size * torch.arange(seq_len_const + 1, dtype=torch.int64, device=doc_count_by_size.device))
        total_mass = int(token_mass_by_size.sum().item())
        if total_mass == 0:
            raise ValueError("No token mass found for doc sizes; cannot build docsize_token_schedule bins")
        cumsum_mass = torch.cumsum(token_mass_by_size.to(dtype=torch.int64), dim=0)
        q = torch.linspace(0, 1, steps=B + 1, device=cumsum_mass.device)
        inner = []
        for k in range(1, B):
            thr = int((q[k].item()) * total_mass)
            idx = int(torch.searchsorted(cumsum_mass, torch.tensor(thr, dtype=torch.int64, device=cumsum_mass.device), right=True).item())
            inner.append(float(idx))
        if len(inner) == 0:
            inner_edges_t = torch.empty(0, dtype=torch.float32, device=cumsum_mass.device)
        else:
            inner_edges_t = torch.tensor(inner, dtype=torch.float32, device=cumsum_mass.device)
        if B > 0:
            edges_full = torch.empty(B + 1, dtype=torch.float32, device='cpu')
            edges_full[0] = float('-inf')
            if inner_edges_t.numel() > 0:
                edges_full[1:-1] = inner_edges_t.cpu()
            else:
                edges_full[1:-1] = 0.0
            edges_full[-1] = float('inf')
            print(f"Docsize token-weighted quantile boundaries (including endpoints): {edges_full.tolist()}")
            print(f"Docsize inner edges ({B - 1}): {inner_edges_t.cpu().tolist()}")
        doc_n_bins = B
        if inner_edges_t.numel() == 0:
            entry_bin_ids = torch.zeros_like(sizes, dtype=torch.long)
        else:
            entry_bin_ids = torch.bucketize(sizes.to(dtype=torch.float32), inner_edges_t, right=True)
        counts_per_row = (crow[1:] - crow[:-1]).to(dtype=torch.long)
        row_ids = torch.repeat_interleave(torch.arange(N, dtype=torch.long, device=counts_per_row.device), counts_per_row)
        # Build dense (N, B) histogram by flattened index_add on GPU
        hist_flat = torch.zeros(N * B, dtype=torch.float32, device=vals.device)
        flat_indices = row_ids * B + entry_bin_ids
        hist_flat.index_add_(0, flat_indices, vals.to(dtype=torch.float32))
        doc_hist_per_seq = hist_flat.view(N, B)
        doc_total_tokens_per_bin = doc_hist_per_seq.sum(dim=0)
        doc_row_norm2 = (doc_hist_per_seq * doc_hist_per_seq).sum(dim=1)

        # Build (C, B) doc_cluster_bin_tokens via aggregation over (cluster, bin)
        C = int(counts_sparse_gpu.size(1))
        col_ids = counts_sparse_gpu.col_indices()
        cluster_bin_flat = torch.zeros(C * B, dtype=torch.float32, device=vals.device)
        flat_cb_indices = col_ids * B + entry_bin_ids
        cluster_bin_flat.index_add_(0, flat_cb_indices, vals.to(dtype=torch.float32))
        doc_cluster_bin_tokens = cluster_bin_flat.view(C, B)
        # Row-normalize to probabilities
        row_sums = doc_cluster_bin_tokens.sum(dim=1, keepdim=True)
        doc_cluster_bin_prob = doc_cluster_bin_tokens / (row_sums + 1e-12)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_dir = out_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    counts_dir = out_dir / "counts"

    error_log_path = str(out_dir / "error_log.jsonl")
    debug_log_path = str(out_dir / "debug.json")
    addition_log_path = str(out_dir / "target_additions.jsonl")
    doc_addition_log_path = str(out_dir / "docsize_additions.jsonl")

    # Determine optional step budget from --total-tokens argument
    seq_len_const = int(counts_all_cpu[0].sum().item())
    total_tokens_arg = int(getattr(args, 'total_tokens', 0) or 0)
    if total_tokens_arg > 0:
        max_sequences = int(total_tokens_arg // seq_len_const)
        if max_sequences <= 0:
            print(f"Requested total tokens {total_tokens_arg} < sequence length {seq_len_const}; producing 0 sequences")
    else:
        max_sequences = None

    order_indices = greedy_gpu_sparse(
        counts_all_gpu,
        counts_sparse_gpu,
        target,
        k_return=1,
        error_log_path=error_log_path,
        chunk_size=args.chunk_size,
        debug_log_path=debug_log_path,
        addition_log_path=addition_log_path,
        doc_addition_log_path=doc_addition_log_path,
        reg_type=args.reg_type,
        reg_lambda=args.reg_lambda,
        l2norms=l2norms,
        total_l2_sum=total_l2_sum,
        bucket_ids=bucket_ids,
        total_bucket_counts=total_bucket_counts,
        n_buckets=n_buckets,
        random_select=bool(getattr(args, 'rand', False)),
        doc_hist_per_seq=doc_hist_per_seq,
        doc_total_tokens_per_bin=doc_total_tokens_per_bin,
        doc_n_bins=doc_n_bins,
        doc_row_norm2=doc_row_norm2,
        doc_cluster_bin_prob=doc_cluster_bin_prob,
        sched_P=sched_P_t,
        sched_knots=sched_knots_t,
        sched_U=sched_U_t,
        sched_inv_du=sched_inv_du_t,
        offset_tokens=int(getattr(args, 'offset', 0) or 0),
        max_sequences=max_sequences,
    )

    # Optionally truncate the final ordered list to a multiple of --truncate-mod
    truncate_mod = int(getattr(args, 'truncate_mod', 0) or 0)
    if truncate_mod > 0:
        remainder = len(order_indices) % truncate_mod
        if remainder != 0:
            orig_len = len(order_indices)
            order_indices = order_indices[:-remainder]
            print(f"Truncated ordered sequences from {orig_len} to {len(order_indices)} (multiple of {truncate_mod})")

    # ------------------------------------------------------------------ write output (using CPU counts for serialization)
    manifest = write_output(order_indices, tokens_all, counts_all_cpu, tokens_dir, counts_dir, args.shard_size)

    # final stats
    cum = counts_all_cpu[order_indices].sum(dim=0).cpu().numpy()
    tot = float(cum.sum())
    write_manifest(manifest, tokens_dir, target, cum, tot)
    print('Done.')

if __name__ == '__main__':
    main()