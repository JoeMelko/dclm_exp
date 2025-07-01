#!/usr/bin/env python3

"""
Compute block-wise regularized inverse Fisher matrices from gradient memmap and a single reference
vector built from clipped, whitened gradients.

The gradient memmap is assumed to be stored row-major with shape:
    (N, 2 * num_blocks, rank * rank)

Each slice data[:, b, :] corresponds to one block (b = 0 ... 2*num_blocks-1)
of flattened projected gradients of dimension d = rank*rank.

For every block, this script:
1. Forms the Fisher block F_b = (X^T X) / N
2. Adds a scalar ridge λ I chosen so that the condition number of
   F_b + λ I equals the user-supplied --cond value.
3. Inverts the regularized block and stores the inverse square root (whitener).
4. Whitens every per-sample gradient, clips it by its L2 norm relative to the
   90th-percentile norm across the dataset, and averages the clipped vectors to
   obtain a single L2-normalised reference direction.
5. Saves the reference direction (.npy, float32, shape (2*num_blocks*rank^2,))
   and the whitener blocks (.npy, float32, shape (2*num_blocks, d, d)).
"""

import argparse
import os
import numpy as np
import math
from tqdm import tqdm
import torch

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute inverse Fisher blocks from gradient memmap.")
    p.add_argument("--mmap-path", type=str, required=True,
                   help="Path to the memory‑mapped file produced during gradient collection.")
    p.add_argument("--rank", type=int, required=True,
                   help="Rank of the projection matrices (d = rank*rank features per block).")
    p.add_argument("--num-blocks", type=int, required=True,
                   help="Number of projection blocks. Total blocks processed = 2 * num_blocks.")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                   help="Storage dtype of the memmap ('fp16' or 'fp32').")
    p.add_argument("--cond", type=float, default=1e4,
                   help="Target condition number after ridge regularization.")
    p.add_argument("--out-path", type=str, required=True,
                   help="Destination .npy file for the final averaged L2-normalized direction.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-block diagnostics.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype = np.float16 if args.dtype == "fp16" else np.float32
    d = args.rank * args.rank
    B = 2 * args.num_blocks

    itemsize = np.dtype(dtype).itemsize
    file_size_bytes = os.path.getsize(args.mmap_path)
    total_elems = file_size_bytes // itemsize

    if file_size_bytes % itemsize != 0:
        raise ValueError("File size is not an integer multiple of element size.")

    if total_elems % (B * d) != 0:
        raise ValueError(
            f"Provided rank/num_blocks do not match file size. "
            f"File has {total_elems} elements, expected multiple of {B * d}."
        )

    N = total_elems // (B * d)
    if args.verbose:
        print(f"Detected shape: (N={N}, 2*num_blocks={B}, d={d})")

    # Read-only memory map of full data (N, B, d) then convert to torch (B, N, d)
    mm = np.memmap(args.mmap_path, dtype=dtype, mode="r", shape=(N, B, d))

    grads = torch.from_numpy(mm).to(torch.float32).permute(1, 0, 2).contiguous()  # (B, N, d)

    if args.verbose:
        print(f"Loaded gradients with shape {grads.shape} (B, N, d)")

    whiteners = torch.empty((B, d, d), dtype=torch.float32)  # stored on CPU

    # Select computation device for heavy linear-algebra kernels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print(f"Using device {device} for Fisher inversion.")

    eye_cpu = torch.eye(d, dtype=torch.float32)
    eye_gpu = eye_cpu.to(device)

    for b in tqdm(range(B)):
        if args.verbose:
            print(f"Processing block {b}/{B-1}")

        # Move this block to GPU for computation
        X = grads[b].to(device, non_blocking=True)          # shape (N, d)
        X /= math.sqrt(X.shape[0])          # normalize by sqrt(N)
        F = (X.T @ X)                       # Fisher block (d, d) on GPU

        # Eigenvalues for condition number and ridge calculation
        w, _ = torch.linalg.eigh(F)
        w_min, w_max = w[0], w[-1]

        # Desired condition number
        c_star = args.cond
        if c_star <= 1.0:
            raise ValueError("--cond must be > 1.")

        # Solve for ridge λ: (λ_max + λ) / (λ_min + λ) = c_star
        lam = (w_max - c_star * w_min) / (c_star - 1.0)
        lam = max(lam, 0.0)   # Never subtract

        print("apply lambda")
        F_reg = F + lam * eye_gpu

        # Compute inverse of F_reg
        F_inv = torch.linalg.inv(F_reg)

        # Eigen-decomposition of the inverse to obtain inverse square root
        evals, evecs = torch.linalg.eigh(F_inv)
        inv_sqrt = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T            # F^{-1/2}

        whitener_b = inv_sqrt.to(torch.float32).cpu()                       # back to CPU
        whiteners[b] = whitener_b

        # Free GPU memory for this block
        del X, F, F_reg, F_inv, evals, evecs, inv_sqrt, w
        torch.cuda.empty_cache() if device.type == "cuda" else None

        if args.verbose:
            cond_before = w_max / w_min
            cond_after = torch.linalg.cond(F_reg)
            print(
                f"Block {b:>3d}: λ={lam:.3e}  "
                f"cond_before={cond_before:.3e}  cond_after={cond_after:.3e}"
            )

    if args.verbose:
        print("Whitening gradients on GPU in batches and computing row norms (single pass).")

    # Move whitener blocks once to GPU for fast batched multiplication
    whiteners_gpu = whiteners.to(device)

    PROC_BS = 8192  # samples per GPU batch – adjust if memory allows
    grad_norms = torch.empty(N, dtype=torch.float32)  # to store L2 norms on CPU

    # Process dataset in chunks along the N dimension
    for start in tqdm(range(0, N, PROC_BS)):
        end = min(start + PROC_BS, N)

        # Slice (B, bs, d) -> move to GPU
        batch = grads[:, start:end, :].to(device)

        # Whiten: (B, bs, d)
        whitened = torch.bmm(batch, whiteners_gpu)

        # Overwrite the original gradients tensor with whitened values (back on CPU)
        grads[:, start:end, :] = whitened.cpu()

        # Compute L2 norm for each sample in this batch (flatten across blocks)
        flat = whitened.permute(1, 0, 2).reshape(end - start, -1)
        grad_norms[start:end] = flat.norm(p=2, dim=1).cpu()

        # Explicitly free GPU memory of this batch
        del batch, whitened, flat
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # 90-th percentile over all norms
    p90 = torch.quantile(grad_norms, 0.9)
    print(f"90th-percentile grad norm = {p90.item():.6e}")

    # ----------------------------
    #   Clip & compute reference direction on CPU
    # ----------------------------

    flat_all = grads.permute(1, 0, 2).reshape(N, -1)   # (N, B*d)
    denom = torch.maximum(grad_norms, p90)             # (N,)

    # Compute in a single matmul: sum_i (flat_all[i] / denom[i])
    inv_weights = 1.0 / denom                          # (N,)
    final_direction = torch.matmul(flat_all.T, inv_weights) / N  # (B*d,)

    final_direction = final_direction / final_direction.norm(p=2, dim=0)

    np.save(args.out_path, final_direction.cpu().numpy().astype(np.float32))
    if args.verbose:
        print(f"Saved reference direction with shape {final_direction.shape} to {args.out_path}")

    # Save whitener matrices for later use (B, d, d)
    whiteners_out_path = args.out_path.replace('.npy', '_whiteners.npy') if args.out_path.endswith('.npy') else args.out_path + '_whiteners.npy'
    np.save(whiteners_out_path, whiteners.numpy())
    if args.verbose:
        print(f"Saved whitener matrices with shape {whiteners.shape} to {whiteners_out_path}")


if __name__ == "__main__":
    main()
