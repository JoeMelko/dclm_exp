#!/usr/bin/env python3

"""
Compute block-wise regularised Fisher (Hessian) matrices for a set of
projected gradients stored in a memory-mapped file.

Pipeline
--------
1. Load the memmap with shape `(N, 2*num_blocks, rank*rank)` (row-major on disk).
2. Compute the L2 norm of every sample (flattening all blocks) on the GPU in 8192-row batches.
3. Clip each sample by scaling so its norm does not exceed the chosen percentile threshold
   (`--clip-percentile`, default **99.9**).
4. For each block `b = 0 … 2*num_blocks-1`:
   a. Form the Fisher block `F_b = X_bᵀ X_b / N` on the GPU.
   b. Add a diagonal ridge `λ I` such that `cond(F_b + λ I) == --cond`.
   c. Add the ridge and store the regularised Fisher block `\tilde{F}_b = F_b + \lambda I` on CPU.
   d. Release GPU memory before processing the next block.
5. Save the stacked Hessian blocks to `--out-path` with shape
   `(2*num_blocks, rank*rank, rank*rank)` (float32).
6. Save the per-sample norm vector via `--norms-path` (default `<out-path>_norms.npy`).

Example
-------

```bash
python -m dclm_exp.clustering.mgd.hessian \
  --mmap-path /path/to/grads.mmap \
  --rank 32 \
  --num-blocks 60 \
  --dtype fp16 \
  --out-path whiteners.npy \
  --cond 1e4 \
  --clip-percentile 99.9 \
  --verbose
```
"""

import argparse
import os
import numpy as np
import math
from tqdm import tqdm
import torch

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute regularised Fisher (Hessian) blocks from gradient memmap.")
    p.add_argument("--mmap-path", type=str, required=True,
                   help="Path to the memory‑mapped file produced during gradient collection.")
    p.add_argument("--rank", type=int, required=True,
                   help="Rank of the projection matrices (d = rank*rank features per block).")
    p.add_argument("--num-blocks", type=int, required=True,
                   help="Number of projection blocks. Total blocks processed = 2 * num_blocks.")
    p.add_argument("--dtype", choices=["fp16", "fp32"], required=True,
                   help="Storage dtype of the memmap ('fp16' or 'fp32').")
    p.add_argument("--cond", type=float, required=True,
                   help="Target condition number after ridge regularization.")
    p.add_argument("--out-path", type=str, required=True,
                   help="Destination .npy file for the whitening matrices (shape=(2*num_blocks, rank^2, rank^2)).")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-block diagnostics.")
    p.add_argument("--clip-percentile", type=float, required=True,
                   help="Percentile used to compute the clipping threshold (e.g. 99.9 means clip to the 99.9th percentile norm).")
    p.add_argument("--norms-path", type=str, default=None,
                   help="Optional path to store the per-sample L2 norms (float32, shape=(N,)). If omitted, '<out-path>_norms.npy' is used.")
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

    if args.verbose:
        print(f"Loaded gradients with shape {mm.shape} (N, B, d)")

    # ----------------------------
    #   1. Clip gradients by L2 norm
    # ----------------------------

    # Compute per-sample L2 norms on GPU in manageable batches
    grads_cpu = torch.from_numpy(mm).to(torch.float32)   # (N, B, d)

    grad_norms = torch.empty(N, dtype=torch.float32)

    PROC_BS = 16384  # rows per GPU batch
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for start in tqdm(range(0, N, PROC_BS), disable=not args.verbose):
        end = min(start + PROC_BS, N)

        batch = grads_cpu[start:end].to(device_gpu)  # (bs, B, d)
        flat = batch.reshape(end - start, -1)                           # (bs, B*d)
        norms = flat.norm(p=2, dim=1)                                   # (bs,)

        grad_norms[start:end] = norms.cpu()

        del batch, flat, norms
        torch.cuda.empty_cache() if device_gpu.type == "cuda" else None

    clip_p = args.clip_percentile / 100.0
    threshold = torch.quantile(grad_norms, clip_p)

    if args.verbose:
        print(f"{args.clip_percentile}-percentile norm = {threshold.item():.6e}")

    # Compute scaling factors ≤ 1.0
    scaling = torch.minimum(torch.ones_like(grad_norms), threshold / grad_norms)

    # Apply clipping in-place (CPU tensor)
    grads_cpu.mul_(scaling.view(-1, 1, 1))

    # Save norms vector if requested
    norms_path = args.norms_path if args.norms_path is not None else (
        args.out_path.replace('.npy', '_norms.npy') if args.out_path.endswith('.npy') else args.out_path + '_norms.npy'
    )
    np.save(norms_path, grad_norms.numpy().astype(np.float32))
    if args.verbose:
        print(f"Saved gradient norms with shape {grad_norms.shape} to {norms_path}")

    # Re-arrange for block-wise processing: (B, N, d)
    grads = grads_cpu.permute(1, 0, 2)

    # Free temporary tensors
    del scaling

    # ----------------------------
    #   2. Compute regularised Fisher (Hessian) blocks
    # ----------------------------

    hessians = torch.empty((B, d, d), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print(f"Using device {device} for Fisher (Hessian) computation.")

    eye_gpu = torch.eye(d, dtype=torch.float32, device=device)

    for b in tqdm(range(B)):

        X_cpu = grads[b]                            # (N, d) on CPU
        X = X_cpu.to(device)     # move to GPU

        F = X.T @ X                                 # Fisher block (d,d)

        # Eigenvalues for ridge calculation
        w, _ = torch.linalg.eigh(F)
        w_min, w_max = w[0], w[-1]

        c_star = args.cond
        if c_star <= 1.0:
            raise ValueError("--cond must be > 1.")

        lam = (w_max - c_star * w_min) / (c_star - 1.0)
        lam = max(lam, 0.0)

        F_reg = F + lam * eye_gpu
        
        # Invert the regularised Fisher block and store the inverse (whitening matrix)
        F_reg_inv = torch.linalg.inv(F_reg)

        # Store the inverse Hessian block
        hessians[b] = F_reg_inv.to(torch.float32).cpu()

        if args.verbose:
            cond_before = w_max / w_min
            cond_after = torch.linalg.cond(F_reg)
            print(
                f"Block {b:>3d}: λ={lam:.3e}  cond_before={cond_before:.3e}  cond_after={cond_after:.3e}"
            )
        # Cleanup GPU memory
        del X, F, F_reg, F_reg_inv, w
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ----------------------------
    #   3. Save Hessian matrices
    # ----------------------------

    np.save(args.out_path, hessians.numpy())
    if args.verbose:
        print(f"Saved Hessian matrices with shape {hessians.shape} to {args.out_path}")


if __name__ == "__main__":
    main()
