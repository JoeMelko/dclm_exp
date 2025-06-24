#!/usr/bin/env python3

"""Compute block-wise regularized inverse Fisher matrices from gradient memmap.

The gradient memmap is assumed to be stored row-major with shape:
    (N, 2 * num_blocks, rank * rank)

Each slice data[:, b, :] corresponds to one block (b = 0 ... 2*num_blocks-1)
of flattened projected gradients of dimension d = rank*rank.

For every block, this script:
1. Forms the Fisher block F_b = (X^T X) / N
2. Adds a scalar ridge λ I chosen so that the condition number of
   F_b + λ I equals the user‑supplied --cond value.
3. Inverts the regularized block.
4. Stores all inverse blocks in a single .npy file with shape
   (2 * num_blocks, d, d) in float32 for efficient later loading.
"""

import argparse
import os
import numpy as np
import math
from tqdm import tqdm
import torch

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute inverse Fisher blocks from gradient memmap.")
    p.add_argument("--mmap_path", type=str, required=True,
                   help="Path to the memory‑mapped file produced during gradient collection.")
    p.add_argument("--rank", type=int, required=True,
                   help="Rank of the projection matrices (d = rank*rank features per block).")
    p.add_argument("--num_blocks", type=int, required=True,
                   help="Number of projection blocks. Total blocks processed = 2 * num_blocks.")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32",
                   help="Storage dtype of the memmap ('fp16' or 'fp32').")
    p.add_argument("--cond", type=float, default=1e6,
                   help="Target condition number after ridge regularization.")
    p.add_argument("--out_path", type=str, required=True,
                   help="Destination .npy file for the inverse blocks.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per‑block diagnostics.")
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

    # Read‑only memory map
    mm = np.memmap(args.mmap_path, dtype=dtype, mode="r", shape=(N, B, d))
    
    grads = torch.from_numpy(mm[:500_000]).permute(1, 0, 2).to(torch.float32, memory_format=torch.contiguous_format)

    print("grads.shape", grads.shape)
 
    inverses = torch.empty((B, d, d), dtype=torch.float32)

    eye = torch.eye(d, dtype=torch.float32)

    for b in tqdm(range(B)):
        # Cast to float64 for numerical robustness during eigen‑decomp & inversion
        print("grads[b].shape", grads[b].shape)
        X = grads[b]      # shape (N, d)
        print("assign X")
        X /= math.sqrt(X.shape[0]) # normalize by sqrt(N)
        print("X.shape", X.shape)
        F = (X.T @ X)                                  # shape (d, d)
        print("F.shape", F.shape)
        # Eigenvalues for condition number and ridge calculation
        w, _ = torch.linalg.eigh(F)
        w_min, w_max = w[0], w[-1]
        print("w_min", w_min)
        print("w_max", w_max)

        # Ensure strictly positive definiteness before conditioning
        '''if w_min <= 0:
            shift = (abs(w_min) + 1e-12)
            F += shift * eye
            w_min += shift
            w_max += shift
            if args.verbose:
                print(f"Block {b}: shifted by {shift:.3e} to make PD.")'''

        # Desired condition number
        c_star = args.cond
        if c_star <= 1.0:
            raise ValueError("--cond must be > 1.")

        # Solve for ridge λ: (λ_max + λ) / (λ_min + λ) = c_star
        print("solve for lambda")
        lam = (w_max - c_star * w_min) / (c_star - 1.0)
        lam = max(lam, 0.0)   # Never subtract

        print("apply lambda")
        F_reg = F + lam * eye

        # Inverse (back to float32)
        print("inverse")
        inverses[b] = torch.linalg.inv(F_reg)

        if args.verbose:
            cond_before = w_max / w_min
            cond_after = torch.linalg.cond(F_reg)
            print(
                f"Block {b:>3d}: λ={lam:.3e}  "
                f"cond_before={cond_before:.3e}  cond_after={cond_after:.3e}"
            )

    inverses = inverses.numpy()
    np.save(args.out_path, inverses)
    if args.verbose:
        print(f"Saved inverse blocks with shape {inverses.shape} to {args.out_path}")
    breakpoint()
    
    # (B, N, d)  →  (B, d)
    mean_vectors = grads.mean(dim=1)                         # μ_b

    # batched matrix-vector:  (B, d, d)  ×  (B, d, 1)  →  (B, d, 1)
    whitened = torch.bmm(inverses, mean_vectors.unsqueeze(-1)).squeeze(-1)

    # concatenate blocks
    desired_direction = whitened.reshape(-1)                 # (B*d,)
    
    # save the desired direction
    np.save(args.out_path.replace('.npy', '_direction.npy'), desired_direction.numpy())
    if args.verbose:
        print(f"Saved desired direction with shape {desired_direction.shape} to {args.out_path.replace('.npy', '_direction.npy')}")


if __name__ == "__main__":
    main()
