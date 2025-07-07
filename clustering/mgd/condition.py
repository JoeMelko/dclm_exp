#!/usr/bin/env python3
"""
condition.py
-------------
Aggregate the per-GPU target direction vectors written by ``launch_get_target.sh``
and post-process them so that they can be used as a *whitened* optimisation
condition.

Workflow
========
1. Load ``dir_0/sum.npy … dir_{N-1}/sum.npy`` (see ``get_target.py``) and sum
   them in float32.
2. Load the inverse Fisher square-root matrices produced by ``hessian.py``
   (``--whiteners-path``).
3. Whiten the aggregate target by multiplying every block vector ``v₍b₎`` with
   its corresponding whitener ``W₍b₎``:

   ``v_hat₍b₎ = W₍b₎  ·  v₍b₎``

4. Concatenate all whitened block vectors, flatten, compute the global ℓ₂-norm,
   and rescale so that the final vector has unit norm.
5. Save the normalised result via ``--out-path`` (default
   ``whitened_target.npy``) as ``float32`` with shape ``(B·d,)`` (a **1-D**
   vector) where ``B = 2 · num_blocks`` and ``d = rank²``.

Example
-------
::

    python -m dclm_exp.clustering.mgd.condition \
        --num-gpus 8 \
        --root-dir /path/to/results \
        --whiteners-path whiteners.npy \
        --out-path whitened_target.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate and whiten target direction vectors.")
    p.add_argument("--num-gpus", type=int, required=True,
                   help="Number of per-GPU result directories (dir_0 … dir_{N-1}).")
    p.add_argument("--root-dir", type=str, default=".",
                   help="Parent directory containing the per-GPU sub-directories.")
    p.add_argument("--whiteners-path", type=str, required=True,
                   help="Path to the (2*num_blocks, d, d) whitener tensor produced by hessian.py.")
    p.add_argument("--out-path", type=str, default="whitened_target.npy",
                   help="Destination .npy file for the whitened, normalised target direction.")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                   help="Output dtype (default: fp32).")
    return p.parse_args()


def load_and_sum_targets(root: Path, num_gpus: int) -> np.ndarray:
    """Load ``sum.npy`` from every ``dir_i`` and return their float32 sum."""
    agg: np.ndarray | None = None

    for gpu_id in range(num_gpus):
        vec_path = root / f"dir_{gpu_id}" / "sum.npy"
        if not vec_path.exists():
            raise FileNotFoundError(f"Missing per-GPU file: {vec_path}")
        vec = np.load(vec_path, mmap_mode=None).astype(np.float32)

        if agg is None:
            agg = np.zeros_like(vec, dtype=np.float32)
        if vec.shape != agg.shape:
            raise ValueError(
                f"Shape mismatch: expected {agg.shape}, got {vec.shape} in {vec_path}")
        agg += vec

    if agg is None:
        raise RuntimeError("No target vectors loaded – check --num-gpus / --root-dir.")
    return agg  # shape (B, d)


def whiten_target(target: np.ndarray, whiteners: np.ndarray) -> np.ndarray:
    """Apply block-wise whitening: ``W_b @ v_b`` for every block ``b``.

    Parameters
    ----------
    target : np.ndarray
        Shape (B, d) – aggregate target direction.
    whiteners : np.ndarray
        Shape (B, d, d) – inverse Fisher square-root matrices.
    """
    if whiteners.ndim != 3 or target.ndim != 2:
        raise ValueError("Unexpected target / whitener dimensions.")
    if whiteners.shape[0:2] != target.shape:
        raise ValueError(
            f"Whitener shape {whiteners.shape} incompatible with target {target.shape}.")

    # Efficient batched matrix-vector multiply: (B, d, d) @ (B, d) -> (B, d)
    whitened = np.einsum("bij,bj->bi", whiteners.astype(np.float32), target)
    return whitened.astype(np.float32)


def l2_normalise(vec: np.ndarray) -> np.ndarray:
    """Flatten ``vec`` and return a unit-norm 1-D vector."""
    flat = vec.reshape(-1).astype(np.float32)
    norm = np.linalg.norm(flat)
    if norm == 0.0:
        raise ValueError("Zero norm encountered during normalisation.")
    return flat / norm


def main() -> None:
    args = parse_args()

    root = Path(args.root_dir)
    target_sum = load_and_sum_targets(root, args.num_gpus)

    whiteners = np.load(args.whiteners_path, mmap_mode=None).astype(np.float32)

    whitened = whiten_target(target_sum, whiteners)
    unit_vec = l2_normalise(whitened)  # shape (B*d,)

    out_arr = unit_vec.astype(np.float16) if args.dtype == "fp16" else unit_vec
    np.save(args.out_path, out_arr)
    print(
        f"[condition] Saved whitened target to '{args.out_path}' with shape {out_arr.shape} and dtype {out_arr.dtype}.")


if __name__ == "__main__":
    main()
