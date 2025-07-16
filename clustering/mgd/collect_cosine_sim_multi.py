#!/usr/bin/env python
"""
collect_cosine_sim_multi.py
---------------------------
Persistent, multi-dataset variant of `collect_cosine_sim_dc.py`.
Run **one instance per GPU**; it loads the model once, then sequentially
processes a contiguous slice of sub-directories inside a parent directory.

Example
~~~~~~~
    python collect_cosine_sim_multi.py \
        --parent-dir /path/to/datasets \
        --start-idx 0 --end-idx 99 \
        --target-vector vec.npy \
        --uuid 123e4567-e89b-12d3-a456-426614174000 \
        --out-dir clustering/mgd \
        --iter 0 --lora-rank 128 --num-blocks 8 --max-items 500

For every dataset directory <d>, the script produces
    <out_dir>/iter_<iter>/<d>.npz
containing the keys ``dot`` and ``l2`` (float32 arrays) – identical to the
single-dataset implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import tqdm
import webdataset as wds

# ---------------------------------------------------------------------------
# Re-use helpers from the original single-dataset script
# ---------------------------------------------------------------------------
from collect_cosine_sim_dc import (
    BATCH_SIZE,
    collate,
    prepare_model,
    load_openlm_model_from_uuid,
)

# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    wds_dir_path: Path,
    *,
    model,
    logger,
    target_vec: torch.Tensor,
    args,
):
    """Featurise `wds_dir_path` with an *already initialised* model & logger.

    The computational core is identical to `collect_cosine_sim_dc.py`; the only
    difference is that the model is **not** rebuilt for every directory.
    """

    tar_files = sorted(wds_dir_path.glob("*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No .tar shards found in {wds_dir_path}")

    # Shuffle samples on the fly to avoid ordering artefacts
    ds = wds.WebDataset([str(p) for p in tar_files]).shuffle(10_000)
    loader = wds.WebLoader(ds, batch_size=BATCH_SIZE, num_workers=8, collate_fn=collate)

    device_t = next(model.parameters()).device
    dot_gpu: List[torch.Tensor] = []
    l2_gpu: List[torch.Tensor] = []

    for toks, labels in loader:
        # Discard incomplete final batch to keep tensor shapes consistent
        if toks.size(0) < BATCH_SIZE:
            print(f"Encountered incomplete batch of size {toks.size(0)} < {BATCH_SIZE}; stopping.")
            break

        toks, labels = toks.to(device_t), labels.to(device_t)
        with torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=toks, labels=labels)
                out.loss.mul_(1024)
                out.loss.backward()

            features = logger.grads.detach().clone().to(torch.float32)

        # Reset buffer for next step *before* further GPU work
        logger.grads.zero_()

        # Rearrange to (batch, 2*B, d) and flatten per-sample features
        feats_per_sample = features.permute(1, 0, 2)
        if (
            feats_per_sample.shape[1] != 2 * args.num_blocks
            or feats_per_sample.shape[2] != args.lora_rank * args.lora_rank
        ):
            raise ValueError(
                "Unexpected feature shape "
                f"{tuple(feats_per_sample.shape)} – expected (*, {2 * args.num_blocks}, {args.lora_rank * args.lora_rank})."
            )

        flat_feats = feats_per_sample.reshape(feats_per_sample.size(0), -1)
        if flat_feats.size(1) != target_vec.numel():
            raise ValueError(
                f"Target vector length {target_vec.numel()} != gradient feature length {flat_feats.size(1)}"
            )

        dots = torch.matmul(flat_feats, target_vec)
        norms = torch.linalg.norm(flat_feats, dim=1)

        dot_gpu.append(dots)
        l2_gpu.append(norms)

        if len(l2_gpu) >= args.max_items:
            break

    if dot_gpu:
        dot = torch.cat(dot_gpu).cpu().numpy().astype(np.float32)
        l2 = torch.cat(l2_gpu).cpu().numpy().astype(np.float32)

        out_dir = Path(args.out_dir).expanduser() / f"iter_{args.iter}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{wds_dir_path.name}.npz"
        np.savez(out_path, dot=dot, l2=l2)
        print(f"saved {dot.shape[0]} per-sample dot products and norms → {out_path}")
    else:
        print(f"No batches processed for {wds_dir_path}; nothing saved.")

    # Tidy up GPU memory before moving to next dataset
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Persistent, multi-dataset gradient-feature collector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--parent-dir", required=True, help="Directory whose first-level sub-directories are processed.")
    ap.add_argument("--start-idx", type=int, required=True, help="Inclusive start index in the sorted sub-directory list.")
    ap.add_argument("--end-idx",   type=int, required=True, help="Inclusive end index in the sorted sub-directory list.")

    # Flags matching the single-dataset script
    ap.add_argument("--target-vector", required=True, help="Path to .npy containing the (whitened & L2-normalised) target vector.")
    ap.add_argument("--uuid", required=True, help="Open-LM run UUID identifying the model checkpoint.")
    ap.add_argument("--iter", type=int, required=True, help="Iteration index added to output path.")
    ap.add_argument("--lora-rank", type=int, required=True, help="LoRA rank for adapters.")
    ap.add_argument("--num-blocks", type=int, required=True, help="Number of transformer blocks logged.")
    ap.add_argument("--out-dir", required=True, help="Base directory where outputs will be written.")
    ap.add_argument("--max-items", type=int, required=True, help="Maximum number of batches processed *per dataset* before early stop.")

    return ap.parse_args()

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    parent_dir = Path(args.parent_dir).expanduser()
    if not parent_dir.is_dir():
        raise SystemExit(f"ERROR: not a directory → {parent_dir}")

    subdirs = sorted(p for p in parent_dir.iterdir() if p.is_dir())
    if not subdirs:
        raise SystemExit(f"No sub-directories found in {parent_dir}")

    total = len(subdirs)
    if args.start_idx < 0 or args.end_idx < args.start_idx or args.end_idx >= total:
        raise SystemExit(f"Invalid slice [{args.start_idx}:{args.end_idx}] for {total} sub-directories.")

    slice_dirs = subdirs[args.start_idx : args.end_idx + 1]

    # ---------------------------------------------------------------------
    # Model initialisation (once per worker)
    # ---------------------------------------------------------------------
    base_model = load_openlm_model_from_uuid(args.uuid)
    model, handler, logger = prepare_model(base_model, args)

    # Target vector (on device)
    device = next(model.parameters()).device
    target_vec = torch.from_numpy(np.load(args.target_vector).astype(np.float32)).to(device).view(-1)

    expected_dim = 2 * args.num_blocks * args.lora_rank * args.lora_rank
    if target_vec.numel() != expected_dim:
        raise ValueError(f"Target vector length {target_vec.numel()} does not match expected dimension {expected_dim}.")

    # ---------------------------------------------------------------------
    # Run through our assigned datasets
    # ---------------------------------------------------------------------
    for wds_dir in tqdm.tqdm(slice_dirs, desc="datasets"):
        try:
            process_dataset(wds_dir, model=model, logger=logger, target_vec=target_vec, args=args)
        except Exception as e:
            print(f"[ERROR] Dataset {wds_dir} failed with: {e}")

    print("Worker slice completed successfully.")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
