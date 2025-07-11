import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import numpy as np

"""Update dataset sampling logits based on similarity scores.

Example
-------
python update_logits \
    --counts path/to/counts.json \
    --scores-dir path/to/scores_dir \
    --out-path path/to/updated_counts.json \
    --means-out path/to/scaled_means.json \
    --lr 1
"""


EPS = 1e-8  # Small constant to avoid log(0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Update logits based on similarity scores.")
    parser.add_argument("--counts", type=str, required=True, help="Path to counts JSON file.")
    parser.add_argument("--scores-dir", type=str, required=True, help="Directory containing dataset{i}.npz score files.")
    parser.add_argument("--out-path", type=str, required=True, help="Path where updated counts JSON should be saved.")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate for logit update (default: 0.2).")
    parser.add_argument("--max-z", type=float, default=5, help="Maximum standard deviation for outlier clipping (default: 5).")
    parser.add_argument("--means-out", type=str, default=None, help="Optional path to save the scaled mean dot values (JSON).")
    return parser.parse_args()


def load_counts(counts_path: str) -> np.ndarray:
    """Load counts JSON and return counts array of shape (10_000,)."""
    with open(counts_path, "r", encoding="utf-8") as fp:
        counts_dict = json.load(fp)

    counts = np.zeros(10_000, dtype=np.float64)
    for i in range(10_000):
        key = f"dataset{i}"
        if key not in counts_dict:
            raise KeyError(f"Missing key '{key}' in counts file {counts_path}")
        counts[i] = counts_dict[key]
    return counts


def counts_to_logits(counts: np.ndarray) -> np.ndarray:
    """Convert counts to logits via probabilities."""
    total = counts.sum()
    if total == 0:
        raise ValueError("Counts sum to zero; cannot convert to probabilities.")
    probs = counts / total
    logits = np.log(probs + EPS)
    return logits


def load_score_files(scores_dir: str) -> (List[np.ndarray], List[np.ndarray]):
    """Load dot and l2 arrays from dataset{i}.npz files (i=0..9999)."""
    dots: List[np.ndarray] = []
    l2s: List[np.ndarray] = []
    for i in range(10_000):
        path = Path(scores_dir) / f"dataset{i}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing score file: {path}")
        data = np.load(path)
        if "dot" not in data or "l2" not in data:
            raise KeyError(f"File {path} must contain 'dot' and 'l2' arrays.")
        dots.append(data["dot"].astype(np.float64))
        l2s.append(data["l2"].astype(np.float64))
    return dots, l2s


def compute_adjustments(dots: List[np.ndarray], l2s: List[np.ndarray], max_z: float) -> np.ndarray:
    """Compute mean of clipped dot/l2 ratios for each dataset, z-score them (mean=0, std=1), then clip to [-max_z, max_z]."""
    # Determine global 99.9th percentile of l2 norms
    all_l2_concat = np.concatenate(l2s)
    l2_thresh = np.percentile(all_l2_concat, 99.9)

    # Compute clipped dot values and dataset means
    means = np.zeros(10_000, dtype=np.float64)
    for idx, (dot_arr, l2_arr) in enumerate(zip(dots, l2s)):
        denom = np.maximum(l2_arr, l2_thresh)
        clipped = dot_arr / denom
        means[idx] = clipped.mean()

    # Standardize to mean 0, std 1
    mean_val = means.mean()
    std_val = means.std()
    if math.isclose(std_val, 0):
        adjusted = np.zeros_like(means)
    else:
        adjusted = (means - mean_val) / std_val

    # Clip to the specified z-range
    adjusted = np.clip(adjusted, -max_z, max_z)

    return adjusted


def logits_to_counts(logits: np.ndarray, total_counts: float) -> np.ndarray:
    """Convert logits back to counts matching the original total (ceil)."""
    probs = np.exp(logits)
    probs_sum = probs.sum()
    if probs_sum == 0:
        raise ValueError("All logits resulted in zero probability mass.")
    probs /= probs_sum

    raw_counts = probs * total_counts
    new_counts = np.ceil(raw_counts).astype(np.int64)

    return new_counts


def save_counts(counts: np.ndarray, out_path: str):
    """Save counts array to JSON with keys dataset{i}."""
    counts_dict = {f"dataset{i}": int(count) for i, count in enumerate(counts)}
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(counts_dict, fp, indent=2)


def save_means(means: np.ndarray, out_path: str):
    """Save scaled mean dot values (length 10_000) to JSON with keys dataset{i}."""
    means_dict = {f"dataset{i}": float(mean) for i, mean in enumerate(means)}
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(means_dict, fp, indent=2)


def main():
    args = parse_args()

    # Step 1: Load counts and convert to logits
    counts = load_counts(args.counts)
    logits = counts_to_logits(counts)

    # Step 2: Load score files
    dots, l2s = load_score_files(args.scores_dir)

    # Step 3: Compute z-score adjustment values and clip
    adjustments = compute_adjustments(dots, l2s, max_z=args.max_z)

    # Optionally save scaled mean dot values
    if args.means_out:
        save_means(adjustments, args.means_out)

    # Step 4: Update logits and convert back to counts
    logits += args.lr * adjustments
    new_counts = logits_to_counts(logits, counts.sum())

    # Step 5: Save updated counts
    save_counts(new_counts, args.out_path)
    print(f"Updated counts saved to {args.out_path}")


if __name__ == "__main__":
    main()
