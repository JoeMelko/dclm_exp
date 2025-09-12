#!/usr/bin/env python3
"""
Analyze chunk_l2 entries from error_log.jsonl files across multiple run directories.

Default usage looks under a base directory for subdirectories matching patterns like
"sub0_*" and "sub0_hist_*", reads their error_log.jsonl, extracts lines containing
chunk-level logging, writes per-run NumPy arrays (including a running mean per
directory), and prints/writes simple stats per directory (count, min, max, mean).

Example:
  python analyze_chunk_l2.py \
    --base-dir /home/ec2-user/jittered_docs_short0_tok_ordered \
    --patterns sub0_* sub0_hist_* \
    --print-stats

Outputs:
- Per-run .npy files (written next to each error_log.jsonl):
  chunk_l2_stats.npy with rows [chunk_l2, chunk_l2_running_mean]
 - Plot saved to PNG (all runs overlaid on one axis):
  chunk_l2_running_mean.png (default in the same directory as --stats-out)
Quick load example for a single run directory:
  >>> import numpy as np, pathlib as pl
  >>> run_dir = pl.Path('/home/ec2-user/jittered_docs_short0_tok_ordered/sub0_0')
  >>> arr = np.load(run_dir / 'chunk_l2_stats.npy')
  >>> # columns: [chunk_l2, chunk_l2_running_mean]
  >>> arr.shape
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Iterator, List, Dict, Any, Tuple


def iter_run_dirs(base_dir: Path, patterns: List[str]) -> Iterator[Path]:
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if path.is_dir():
                yield path


def parse_chunk_entries(log_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield dicts only for lines that contain chunk-level fields.

    Each yielded dict has keys: step, chunk_size, chunk_tokens, chunk_l2, and
    optionally chunk_reg_error if present in the log.
    """
    if not log_path.exists():
        return
    with log_path.open("r") as f:
        for line in f:
            # Quick filter to skip non-chunk lines early
            if '"chunk_size"' not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only yield entries that include chunk_l2 (and related fields)
            if (
                isinstance(obj, dict)
                and "chunk_l2" in obj
                and "chunk_size" in obj
                and "chunk_tokens" in obj
                and "step" in obj
            ):
                entry: Dict[str, Any] = {
                    "step": int(obj["step"]),
                    "chunk_size": int(obj["chunk_size"]),
                    "chunk_tokens": int(obj["chunk_tokens"]),
                    "chunk_l2": float(obj["chunk_l2"]),
                }
                if "chunk_reg_error" in obj:
                    try:
                        entry["chunk_reg_error"] = float(obj["chunk_reg_error"])
                    except Exception:
                        pass
                yield entry


def write_consolidated_npy(rows: Iterable[Tuple[float, float]], out_npy: Path) -> None:
    # No-op: consolidated output removed; keep function for backward compatibility if imported elsewhere.
    pass


def compute_stats_by_dir(values_by_dir: Dict[str, List[float]]) -> List[Tuple[str, int, float, float, float, float]]:
    stats: List[Tuple[str, int, float, float, float, float]] = []
    for d, vals in values_by_dir.items():
        if not vals:
            continue
        stats.append((d, len(vals), min(vals), max(vals), mean(vals), median(vals)))
    # Sort by directory name for stable output
    stats.sort(key=lambda x: x[0])
    return stats


def write_stats_csv(stats: List[Tuple[str, int, float, float, float, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dir", "n", "min", "max", "mean", "median"])
        writer.writerows(stats)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Aggregate and summarize chunk_l2 from error logs.")
    ap.add_argument("--base-dir", type=Path, default=Path("/home/ec2-user/jittered_docs_short0_tok_ordered"),
                    help="Base directory containing run subdirectories (default: %(default)s)")
    ap.add_argument("--patterns", nargs="*", default=["sub0_*", "sub0_hist_*"],
                    help="Glob patterns (relative to base-dir) for run directories")
    ap.add_argument("--stats-out", type=Path, default=Path("/home/ec2-user/chunk_l2_stats.csv"),
                    help="Path to write per-directory chunk_l2 stats CSV (csv text)")
    ap.add_argument("--reg-stats-out", type=Path, default=Path("/home/ec2-user/chunk_reg_stats.csv"),
                    help="Path to write per-directory chunk_reg_error stats CSV (csv text)")
    ap.add_argument("--print-stats", action="store_true", help="Also print stats to stdout")
    ap.add_argument("--plot-out", type=Path, default=None,
                    help="PNG path for overlaid running means (default: directory of --stats-out)")
    ap.add_argument("--reg-plot-out", type=Path, default=None,
                    help="PNG path for overlaid running means (all runs) for chunk_reg_error (default: directory of --reg-stats-out)")
    ap.add_argument("--exclude-final-frac", type=float, default=0.0,
                    help="Fraction in [0,1] of final entries to drop after sorting by step (default: 0.0)")
    args = ap.parse_args(argv)

    rows: List[Tuple[float, float]] = []  # legacy consolidated rows for chunk_l2
    values_by_dir: Dict[str, List[float]] = defaultdict(list)  # chunk_l2 values
    values_by_dir_reg: Dict[str, List[float]] = defaultdict(list)  # chunk_reg_error values
    entries_by_dir: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    running_means_by_dir: Dict[str, List[float]] = {}
    running_means_by_dir_reg: Dict[str, List[float]] = {}

    for run_dir in iter_run_dirs(args.base_dir, args.patterns):
        log_path = run_dir / "error_log.jsonl"
        for e in parse_chunk_entries(log_path):
            entries_by_dir[str(run_dir)].append(e)

    # For each directory, sort by step and compute running mean over chunk_l2 and chunk_reg_error (if present)
    for d, entries in entries_by_dir.items():
        if not entries:
            continue
        entries.sort(key=lambda x: x["step"])  # ensure monotonic order
        # Optionally truncate the final fraction of examples
        exclude_frac = args.exclude_final_frac if hasattr(args, "exclude_final_frac") else 0.0
        if exclude_frac < 0.0:
            exclude_frac = 0.0
        if exclude_frac > 1.0:
            exclude_frac = 1.0
        keep_count = int(len(entries) * (1.0 - exclude_frac))
        if keep_count < len(entries):
            entries = entries[:keep_count]
        running_sum = 0.0
        count = 0
        npy_rows: List[Tuple[float, float]] = []
        running_sum_reg = 0.0
        count_reg = 0
        npy_rows_reg: List[Tuple[float, float]] = []
        for e in entries:
            v = float(e["chunk_l2"]) 
            count += 1
            running_sum += v
            running_mean = running_sum / count
            rows.append((v, running_mean))
            npy_rows.append((v, running_mean))
            values_by_dir[d].append(v)
            # Optional reg error handling if present
            if "chunk_reg_error" in e:
                vr = float(e["chunk_reg_error"])  # may be 0.0 if disabled
                count_reg += 1
                running_sum_reg += vr
                running_mean_reg = running_sum_reg / count_reg
                npy_rows_reg.append((vr, running_mean_reg))
                values_by_dir_reg[d].append(vr)

        # Write per-directory 2D numpy array next to error_log.jsonl
        if npy_rows:
            arr = np.asarray(npy_rows, dtype=np.float64)
            np.save(Path(d) / "chunk_l2_stats.npy", arr)
            running_means_by_dir[d] = [float(x[1]) for x in npy_rows]
        if npy_rows_reg:
            arr_reg = np.asarray(npy_rows_reg, dtype=np.float64)
            np.save(Path(d) / "chunk_reg_stats.npy", arr_reg)
            running_means_by_dir_reg[d] = [float(x[1]) for x in npy_rows_reg]

    # No consolidated output; per-run .npy files are written next to each log

    # Compute and write stats
    stats = compute_stats_by_dir(values_by_dir)
    write_stats_csv(stats, args.stats_out)
    stats_reg = compute_stats_by_dir(values_by_dir_reg)
    write_stats_csv(stats_reg, args.reg_stats_out)

    if args.print_stats:
        # Print as aligned text for quick readability
        print("dir,n,min,max,mean,median (chunk_l2)")
        for d, n, vmin, vmax, vmean, vmedian in stats:
            print(f"{d},{n},{vmin:.6f},{vmax:.6f},{vmean:.6f},{vmedian:.6f}")
        if stats_reg:
            print("\ndir,n,min,max,mean,median (chunk_reg_error)")
            for d, n, vmin, vmax, vmean, vmedian in stats_reg:
                print(f"{d},{n},{vmin:.6f},{vmax:.6f},{vmean:.6f},{vmedian:.6f}")

    # Plot overlaid running means across all directories
    if running_means_by_dir:
        default_plot_dir = args.stats_out.parent if hasattr(args, 'stats_out') and args.stats_out is not None else args.base_dir
        plot_path = args.plot_out or (default_plot_dir / "chunk_l2_running_mean.png")
        plt.figure(figsize=(10, 6))
        for d in sorted(running_means_by_dir.keys()):
            vals = running_means_by_dir[d]
            if not vals:
                continue
            label = Path(d).name
            xs = list(range(1, len(vals) + 1))
            plt.plot(xs, vals, label=label, linewidth=1.5)
        plt.xlabel("Chunk index")
        plt.ylabel("Running mean of chunk_l2")
        plt.title("Running mean of chunk_l2 per run")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150)
        print(f"Wrote plot PNG to: {plot_path}")

    if running_means_by_dir_reg:
        default_plot_dir = args.reg_stats_out.parent if hasattr(args, 'reg_stats_out') and args.reg_stats_out is not None else args.base_dir
        plot_path_reg = args.reg_plot_out or (default_plot_dir / "chunk_reg_running_mean.png")
        plt.figure(figsize=(10, 6))
        for d in sorted(running_means_by_dir_reg.keys()):
            vals = running_means_by_dir_reg[d]
            if not vals:
                continue
            label = Path(d).name
            xs = list(range(1, len(vals) + 1))
            plt.plot(xs, vals, label=label, linewidth=1.5)
        plt.xlabel("Chunk index")
        plt.ylabel("Running mean of chunk_reg_error")
        plt.title("Running mean of chunk_reg_error per run")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(str(plot_path_reg), dpi=150)
        print(f"Wrote plot PNG to: {plot_path_reg}")

    print(f"Wrote stats CSV to: {args.stats_out}")
    print(f"Wrote reg stats CSV to: {args.reg_stats_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


