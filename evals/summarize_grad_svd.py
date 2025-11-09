#!/usr/bin/env python
"""
summarize_grad_svd.py
---------------------
Read JSONL output from eval_grad_svd.py and compute summary statistics
(mean, median, min, max) for sv_max and stable_rank for each layer and
each of the tracked matrices: attn_in, attn_out, ffn_in, ffn_out.

Usage:
  python -m evals.summarize_grad_svd --log-json /path/to/grad_svd_logs.jsonl
  python -m evals.summarize_grad_svd --log-json logs1.jsonl logs2.jsonl --out-json summary.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple


def _is_finite_number(x) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def _safe_stats(values: List[float]) -> Dict[str, float]:
    """Compute count, mean, median, min, max for a list, ignoring NaNs/Infs."""
    finite_vals = [float(v) for v in values if _is_finite_number(v)]
    if not finite_vals:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    finite_vals.sort()
    return {
        "count": len(finite_vals),
        "mean": float(mean(finite_vals)),
        "median": float(median(finite_vals)),
        "min": float(finite_vals[0]),
        "max": float(finite_vals[-1]),
    }


def _iter_records(paths: Iterable[Path]) -> Iterable[dict]:
    """Yield parsed JSON objects from given JSONL file paths."""
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _collect_values(records: Iterable[dict]) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """
    Build a nested mapping:
      layer_idx -> section (attn_in/attn_out/ffn_in/ffn_out) -> metric -> list of values
    """
    by_layer: Dict[int, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    wanted_sections = ("attn_in", "attn_out", "ffn_in", "ffn_out")
    wanted_metrics = ("sv_max", "stable_rank")

    for rec in records:
        per_layer = rec.get("per_layer")
        if not isinstance(per_layer, list):
            continue
        for layer_obj in per_layer:
            if not isinstance(layer_obj, dict):
                continue
            layer_idx = layer_obj.get("layer")
            if not isinstance(layer_idx, int):
                continue
            for section in wanted_sections:
                sec_obj = layer_obj.get(section)
                if not isinstance(sec_obj, dict):
                    continue
                for metric in wanted_metrics:
                    value = sec_obj.get(metric)
                    if _is_finite_number(value):
                        by_layer[layer_idx][section][metric].append(float(value))
                    else:
                        # Keep NaNs/invalids out of stats; they are ignored
                        continue
    return by_layer


def _summarize(by_layer: Dict[int, Dict[str, Dict[str, List[float]]]]) -> Dict[str, dict]:
    """
    Convert collected lists into summary statistics.
    Output schema:
      {
        "layers": {
          "<layer_idx>": {
            "<section>": {
              "sv_max": {count, mean, median, min, max},
              "stable_rank": {count, mean, median, min, max}
            },
            ...
          },
          ...
        }
      }
    """
    summary = {"layers": {}}
    for layer_idx in sorted(by_layer.keys()):
        layer_entry = {}
        for section, metrics_map in by_layer[layer_idx].items():
            section_entry = {}
            for metric_name, values in metrics_map.items():
                section_entry[metric_name] = _safe_stats(values)
            layer_entry[section] = section_entry
        summary["layers"][str(layer_idx)] = layer_entry
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-json",
        nargs="+",
        required=True,
        help="Path(s) to JSONL produced by eval_grad_svd.py",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to write the summary JSON. Prints to stdout if omitted.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.log_json]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    records = _iter_records(paths)
    collected = _collect_values(records)
    summary = _summarize(collected)

    out_str = json.dumps(summary, indent=2, sort_keys=True)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_str + "\n")
    else:
        print(out_str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


