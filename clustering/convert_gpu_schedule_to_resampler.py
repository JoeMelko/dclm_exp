#!/usr/bin/env python3
"""
convert_gpu_schedule_to_resampler.py
------------------------------------

Convert a GPU-style dict-of-knots token-ratio schedule into the per-dataset
per-chunk keep ratios expected by line_ratio_resampler_curric.py (schedule mode),
with intentional drift allowed across clusters.

Semantics
- Interpolates the knot schedule linearly in log-token space (matching the
  greedy GPU script semantics).
- Integrates expected tokens per dataset over equal-token chunks to get per-chunk
  average token shares alpha[i, t].
- Emits r[i, t] = alpha[i, t] / (T * ratios0[i]) where T is --chunks and
  ratios0 is the baseline cluster mix. If ratios0[i] == 0, r[i, t] is set to 0.

Usage (module)
  python -m dclm_exp.clustering.convert_gpu_schedule_to_resampler \
    --gpu-schedule /path/to/knot_schedule.json \
    --ratios0 /path/to/ratios0.json \
    --total-toks 500000000 \
    --chunks 16 \
    --out /path/to/resampler_schedule.json \
    --pretty

Inputs
- --gpu-schedule: JSON dict of {"<knot_tokens>": {"dataset0": r0, ..., "dataset{C-1}": rC-1}}
- --ratios0: JSON dict baseline {"dataset0": p0, ..., "dataset{C-1}": pC-1} (will be normalized)
- --total-toks: total tokens across all chunks (float/int)
- --chunks: number of equal-token chunks T

Output
- --out: JSON schedule {"dataset<i>": [r[i,0], ..., r[i,T-1]]} consumable by
  line_ratio_resampler_curric.py in --ratio-schedule mode.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_gpu_knot_schedule(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read the dict-of-knots GPU schedule and return (knots, P, dataset_cols).

    Accepts only contiguous dataset{i} keys starting at 0; caller guarantees
    inputs are well-formed. No additional runtime validation.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) == 0:
        raise ValueError("GPU schedule must be a non-empty JSON dict of {knot_tokens: {dataset{i}: ratio, ...}}")

    knots: List[float] = []
    per_knot_dicts: List[Dict[str, float]] = []
    for k, v in raw.items():
        if not isinstance(v, dict) or len(v) == 0:
            raise ValueError("Each knot must map to a non-empty dict of dataset ratios")
        try:
            k_float = float(k)
        except Exception:
            raise ValueError("Invalid knot key")
        if not math.isfinite(k_float) or k_float <= 0.0:
            raise ValueError("Invalid knot value")
        knots.append(k_float)
        per_knot_dicts.append(v)

    order = np.argsort(np.asarray(knots, dtype=np.float64))
    knots_sorted = [knots[i] for i in order]
    dicts_sorted = [per_knot_dicts[i] for i in order]

    # Infer C from the first knot dict and use fixed column names dataset0..dataset{C-1}
    first_dict = dicts_sorted[0]
    C = len(first_dict)
    cols_expected = [f"dataset{i}" for i in range(C)]

    K = len(knots_sorted)
    P = np.empty((K, C), dtype=np.float32)
    for i in range(K):
        d = dicts_sorted[i]
        row = [float(d[name]) for name in cols_expected]
        P[i, :] = np.asarray(row, dtype=np.float32)

    return np.asarray(knots_sorted, dtype=np.float64), P, cols_expected


def _load_baseline_ratios(path: Path, expected_cols: List[str]) -> np.ndarray:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) == 0:
        raise ValueError("Invalid baseline ratios")
    ratios0 = np.empty((len(expected_cols),), dtype=np.float64)
    for i, name in enumerate(expected_cols):
        val = raw.get(name, None)
        if val is None:
            raise ValueError("Invalid baseline ratios")
        try:
            x = float(val)
        except Exception:
            raise ValueError("Invalid baseline ratios")
        if not math.isfinite(x) or x < 0.0:
            raise ValueError("Invalid baseline ratios")
        ratios0[i] = x
    total = float(ratios0.sum())
    if total <= 0.0:
        raise ValueError("Invalid baseline ratios")
    # Normalize to sum ~1 if not already
    ratios0 = ratios0 / total
    return ratios0.astype(np.float64)


def _integrate_piece_linear_in_logN(Na: float, Nb: float, U0: float, U1: float, P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
    """Exact integral of p(N) over [Na,Nb] when p is linear in log N over [U0,U1].

    p(u) = (1 - t) P0 + t P1,  t = (u - U0) / (U1 - U0).  We compute ∫ p(N) dN = ∫ p(u) e^u du.
    Closed-form: If we let p(u) = a + b u, then ∫ (a + b u) e^u du = a e^u + b e^u (u - 1).
    """
    assert Na <= Nb
    assert U0 < U1
    du = (U1 - U0)
    # a = P0 - ((P1 - P0) * U0) / du;  b = (P1 - P0) / du
    diff = (P1 - P0)
    b = diff / du
    a = P0 - (b * U0)
    ua = math.log(Na)
    ub = math.log(Nb)
    # Evaluate primitive at bounds
    def F(u: float) -> np.ndarray:
        eu = math.exp(u)
        return a * eu + b * eu * (u - 1.0)
    return F(ub) - F(ua)


def _integrate_schedule_over_interval(Na: float, Nb: float, knots: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Integrate p(N) dN over [Na, Nb] using exact expressions with log-token linear interpolation.

    Handles clamping below the first knot and above the last knot (constant p).
    Returns a vector of expected tokens per dataset for the interval.
    """
    assert Na < Nb
    # Below first knot: constant P[0]
    total = np.zeros((P.shape[1],), dtype=np.float64)
    if Nb <= knots[0]:
        return (Nb - Na) * P[0].astype(np.float64)
    if Na >= knots[-1]:
        return (Nb - Na) * P[-1].astype(np.float64)

    curr_a = Na
    # Segment 1: up to first knot
    if curr_a < knots[0]:
        seg_b = min(Nb, float(knots[0]))
        total += (seg_b - curr_a) * P[0].astype(np.float64)
        curr_a = seg_b
        if curr_a >= Nb:
            return total

    # Middle segments between knots
    for k in range(len(knots) - 1):
        seg_left = float(knots[k])
        seg_right = float(knots[k + 1])
        if curr_a >= Nb:
            break
        if Nb <= seg_left or curr_a >= seg_right:
            continue
        a = max(curr_a, seg_left)
        b = min(Nb, seg_right)
        if a < b:
            U0 = math.log(seg_left)
            U1 = math.log(seg_right)
            contrib = _integrate_piece_linear_in_logN(a, b, U0, U1, P[k].astype(np.float64), P[k + 1].astype(np.float64))
            total += contrib
            curr_a = b

    if curr_a < Nb:
        # Tail after last knot: constant P[-1]
        total += (Nb - curr_a) * P[-1].astype(np.float64)
    return total


def compute_resampler_schedule(
    knots: np.ndarray,
    P: np.ndarray,
    ratios0: np.ndarray,
    total_tokens: float,
    chunks: int,
    eps: float = 1e-12,
) -> Dict[str, List[float]]:
    """Compute r[i,t] = alpha[i,t] / (chunks * ratios0[i]) and return as dict of lists.

    alpha[i,t] is the average token share for cluster i in chunk t, obtained by exact
    integration of the knot schedule over the token interval for that chunk.
    """
    C = int(P.shape[1])
    T = int(chunks)
    if total_tokens <= 0 or T <= 0:
        raise ValueError("Invalid arguments")

    # Equal-token chunk boundaries
    boundaries = np.linspace(0.0, float(total_tokens), num=T + 1, dtype=np.float64)
    E = np.zeros((C, T), dtype=np.float64)
    for t in range(T):
        Na = float(boundaries[t])
        Nb = float(boundaries[t + 1])
        if Nb <= Na:
            continue
        # Integrate expected tokens per dataset over [Na, Nb]
        integ = _integrate_schedule_over_interval(Na, Nb, knots, P)
        E[:, t] = integ

    # Convert to shares per chunk
    deltaN = (float(total_tokens) / T)
    alpha = E / max(deltaN, eps)
    # Numerical safety: clamp small negatives and renormalize each chunk to sum ~1
    alpha = np.clip(alpha, 0.0, None)
    col_sums = alpha.sum(axis=0, keepdims=True)
    nz = np.where(col_sums > 0.0, col_sums, 1.0)
    alpha = alpha / nz

    # r[i,t] = alpha[i,t] / (T * ratios0[i])
    denom = ratios0.reshape(C, 1)
    r = np.empty_like(alpha)
    zero_mask = denom <= eps
    # For datasets with zero baseline, force r=0 (treated as excluded)
    r[zero_mask[:, 0], :] = 0.0
    # For others, divide by (T * ratios0)
    safe_denom = np.where(zero_mask, 1.0, denom)
    r = np.where(zero_mask, 0.0, alpha / (T * safe_denom))

    # Emit as dict of lists for line_ratio_resampler_curric.py
    out: Dict[str, List[float]] = {}
    for i in range(C):
        out[f"dataset{i}"] = [float(x) for x in r[i, :].tolist()]
    return out


def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(description="Convert GPU knot schedule to resampler per-chunk keep ratios (with drift)")
    ap.add_argument("--gpu-schedule", type=Path, required=True, help="Path to dict-of-knots JSON schedule for greedy GPU script")
    ap.add_argument("--ratios0", type=Path, required=True, help="Path to baseline ratios JSON {dataset{i}: ratio}")
    ap.add_argument("--total-toks", type=float, required=True, help="Total tokens across all chunks")
    ap.add_argument("--chunks", type=int, required=True, help="Number of chunks T")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path for resampler schedule")
    ap.add_argument("--eps", type=float, default=1e-12, help="Small epsilon for stability when dividing by ratios0")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = ap.parse_args(argv)

    # Expand ~ and ensure .json extension for out if missing
    args.gpu_schedule = args.gpu_schedule.expanduser()
    args.ratios0 = args.ratios0.expanduser()
    args.out = args.out.expanduser()
    if args.out.suffix.lower() != ".json":
        args.out = args.out.with_suffix(".json")

    knots, P, cols = _load_gpu_knot_schedule(args.gpu_schedule)
    ratios0 = _load_baseline_ratios(args.ratios0, cols)
    if len(cols) != P.shape[1]:
        raise AssertionError("Column mismatch")

    schedule = compute_resampler_schedule(knots, P, ratios0, float(args.total_toks), int(args.chunks), float(args.eps))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(schedule, f, indent=2, ensure_ascii=False)
        else:
            json.dump(schedule, f, separators=(",", ":"), ensure_ascii=False)
            f.write("\n")
    print(f"Wrote resampler schedule to {args.out}")


if __name__ == "__main__":
    main()


