#!/usr/bin/env python3
"""
convert_gpu_schedule_to_resampler.py
------------------------------------

Convert a GPU-style dict-of-knots token-ratio schedule into the per-dataset
per-chunk keep ratios expected by line_ratio_resampler_curric.py (schedule mode),
with intentional drift allowed across clusters.

This implementation:
- Interpolates *log-probabilities* linearly in *log-token* space between knots,
  then renormalizes via softmax at every evaluation point.
- Integrates expected tokens per dataset over equal-token chunks using a
  composite trapezoidal rule with a **fixed absolute step size** equal to
  --seq-len (default: 2048 tokens), aligned to a global grid defined by
  G(m) = --offset + m * --seq-len.
  This mirrors the greedy GPU script’s per-sequence trapezoid step.
- Emits r[i, t] = alpha[i, t] / (T * ratios0[i]) where T is --chunks and
  ratios0 is the baseline cluster mix. If ratios0[i] == 0, r[i, t] is set to 0.

Usage (module)
  python -m dclm_exp.clustering.convert_gpu_schedule_to_resampler \
    --gpu-schedule /path/to/knot_schedule.json \
    --ratios0 /path/to/ratios0.json \
    --total-toks 500000000 \
    --chunks 16 \
    --out /path/to/resampler_schedule.json \
    --seq-len 2048 \
    --offset 0 \
    --pretty

Inputs
- --gpu-schedule: JSON dict {"<knot_tokens>": {"dataset0": r0, ..., "dataset{C-1}": rC-1}}
- --ratios0: JSON dict baseline {"dataset0": p0, ..., "dataset{C-1}": pC-1} (will be normalized)
- --total-toks: total tokens across all chunks (float/int)
- --chunks: number of equal-token chunks T
- --seq-len: integration step size in tokens (default 2048). Also the grid period.
- --offset: absolute-token offset for the fixed step grid (default 0).

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

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback when tqdm not available
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Interpolation & Integration                                                 #
# --------------------------------------------------------------------------- #
def _normalize_row(row: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a nonnegative row to sum ~ 1 with epsilon guard."""
    s = float(row.sum())
    if not math.isfinite(s) or s <= eps:
        C = row.shape[0]
        return np.full((C,), 1.0 / max(C, 1), dtype=np.float64)
    return (row / s).astype(np.float64)


def _p_at_N_logprob(N: float, knots: np.ndarray, P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Instantaneous mix p(N) using log-prob interpolation in log-token space.

    - If N <= first knot: return normalized P[0]
    - If N >= last knot : return normalized P[-1]
    - Else on interval [k_i, k_{i+1}]:
        u = log N, U0 = log k_i, U1 = log k_{i+1}
        l(u) = (1-t) * log P[i] + t * log P[i+1],  t = (u-U0)/(U1-U0)
        p(N) = softmax(l(u))
    """
    if N <= knots[0]:
        return _normalize_row(P[0].astype(np.float64), eps)
    if N >= knots[-1]:
        return _normalize_row(P[-1].astype(np.float64), eps)

    # Find active interval
    K = knots.shape[0]
    i = int(np.searchsorted(knots, N, side="right") - 1)
    i = max(0, min(i, K - 2))

    U0 = math.log(float(knots[i]))
    U1 = math.log(float(knots[i + 1]))
    u = math.log(max(N, eps))
    t = (u - U0) / max(U1 - U0, eps)

    # Interpolate logits, then softmax
    l0 = np.log(np.clip(P[i].astype(np.float64),     eps, None))
    l1 = np.log(np.clip(P[i + 1].astype(np.float64), eps, None))
    l  = (1.0 - t) * l0 + t * l1
    m  = float(np.max(l))
    ex = np.exp(l - m)
    Z  = float(ex.sum())
    if Z <= eps or not math.isfinite(Z):
        return _normalize_row(P[i].astype(np.float64), eps)  # fallback
    return (ex / Z).astype(np.float64)


def _integrate_fixedstep_trapezoid(
    Na: float,
    Nb: float,
    knots: np.ndarray,
    P: np.ndarray,
    seq_len: float,
    offset: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Integrate ∫_{Na}^{Nb} p(N) dN using composite trapezoid with a fixed global step.

    - Step size h = seq_len (tokens)
    - Grid is G(m) = offset + m*h (m integer)
    - Intervals are [Na, first_grid), then full steps on aligned [g, g+h], then [last_grid, Nb)
    - p(N) is evaluated via log-prob interpolation in log-token space.
    """
    assert Nb > Na, "Empty interval"
    h = float(seq_len)
    if h <= 0.0 or not math.isfinite(h):
        raise ValueError("seq_len must be positive and finite")
    off = float(offset)

    C = int(P.shape[1])
    total = np.zeros((C,), dtype=np.float64)

    # Helper: one trapezoid contribution on [a,b]
    def trap(a: float, b: float) -> np.ndarray:
        if b <= a:
            return np.zeros((C,), dtype=np.float64)
        pa = _p_at_N_logprob(a, knots, P, eps=eps)
        pb = _p_at_N_logprob(b, knots, P, eps=eps)
        return 0.5 * (pa + pb) * (b - a)

    # First grid boundary >= Na
    m_start = math.ceil((Na - off) / h - 1e-12)
    g0 = off + m_start * h

    # Last grid boundary <= Nb
    m_end = math.floor((Nb - off) / h + 1e-12)
    g_last = off + m_end * h

    # Initial partial segment
    if g0 > Na:
        total += trap(Na, min(Nb, g0))

    # Full aligned steps
    if g0 < Nb and m_end >= m_start:
        iter_range = range(m_start, m_end)
        total_steps = max(m_end - m_start, 0)
        for m in tqdm(iter_range, total=total_steps, desc="Trapezoids", unit="step", leave=False):
            a = off + m * h
            b = a + h
            if a >= Nb:
                break
            total += trap(max(a, Na), min(b, Nb))

    # Tail partial segment
    if g_last < Nb:
        total += trap(max(g_last, Na), Nb)

    return total


# --------------------------------------------------------------------------- #
# Schedule computation                                                        #
# --------------------------------------------------------------------------- #
def compute_resampler_schedule(
    knots: np.ndarray,
    P: np.ndarray,
    ratios0: np.ndarray,
    total_tokens: float,
    chunks: int,
    seq_len: int = 2048,
    offset: float = 0.0,
    ratio_scale: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, List[float]]:
    """Compute r[i,t] = alpha[i,t] / (chunks * ratios0[i]) and return as dict of lists.

    alpha[i,t] is the average token share for cluster i in chunk t, obtained by
    fixed-step trapezoidal integration with step = seq_len, aligned to the
    global grid offset + m*seq_len.
    """
    C = int(P.shape[1])
    T = int(chunks)
    if total_tokens <= 0 or T <= 0:
        raise ValueError("Invalid arguments")

    # Equal-token chunk boundaries
    boundaries = np.linspace(0.0, float(total_tokens), num=T + 1, dtype=np.float64)
    E = np.zeros((C, T), dtype=np.float64)

    for t in range(T):
        print(f"Computing chunk {t} of {T}")
        Na = float(boundaries[t])
        Nb = float(boundaries[t + 1])
        if Nb <= Na:
            continue
        integ = _integrate_fixedstep_trapezoid(
            Na=Na, Nb=Nb, knots=knots, P=P, seq_len=float(seq_len), offset=float(offset), eps=eps
        )
        E[:, t] = integ

    # Convert to shares per chunk
    deltaN = float(total_tokens) / T
    alpha = E / max(deltaN, eps)

    # Numerical safety: clamp tiny negatives and renormalize each chunk to sum ~1
    alpha = np.clip(alpha, 0.0, None)
    col_sums = alpha.sum(axis=0, keepdims=True)
    nz = np.where(col_sums > 0.0, col_sums, 1.0)
    alpha = alpha / nz

    # r[i,t] = alpha[i,t] / (T * ratios0[i])
    denom = ratios0.reshape(C, 1)
    zero_mask = denom <= eps

    r = np.empty_like(alpha)
    # For datasets with zero baseline, force r=0 (treated as excluded)
    r[zero_mask[:, 0], :] = 0.0
    # For others, divide by (T * ratios0)
    safe_denom = np.where(zero_mask, 1.0, denom)
    r = np.where(zero_mask, 0.0, alpha / (T * safe_denom))

    # Apply constant scaling to all output ratios
    r = r * float(ratio_scale)

    # Emit as dict of lists for line_ratio_resampler_curric.py
    out: Dict[str, List[float]] = {}
    for i in range(C):
        out[f"dataset{i}"] = [float(x) for x in r[i, :].tolist()]
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(
        description="Convert GPU knot schedule to resampler per-chunk keep ratios (with drift)"
    )
    ap.add_argument("--gpu-schedule", type=Path, required=True,
                    help="Path to dict-of-knots JSON schedule for greedy GPU script")
    ap.add_argument("--ratios0", type=Path, required=True,
                    help="Path to baseline ratios JSON {dataset{i}: ratio}")
    ap.add_argument("--total-toks", type=float, required=True,
                    help="Total tokens across all chunks")
    ap.add_argument("--chunks", type=int, required=True,
                    help="Number of chunks T")
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="Fixed integration step size (tokens). Default: 2048")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="Absolute-token offset for the step grid (default 0)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path for resampler schedule")
    ap.add_argument("--eps", type=float, default=1e-12,
                    help="Small epsilon for stability (clamps, divisions)")
    ap.add_argument("--ratio-scale", type=float, default=1.0,
                    help="Multiply all output ratios by this constant")
    ap.add_argument("--pretty", action="store_true",
                    help="Pretty-print JSON output")
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

    schedule = compute_resampler_schedule(
        knots=knots,
        P=P,
        ratios0=ratios0,
        total_tokens=float(args.total_toks),
        chunks=int(args.chunks),
        seq_len=int(args.seq_len),
        offset=float(args.offset),
        ratio_scale=float(args.ratio_scale),
        eps=float(args.eps),
    )

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
