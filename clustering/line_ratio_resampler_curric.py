import argparse, math, random, json, io
import hashlib
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable
"""
Replicate lines across up to 10 000 datasets using either:
  (a) a single per-dataset ratio (legacy "--ratios" mode), or
  (b) a time-varying per-chunk ratio schedule ("--ratio-schedule" mode).

Usage (legacy ratios mode)
--------------------------
    python -m dclm_exp.clustering.line_ratio_resampler_curric \
        --input-root /data/my_datasets \
        --ratios /path/to/ratios.json \
        --output-dir /data/processed \
        --num-shards 16 \
        --num-dirs 4 \
        --first-third  # optional: only use first third of each dataset \
        --jitter       # optional: nested jitter interleave \
        --write-clusters \
        --seed 42

The ratios JSON must map dataset directory names to floating-point ratios, e.g.::

    {
        "dataset0": 1.25,
        "dataset1": 0.0,
        ...,
        "dataset9999": 0.75
    }

Each line in dataset<i> appears ⌊ratio⌋ or ⌈ratio⌉ times (difference ≤ 1 over
lines). Output is globally ordered (shuffle or jitter), then evenly divided into
``--num-shards`` shards (sizes differ by ≤ 1), optionally spread across
``--num-dirs`` subdirectories named ``sub_dir*``.

Usage (time-varying schedule mode)
----------------------------------
    python -m dclm_exp.clustering.line_ratio_resampler_curric \
        --input-root /data/my_datasets \
        --ratio-schedule /path/to/schedule.json \
        --output-dir /data/processed \
        --jitter --write-clusters --seed 42

The schedule JSON must map dataset directory names to arrays of per-chunk
ratios::

    {
        "dataset0": [r00, r01, ..., r0,T-1],
        "dataset1": [r10, r11, ..., r1,T-1],
        ...
    }

All arrays must have the same length T (number of shards). In this mode,
``--num-shards`` (and ``--num-dirs``) are ignored; exactly T shards are
emitted in a single pass as ``shard_{t:08d}_processed.jsonl.zstd`` for
``t = 0..T-1``. If ``--write-clusters`` is set, aligned
``shard_{t:08d}_processed_clusters.npy`` files are also written.

Schedule semantics
------------------
- Deterministic given ``--seed``; per-dataset permutations and tie-breakers are
  seeded deterministically, and per-chunk jitter uses ``seed ^ t``.
- Chunk t contains approximately ``r[i][t] * N_i`` items from dataset i, with
  exact integer counts via fair cumulative rounding.
- No repeats from dataset i until all ``N_i`` unique lines have appeared;
  repeats, when necessary, are introduced minimally and evenly spaced.
"""
from pathlib import Path
from typing import List, Sequence, Tuple, Dict

import zstandard as zstd
import numpy as np
import heapq
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Helpers – IO for .jsonl.zstd files
# ─────────────────────────────────────────────────────────────────────────────

def _open_zstd_text(path: Path):
    """Yield *decoded* text lines from a ``.zstd`` compressed file."""
    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                # Preserve the original newline so downstream tools see the same
                # bytes that were present in the source file.
                yield line


def _write_zstd_lines(lines: Sequence[str], out_path: Path):
    """Write *lines* to *out_path* as UTF-8 text compressed with zstd."""
    cctx = zstd.ZstdCompressor(level=3)
    with out_path.open("wb") as fh:
        with cctx.stream_writer(fh) as writer:
            for line in tqdm(lines, desc="Writing lines...", total=len(lines)):
                if not line.endswith("\n"):
                    line = line + "\n"
                writer.write(line.encode("utf-8"))

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic 64-bit hashing
# ─────────────────────────────────────────────────────────────────────────────

def _hash64(*parts: object) -> int:
    """Return a stable 64-bit hash of heterogeneous parts.

    Uses BLAKE2b with 8-byte digest for determinism across Python versions.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        elif isinstance(p, str):
            h.update(b"S\0")
            h.update(p.encode("utf-8"))
            h.update(b"\0")
        elif isinstance(p, int):
            h.update(b"I\0")
            h.update(p.to_bytes(8, byteorder="little", signed=True))
            h.update(b"\0")
        else:
            # Fallback to repr for other simple types
            s = repr(p)
            h.update(b"R\0")
            h.update(s.encode("utf-8"))
            h.update(b"\0")
    return int.from_bytes(h.digest(), byteorder="little", signed=False)

# ─────────────────────────────────────────────────────────────────────────────
# Core sampling logic (line-based, not character-based)
# ─────────────────────────────────────────────────────────────────────────────

def replicate_lines_for_ratio(lines: Sequence[str], ratio: float, rng: random.Random) -> List[str]:
    """Return a list where every *original* line appears either ⌊ratio⌋ or ⌈ratio⌉ times.

    The total number of returned lines is ``ceil(ratio * len(lines))``.
    """
    if ratio <= 0 or not lines:
        return []

    n = len(lines)
    base = int(math.floor(ratio))               # repeated for every line
    total_needed = math.ceil(ratio * n)
    extra = total_needed - base * n             # number of lines that get +1 copy

    out: List[str] = []

    # Duplicate every line *base* times (may be zero).
    if base:
        for _ in range(base):
            out.extend(lines)

    # Pick *extra* distinct lines at random to receive one additional copy.
    if extra:
        extra_indices = rng.sample(range(n), extra)
        out.extend(lines[i] for i in extra_indices)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Optional 2-way jitter shuffle helpers (borrowed from order_ds.py)
# ─────────────────────────────────────────────────────────────────────────────

def _jittered(ids: Sequence[int], seed: int | None) -> np.ndarray:
    """Return a jittered permutation of *ids* that spaces duplicates.

    If an element appears *f* times in *ids*, the returned order guarantees that
    those *f* occurrences are placed roughly 1/*f* apart, with a random phase
    offset in [0, 1).*
    """
    rng = np.random.default_rng(seed)
    counts = Counter(ids)

    timetable: list[tuple[float, int]] = []
    for idx, f in counts.items():
        u = rng.random()               # random phase offset
        ts = (np.arange(f) + u) / f    # f equi-distant time keys in (0,1]
        timetable.extend(zip(ts, [idx] * f))

    timetable.sort(key=lambda p: p[0])
    return np.fromiter((idx for _, idx in timetable), dtype=np.asarray(ids).dtype)


def nested_jittered_order(
    cluster_ids: Sequence[int],
    sample_ids: Sequence[int],
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Compute a permutation with balanced intra-/inter-cluster spacing.

    Returns an array of indices into *(cluster_ids, sample_ids)* defining
    the jittered order.
    """
    rng = np.random.default_rng(seed)

    # 1) group sample indices by cluster
    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, c in enumerate(cluster_ids):
        clusters[c].append(idx)

    # 2) intra-cluster jitter
    per_cluster: Dict[int, np.ndarray] = {}
    for c, idxs in clusters.items():
        ids_this_cluster = [sample_ids[i] for i in idxs]
        permuted_ids = _jittered(ids_this_cluster, rng.integers(2 ** 63))

        pos_lookup: Dict[int, list[int]] = defaultdict(list)
        for pos in idxs:
            pos_lookup[sample_ids[pos]].append(pos)

        ordered_positions = [pos_lookup[id_].pop() for id_ in permuted_ids]
        per_cluster[c] = np.asarray(ordered_positions, dtype=np.int64)

    # 3) inter-cluster merge via heap
    heap: List[Tuple[float, int, int]] = []
    offsets: Dict[int, float] = {}
    for c, order in per_cluster.items():
        f = len(order)
        u = rng.random()
        offsets[c] = u
        heap.append(((u / f), c, 0))
    heapq.heapify(heap)

    out: List[int] = []
    while heap:
        _, c, p = heapq.heappop(heap)
        idx = per_cluster[c][p]
        out.append(idx)
        p += 1
        if p < len(per_cluster[c]):
            f = len(per_cluster[c])
            new_t = (p + offsets[c]) / f
            heapq.heappush(heap, (new_t, c, p))
    return np.array(out, dtype=np.int64)

# ─────────────────────────────────────────────────────────────────────────────
# High-level workflow helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_lines(ds_dir: Path) -> List[str]:
    """Load *all* lines from ``combined.jsonl.zstd`` inside *ds_dir*."""
    src = ds_dir / "combined.jsonl.zstd"
    if not src.exists():
        src = ds_dir / "shard_00000000_processed.jsonl.zstd"
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    return list(_open_zstd_text(src))


# ============================================================================
# Shard writer – with optional cluster index export
# ============================================================================


def write_shards(
    all_lines: Sequence[str],
    cluster_ids: Sequence[int] | None,
    out_dir: Path,
    num_shards: int,
    num_dirs: int = 1,
):
    """Write *all_lines* evenly to *num_shards* across *num_dirs* sub-directories.

    Every shard receives either ⌊N/S⌋ or ⌈N/S⌉ lines (*N* = total lines,
    *S* = *num_shards*), guaranteeing that shard sizes differ by **at most 1**.

    If ``num_dirs`` > 1, shards are assigned **sequentially** to directories
    named ``sub_dir0``, ``sub_dir1`` … ``sub_dir<num_dirs-1>`` such that
    directory *d* contains shards in the closed interval
    ``[d * (S / D), (d + 1) * (S / D))`` (example: ``S=16``, ``D=4`` → 0–3,
    4–7, 8–11, 12–15).
    """

    if num_shards <= 0:
        raise ValueError("--num-shards must be a positive integer")
    if num_dirs <= 0:
        raise ValueError("--num-dirs must be a positive integer")

    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(all_lines)
    if cluster_ids is not None and len(cluster_ids) != total:
        raise ValueError("cluster_ids length must match number of lines")

    base, remainder = divmod(total, num_shards)  # each shard gets *base* or +1

    # Pre-compute how many shards each directory should hold (ceil division).
    shards_per_dir = math.ceil(num_shards / num_dirs)

    shard_counts: List[int] = []
    pos = 0
    for shard_idx in range(num_shards):
        this_size = base + (1 if shard_idx < remainder else 0)
        shard_lines = all_lines[pos : pos + this_size]
        shard_clusters = (
            cluster_ids[pos : pos + this_size] if cluster_ids is not None else None
        )

        dir_idx = shard_idx // shards_per_dir
        sub_dir = out_dir / f"sub_dir{dir_idx}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        out_path = sub_dir / f"shard_{shard_idx:08d}_processed.jsonl.zstd"
        _write_zstd_lines(shard_lines, out_path)

        # Optional: save cluster indices alongside shard
        if shard_clusters is not None:
            clusters_out = sub_dir / f"shard_{shard_idx:08d}_processed_clusters.npy"
            np.save(clusters_out, np.asarray(shard_clusters, dtype=np.int32))

        shard_counts.append(len(shard_lines))
        pos += this_size

    # Disabled manifest.jsonl writing; previously wrote manifest.jsonl
    # manifest = out_dir / "manifest.jsonl"
    # with manifest.open("w", encoding="utf-8") as mf:
    #     for i, cnt in enumerate(shard_counts):
    #         shard_name = f"shard_{i:08d}_processed"
    #         json.dump({"shard": shard_name, "num_lines": cnt}, mf)
    #         mf.write("\n")

    print(
        f"✅  Wrote {sum(shard_counts):,} lines to {len(shard_counts)} shards across "
        f"{num_dirs} dir(s) in {out_dir}"
        + (
            " with cluster indices" if cluster_ids is not None else ""
        )
    )


def write_single_shard(
    lines: Sequence[str],
    clusters: Sequence[int] | None,
    out_dir: Path,
    t: int,
):
    """Write exactly one shard named by index t into out_dir.

    Writes shard_{t:08d}_processed.jsonl.zstd and, if clusters is provided,
    shard_{t:08d}_processed_clusters.npy aligned to the final order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{t:08d}_processed.jsonl.zstd"
    _write_zstd_lines(lines, out_path)
    if clusters is not None:
        clusters_out = out_dir / f"shard_{t:08d}_processed_clusters.npy"
        np.save(clusters_out, np.asarray(clusters, dtype=np.int32))

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Replicate lines in 10k datasets according to per-dataset ratios.")

    p.add_argument("--input-root", type=Path, required=True,
                   help="Root directory containing dataset{0..9999} sub-dirs with combined.jsonl.zstd files")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ratios", type=Path,
                       help="Path to JSON file mapping 'dataset<i>' → ratio")
    group.add_argument("--ratio-schedule", dest="ratio_schedule", type=Path,
                       help="Path to JSON mapping 'dataset<i>' → [r[i][0],...,r[i][T-1]] (all arrays same length T)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Destination directory for the processed shards")
    p.add_argument("--num-shards", dest="num_shards", type=int, default=None,
                   help="Number of output shards (ignored in --ratio-schedule mode)")
    p.add_argument("--num-dirs", dest="num_dirs", type=int, default=1,
                   help="Number of subdirectories to split shards into (sequential assignment)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for shuffling & extra-line selection")
    p.add_argument("--first-third", action="store_true",
                   help="Only consider the first third of lines in each dataset when applying ratios")
    p.add_argument("--jitter", action="store_true",
                   help="Apply 2-way jitter (balanced intra-/inter-dataset spacing) instead of uniform shuffle")

    # New: write cluster indices for ordered_tokenize.py
    p.add_argument("--write-clusters", action="store_true",
                   help="Emit *_clusters.npy files with the dataset index corresponding to each line")

    args = p.parse_args(argv)
    # Schedule mode takes precedence if provided
    if getattr(args, "ratio_schedule", None) is not None:
        # Parse & validate schedule JSON
        if args.ratio_schedule.suffix != ".json":
            raise SystemExit("--ratio-schedule must point to a .json file mapping 'dataset<i>' → [ratios]")

        with args.ratio_schedule.open("r", encoding="utf-8") as f:
            sched_raw = json.load(f)
        if not isinstance(sched_raw, dict):
            raise SystemExit("Schedule JSON must be an object mapping dataset names to arrays of floats")

        # Normalize: keys to str, values to list[float]
        schedule: Dict[str, List[float]] = {}
        T: int | None = None
        for k, v in sched_raw.items():
            key = str(k)
            if not isinstance(v, list) or not v:
                raise SystemExit(f"Schedule for {key} must be a non-empty list of floats")
            arr = [float(x) for x in v]
            if T is None:
                T = len(arr)
            elif len(arr) != T:
                raise SystemExit("All schedule arrays must have identical length T")
            schedule[key] = arr
        assert T is not None and T >= 1

        if args.num_shards is not None:
            print(f"ℹ️  --num-shards={args.num_shards} ignored in schedule mode (using T={T}).")

        # Initialize T buffers
        print(f"Initializing {T} buffers...")
        chunk_lines: List[List[str]] = [[] for _ in range(T)]
        chunk_cluster_ids: List[List[int]] = [[] for _ in range(T)]
        chunk_sample_ids: List[List[int]] = [[] for _ in range(T)]

        # Iterate datasets 0..9999
        for idx in tqdm(range(10000), desc="Processing datasets...", total=10000):
            key = f"dataset{idx}"
            ratios = schedule.get(key)
            if ratios is None:
                print(f"⚠️  No schedule provided for {key} – skipping.")
                continue

            ds_dir = args.input_root / key
            if not ds_dir.exists():
                print(f"⚠️  Dataset directory missing: {ds_dir} – skipping.")
                continue
            lines_full = load_dataset_lines(ds_dir)
            visible = lines_full[: len(lines_full) // 3] if args.first_third else lines_full
            N_i = len(visible)
            if N_i == 0:
                print(f"⚠️  {key} has 0 visible lines – skipping.")
                continue

            total_ratio = sum(ratios)
            if total_ratio > 1.0 + 1e-12:
                print(f"⚠️  Sum of ratios for {key} is {total_ratio:.6f} (>1.0): repeats will occur.")

            # Deterministic per-dataset seeds
            seed_perm = _hash64("perm", int(args.seed), int(idx))
            seed_u = _hash64("u", int(args.seed), int(idx))

            # Build one permutation P_i of indices 0..N_i-1
            #print(f"Building permutation for {key}...")
            rng_np = np.random.default_rng(seed_perm)
            P_i = np.arange(N_i, dtype=np.int64)
            rng_np.shuffle(P_i)

            # Draw u_i ∈ [0,1)
            rng_u = np.random.default_rng(seed_u)
            u_i = float(rng_u.random())

            # Precompute cumulative rounding M_i,t and chunk counts Δ_i,t
            cumulative = 0.0
            M_prev = 0
            c_i = 0  # total emitted so far for this dataset
            for t in range(T):
                r_it = float(ratios[t])
                cumulative += r_it
                M_t = int(math.floor(N_i * cumulative + u_i))
                delta = M_t - M_prev
                if delta < 0:
                    # Numerical issues should not cause negative deltas; clamp
                    delta = 0
                if delta > 0:
                    # Select next delta indices without premature repeats
                    for k in range(delta):
                        idx_k = int(P_i[(c_i + k) % N_i])
                        chunk_lines[t].append(visible[idx_k])
                        chunk_cluster_ids[t].append(idx)
                        chunk_sample_ids[t].append(idx_k)
                    c_i += delta
                M_prev = M_t

            # Validation: total emitted equals M_i,T-1
            assert c_i == M_prev, f"Cumulative count mismatch for {key}: emitted={c_i}, target={M_prev}"

        # Per-chunk ordering and writing
        for t in range(T):
            lines_t = chunk_lines[t]
            cl_ids_t = chunk_cluster_ids[t]
            smp_ids_t = chunk_sample_ids[t]

            if args.jitter:
                order = nested_jittered_order(cl_ids_t, smp_ids_t, seed=(int(args.seed) ^ int(t)))
                lines_t = [lines_t[i] for i in order]
                cl_final = [cl_ids_t[i] for i in order] if args.write_clusters else None
            else:
                rng_local = random.Random(int(args.seed) ^ int(t))
                if args.write_clusters:
                    perm = list(range(len(lines_t)))
                    rng_local.shuffle(perm)
                    lines_t = [lines_t[i] for i in perm]
                    cl_final = [cl_ids_t[i] for i in perm]
                else:
                    rng_local.shuffle(lines_t)
                    cl_final = None

            write_single_shard(lines_t, cl_final, args.output_dir, t)

        print(f"✅  Wrote {T} shards to {args.output_dir} (schedule mode)")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy ratios mode (unchanged behavior)
    # ─────────────────────────────────────────────────────────────────────────
    if args.num_shards is None:
        raise SystemExit("--num-shards is required when using --ratios mode")

    rng = random.Random(args.seed)

    # Load JSON mapping dataset<i> → ratio
    if args.ratios.suffix != ".json":
        raise SystemExit("--ratios must point to a .json file containing a mapping 'dataset<i>' → ratio")

    with args.ratios.open("r", encoding="utf-8") as f:
        ratio_map_raw = json.load(f)

    if not isinstance(ratio_map_raw, dict):
        raise SystemExit("Ratios JSON must be an object mapping dataset names to float ratios")

    # Normalise keys to strings and ensure float values
    ratio_map = {str(k): float(v) for k, v in ratio_map_raw.items()}

    all_lines: List[str] = []
    all_clusters: List[int] = []  # aligns with *all_lines*
    cluster_ids: List[int] = []
    sample_ids: List[int] = []
    sample_lines: List[str] = []

    for idx in range(10000):
        key = f"dataset{idx}"
        ratio = ratio_map.get(key)
        if ratio is None:
            print(f"⚠️  No ratio specified for {key} – skipping.")
            continue

        ds_dir = args.input_root / key
        if not ds_dir.exists():
            print(f"⚠️  Dataset directory missing: {ds_dir} – skipping.")
            continue

        print(f"📂 Processing {ds_dir}  (ratio={ratio})")
        lines_full = load_dataset_lines(ds_dir)
        visible = lines_full[: len(lines_full) // 3] if args.first_third else lines_full

        # Build selected indices via replicate_lines_for_ratio
        idxs = list(range(len(visible)))
        selected_idxs = replicate_lines_for_ratio(idxs, ratio, rng)
        selected_lines = [visible[i] for i in selected_idxs]

        print(f"   Selected {len(selected_lines):,} lines out of {len(lines_full):,} (visible={len(visible):,})")

        if args.jitter:
            cluster_ids.extend([idx] * len(selected_idxs))
            sample_ids.extend(selected_idxs)
            sample_lines.extend(selected_lines)
        else:
            all_lines.extend(selected_lines)
            if args.write_clusters:
                all_clusters.extend([idx] * len(selected_lines))

    # Order the final lines
    if args.jitter:
        order = nested_jittered_order(cluster_ids, sample_ids, seed=args.seed)
        all_lines = [sample_lines[i] for i in order]
    else:
        if args.write_clusters:
            perm = list(range(len(all_lines)))
            rng.shuffle(perm)
            all_lines = [all_lines[i] for i in perm]
            all_clusters = [all_clusters[i] for i in perm]
        else:
            rng.shuffle(all_lines)

    clusters_param = None
    if args.write_clusters:
        if args.jitter:
            # clusters derived after jitter ordering
            clusters_param = [cluster_ids[i] for i in order]
        else:
            clusters_param = all_clusters

    write_shards(all_lines, clusters_param, args.output_dir, args.num_shards, args.num_dirs)


if __name__ == "__main__":
    main() 