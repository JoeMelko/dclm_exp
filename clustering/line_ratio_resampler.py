import argparse, math, random, json, io
"""
Replicate lines across up to 10 000 datasets according to per-dataset ratios.

Usage example
-------------
    python -m dclm_exp.clustering.line_ratio_resampler \
        --input-root /data/my_datasets \
        --ratios ratios.json \
        --output-dir /data/processed \
        --num-shards 16 \
        --num-dirs 4 \
        --first-third  # optional: only use first third of each dataset \
        --seed 42

The JSON file (provided via --ratios) must map dataset directory names to
floating-point ratios, e.g.::

    {
        "dataset0": 1.25,
        "dataset1": 0.0,
        ...,
        "dataset9999": 0.75
    }

Each line in dataset<i> will appear ⌊ratio⌋ or ⌈ratio⌉ times such that the
maximum difference in per-line counts is 1.  Output is shuffled and written to
shards named ``shard_XXXXXXXX_processed.jsonl.zstd`` with at most 100 000 lines
per shard, along with a manifest.jsonl.
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
            for line in lines:
                if not line.endswith("\n"):
                    line = line + "\n"
                writer.write(line.encode("utf-8"))

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

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Replicate lines in 10k datasets according to per-dataset ratios.")

    p.add_argument("--input-root", type=Path, required=True,
                   help="Root directory containing dataset{0..9999} sub-dirs with combined.jsonl.zstd files")
    p.add_argument("--ratios", type=Path, required=True,
                   help="Path to JSON file mapping 'dataset<i>' → ratio")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Destination directory for the processed shards")
    p.add_argument("--num-shards", dest="num_shards", type=int, required=True,
                   help="Number of output shards; lines are divided as evenly as possible (sizes differ by ≤1)")
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