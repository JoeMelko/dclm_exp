#!/usr/bin/env python3
"""order_ds.py  –  Assemble a *single* WebDataset from several per-cluster
sub-datasets while guaranteeing periodic inter- and intra-cluster spacing.

The script expects **already tokenised** WebDataset directories produced by
``create_cluster_ds.py`` → ``tokshuf_dir.sh`` or any equivalent workflow.
Each cluster lives in its *own* directory that contains any number of
``*.tar`` shards plus a ``manifest.jsonl`` (the manifest is ignored here).
You can either:

• pass an **explicit** list of cluster directories via ``--input-dirs``; or
• pass a **parent** directory via ``--input-root`` whose *immediate*
  sub-directories are treated as the clusters.

For every cluster *i* the caller specifies a target number of samples *n_i*.
If *n_i* ≤ |D_i|, a random subset of that many unique samples is drawn.
If *n_i* > |D_i|, every sample is repeated ⌊n_i / |D_i|⌋ times and the
remainder is filled by sampling **without replacement** from the original
samples.  The final goal is ∑_i n_i samples.

Two levels of jitter are applied:

1. Intra-cluster  (handled by `_jittered`)  –  guarantees that if a sample
   occurs *f* times inside its cluster, the temporal offsets between those
   occurrences are as even as possible (i.e. a jittered linspace).
2. Inter-cluster  (handled by `nested_jittered_order`)  –  merges the pre-
   jittered per-cluster orders such that clusters themselves are evenly
   inter-leaved as well.

The resulting order is written *sequentially* to a new WebDataset via
:class:`webdataset.ShardWriter`.  All runtime shuffling should therefore be
turned off during training so that the model sees samples in this exact order.

Examples
--------

Explicit directory list::

    python order_ds.py \
        --input-dirs   cluster0 cluster1 cluster2 \
        --counts-json  n_per_cluster.json         \
        --output-dir   merged_balanced_ds         \
        --shard-size   8192                       \
        --seed         42

Parent directory::

    python order_ds.py \
        --input-root   clusters_parent_dir \
        --counts-json  n_per_cluster.json  \
        --output-dir   merged_balanced_ds  \
        --seed         42 \
        --shard-size   64

``n_per_cluster.json`` must map *directory basenames* to integers, e.g.::

    {
        "cluster0": 5000000,
        "cluster1": 7500000,
        "cluster2": 3000000
    }
"""
from __future__ import annotations

import argparse, json, math, heapq, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
import webdataset as wds

# ─────────────────────────────────────────────────────────────────────────────
# Jitter helpers (borrowed from https://github.com/facebookresearch/llama-rec? …)
# ─────────────────────────────────────────────────────────────────────────────

def _jittered(ids: Sequence[int], seed: int | None) -> np.ndarray:
    """Return a jittered permutation of *ids* that spaces duplicates.

    If an element appears *f* times in *ids*, the returned order guarantees that
    those *f* occurrences are placed roughly 1/*f* apart, with a random phase
    offset in [0, 1).*.
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
    """Compute a one-epoch permutation with balanced intra-/inter-cluster spacing.

    Parameters
    ----------
    cluster_ids
        Array-like of the same length as *sample_ids* assigning each sample to a
        cluster.
    sample_ids
        Identifiers of the *actual* samples (may contain duplicates).
    seed
        Optional RNG seed;  *None* → non-deterministic.

    Returns
    -------
    np.ndarray
        The permuted **indices** into *(cluster_ids, sample_ids)* (dtype = int64).
    """
    rng = np.random.default_rng(seed)

    # 1) group sample indices by cluster
    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, c in enumerate(cluster_ids):
        clusters[c].append(idx)

    # 2) pre-compute intra-cluster orders (indices!)
    per_cluster: Dict[int, np.ndarray] = {}
    for c, idxs in clusters.items():
        # Collect the sample IDs for these positions – may contain duplicates
        ids_this_cluster = [sample_ids[i] for i in idxs]

        # Compute a jittered permutation *of the IDs* ( preserves duplicates )
        permuted_ids = _jittered(ids_this_cluster, rng.integers(2 ** 63))

        # Map each ID to the list of positions that carry it
        pos_lookup: Dict[int, list[int]] = defaultdict(list)
        for pos in idxs:
            pos_lookup[sample_ids[pos]].append(pos)

        # Build the final order of *positions* for this cluster by popping
        # from the lookup in the sequence dictated by `permuted_ids`.
        ordered_positions = [pos_lookup[id_].pop() for id_ in permuted_ids]
        per_cluster[c] = np.asarray(ordered_positions, dtype=np.int64)

    # 3) build heap of (time_key, cluster, ptr)
    heap: List[Tuple[float, int, int]] = []
    offsets: Dict[int, float] = {}
    for c, order in per_cluster.items():
        f = len(order)
        u = rng.random()
        offsets[c] = u
        heap.append(((u / f), c, 0))
    heapq.heapify(heap)

    # 4) global merge – produces indices into original arrays
    out: List[int] = []
    while heap:
        t, c, p = heapq.heappop(heap)
        idx = per_cluster[c][p]
        out.append(idx)
        p += 1
        if p < len(per_cluster[c]):
            f = len(per_cluster[c])
            new_t = (p + offsets[c]) / f
            heapq.heappush(heap, (new_t, c, p))
    return np.array(out, dtype=np.int64)

# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_samples(shard_dir: Path) -> List[dict]:
    """Load **all** samples from ``shard_dir`` into memory.

    We expand the ``*.tar`` glob ourselves so that we can detect the error case
    where no shards are present and raise a *clear* exception instead of the
    less helpful FileNotFoundError coming from WebDataset.
    """
    paths = sorted(p for p in shard_dir.glob("*.tar") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")

    # Disable internal shuffling – we want deterministic original order.
    ds = wds.WebDataset([str(p) for p in paths], shardshuffle=False, handler=wds.warn_and_continue)
    ds = ds.with_length(None)  # keep raw bytes, no len information
    return list(ds)


def select_n_samples(
    samples: List[dict],
    n_target: int,
    rng: np.random.Generator,
) -> List[int]:
    """Return a list of *indices* (into *samples*) of length *n_target*.

    Implements the over-sampling rule described in the docstring.
    """
    n_src = len(samples)
    if n_target <= n_src:
        return rng.choice(n_src, size=n_target, replace=False).tolist()

    k, r = divmod(n_target, n_src)      # each sample repeated k times; r extras
    idxs = list(range(n_src)) * k
    idxs.extend(rng.choice(n_src, size=r, replace=False))
    rng.shuffle(idxs)
    return idxs

# ─────────────────────────────────────────────────────────────────────────────
# Main assembly logic
# ─────────────────────────────────────────────────────────────────────────────

def assemble_dataset(
    input_dirs: List[Path],
    counts: Dict[str, int],
    out_dir: Path,
    maxcount: int,
    seed: int,
    order_mode: str = "jittered",
):
    rng = np.random.default_rng(seed)

    # Step A – load & sample per cluster
    per_cluster_samples: Dict[int, List[dict]] = {}
    per_cluster_selected_idxs: Dict[int, List[int]] = {}

    for cid, in_dir in enumerate(input_dirs):
        dir_name = in_dir.name
        if dir_name not in counts:
            raise SystemExit(f"No sample-count specified for cluster '{dir_name}'")
        n_target = counts[dir_name]
        print(f"📥  Loading cluster '{dir_name}' (id={cid}) – target {n_target} samples")

        samples = load_all_samples(in_dir)
        per_cluster_samples[cid] = samples

        selected_idxs = select_n_samples(samples, n_target, rng)
        per_cluster_selected_idxs[cid] = selected_idxs

        print(f"    original |D| = {len(samples):,}; selected {len(selected_idxs):,} samples")

    # Step B – build arrays for jittered ordering
    cluster_ids: List[int] = []
    sample_ids: List[Tuple[int, int]] = []      # (cluster_id, idx_within_cluster)

    for cid, idxs in per_cluster_selected_idxs.items():
        cluster_ids.extend([cid] * len(idxs))
        sample_ids.extend([(cid, idx) for idx in idxs])

    # For the *intra* jitter we only need a 1-D array; we encode each sample as
    # a running integer index.
    linear_sample_ids = list(range(len(sample_ids)))

    if order_mode == "random":
        order = rng.permutation(len(sample_ids))
    else:
        order = nested_jittered_order(cluster_ids, linear_sample_ids, seed=seed)

    print("🧮  Final permutation length:", len(order))

    # Step C – write to output WebDataset
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = wds.ShardWriter(str(out_dir / "shard_%08d.tar"), maxcount=maxcount, verbose=1, encoder=False)

    for pos in order:
        cid, local_idx = sample_ids[pos]
        sample = per_cluster_samples[cid][local_idx]
        # ShardWriter takes a dict-like sample; we *must* ensure keys are bytes
        sink.write(sample)
    sink.close()

    # Step D – write manifest.jsonl describing the new shards
    total_samples = len(order)
    num_full, remainder = divmod(total_samples, maxcount)
    shard_counts = [maxcount] * num_full
    if remainder:
        shard_counts.append(remainder)

    manifest_out = out_dir / "manifest.jsonl"
    with manifest_out.open("w", encoding="utf-8") as mf:
        for i, count in enumerate(shard_counts):
            shard_name = f"shard_{i:08d}"
            json.dump({"shard": shard_name, "num_sequences": count}, mf)
            mf.write("\n")

    print(f"✅  Wrote {len(order):,} samples to {out_dir} across {len(shard_counts)} shards; manifest.jsonl created")

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_counts(fp: Path) -> Dict[str, int]:
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("counts-json must be an object mapping name → int")
    return {str(k): int(v) for k, v in data.items()}


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser("merge clusters into one WebDataset with balanced order")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dirs", nargs="+", type=Path,
                       help="Explicit list of per-cluster dataset directories (tokenised WebDataset)")
    group.add_argument("--input-root", type=Path,
                       help="Parent directory whose *immediate* sub-directories are the per-cluster datasets")
    p.add_argument("--counts-json", type=Path, required=True,
                   help="JSON mapping <basename(input_dir)> → target sample count n_i")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Destination directory for the merged dataset")
    p.add_argument("--shard-size", "--maxcount", dest="maxcount", type=int, default=1024,
                   help="Maximum #samples per output shard (ShardWriter). Alias: --shard-size")
    p.add_argument("--seed", type=int, required=True,
                   help="Global RNG seed")
    p.add_argument("--order", choices=["jittered", "random"], default="jittered",
                   help="Ordering of merged samples: balanced 'jittered' (default) or fully 'random'")

    args = p.parse_args(argv)

    counts = parse_counts(args.counts_json)

    # resolve input directories
    input_dirs: List[Path]
    if args.input_dirs is not None:
        input_dirs = args.input_dirs
    else:
        if not args.input_root.exists():
            raise SystemExit(f"--input-root directory not found: {args.input_root}")
        input_dirs = sorted(p for p in args.input_root.iterdir() if p.is_dir())
        if not input_dirs:
            raise SystemExit(f"--input-root contains no sub-directories: {args.input_root}")

    assemble_dataset(
        input_dirs=input_dirs,
        counts=counts,
        out_dir=args.output_dir,
        maxcount=args.maxcount,
        seed=args.seed,
        order_mode=args.order,
    )


if __name__ == "__main__":
    main()
