#!/usr/bin/env python3
"""
greedy_order_gpu.py
===================

Multi-GPU greedy ordering for ≈10 M sequences × 10 k clusters.

Key ideas
---------
∘ Counts are stored **once** per GPU as int16 – memory ≈ 25 GB / GPU
∘ A Boolean *used* mask on each GPU guarantees every sequence is picked once.
∘ Each iteration:
   1. Current global token totals are broadcast to every GPU
   2. Every GPU computes the error for **all of its local sequences** in one
      vectorised kernel and returns the *k* best (k ≪ N)
   3. CPU merges the candidates, chooses the global best, updates state
   4. That row’s error is flushed to +∞ on its GPU (mask = True)

Dependencies
------------
`pip install torch numpy tqdm`

No Faiss / no scipy required.

Usage
-----
 Run on 8 H100 GPUs:
 
     python greedy_order.py \
         --input-dir   /path/to/tokenized_shards \
         --ratio-file  cluster_ratios.json \
         --output-dir  /path/to/ordered_tokens \  # tokens shards
         --counts-dir  /path/to/ordered_counts \  # parallel counts shards
         --shard-size 8192 \
         --gpus 8
 
This writes token shards into ``--output-dir`` and matching counts-only
shards into ``--counts-dir``.
"""

from __future__ import annotations
import argparse, json, gzip, tarfile, uuid, os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
import webdataset as wds
import faiss  # Facebook AI similarity search


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class SequenceMeta:
    shard_name : str   # original tar file
    uuid_str   : str   # uuid inside that tar


def read_ratio_file(path: Path) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    ratios = np.asarray(data['ratios'] if isinstance(data, dict) else data, dtype=np.float32)
    ratios /= ratios.sum()
    return ratios


# --------------------------------------------------------------------------- #
# Fast streaming loader                                                      #
# --------------------------------------------------------------------------- #

def stream_shards_to_arrays(
    shard_paths: list[Path],
    n_cluster: int,
    workers: int,
    prefetch: int,
) -> tuple[torch.Tensor, list[bytes], list[SequenceMeta]]:
    """Stream shards into a contiguous counts matrix + tokens list.

    Returns
    -------
    counts_all : torch.Tensor (N, n_cluster) float32 (CPU)
    tokens_all : List[bytes]              original gzip payload per seq
    meta_all   : List[SequenceMeta]       one per sequence
    """

    # ------------------------------------------------------------------ load counts tensors (one per shard) – stay in torch
    counts_tensor_list: list[torch.Tensor] = []
    for p in tqdm(shard_paths, desc="Loading counts", unit="shard"):
        counts_path = p.parent / f"{p.stem}_counts.pt"
        if not counts_path.exists():
            raise FileNotFoundError(f"Expected counts file {counts_path} for shard {p}")
        tens: torch.Tensor = torch.load(counts_path, map_location="cpu")  # type: ignore
        if tens.ndim != 2 or tens.shape[1] != n_cluster:
            raise ValueError(
                f"Counts tensor shape mismatch in {counts_path}; expected (_, {n_cluster})"
            )
        counts_tensor_list.append(tens.to(dtype=torch.float32, copy=False))

    counts_all: torch.Tensor = torch.cat(counts_tensor_list, dim=0)

    # ------------------------------------------------------------------ stream tokens (only) with WebDataset
    dataset = (
        wds.WebDataset([str(p) for p in shard_paths], shardshuffle=False, handler=wds.warn_and_continue)
        .to_tuple("__key__", "tokens.json.gz")
        .with_length(None)
    )

    loader = wds.WebLoader(
        dataset,
        num_workers=workers,
        batch_size=None,
        prefetch_factor=prefetch,
    )

    tokens_all: list[bytes] = []
    meta_all  : list[SequenceMeta] = []

    for key, tok_bytes in tqdm(loader, desc="Streaming tokens", unit="seq"):
        tokens_all.append(tok_bytes)
        meta_all.append(SequenceMeta(key.split("@")[0], key))

    if len(tokens_all) != counts_all.shape[0]:
        raise RuntimeError(
            f"Mismatch between token sequences ({len(tokens_all)}) and counts ({counts_all.shape[0]})"
        )

    return counts_all, tokens_all, meta_all


# --------------------------------------------------------------------------- #
# Build Faiss multi-GPU HNSW index                                            #
# --------------------------------------------------------------------------- #

def build_faiss_index(counts_all: torch.Tensor, M: int, ef_construction: int, ngpu: int):
    """Return a faiss.Index distributed across GPUs (shards) using HNSW."""

    # Convert to NumPy float32 for Faiss
    counts_np = counts_all.cpu().numpy().astype("float32", copy=False)
    d = counts_np.shape[1]

    index_cpu = faiss.IndexHNSWFlat(d, M)
    index_cpu.hnsw.efConstruction = ef_construction
    index_cpu.verbose = False
    # Add vectors in batches to reduce memory spikes
    batch = 1024
    for i in tqdm(range(0, counts_np.shape[0], batch), desc="Adding vectors to Faiss"):
        index_cpu.add(counts_np[i : i + batch])

    # Shard index across GPUs
    res = [faiss.StandardGpuResources() for _ in range(ngpu)]
    gpu_index = faiss.index_cpu_to_gpus_list(index_cpu, res, list(range(ngpu)))
    return gpu_index


# --------------------------------------------------------------------------- #
# Main greedy loop                                                            #
# --------------------------------------------------------------------------- #
def greedy_faiss(counts_all: torch.Tensor, target: np.ndarray, ngpu: int, k_return: int = 8,
                 M: int = 32, ef_construction: int = 40, ef_search: int = 128) -> list[int]:
    """Greedy ordering using a Faiss multi-GPU HNSW index."""

    target_t = torch.as_tensor(target, dtype=torch.float32, device='cuda:0')
    n_cluster = target.shape[0]

    # Build index (CPU -> GPUs)
    index = build_faiss_index(counts_all, M=M, ef_construction=ef_construction, ngpu=ngpu)
    index.hnsw.efSearch = ef_search

    cum_counts = torch.zeros(n_cluster, dtype=torch.float32, device='cuda:0')
    seq_len_const = 2049  # constant sequence length

    n_total = counts_all.shape[0]
    order: list[int] = []

    pbar = tqdm(total=n_total, desc="Ordering sequences (Faiss)")

    used_mask = np.zeros(n_total, dtype=bool)

    counts_all_cuda = counts_all.to(device='cuda:0')  # for cum_counts updates

    while len(order) < n_total:
        desired_remaining = target_t * (n_total * seq_len_const) - cum_counts

        # Query faiss with the desired vector
        query = desired_remaining.cpu().numpy().astype('float32').reshape(1, -1)
        D, I = index.search(query, k_return)
        candidates = I[0]

        chosen = -1
        for idx in candidates:
            if idx < 0:
                continue
            if not used_mask[idx]:
                chosen = int(idx)
                break

        if chosen == -1:
            # Fallback: search more neighbors
            D, I = index.search(query, k_return * 4)
            for idx in I[0]:
                if idx >= 0 and not used_mask[idx]:
                    chosen = int(idx)
                    break
            if chosen == -1:
                raise RuntimeError("No selectable sequence found – logic error (faiss)")

        # Mark as used and remove from index
        used_mask[chosen] = True
        idx_remove = np.array([chosen], dtype='int64')
        index.remove_ids(idx_remove)

        # Update cum_counts and order
        cum_counts += counts_all_cuda[chosen]
        order.append(chosen)

        pbar.update(1)

    pbar.close()
    return order


# --------------------------------------------------------------------------- #
# Tar writing & manifest                                                      #
# --------------------------------------------------------------------------- #
def write_output(order: list[int], tokens_all: List[bytes], counts_all: torch.Tensor,
                 out_sequences_dir: Path, counts_dir: Path, chunk: int) -> list[Dict]:
    out_sequences_dir.mkdir(parents=True, exist_ok=True)
    counts_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[Dict] = []
    shard_id = 0
    buf_tokens, buf_counts = [], []

    def flush():
        nonlocal shard_id, buf_tokens, buf_counts
        if not buf_tokens:
            return
        shard_path = out_sequences_dir / f'shard_{shard_id:08d}.tar'
        with tarfile.open(shard_path, 'w') as tar:
            for tok, cnt in zip(buf_tokens, buf_counts):
                uid = uuid.uuid4().hex
                tb = tok  # already gzipped bytes from source
                cb = gzip.compress(json.dumps([int(x) for x in cnt]).encode())
                ti = tarfile.TarInfo(f'{uid}.tokens.json.gz'); ti.size=len(tb)
                ci = tarfile.TarInfo(f'{uid}.counts.json.gz'); ci.size=len(cb)
                tar.addfile(ti, BytesIO(tb)); tar.addfile(ci, BytesIO(cb))
        manifest.append({'shard_id': shard_id,
                         'shard_name': shard_path.name,
                         'num_sequences': len(buf_tokens)})
        shard_id += 1
        buf_tokens, buf_counts = [], []

    counts_writer = wds.ShardWriter(str(counts_dir / "shard_%08d.tar"), maxcount=chunk, encoder=False)

    for idx in order:
        tok_bytes = tokens_all[idx]
        cnt = counts_all[idx].to(dtype=torch.int32, device='cpu').tolist()
        buf_tokens.append(tok_bytes); buf_counts.append(cnt)

        # Write counts sample immediately via ShardWriter to avoid big memory
        uid = uuid.uuid4().hex
        counts_writer.write({"__key__": uid, "counts.json.gz": gzip.compress(json.dumps(cnt).encode())})

        if len(buf_tokens) == chunk:
            flush()
    flush()
    counts_writer.close()
    return manifest


def write_manifest(manifest: list[Dict], out_dir: Path,
                   target: np.ndarray, cum_counts: np.ndarray, tot_tokens: int):
    fout = out_dir / 'manifest.jsonl'
    final_ratio = (cum_counts / tot_tokens).tolist()
    with open(fout, 'w') as f:
        summary = dict(summary=dict(
            num_shards      = len(manifest),
            num_sequences   = sum(m['num_sequences'] for m in manifest),
            num_tokens      = int(tot_tokens),
            target_cluster_ratios = target.tolist(),
            actual_cluster_ratios = final_ratio,
            total_squared_error   = float(np.square(
                np.asarray(final_ratio) - target).sum())))
        f.write(json.dumps(summary) + '\n')
        for m in manifest:
            f.write(json.dumps(m)+'\n')
    print(f'Wrote manifest to {fout}')


# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--ratio-file', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--counts-dir', required=True,
                    help="Directory to write counts-only shards (separate dataset)")
    ap.add_argument('--shard-size', type=int, default=8192,
                    help="Number of sequences per output shard")
    ap.add_argument('--gpus', type=int, default=torch.cuda.device_count(),
                    help='GPUs to use for Faiss index (default: all visible)')
    ap.add_argument('--hnsw-M', type=int, default=32, help='HNSW M parameter')
    ap.add_argument('--hnsw-ef-construction', type=int, default=40, help='HNSW efConstruction')
    ap.add_argument('--hnsw-ef-search', type=int, default=128, help='HNSW efSearch during queries')
    ap.add_argument('--loader-workers', type=int, default=8,
                    help='Worker processes for WebLoader streaming')
    ap.add_argument('--loader-prefetch', type=int, default=8,
                    help='Prefetch batches per worker (WebLoader)')
    args = ap.parse_args()

    target = read_ratio_file(Path(args.ratio_file))
    n_cluster = len(target)
    print(f'Clusters: {n_cluster}')

    # ------------------------------------------------------------------ load tars
    shard_paths = sorted(Path(args.input_dir).glob('shard_*.tar'))
    print(f"Found {len(shard_paths)} shards; loading counts tensors and streaming token data …")

    counts_all, tokens_all, meta_all = stream_shards_to_arrays(
        shard_paths,
        n_cluster=n_cluster,
        workers=args.loader_workers,
        prefetch=args.loader_prefetch,
    )

    # ------------------------------------------------------------------ greedy order with Faiss HNSW
    torch.cuda.empty_cache()
    order_indices = greedy_faiss(counts_all, target, args.gpus, k_return=8,
                                 M=args.hnsw_M, ef_construction=args.hnsw_ef_construction,
                                 ef_search=args.hnsw_ef_search)

    # ------------------------------------------------------------------ write output
    out_dir = Path(args.output_dir)
    counts_dir = Path(args.counts_dir)
    manifest = write_output(order_indices, tokens_all, counts_all, out_dir, counts_dir, args.shard_size)

    # final stats
    cum = counts_all[order_indices].sum(dim=0).cpu().numpy()
    tot = float(cum.sum())
    write_manifest(manifest, out_dir, target, cum, tot)
    print('Done.')

if __name__ == '__main__':
    main() 