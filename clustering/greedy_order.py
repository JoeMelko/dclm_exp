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
) -> tuple[np.ndarray, list[bytes], list[SequenceMeta]]:
    """Stream shards into a contiguous counts matrix + tokens list.

    Returns
    -------
    counts_all : np.ndarray (N, n_cluster) float32
    tokens_all : List[bytes]              original gzip payload per seq
    meta_all   : List[SequenceMeta]       one per sequence
    """

    dataset = (
        wds.WebDataset([str(p) for p in shard_paths], shardshuffle=False, handler=wds.warn_and_continue)
        .to_tuple("__key__", "tokens.json.gz", "counts.json.gz")
        .with_length(None)
    )

    loader = wds.WebLoader(dataset,
                           num_workers=workers,
                           batch_size=None,
                           prefetch_factor=prefetch)

    counts_list: list[np.ndarray] = []
    tokens_all: list[bytes] = []
    meta_all  : list[SequenceMeta] = []

    for key, tok_bytes, cnt_bytes in tqdm(loader, desc="Streaming", unit="seq"):
        counts_vec = np.asarray(json.loads(gzip.decompress(cnt_bytes).decode()), dtype=np.float32)
        if counts_vec.size != n_cluster:
            raise ValueError(f"counts length mismatch in sample {key}")
        counts_list.append(counts_vec)
        tokens_all.append(tok_bytes)
        meta_all.append(SequenceMeta(key.split("@")[0], key))

    counts_all = np.vstack(counts_list).astype(np.float32, copy=False)
    return counts_all, tokens_all, meta_all


# --------------------------------------------------------------------------- #
# GPU helper                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class GPUPartition:
    counts  : torch.Tensor  # (n_seq, n_cluster) float32 (tokens per cluster)
    lengths : torch.Tensor  # (n_seq,)  int32  (total tokens per seq)
    used    : torch.Tensor  # (n_seq,)  bool
    meta    : list[SequenceMeta]
    indices : np.ndarray   # global indices

def split_across_gpus(counts_all: np.ndarray, meta_all: list[SequenceMeta],
                      ngpu: int) -> list[GPUPartition]:
    parts: list[GPUPartition] = []
    idxs = np.array_split(np.arange(len(counts_all)), ngpu)
    for dev_id, idx in enumerate(idxs):
        device = f'cuda:{dev_id}'
        part_counts_f32 = torch.from_numpy(counts_all[idx]).to(device=device, dtype=torch.float32)
        part_len        = part_counts_f32.sum(dim=1, dtype=torch.int32)
        parts.append(GPUPartition(
            counts  = part_counts_f32,
            lengths = part_len,
            used    = torch.zeros(len(idx), dtype=torch.bool, device=device),
            meta    = [meta_all[i] for i in idx.tolist()],
            indices = idx,
        ))
    return parts


# --------------------------------------------------------------------------- #
# Main greedy loop                                                            #
# --------------------------------------------------------------------------- #
def greedy_gpu(parts: list[GPUPartition], target: np.ndarray,
               k_return: int = 8) -> list[int]:
    """
    Returns the *global* order (global-sequence indices) in which sequences
    should be played out.
    """
    target_t = torch.as_tensor(target, dtype=torch.float32, device='cuda:0')
    n_cluster = target.shape[0]

    # Per-cluster cumulative counts and token totals as float32 for arithmetic
    cum_counts = torch.zeros(n_cluster, dtype=torch.float32, device='cuda:0')
    total_tokens = torch.tensor(0.0, dtype=torch.float32, device='cuda:0')

    order: list[int] = []  # global indices
    n_total = sum(p.counts.shape[0] for p in parts)

    # Helper: move small scalars to CPU
    def _cpu(x): return x.detach().to('cpu', non_blocking=True)

    # For progress bar
    pbar = tqdm(total=n_total, desc='Ordering sequences (GPU)')

    while len(order) < n_total:
        # Desired count vector after adding one *average* sequence
        avg_len = float((_cpu(total_tokens) / len(order)).item() if order else
                        sum(_cpu(p.lengths).sum() for p in parts) / n_total)
        desired = target_t * (total_tokens + avg_len) - cum_counts
        desired = torch.clamp(desired, min=0)
        desired_ratio = (desired / desired.sum()).float() if desired.sum() > 0 else target_t

        best_err   = torch.tensor(float('inf'), device='cuda:0')
        best_dev   = -1
        best_local = -1

        # Parallel: each GPU returns its k best unused candidates
        for dev_id, part in enumerate(parts):
            if (~part.used).sum() == 0:
                continue

            counts_f = part.counts   # already float32
            lengths  = part.lengths

            # Vectorised error for ALL sequences
            numer   = counts_f + cum_counts.to(part.counts.device)
            denom   = (lengths.to(torch.float32) + total_tokens).unsqueeze(1)
            diff    = numer / denom - target_t.to(part.counts.device)
            errs    = (diff ** 2).sum(dim=1)
            errs.masked_fill_(part.used, float('inf'))

            # k smallest
            vals, idxs = torch.topk(errs, k_return, largest=False)
            for v, i in zip(vals, idxs):
                if v < best_err:
                    best_err   = v
                    best_dev   = dev_id
                    best_local = i.item()

        if best_dev == -1:
            raise RuntimeError('No selectable sequence found – logic error')

        # Mark chosen
        chosen_part = parts[best_dev]
        chosen_part.used[best_local] = True

        # Update globals
        seq_cnt   = chosen_part.counts[best_local]
        seq_len   = chosen_part.lengths[best_local].to(dtype=torch.float32)
        cum_counts += seq_cnt.to(cum_counts.device)
        total_tokens += seq_len
        global_idx = parts[best_dev].indices[best_local].item()
        order.append(global_idx)

        pbar.update(1)

    pbar.close()
    return order


# --------------------------------------------------------------------------- #
# Tar writing & manifest                                                      #
# --------------------------------------------------------------------------- #
def write_output(order: list[int], tokens_all: List[bytes], counts_all: np.ndarray,
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
        cnt = counts_all[idx].astype(np.int32).tolist()
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
                    help='GPUs to use (default: all visible)')
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
    print(f"Found {len(shard_paths)} shards; streaming with WebDataset …")

    counts_all, tokens_all, meta_all = stream_shards_to_arrays(
        shard_paths,
        n_cluster=n_cluster,
        workers=args.loader_workers,
        prefetch=args.loader_prefetch,
    )

    # ------------------------------------------------------------------ distribute to GPUs
    torch.cuda.empty_cache()
    parts = split_across_gpus(counts_all, meta_all, args.gpus)
    print('GPU partitions:', [p.counts.shape[0] for p in parts])

    # ------------------------------------------------------------------ greedy order
    order_indices = greedy_gpu(parts, target)

    # ------------------------------------------------------------------ write output
    out_dir = Path(args.output_dir)
    counts_dir = Path(args.counts_dir)
    manifest = write_output(order_indices, tokens_all, counts_all, out_dir, counts_dir, args.shard_size)

    # final stats
    cum = sum(p.counts[p.used].sum(dim=0).to('cpu',dtype=torch.float32)
              for p in parts).numpy()
    tot = float(sum(p.counts[p.used].sum().item() for p in parts))
    write_manifest(manifest, out_dir, target, cum, tot)
    print('Done.')

if __name__ == '__main__':
    main() 