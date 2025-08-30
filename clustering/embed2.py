#!/usr/bin/env python
"""
embed_datacomp_qwen3.py
-----------------------
Stream Datacomp‑LM (baseline global‑1 / local‑1, 279 *.jsonl.zstd shards),
embed the `text` field with Qwen/Qwen3‑Embedding‑0.6B, and store
either a single `.npy` matrix (default) or a raw fp16 binary (`--raw_out`),
whose rows exactly follow source order.

Launch on one 8‑GPU node (example):

  torchrun --standalone --nproc_per_node 8 \
      clustering/embed2.py \
      --data_dir   /mnt/one/home/jmelko/dclm_exp/data/gs01_ls1 \
      --out_dir    /mnt/one/home/jmelko/dclm_exp/data/gs01_ls1/embeddings \
      --batch_size 64 \
      --max_len    1024 \
      --workers    4 \
      --fp16             # optional: run model in fp16 (default bf16) \
      --raw_out          # optional: write raw fp16 .fp16 instead of .npy

Flags
-----
--data_dir   PATH   Directory of input shards (expects shard_*_processed.jsonl.zstd)
--out_dir    PATH   Output directory (created if missing)
--batch_size INT    Per‑GPU batch size (default: 64)
--max_len    INT    Max tokens for truncation (default: 1024)
--workers    INT    DataLoader workers per process (default: 4)
--fp16              Use fp16 compute (default: bf16)
--raw_out           Save final embeddings as raw fp16 binary (.fp16). By default writes .npy
"""

from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import zstandard as zstd
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import builtins

# --------------------------------------------------------------------------- #
#                               DATASET                                       #
# --------------------------------------------------------------------------- #
class ShardedJsonl(IterableDataset):
    """Stream *.jsonl.zstd shards, yield (global_idx, text) in a
    distributed‑ and dataloader‑worker‑safe way."""

    def __init__(self, shards: list[str], rank: int, world: int):
        self.shards = shards
        self.rank = rank
        self.world = world

    def __iter__(self) -> Iterator[Tuple[int, str]]:
        w_info      = get_worker_info()
        worker_id   = w_info.id if w_info else 0
        workers_per = w_info.num_workers if w_info else 1

        total_replicas = self.world * workers_per
        my_id          = self.rank * workers_per + worker_id

        dctx = zstd.ZstdDecompressor()
        idx  = 0
        for shard in self.shards:
            # Update "shard" progress bar (only once per physical shard).
            pbar = getattr(builtins, "SHARD_PBAR", None)
            if pbar is not None and my_id == 0:
                pbar.update(1)
            # open() from the high‑level zstd API transparently streams decompression
            with zstd.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if idx % total_replicas == my_id:
                        try:
                            text = json.loads(line)["text"]
                            yield idx, text
                        except Exception:
                            pass  # malformed line → skip but keep idx consistent
                    idx += 1

# --------------------------------------------------------------------------- #
#                             DISTRIBUTED SETUP                               #
# --------------------------------------------------------------------------- #
def ddp_init() -> tuple[int, int]:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world

# --------------------------------------------------------------------------- #
#                            COLLATE & EMBED                                  #
# --------------------------------------------------------------------------- #
def collate(batch, tok, max_len):
    idxs, texts = zip(*batch)
    toks = tok(list(texts),
               padding=True,
               truncation=True,
               max_length=max_len,
               return_tensors="pt")
    return torch.tensor(idxs, dtype=torch.int64), toks

def embed_loop(loader: DataLoader,
               model: AutoModel,
               fp16: bool = False,
               row_pbar: tqdm | None = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    dtype = torch.float16 if fp16 else torch.bfloat16
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for idx, toks in loader:
            idx = idx.cpu().numpy()
            toks = {k: v.to(model.device, non_blocking=True) for k, v in toks.items()}
            out  = model(**toks, return_dict=True)
            emb  = out.last_hidden_state.mean(dim=1)          # (B, hidden)
            # Update overall row progress bar if present.
            if row_pbar is not None:
                row_pbar.update(len(idx))
            yield idx, emb.cpu().to(torch.float16).numpy()    # save space

# --------------------------------------------------------------------------- #
#                               MAIN                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len",    type=int, default=1024)
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--fp16",       action="store_true")
    ap.add_argument("--raw_out",    action="store_true",
                    help="Write final embeddings as raw fp16 binary (row-major) for torch.from_file.")
    args = ap.parse_args()

    rank, world = ddp_init()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # --------------------- data (shard list) ------------------------------- #
    shards = sorted(glob.glob(os.path.join(args.data_dir,
                                           "shard_*_processed.jsonl.zstd")))

    # --------------------- progress bars (rank-0 only) --------------------- #
    if rank == 0:
        shard_pbar = tqdm(total=len(shards), desc="Shards", position=0, leave=True)
        row_pbar   = tqdm(total=0, desc="Rows", unit="rows", position=1, leave=True)
    else:
        shard_pbar = None
        row_pbar   = None

    # Make shard progress bar globally visible so that Dataset workers can update it.
    builtins.SHARD_PBAR = shard_pbar

    # --------------------- model ------------------------------------------- #
    model_name = "Qwen/Qwen3-Embedding-0.6B"          # <- verified model id
    tokenizer  = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if not args.fp16 else torch.float16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    # Place the entire model on the GPU that corresponds to this processʼ rank.
    model.to(f"cuda:{rank}").eval()

    # --------------------- data loader ------------------------------------- #
    ds     = ShardedJsonl(shards, rank, world)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate(b, tokenizer, args.max_len),
        persistent_workers=args.workers > 0,
    )

    # --------------------- embed & dump ------------------------------------ #
    tmp_path = Path(args.out_dir) / f"rank{rank:02d}.npz"
    idx_buf, emb_buf = [], []
    for idx, emb in embed_loop(loader, model, args.fp16, row_pbar=row_pbar):
        idx_buf.append(idx)
        emb_buf.append(emb)

    # Ensure we always create a file so rank-0 can merge without missing paths.
    if idx_buf:
        np.savez_compressed(tmp_path,
                            idx=np.concatenate(idx_buf),
                            emb=np.concatenate(emb_buf))
    else:
        hdim = model.config.hidden_size
        np.savez_compressed(tmp_path,
                            idx=np.empty(0, dtype=np.int64),
                            emb=np.empty((0, hdim), dtype=np.float16))

    # Close progress bars cleanly (rank-0).
    if rank == 0:
        shard_pbar.close(); row_pbar.close()

    dist.barrier()

    # --------------------- merge on rank-0 --------------------------------- #
    if rank == 0:
        tmp_files = [Path(args.out_dir) / f"rank{r:02d}.npz" for r in range(world)]
        idx, emb = [], []
        for f in tmp_files:
            d = np.load(f)
            idx.append(d["idx"]); emb.append(d["emb"])
        idx = np.concatenate(idx)
        emb = np.concatenate(emb)

        # Guard against accidental duplicate indices (should not happen).
        uniq_idx, first_pos = np.unique(idx, return_index=True)
        if len(uniq_idx) != len(idx):
            dup_count = len(idx) - len(uniq_idx)
            print(f"[rank-0] ⚠️  Found {dup_count} duplicate row indices; keeping first occurrence.")
            emb = emb[first_pos]
            idx = uniq_idx

        order = idx.argsort(kind="mergesort")
        sorted_emb = emb[order]

        if args.raw_out:
            # Write raw fp16 binary, row-major. Compatible with torch.from_file in kmeans.py
            final_raw = Path(args.out_dir) / "datacomp_glob1_local1_qwen3_0.6B.fp16"
            sorted_emb.astype(np.float16, copy=False).tofile(final_raw)
            print(f"[rank-0] ✅ All done. Final raw fp16 matrix -> {final_raw}")
        else:
            final_npy = Path(args.out_dir) / "datacomp_glob1_local1_qwen3_0.6B.npy"
            np.save(final_npy, sorted_emb)
            print(f"[rank-0] ✅ All done. Final matrix -> {final_npy}")

        # Clean up per-rank temporary files.
        for f in tmp_files:
            try:
                f.unlink()  # delete file
            except FileNotFoundError:
                pass

    # Finalize distributed backend. A second barrier may deadlock if rank-0
    # spends a long time merging / writing the final matrix, so we simply
    # destroy the process group instead of synchronizing once more.
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
