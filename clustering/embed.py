#!/usr/bin/env python
"""
Compute 1 024‑D fp16 embeddings for every JSON line in a Datacomp‑LM shard
directory.  Requires ≈82 GB of free disk for the mem‑map when N≈40 M.

Outputs
-------
embeddings.fp16   # memmap, shape = (N_docs, 1024), dtype=float16
doc_index.tsv     # row_id <TAB> relative_path:<line_no>\n
"""

import argparse, json, os, zstandard as zstd, numpy as np, torch, tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

MAXLEN  = 32_768
EMBDIM  = 1_024
DTYPE   = np.float16
BUFF_SZ = 2_048          # shuffle buffer for batching

def iter_docs(shard_root: Path):
    """Yield (rel_path, line_no, text) for every JSON line in every .zst file."""
    for p in sorted(shard_root.glob("*.jsonl.zst")):
        rel = p.name
        with open(p, "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as zr:
            for i, ln in enumerate(zr):
                try:
                    yield rel, i, json.loads(ln)["text"]
                except Exception:           # malformed JSON → skip
                    continue

def count_docs(shard_root: Path) -> int:
    """One quick pass to count lines; 65 k lines/s on a single core."""
    n = 0
    for p in shard_root.glob("*.jsonl.zst"):
        with open(p, "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as zr:
            for _ in zr:
                n += 1
    return n

def main(args):
    root = Path(args.shard_dir).resolve()
    print(f"Scanning {root} …")
    N = count_docs(root)
    print(f"Found {N:,} documents")

    emb = np.memmap("embeddings.fp16", mode="w+", dtype=DTYPE, shape=(N, EMBDIM))
    idx = open("doc_index.tsv", "w")

    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs=dict(attn_implementation="flash_attention_2",
                          device_map="auto",
                          torch_dtype=torch.bfloat16),
        tokenizer_kwargs={"padding_side": "left"},
    )

    batch_docs, batch_meta = [], []
    row = 0
    for rel, i, txt in tqdm.tqdm(iter_docs(root), total=N):
        batch_docs.append(txt)
        batch_meta.append(f"{rel}:{i}")
        if sum(len(t)//4 for t in batch_docs) >= args.tok_budget or len(batch_docs) >= args.max_batch:
            vecs = model.encode(batch_docs, max_length=MAXLEN,
                                normalize_embeddings=True, show_progress_bar=False)
            emb[row:row+len(vecs)] = vecs.astype(DTYPE)
            for j, meta in enumerate(batch_meta):
                idx.write(f"{row+j}\t{meta}\n")
            row += len(vecs)
            batch_docs, batch_meta = [], []
    if batch_docs:                                     # flush remainder
        vecs = model.encode(batch_docs, max_length=MAXLEN,
                            normalize_embeddings=True)
        emb[row:row+len(vecs)] = vecs.astype(DTYPE)
        for j, meta in enumerate(batch_meta):
            idx.write(f"{row+j}\t{meta}\n")

    emb.flush(); idx.close()
    print("Done – embeddings.fp16 + doc_index.tsv written.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True,
                    help="local-shard directory containing *.jsonl.zst")
    ap.add_argument("--tok-budget", type=int, default=MAXLEN*2,
                    help="approx tokens per encode() call per GPU")
    ap.add_argument("--max-batch", type=int, default=128,
                    help="cap on docs per encode() call")
    main(ap.parse_args())
