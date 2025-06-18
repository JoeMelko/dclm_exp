#!/usr/bin/env python
"""
Cluster 40 M fp16 embeddings (1024‑D) into 10 k clusters with FAISS GPU K‑means
and write a mapping row_id -> cluster_id (tab‑separated).

  python kmeans40M.py \
         --embeddings embeddings.fp16 \
         --n-clusters 10000 \
         --sample 3000000 \
         --out clusters.tsv
"""
import argparse, time, numpy as np, faiss, torch, tqdm

def train_kmeans(memmap, n_clusters, sample, seed=0):
    np.random.seed(seed)
    N, d = memmap.shape
    idx  = np.random.choice(N, size=sample, replace=False)
    x    = np.asarray(memmap[idx], dtype=np.float32)
    kmeans = faiss.Kmeans(
        d, n_clusters,
        gpu=True,
        niter=25,   # outer Lloyd iterations
        nredo=3,
        verbose=True,
        seed=seed,
        max_points_per_centroid=100_000  # keep RAM low
    )
    kmeans.train(x)
    return kmeans.centroids

def assign_chunks(memmap, centroids, chunk, out_path):
    N, d = memmap.shape
    res  = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatL2(d)
    index_cpu.add(centroids)
    index = faiss.index_cpu_to_all_gpus(index_cpu)   # replicate on all GPUs

    with open(out_path, "w") as fh:
        for base in tqdm.trange(0, N, chunk, desc="assign"):
            x = np.asarray(memmap[base:base+chunk], dtype=np.float32)
            _, labs = index.search(x, 1)            # (chunk, 1)
            for i, c in enumerate(labs.ravel()):
                fh.write(f"{base+i}\t{int(c)}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--n-clusters", type=int, default=10_000)
    ap.add_argument("--sample", type=int, default=3_000_000)
    ap.add_argument("--chunk",  type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    emb = np.memmap(args.embeddings, dtype=np.float16, mode="r")
    emb = emb.reshape(-1, 1_024)   # (N, 1024)

    t0 = time.time()
    print("Training K‑means on sample …")
    cents = train_kmeans(emb, args.n_clusters, args.sample)
    print(f"done in {time.time()-t0:.1f}s")

    print("Assigning all vectors …")
    assign_chunks(emb, cents, args.chunk, args.out)
    print(f"Finished – wrote {args.out}")
