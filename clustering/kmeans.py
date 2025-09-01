#!/usr/bin/env python
"""
Cluster large embedding matrices with FAISS spherical K‑means on GPU and write
the mapping row_id -> cluster_id.

Usage:
  python kmeans.py --embeddings PATH --n-clusters K --sample N --chunk M --out OUTPUT

Inputs (embeddings):
  - .npy: NumPy array of shape (N, 1024), dtype float16/float32 (memory-mapped when possible)
  - .npz: First array in the archive is used; must be shape (N, 1024)
  - Raw binary fp16: Row-major float16 file containing (N x 1024) vectors; N is
    inferred from file size. 1024 dimensions are assumed.

Processing pipeline:
  - Embeddings are upcast to float32 and L2-normalised in memory
  - K‑means is trained on a random sample of size --sample using FAISS with
    spherical=True (cosine-distance k‑means)
  - All vectors are assigned on the GPU in chunks of size --chunk

Outputs (out):
  - If OUTPUT ends with .npy: saves a 1D int32 array where labels[i] is the
    cluster id for row i
  - Otherwise (e.g., .tsv): writes lines of the form "row_id\tcluster_id"
  - Also writes a cluster-size histogram PNG next to OUTPUT as OUTPUT_hist.png

Arguments:
  --embeddings PATH     Path to .npy, .npz, or raw fp16 binary (N x 1024)
  --n-clusters K        Number of clusters (default: 10000)
  --sample N            Number of points to sample for training (default: 3,000,000)
  --chunk M             Assignment chunk size for GPU (default: 1,000,000)
  --out OUTPUT          Output path (.npy for label array, or .tsv for text)

Requirements:
  - CUDA-capable GPU and faiss-gpu installed
  - PyTorch (for tensor interop and normalisation), NumPy, tqdm, matplotlib

Examples:
  # .npy → .npy labels
  python kmeans.py \
    --embeddings embeddings/my_embeds.npy \
    --n-clusters 10000 \
    --sample 10000000 \
    --chunk 1000000 \
    --out clusters.npy

  # .npy → .tsv labels
  python kmeans.py \
    --embeddings embeddings/my_embeds.npy \
    --n-clusters 5000 \
    --sample 3000000 \
    --out clusters.tsv

  # .npz input (first array used) → .npy labels
  python kmeans.py \
    --embeddings embeddings/my_embeds.npz \
    --n-clusters 8000 \
    --sample 5000000 \
    --out clusters.npy

  # raw fp16 binary (N x 1024) → .tsv labels
  python kmeans.py \
    --embeddings embeddings/my_embeds.fp16 \
    --n-clusters 10000 \
    --sample 10000000 \
    --out clusters.tsv

Notes:
  - Increase --chunk if GPU memory allows for faster assignment; decrease to
    avoid OOM
  - Ensure --sample ≤ N; larger samples improve quality at the cost of time
  - Script assumes 1024‑D vectors; adjust load_embeddings() if your dimension differs
"""
import argparse, os, time, numpy as np, faiss, torch, tqdm, matplotlib.pyplot as plt
# Enable FAISS <-> PyTorch tensor interoperability (search/index functions accept tensors)
import faiss.contrib.torch_utils  # noqa: F401 – side-effect import

import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility: in-place L2 normalization for embedding matrices stored as
# NumPy arrays (supports fp32 and fp16). Uses torch for speed and thread-parallelism.
# -----------------------------------------------------------------------------

def l2_normalize_inplace(arr: np.ndarray):
    """In-place L2 normalization using PyTorch – supports fp32 and fp16 arrays."""

    t = torch.from_numpy(arr)  # shares memory with NumPy array (CPU tensor)

    if t.dtype == torch.float16:
        t_norm = F.normalize(t.float(), p=2, dim=1, eps=1e-12)
        t.copy_(t_norm.to(dtype=torch.float16))
    elif t.dtype == torch.float32:
        t_norm = F.normalize(t, p=2, dim=1, eps=1e-12)
        t.copy_(t_norm)
    else:
        raise ValueError("Unsupported dtype for normalization: %s" % t.dtype)

def load_embeddings(path):
    """Load embeddings and return a *torch* tensor (fp32, L2-normalised).

    Supported formats:
      • Raw binary / mem-mapped fp16 (preferred for speed).
      • .npy / .npz archives (will be memory-mapped with NumPy then wrapped).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in {".npy", ".npz"}:
        arr = np.asarray(np.load(path))
        # np.load on .npz returns a dict-like object; grab the first array
        if not isinstance(arr, np.ndarray):
            key = list(arr.keys())[0]
            arr = arr[key]

        emb = torch.from_numpy(arr)  # zero-copy wrapper (CPU tensor)
    else:
        # Raw binary fp16 (row-major, 1024 cols). Fast path.
        num_elems = os.path.getsize(path) // 2  # fp16 = 2 bytes
        emb = torch.from_file(path, dtype=torch.float16, size=num_elems)
        emb = emb.reshape(-1, 1_024)

    # Convert to fp32 *once* (new view when possible)
    if emb.dtype != torch.float32:
        emb = emb.float()

    if emb.shape[1] != 1_024:
        raise ValueError(f"Expected embeddings shape (N, 1024); got {tuple(emb.shape)}")

    # L2 normalise in-place
    emb.copy_(F.normalize(emb, p=2, dim=1, eps=1e-12))
    return emb

def train_kmeans(memmap, n_clusters, sample, seed=0):
    np.random.seed(seed)
    N, d = memmap.shape
    idx = np.random.choice(N, size=sample, replace=False)
    print(f"creating random sample of size {sample}")

    if isinstance(memmap, torch.Tensor):
        x = memmap[idx].cpu().numpy()  # already fp32
    else:
        x = np.asarray(memmap[idx], dtype=np.float32)

    print("training spherical kmeans")
    kmeans = faiss.Kmeans(
        d, n_clusters,
        gpu=True,
        niter=25,   # outer Lloyd iterations
        nredo=3,
        verbose=True,
        seed=seed,
        spherical=True,                   # cosine distance
        max_points_per_centroid=100_000,  # keep RAM low
    )
    kmeans.train(x)
    return kmeans.centroids

def assign_chunks(memmap, centroids, chunk, out_path, plot_path=None):
    """Assign every vector to its nearest centroid using GPUs.

    Output format is inferred from the extension of ``out_path``:
      • ``.npy`` → writes a 1D NumPy array of dtype int32 where index == row_id
      • other (e.g. ``.tsv``) → writes tab-separated ``row_id\tcluster_id`` lines

    Each chunk is converted to a CUDA tensor (float32) which avoids the
    single-threaded NumPy → GPU copy bottleneck and leverages FAISS' tensor
    interface enabled by importing faiss.contrib.torch_utils.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for torch-accelerated assignment.")

    N, d = memmap.shape

    # Build the index on GPU 0 (works with torch tensor inputs)
    index_cpu = faiss.IndexFlatL2(d)
    index_cpu.add(centroids)

    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    counts = np.zeros(centroids.shape[0], dtype=np.int64)

    write_npy = out_path.lower().endswith(".npy")
    labels_arr = np.empty(N, dtype=np.int32) if write_npy else None

    fh = None
    try:
        if not write_npy:
            fh = open(out_path, "w")

        for base in tqdm.trange(0, N, chunk, desc="assign"):
            # Slice fp16 view, then move to GPU and cast to fp32 in one call
            if isinstance(memmap, torch.Tensor):
                x_chunk = memmap[base:base + chunk].to(
                    device="cuda", dtype=torch.float32, non_blocking=True
                )
            else:
                x_chunk = torch.from_numpy(memmap[base:base + chunk]).to(
                    device="cuda", dtype=torch.float32, non_blocking=True
                )

            # Search – FAISS returns torch tensors because input is a tensor
            _, labs = index.search(x_chunk, 1)  # (chunk, 1) on GPU

            labs_np = labs.cpu().numpy().ravel()
            # Bring labels back to CPU for writing & counting
            counts += np.bincount(labs_np, minlength=counts.size)

            if write_npy:
                labels_arr[base:base + labs_np.size] = labs_np
            else:
                for i, c in enumerate(labs_np):
                    fh.write(f"{base + i}\t{c}\n")
    finally:
        if fh is not None:
            fh.close()

    if write_npy:
        np.save(out_path, labels_arr)

    # Plot distribution if requested
    if plot_path is not None:
        plt.figure(figsize=(8,6))
        plt.hist(counts, bins=50, color="steelblue", edgecolor="black")
        plt.xlabel("Points per cluster")
        plt.ylabel("Number of clusters")
        plt.title("Cluster size distribution")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Saved cluster-size histogram to {plot_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--n-clusters", type=int, default=10_000)
    ap.add_argument("--sample", type=int, default=3_000_000)
    ap.add_argument("--chunk",  type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load the entire embedding matrix into RAM. Supports both raw fp16 binaries and .npy/.npz files.
    print("Loading embeddings …")
    emb = load_embeddings(args.embeddings)   # (N, 1024) fp32 in RAM
    print(f"Loaded embeddings with shape {emb.shape} and dtype {emb.dtype}")

    t0 = time.time()
    print("Training K‑means on sample …")
    cents = train_kmeans(emb, args.n_clusters, args.sample)
    print(f"done in {time.time()-t0:.1f}s")

    print("Assigning all vectors …")
    # Determine plot output path automatically
    plot_path = os.path.splitext(args.out)[0] + "_hist.png"
    assign_chunks(emb, cents, args.chunk, args.out, plot_path=plot_path)
    print(f"Finished – wrote {args.out}")
