#!/usr/bin/env python3
"""
ordered_tokenize.py
===================

Utility for building **ordered** tokenized web-dataset style shards from
jsonl input.

Differences from the original `tokshuf` workflow:

1.  Documents are processed **in order** – no shuffling at any stage.
2.  For every produced 2048-token sequence we also track *how many tokens
    originate from each cluster* and store this information in a parallel
    object alongside the token sequence.

Inputs
------
The program expects **pairs** of files – a jsonl file containing one
JSON object per line with a "text" field and an accompanying NumPy
binary (`.npy`) containing the integer cluster id for each respective
line.

For instance a shard may look like::

    /datasets/dclm-baseline-1.0/global-shard_01_of_10/local-shard_0_of_10.jsonl
    /datasets/dclm-baseline-1.0/global-shard_01_of_10/local-shard_0_of_10_clusters.npy

The stem must match; the suffix `_clusters.npy` is automatically
resolved.

Outputs
-------
Web-dataset style tar archives named ``shard_XXXXXXXX.tar`` (chunk size
8192 sequences by default).  Every sequence is stored inside the tar as
*two* gzip-compressed JSON files:

* ``<uuid>.tokens.json.gz``  → list of ints (length 2049)
* ``<uuid>.counts.json.gz``  → list of ints (length=num_clusters)

The relative order of pairs inside each shard mirrors the order of
creation, which itself mirrors the order of the input documents.

Example
-------
    python ordered_tokenize.py \
        --input-dir  /datasets/dclm-baseline-1.0 \
        --output-dir /tmp/ordered_shards \
        --tokenizer-name EleutherAI/gpt-neox-20b

Requirements
------------
``pip install numpy tqdm transformers``
"""
import argparse
import gzip
import json
import tarfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

# ---------------------------------------------------------------------------
# External deps
# ---------------------------------------------------------------------------

import numpy as np
from tqdm import tqdm
# Added for simple parallel processing
import torch  # Added for saving counts tensors alongside shards

# Hugging-Face fast tokenizer (Rust backend)
try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError as e:  # nocov
    raise SystemExit(
        "transformers package not found. Install with `pip install transformers`.") from e

# Optional but recommended for .zstd support (kept optional to avoid hard dep)
try:
    import zstandard as zstd  # type: ignore
except ImportError:
    zstd = None  # type: ignore


###############################################################################
# Tokenizer helper
###############################################################################


def build_tokenizer(args: argparse.Namespace):
    """Return (encode_fn, vocab_size) using Hugging-Face fast tokenizer."""

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    # Disable automatic BOS/EOS insertion – we append our own EOT token.
    def encode_fn(text: str):
        return tok.encode(text, add_special_tokens=False)

    return encode_fn, tok.vocab_size


###############################################################################
# I/O helpers
###############################################################################


def iter_lines(path: Path) -> Iterator[str]:
    """Yield decoded lines from a potentialy-compressed jsonl(.zstd|.gz) file."""

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield line
    elif path.suffix == ".zstd":
        if zstd is None:
            raise RuntimeError(
                "File ends with .zstd but the python-zstandard package is not installed."
            )
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with open(path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                import io
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    yield line
    else:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                yield line


###############################################################################
# Core processing
###############################################################################

EOT_TOKEN_ID = 0  # matches Rust code
SEQLEN = 2049
CHUNK_SIZE = 8192  # sequences per tar shard


class SequenceWriter:
    """Accumulate sequences and flush to tar shards."""

    def __init__(self, output_dir: Path, num_clusters: int):
        self.output_dir = output_dir
        self.num_clusters = num_clusters
        self.shard_id = 0
        self.buffer: List[Tuple[List[int], List[int]]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    def add(self, tokens: List[int], counts: List[int]):
        assert len(tokens) == SEQLEN, "sequence length mismatch"
        assert len(counts) == self.num_clusters, "counts vector length mismatch"
        self.buffer.append((tokens, counts))
        if len(self.buffer) >= CHUNK_SIZE:
            self.flush()

    # ---------------------------------------------------------------------
    def flush(self):
        if not self.buffer:
            return
        shard_path = self.output_dir / f"shard_{self.shard_id:08d}.tar"
        # Save counts as a 2-D PyTorch tensor (num_sequences, num_clusters)
        counts_path = self.output_dir / f"shard_{self.shard_id:08d}_counts.pt"
        counts_tensor = torch.tensor([cnt for _, cnt in self.buffer], dtype=torch.int16)
        torch.save(counts_tensor, counts_path)
        # Write tokens only into the tar archive; counts are stored separately
        with tarfile.open(shard_path, "w") as tar:
            for tokens, _ in self.buffer:
                uid = uuid.uuid4().hex
                # tokens
                tok_bytes = gzip.compress(json.dumps(tokens).encode("utf-8"))
                tok_info = tarfile.TarInfo(f"{uid}.tokens.json.gz")
                tok_info.size = len(tok_bytes)
                tar.addfile(tok_info, BytesIO(tok_bytes))
        self.buffer.clear()
        self.shard_id += 1

    # ---------------------------------------------------------------------
    def close(self):
        self.flush()


###############################################################################
# Main driver
###############################################################################


def process_pair(
    jsonl_path: Path,
    clusters_path: Path,
    encode_fn,
    seq_writer: SequenceWriter,
):
    # Load clusters array
    cluster_ids = np.load(clusters_path)
    assert cluster_ids.ndim == 1, "clusters array must be 1-D"

    # Count docs for quick sanity check
    n_docs = sum(1 for _ in iter_lines(jsonl_path))
    assert n_docs == len(cluster_ids), (
        f"Mismatch between number of documents ({n_docs}) and cluster array ({len(cluster_ids)}) in {jsonl_path}"
    )

    # Go back to beginning to actually process
    seq_tokens: List[int] = []
    seq_counts: List[int] = [0] * seq_writer.num_clusters

    # Progress bar for this particular shard
    for idx, line in enumerate(
        tqdm(
            iter_lines(jsonl_path),
            total=n_docs,
            desc=f"Shard {jsonl_path.name}",
            leave=False,
        )
    ):
        # Parse json and extract text
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as ex:
            raise RuntimeError(f"JSON decode error in {jsonl_path} line {idx}") from ex
        text = obj.get("text", "")
        doc_tokens = encode_fn(text) + [EOT_TOKEN_ID]
        cluster = int(cluster_ids[idx])

        # Ensure counts vector large enough (dynamic cluster id growth support)
        if cluster >= seq_writer.num_clusters:
            raise ValueError(
                f"Encountered cluster id {cluster} which exceeds num_clusters {seq_writer.num_clusters}"
            )

        ptr = 0
        while ptr < len(doc_tokens):
            remaining = SEQLEN - len(seq_tokens)
            take = min(remaining, len(doc_tokens) - ptr)
            seq_tokens.extend(doc_tokens[ptr : ptr + take])
            seq_counts[cluster] += take
            ptr += take

            if len(seq_tokens) == SEQLEN:
                seq_writer.add(seq_tokens, seq_counts)
                seq_tokens = []
                seq_counts = [0] * seq_writer.num_clusters

        # Drop leftovers (< SEQLEN) – no padding to keep strict length


def discover_shards(input_dir: Path) -> List[Tuple[Path, Path]]:
    """Return list of (jsonl_path, clusters_path) pairs."""
    pairs = []
    candidate_files = (
        sorted(input_dir.rglob("*processed.jsonl.zstd"))
        + sorted(input_dir.rglob("*.jsonl"))
    )

    for jsonl_path in candidate_files:
        # Determine cluster file location regardless of compression suffixes
        basename = jsonl_path.name
        if "jsonl" not in basename:
            continue

        prefix = basename.split(".jsonl")[0]
        clusters_name = prefix + "_clusters.npy"
        clusters_path = jsonl_path.with_name(clusters_name)
        if not clusters_path.exists():
            raise FileNotFoundError(f"Expected cluster file {clusters_path} for {jsonl_path}")
        pairs.append((jsonl_path, clusters_path))
    if not pairs:
        raise RuntimeError("No jsonl(.zstd) files found in input_dir")
    return pairs


def determine_num_clusters(pairs: List[Tuple[Path, Path]]) -> int:
    max_cluster = -1
    for _, cluster_path in pairs:
        arr = np.load(cluster_path, mmap_mode="r")  # cheap
        if arr.size == 0:
            continue
        max_cluster = max(max_cluster, int(arr.max()))
    if max_cluster < 0:
        raise RuntimeError("Could not determine num_clusters – no clusters found")
    return max_cluster + 1


###############################################################################
# Entry point
###############################################################################


def main():
    parser = argparse.ArgumentParser(description="Ordered tokenizer & sharder with cluster counts")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing jsonl(.zstd) shards")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to place output .tar shards")
    parser.add_argument("--tokenizer-name", type=str, default="EleutherAI/gpt-neox-20b", help="Any Hugging-Face tokenizer id")
    parser.add_argument("--n-chunks", type=int, default=1, help="Total number of chunks in the global split")
    parser.add_argument("--chunk", type=int, default=0, help="Zero-based index of the chunk this process should handle")
    args = parser.parse_args()

    # Build a tokenizer once in the main process (informational)
    encode_fn, vocab_size = build_tokenizer(args)
    print(f"Loaded tokenizer (vocab size {vocab_size})")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    all_pairs = discover_shards(input_dir)
    num_clusters = determine_num_clusters(all_pairs)
    print(f"Detected {num_clusters} clusters across shards")

    num_chunks = max(1, args.n_chunks)
    if args.chunk < 0 or args.chunk >= num_chunks:
        raise ValueError(f"--chunk must be in [0, {num_chunks}) but got {args.chunk}")

    chunk_size = (len(all_pairs) + num_chunks - 1) // num_chunks  # ceil division

    chunk_idx = args.chunk
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, len(all_pairs))
    chunk_pairs = all_pairs[start:end]
    print(f"Processing chunk {chunk_idx}/{num_chunks}: indices [{start}, {end}) of {len(all_pairs)} total shards")

    chunk_out_root = output_dir / f"sub{chunk_idx}"
    chunk_out_root.mkdir(parents=True, exist_ok=True)

    # Single-process: one writer; process pairs sequentially
    writer = SequenceWriter(chunk_out_root, num_clusters)
    for j_path, c_path in tqdm(chunk_pairs, desc=f"Chunk {chunk_idx}: processing", leave=False):
        process_pair(Path(j_path), Path(c_path), encode_fn, writer)
    writer.close()

    print(f"Done. Wrote {writer.shard_id} tar shards in chunk directory sub{chunk_idx}.")


if __name__ == "__main__":
    main() 