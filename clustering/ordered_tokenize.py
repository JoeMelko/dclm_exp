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

* ``<uuid>.tokens.json.gz``  → list of ints (length 2048)
* ``<uuid>.counts.json.gz``  → list of ints (length=num_clusters)

The relative order of pairs inside each shard mirrors the order of
creation, which itself mirrors the order of the input documents.

Example
-------
    python ordered_tokenize.py \
        --input-dir /datasets/dclm-baseline-1.0 \
        --output-dir /tmp/ordered_shards \
        --tokenizer-eleuther-path EleutherAI_gpt-neox-20b.tiktoken

Requirements
------------
``pip install numpy tqdm tiktoken``
"""
import argparse
import gzip
import json
import tarfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import numpy as np
from tqdm import tqdm

try:
    # tiktoken is preferred – identical to the Rust implementation
    import tiktoken  # type: ignore
except ImportError as e:
    raise SystemExit("tiktoken package not found. Install with `pip install tiktoken`." ) from e

# Optional but recommended for .zstd support
try:
    import zstandard as zstd  # type: ignore
except ImportError:
    zstd = None  # type: ignore


###############################################################################
# Tokenizer helpers
###############################################################################


def load_tiktoken_from_file(tiktoken_path: Path, regex_pattern: str):
    """Load a CoreBPE from a raw *.tiktoken* file.

    The format matches that used by the Rust implementation.
    """

    from tiktoken.core_bpe import CoreBPE  # pyright: ignore

    encoder = {}
    with open(tiktoken_path, "r", encoding="utf-8") as fh:
        for line in fh:
            raw, rank = line.strip().split(" ")
            # we cannot rely on base64 from tiktoken internal. Use python stdlib
            import base64

            token_bytes = base64.b64decode(raw)
            encoder[token_bytes] = int(rank)

    return CoreBPE(encoder, {}, regex_pattern)


def build_tokenizer(args: argparse.Namespace):
    """Return (encode_fn, vocab_size) pair."""

    if args.tokenizer_eleuther_path:
        # Build CoreBPE from provided *.tiktoken* file
        enc_name = args.tokenizer_name
        if enc_name == "EleutherAI/gpt-neox-20b":
            regex_pattern = (
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            )
            core_bpe = load_tiktoken_from_file(Path(args.tokenizer_eleuther_path), regex_pattern)
            return core_bpe.encode, 50254  # GPT-NeoX vocab size
        else:
            raise ValueError(
                "Custom *.tiktoken* loading currently supported only for EleutherAI/gpt-neox-20b"
            )
    else:
        # Fallback – attempt to load via tiktoken's built-in encodings
        try:
            enc: tiktoken.Encoding = tiktoken.encoding_for_model("gpt-neox")
        except KeyError:
            enc = tiktoken.get_encoding("gpt2")
        return enc.encode, enc.n_vocab


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
SEQLEN = 2048
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
        with tarfile.open(shard_path, "w") as tar:
            for tokens, counts in self.buffer:
                uid = uuid.uuid4().hex
                # tokens
                tok_bytes = gzip.compress(json.dumps(tokens).encode("utf-8"))
                tok_info = tarfile.TarInfo(f"{uid}.tokens.json.gz")
                tok_info.size = len(tok_bytes)
                tar.addfile(tok_info, BytesIO(tok_bytes))
                # counts
                cnt_bytes = gzip.compress(json.dumps(counts).encode("utf-8"))
                cnt_info = tarfile.TarInfo(f"{uid}.counts.json.gz")
                cnt_info.size = len(cnt_bytes)
                tar.addfile(cnt_info, BytesIO(cnt_bytes))
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

    for idx, line in enumerate(iter_lines(jsonl_path)):
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
    parser.add_argument("--tokenizer-name", type=str, default="EleutherAI/gpt-neox-20b", help="Tokenizer id (informational)")
    parser.add_argument("--tokenizer-eleuther-path", type=str, default="", help="Optional path to *.tiktoken file for EleutherAI/gpt-neox-20b")
    args = parser.parse_args()

    encode_fn, vocab_size = build_tokenizer(args)
    print(f"Loaded tokenizer (vocab size {vocab_size})")

    input_dir = Path(args.input_dir)
    pairs = discover_shards(input_dir)
    num_clusters = determine_num_clusters(pairs)
    print(f"Detected {num_clusters} clusters across shards")

    writer = SequenceWriter(Path(args.output_dir), num_clusters)

    for jsonl_path, clusters_path in tqdm(pairs, desc="Processing shards"):
        process_pair(jsonl_path, clusters_path, encode_fn, writer)

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main() 