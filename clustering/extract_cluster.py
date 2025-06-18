# file: extract_cluster.py
import argparse, json, zstandard as zstd, webdataset as wds, numpy as np
from pathlib import Path, PurePosixPath

def load_index(path="doc_index.tsv"):
    return [line.rstrip().split("\t")[1] for line in open(path)]

def main(args):
    idx = load_index()
    wanted = set(int(x) for x in open(args.row_ids))
    out = open(args.out_jsonl, "w")
    by_shard = {}
    for row_id in wanted:
        shard, ln = idx[row_id].split(":")
        by_shard.setdefault(shard, []).append(int(ln))
    for shard, lines in by_shard.items():
        path = args.shard_root / shard
        targets = set(lines)
        with open(path, "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as zr:
            for i, ln in enumerate(zr):
                if i in targets:
                    out.write(ln.decode("utf-8"))
    out.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-ids", required=True, help="cluster_*.txt")
    ap.add_argument("--shard-root", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True)
    main(ap.parse_args())
