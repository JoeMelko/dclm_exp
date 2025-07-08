# file: tools/balanced_resample_wds.py
"""
Create an output WebDataset that contains exactly N_target samples
drawn from an *already tokenised* WebDataset directory.  If
N_target > |D|, samples are duplicated such that every original
sample appears either k or k+1 times (where k = ⌊N_target / |D|⌋).
"""
import argparse, math, random, json, re, webdataset as wds
from pathlib import Path

def count_samples(manifest: Path) -> int:
    """Sum '#samples' column in the manifest produced by make_wds_manifest.py."""
    total = 0
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # Preferred format: JSON lines with a "num_sequences" key.
            # Fallback: legacy CSV line "<path>,<num_samples>".
            try:
                obj = json.loads(line)
                if "num_sequences" not in obj:
                    raise KeyError  # fall back below
                total += int(obj["num_sequences"])
                continue
            except (json.JSONDecodeError, KeyError):
                pass

            # Legacy CSV fallback
            parts = line.rstrip().split(",")
            if len(parts) < 2 or not parts[1].isdigit():
                raise ValueError(f"Malformed manifest line: {line.strip()}")
            total += int(parts[1])
    return total

def balanced_resample(indir: Path, outdir: Path, n_target: int, maxcount=8192, seed=0):
    random.seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = indir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"manifest.jsonl not found in {indir}")

    n_src = count_samples(manifest)
    k, r = divmod(n_target, n_src)          # every sample repeated k times; r of them (≤n_src-1) once more
    extra = set(random.sample(range(n_src), r))  # indices that get the +1 copy

    # ────────────────────────────────────────────────────────────────────
    # Determine shard filename pattern (prefix + zero-padded index width)
    # based on the *input* dataset so that we keep naming consistent.
    # Example: shard_00000000.tar  → prefix="shard_", width=8
    #          cc_en-000123.tar    → prefix="cc_en-", width=6
    # Fallback default: prefix="shard_", width=8 if inference fails.
    # ────────────────────────────────────────────────────────────────────

    tar_files = sorted(indir.glob("*.tar"))
    if not tar_files:
        raise SystemExit(f"No .tar files found in {indir}")

    def infer_prefix_and_width(files):
        for tar in files:
            m = re.match(r"^(.*?)(\d+)\.tar$", tar.name)
            if m:
                return m.group(1), len(m.group(2))
        return "shard_", 8

    prefix, width = infer_prefix_and_width(tar_files)
    shard_pattern = f"{outdir}/{prefix}%0{width}d.tar"

    # Disable automatic per-extension re-encoding to keep the byte streams
    # untouched (especially important for pre-compressed files like .json.gz).
    #
    # Also keep metadata keys such as __key__ exactly as provided, except when
    # making duplicates (see below).

    sink = wds.ShardWriter(shard_pattern, maxcount=maxcount, encoder=False)
    ds   = wds.WebDataset([str(p) for p in tar_files], shardshuffle=False, handler=wds.warn_and_continue)

    dup_counters = {}

    for idx, sample in enumerate(ds):       # idx is running sample index
        reps = k + (1 if idx in extra else 0)
        base_key = sample["__key__"]

        for r in range(reps):
            if r == 0:
                sink.write(sample)
            else:
                # Avoid duplicate filenames within a shard by generating a new key.
                dup_index = dup_counters.get(base_key, 1)
                dup_counters[base_key] = dup_index + 1

                dup_sample = sample.copy()
                dup_sample["__key__"] = f"{base_key}_dup{dup_index}"
                sink.write(dup_sample)
    sink.close()

    # Create a new manifest.jsonl compatible with the format used by
    # open_lm.utils.make_wds_manifest: JSON lines containing keys "shard" and
    # "num_sequences".  Since shards are written sequentially with at most
    # `maxcount` samples, we can infer counts deterministically from
    # `n_target` and `maxcount`.

    num_full, remainder = divmod(n_target, maxcount)
    shard_counts = [maxcount] * num_full
    if remainder:
        shard_counts.append(remainder)

    manifest_out = outdir / "manifest.jsonl"
    with manifest_out.open("w", encoding="utf-8") as mf:
        for i, count in enumerate(shard_counts):
            shard_name = f"{prefix}{i:0{width}d}"
            json.dump({"shard": shard_name, "num_sequences": count}, mf)
            mf.write("\n")

    print(f"{outdir}: wrote {n_target} samples from {n_src} originals (k={k}, r={r})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir",  required=True, help="dir with .tar shards + manifest.jsonl")
    ap.add_argument("--outdir", required=True, help="output dir for re-sampled shards")
    ap.add_argument("--n",      type=int, required=True, help="target # samples")
    ap.add_argument("--maxcount", type=int, default=8192)
    ap.add_argument("--seed",     type=int, default=0)
    balanced_resample(Path(ap.parse_args().indir),
                      Path(ap.parse_args().outdir),
                      ap.parse_args().n,
                      ap.parse_args().maxcount,
                      ap.parse_args().seed)
