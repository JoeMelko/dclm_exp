# file: tools/balanced_resample_wds.py
"""
Create an output WebDataset that contains exactly N_target samples
drawn from an *already tokenised* WebDataset directory.  If
N_target > |D|, samples are duplicated such that every original
sample appears either k or k+1 times (where k = ⌊N_target / |D|⌋).
"""
import argparse, math, random, webdataset as wds
from pathlib import Path

def count_samples(manifest: Path) -> int:
    """Sum '#samples' column in the manifest produced by make_wds_manifest.py."""
    total = 0
    with manifest.open() as f:
        for line in f:
            # each line:  /abs/path/shard-000123.tar,<num_samples>\n
            total += int(line.rstrip().split(",")[1])
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

    sink = wds.ShardWriter(f"{outdir}/resamp-%06d.tar", maxcount=maxcount)
    ds   = wds.WebDataset(str(indir / "*.tar"), shardshuffle=False, handler=wds.warn_and_continue)

    for idx, sample in enumerate(ds):       # idx is running sample index
        reps = k + (1 if idx in extra else 0)
        for _ in range(reps):
            sink.write(sample)
    sink.close()
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
