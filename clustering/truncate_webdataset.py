#!/usr/bin/env python3
"""
truncate_webdataset.py
======================

Truncate an existing WebDataset of token sequences (and optionally counts)
to a multiple of a provided modulus and write a new dataset with a fresh
manifest.

Default behavior is storage-efficient: it creates a truncated dataset by
symlinking the selected full shards from the source and writing a new
manifest. You can force materialization (copying) with --copy.

Rules (symlink and copy modes):
- If the final shard is partial, drop it first.
- Ensure the remaining number of full shards is a multiple of
  (--truncate-mod / shard-size); trim extra shards from the end.
- --truncate-mod must be a multiple of the shard size.
- Shard size is inferred from the source manifest when present; otherwise
  it is estimated from shard counts.

Single dataset (symlink mode, default):
  python truncate_webdataset.py \
    --input-tokens /path/to/dataset/tokens \
    --input-counts /path/to/dataset/counts \
    --out-dir /path/to/dataset_trunc_128 \
    --truncate-mod 128

Single dataset (force copying/materialization):
  python truncate_webdataset.py \
    --input-tokens /path/to/dataset/tokens \
    --input-counts /path/to/dataset/counts \
    --out-dir /path/to/dataset_trunc_128 \
    --truncate-mod 128 \
    --copy

Single dataset (concatenate after truncation into one shard):
  python truncate_webdataset.py \
    --input-tokens /path/to/dataset/tokens \
    --input-counts /path/to/dataset/counts \
    --out-dir /path/to/dataset_trunc_128_concat \
    --truncate-mod 128 \
    --concat-single

Batch mode over multiple roots (writes to <top>_trunc_<mod>):
  python truncate_webdataset.py \
    --top-paths /data/runA /data/runB \
    --truncate-mod 128

Batch mode + concatenate (each root becomes a single-shard dataset):
  python truncate_webdataset.py \
    --top-paths /data/runA /data/runB \
    --truncate-mod 128 \
    --concat-single
"""

from __future__ import annotations
import argparse, gzip, json, tarfile, uuid, os, shutil
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
import webdataset as wds


def read_total_sequences_from_manifest(tokens_dir: Path) -> Optional[int]:
    manifest_path = tokens_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return None
    total = 0
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                total += int(obj.get("num_sequences", 0))
            except Exception:
                continue
    return total


def detect_default_shard_size(tokens_dir: Path, fallback: int = 64) -> int:
    manifest_path = tokens_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return fallback
    counts: Dict[int, int] = {}
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                n = int(json.loads(line).get("num_sequences", 0))
                if n > 0:
                    counts[n] = counts.get(n, 0) + 1
            except Exception:
                pass
    if not counts:
        return fallback
    # return the most frequent shard size observed
    return sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]


def read_manifest_entries(tokens_dir: Path) -> Optional[List[Dict]]:
    manifest_path = tokens_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return None
    entries: List[Dict] = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and 'shard' in obj and 'num_sequences' in obj:
                    entries.append({'shard': str(obj['shard']), 'num_sequences': int(obj['num_sequences'])})
            except Exception:
                continue
    return entries


def write_manifest(manifest: List[Dict], out_tokens_dir: Path,
                   cum_counts: Optional[np.ndarray], tot_tokens: Optional[int]):
    fout = out_tokens_dir / 'manifest.jsonl'
    with open(fout, 'w') as f:
        for m in manifest:
            f.write(json.dumps(m) + '\n')
    summary = {
        'num_shards': len(manifest),
        'num_sequences': int(sum(m['num_sequences'] for m in manifest)),
    }
    if cum_counts is not None and tot_tokens is not None and tot_tokens > 0:
        final_ratio = (cum_counts / float(tot_tokens)).tolist()
        summary.update({
            'num_tokens': int(tot_tokens),
            'actual_cluster_ratios': final_ratio,
        })
    print("Manifest summary:", json.dumps(summary))
    print(f"Wrote manifest to {fout}")


def _select_shards_for_truncation(
    in_tokens_dir: Path,
    trunc_mod: int,
    shard_size_opt: int,
) -> tuple[List[Path], int, Dict[str, int]]:
    token_shards = sorted(in_tokens_dir.glob('shard_*.tar'))
    if len(token_shards) == 0:
        raise FileNotFoundError(f"No token shards found in {in_tokens_dir}")

    # Determine shard_size
    shard_size = int(shard_size_opt or 0)
    if shard_size <= 0:
        shard_size = detect_default_shard_size(in_tokens_dir, fallback=64)

    # Read manifest entries if present to get per-shard counts
    entries = read_manifest_entries(in_tokens_dir)
    stem_to_count: Dict[str, int] = {}
    if entries is not None:
        for e in entries:
            stem_to_count[str(e['shard'])] = int(e['num_sequences'])

    # Determine if last shard is partial
    def count_entries_in_tar(path: Path) -> int:
        try:
            with tarfile.open(path, 'r') as tar:
                return sum(1 for m in tar if m.isfile())
        except Exception:
            return shard_size  # fall back

    last_stem = token_shards[-1].stem
    last_count = stem_to_count.get(last_stem)
    if last_count is None:
        last_count = count_entries_in_tar(token_shards[-1])

    full_shards = list(token_shards)
    if trunc_mod > 0 and last_count < shard_size:
        full_shards = token_shards[:-1]

    if trunc_mod > 0:
        if trunc_mod % shard_size != 0:
            raise ValueError(f"truncate-mod ({trunc_mod}) must be a multiple of shard-size ({shard_size}) for symlink/concat mode")
        block_size = trunc_mod // shard_size
        num_full = len(full_shards)
        desired_shards = (num_full // block_size) * block_size
    else:
        desired_shards = len(token_shards)

    selected_shards = full_shards[:desired_shards]
    return selected_shards, shard_size, stem_to_count


def process_one_dataset(
    in_tokens_dir: Path,
    in_counts_dir: Optional[Path],
    out_base: Path,
    trunc_mod: int,
    shard_size_opt: int,
    loader_workers: int,
    loader_prefetch: int,
) -> None:
    out_tokens_dir = out_base / 'tokens'
    out_counts_dir = out_base / 'counts'
    out_tokens_dir.mkdir(parents=True, exist_ok=True)
    if in_counts_dir is not None:
        out_counts_dir.mkdir(parents=True, exist_ok=True)

    # Determine total sequences and target truncation size
    total_seq = read_total_sequences_from_manifest(in_tokens_dir)
    if total_seq is None:
        # Fallback: scan quickly using WebDataset
        shard_paths = sorted(in_tokens_dir.glob('shard_*.tar'))
        if len(shard_paths) == 0:
            raise FileNotFoundError(f"No token shards found in {in_tokens_dir}")
        ds_count = (
            wds.WebDataset([str(p) for p in shard_paths], shardshuffle=False, handler=wds.warn_and_continue)
            .to_tuple('json.gz')
            .with_length(None)
        )
        total_seq = sum(1 for _ in ds_count)

    desired_total = int(total_seq)
    if trunc_mod > 0:
        desired_total = desired_total - (desired_total % trunc_mod)

    # Determine output shard size
    shard_size = int(shard_size_opt or 0)
    if shard_size <= 0:
        shard_size = detect_default_shard_size(in_tokens_dir, fallback=64)

    print(f"Input: {in_tokens_dir.parent} | sequences: {total_seq}; truncate_mod={trunc_mod}; writing {desired_total}; shard_size={shard_size}")

    token_shards = sorted(in_tokens_dir.glob('shard_*.tar'))
    if len(token_shards) == 0:
        raise FileNotFoundError(f"No token shards found in {in_tokens_dir}")

    # Build datasets for streaming
    ds_tokens = (
        wds.WebDataset([str(p) for p in token_shards], shardshuffle=False, handler=wds.warn_and_continue)
        .to_tuple('json.gz')
        .with_length(None)
    )

    if in_counts_dir is not None:
        count_shards = sorted(in_counts_dir.glob('shard_*.tar'))
        if len(count_shards) == 0:
            raise FileNotFoundError(f"No counts shards found in {in_counts_dir}")
        ds_counts = (
            wds.WebDataset([str(p) for p in count_shards], shardshuffle=False, handler=wds.warn_and_continue)
            .to_tuple('counts.json.gz')
            .with_length(None)
        )
        counts_iter = iter(ds_counts)
        counts_writer = wds.ShardWriter(str(out_counts_dir / 'shard_%08d.tar'), maxcount=shard_size, encoder=False)
    else:
        counts_iter = None
        counts_writer = None

    manifest: List[Dict] = []
    shard_id = 0
    buf_tokens: List[bytes] = []

    cum_counts: Optional[np.ndarray] = None
    tot_tokens: Optional[int] = None

    def flush_tokens():
        nonlocal shard_id, buf_tokens
        if not buf_tokens:
            return
        shard_path = out_tokens_dir / f'shard_{shard_id:08d}.tar'
        with tarfile.open(shard_path, 'w') as tar:
            for tok in buf_tokens:
                uid = uuid.uuid4().hex
                tb = tok
                ti = tarfile.TarInfo(f'{uid}.json.gz'); ti.size = len(tb)
                tar.addfile(ti, BytesIO(tb))
        manifest.append({'shard': shard_path.stem, 'num_sequences': len(buf_tokens)})
        shard_id += 1
        buf_tokens = []

    written = 0
    token_iter = iter(ds_tokens)

    pbar = tqdm(total=desired_total, desc=f'Truncating WebDataset -> {out_base}', unit='seq')

    while written < desired_total:
        try:
            (tok_bytes,) = next(token_iter)
        except StopIteration:
            break

        buf_tokens.append(tok_bytes)

        if counts_iter is not None and counts_writer is not None:
            try:
                (cnt_gz_bytes,) = next(counts_iter)
            except StopIteration:
                raise RuntimeError("Counts dataset ended before tokens dataset; datasets are misaligned")
            # write counts record as-is (gz bytes), with a fresh uuid key
            counts_writer.write({"__key__": uuid.uuid4().hex, "counts.json.gz": cnt_gz_bytes})

            # accumulate summary stats if possible
            try:
                cnt = json.loads(gzip.decompress(cnt_gz_bytes).decode('utf-8'))
                arr = np.asarray(cnt, dtype=np.int64)
                if cum_counts is None:
                    cum_counts = arr.astype(np.int64)
                else:
                    if cum_counts.shape != arr.shape:
                        raise ValueError("Counts dimensionality mismatch across records; cannot summarize")
                    cum_counts += arr
                tot_tokens = int((tot_tokens or 0) + int(arr.sum()))
            except Exception:
                pass

        if len(buf_tokens) == shard_size:
            flush_tokens()

        written += 1
        pbar.update(1)

    pbar.close()

    # finalize
    flush_tokens()
    if counts_writer is not None:
        counts_writer.close()

    # write manifest + summary
    write_manifest(manifest, out_tokens_dir, cum_counts, tot_tokens)
    print(f'Done writing to {out_base}')


def process_one_dataset_symlink(
    in_tokens_dir: Path,
    in_counts_dir: Optional[Path],
    out_base: Path,
    trunc_mod: int,
    shard_size_opt: int,
) -> None:
    out_tokens_dir = out_base / 'tokens'
    out_counts_dir = out_base / 'counts'
    out_tokens_dir.mkdir(parents=True, exist_ok=True)
    if in_counts_dir is not None:
        out_counts_dir.mkdir(parents=True, exist_ok=True)

    selected_shards, shard_size, stem_to_count = _select_shards_for_truncation(
        in_tokens_dir=in_tokens_dir,
        trunc_mod=trunc_mod,
        shard_size_opt=shard_size_opt,
    )

    # Create symlinks and manifest
    manifest: List[Dict] = []
    for src_path in selected_shards:
        dst_path = out_tokens_dir / src_path.name
        if dst_path.exists() or dst_path.is_symlink():
            try:
                dst_path.unlink()
            except FileNotFoundError:
                pass
        os.symlink(src_path.resolve(), dst_path)
        nseq = stem_to_count.get(src_path.stem, shard_size)
        manifest.append({'shard': src_path.stem, 'num_sequences': int(nseq)})

    if in_counts_dir is not None:
        for src_path in selected_shards:
            counts_src = in_counts_dir / src_path.name
            if not counts_src.exists():
                raise FileNotFoundError(f"Counts shard missing for {src_path.name}: {counts_src}")
            counts_dst = out_counts_dir / counts_src.name
            if counts_dst.exists() or counts_dst.is_symlink():
                try:
                    counts_dst.unlink()
                except FileNotFoundError:
                    pass
            os.symlink(counts_src.resolve(), counts_dst)

    write_manifest(manifest, out_tokens_dir, cum_counts=None, tot_tokens=None)
    print(f"Symlinked {len(selected_shards)} shards to {out_base}")


def process_one_dataset_concat_single(
    in_tokens_dir: Path,
    in_counts_dir: Optional[Path],
    out_base: Path,
    trunc_mod: int,
    shard_size_opt: int,
) -> None:
    out_tokens_dir = out_base / 'tokens'
    out_counts_dir = out_base / 'counts'
    out_tokens_dir.mkdir(parents=True, exist_ok=True)
    if in_counts_dir is not None:
        out_counts_dir.mkdir(parents=True, exist_ok=True)

    selected_shards, shard_size, stem_to_count = _select_shards_for_truncation(
        in_tokens_dir=in_tokens_dir,
        trunc_mod=trunc_mod,
        shard_size_opt=shard_size_opt,
    )

    # Tokens: concatenate all entries into a single tar
    dst_tokens = out_tokens_dir / 'shard_00000000.tar'
    total_seq = 0
    with tarfile.open(dst_tokens, 'w') as out_tar:
        for src_path in selected_shards:
            with tarfile.open(src_path, 'r') as in_tar:
                for member in in_tar:
                    if not member.isfile():
                        continue
                    f = in_tar.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                    # preserve suffix after first dot; default to .json.gz if absent
                    name = member.name
                    if '.' in name:
                        suffix = name[name.find('.'):]
                    else:
                        suffix = '.json.gz'
                    uid = uuid.uuid4().hex
                    ti = tarfile.TarInfo(f'{uid}{suffix}')
                    ti.size = len(data)
                    out_tar.addfile(ti, BytesIO(data))
                    total_seq += 1

    # Counts: concatenate if provided
    if in_counts_dir is not None:
        dst_counts = out_counts_dir / 'shard_00000000.tar'
        with tarfile.open(dst_counts, 'w') as out_ctar:
            for src_path in selected_shards:
                counts_src = in_counts_dir / src_path.name
                if not counts_src.exists():
                    raise FileNotFoundError(f"Counts shard missing for {src_path.name}: {counts_src}")
                with tarfile.open(counts_src, 'r') as in_ctar:
                    for member in in_ctar:
                        if not member.isfile():
                            continue
                        f = in_ctar.extractfile(member)
                        if f is None:
                            continue
                        data = f.read()
                        name = member.name
                        if '.' in name:
                            suffix = name[name.find('.'):]
                        else:
                            suffix = '.counts.json.gz'
                        uid = uuid.uuid4().hex
                        ti = tarfile.TarInfo(f'{uid}{suffix}')
                        ti.size = len(data)
                        out_ctar.addfile(ti, BytesIO(data))

    # Write manifest with single shard entry
    manifest = [{'shard': 'shard_00000000', 'num_sequences': int(total_seq)}]
    write_manifest(manifest, out_tokens_dir, cum_counts=None, tot_tokens=None)
    print(f"Concatenated {len(selected_shards)} shards -> {dst_tokens} with {total_seq} sequences")


def main():
    ap = argparse.ArgumentParser()
    # Single-dataset mode
    ap.add_argument('--input-tokens', required=False,
                    help='Directory containing input token shards and manifest.jsonl')
    ap.add_argument('--input-counts', required=False,
                    help='Optional directory containing input counts shards')
    ap.add_argument('--out-dir', required=False,
                    help='Output base directory; will create tokens/ and counts/ subdirs')
    # Batch mode
    ap.add_argument('--top-paths', nargs='+', default=None,
                    help='One or more dataset roots containing tokens/ (and optionally counts/) to truncate')
    ap.add_argument('--concat-datasets', action='store_true',
                    help='With --top-paths, combine truncated token shards from all roots into one output dir (tokens only)')
    # Common
    ap.add_argument('--truncate-mod', type=int, default=0,
                    help='If > 0, truncate total sequences to a multiple of this value')
    ap.add_argument('--shard-size', type=int, default=0,
                    help='Output shard size; if 0, inferred from input manifest or defaults to 64')
    ap.add_argument('--loader-workers', type=int, default=8,
                    help='Workers for WebDataset streaming (reading)')
    ap.add_argument('--loader-prefetch', type=int, default=8,
                    help='Prefetch factor per worker for reading')
    ap.add_argument('--copy', action='store_true',
                    help='Materialize truncated dataset (copy) instead of creating symlinks')
    ap.add_argument('--concat-single', action='store_true',
                    help='After truncation, concatenate all selected shards into a single shard dataset')
    args = ap.parse_args()

    # Batch mode
    if args.top_paths and args.concat_datasets:
        trunc_mod = int(args.truncate_mod or 0)
        if not args.out_dir:
            raise SystemExit('With --concat-datasets, provide a single --out-dir for the combined dataset')
        top_paths = [Path(p) for p in args.top_paths]
        # Validate inputs
        for top in top_paths:
            tokens_dir = Path(top) / 'tokens'
            if not tokens_dir.exists() or not any(tokens_dir.glob('shard_*.tar')):
                raise FileNotFoundError(f"No token shards found in {tokens_dir}")

        out_base = Path(args.out_dir)
        out_tokens_dir = out_base  # tokens written directly to out_dir (no tokens subdir)
        out_tokens_dir.mkdir(parents=True, exist_ok=True)

        manifest: List[Dict] = []
        shard_id = 0

        for top in top_paths:
            in_tokens_dir = Path(top) / 'tokens'
            selected_shards, shard_size, stem_to_count = _select_shards_for_truncation(
                in_tokens_dir=in_tokens_dir,
                trunc_mod=trunc_mod,
                shard_size_opt=int(args.shard_size or 0),
            )
            print(f"Selected {len(selected_shards)} shards from {top}")
            for src_path in selected_shards:
                dst_name = f'shard_{shard_id:08d}.tar'
                dst_tokens = out_tokens_dir / dst_name
                if dst_tokens.exists() or dst_tokens.is_symlink():
                    try:
                        dst_tokens.unlink()
                    except FileNotFoundError:
                        pass
                if args.copy:
                    shutil.copy2(src_path, dst_tokens)
                else:
                    os.symlink(src_path.resolve(), dst_tokens)

                nseq = stem_to_count.get(src_path.stem, shard_size)
                manifest.append({'shard': dst_tokens.stem, 'num_sequences': int(nseq)})

                shard_id += 1

        write_manifest(manifest, out_tokens_dir, cum_counts=None, tot_tokens=None)
        print(f"Concatenated {len(top_paths)} datasets (tokens only) into {out_base} with {len(manifest)} shards")
        return

    if args.top_paths:
        trunc_mod = int(args.truncate_mod or 0)
        for top in args.top_paths:
            top_path = Path(top)
            in_tokens_dir = top_path / 'tokens'
            in_counts_dir = top_path / 'counts'
            if not in_tokens_dir.exists():
                raise FileNotFoundError(f"Tokens directory not found: {in_tokens_dir}")
            if not any(in_tokens_dir.glob('shard_*.tar')):
                raise FileNotFoundError(f"No token shards found in {in_tokens_dir}")
            counts_dir_opt: Optional[Path] = in_counts_dir if in_counts_dir.exists() and any(in_counts_dir.glob('shard_*.tar')) else None
            out_base = top_path.parent / f"{top_path.name}_trunc_{trunc_mod}"
            print(f"Processing: {top_path} -> {out_base}")
            if args.copy and not args.concat_single:
                process_one_dataset(
                    in_tokens_dir=in_tokens_dir,
                    in_counts_dir=counts_dir_opt,
                    out_base=out_base,
                    trunc_mod=trunc_mod,
                    shard_size_opt=int(args.shard_size or 0),
                    loader_workers=int(args.loader_workers or 8),
                    loader_prefetch=int(args.loader_prefetch or 8),
                )
            elif args.concat_single:
                process_one_dataset_concat_single(
                    in_tokens_dir=in_tokens_dir,
                    in_counts_dir=counts_dir_opt,
                    out_base=out_base,
                    trunc_mod=trunc_mod,
                    shard_size_opt=int(args.shard_size or 0),
                )
            else:
                process_one_dataset_symlink(
                    in_tokens_dir=in_tokens_dir,
                    in_counts_dir=counts_dir_opt,
                    out_base=out_base,
                    trunc_mod=trunc_mod,
                    shard_size_opt=int(args.shard_size or 0),
                )
        return

    # Single-dataset mode
    if not args.input_tokens or not args.out_dir:
        raise SystemExit("Provide either --top-paths (one or more) OR --input-tokens and --out-dir")

    if args.copy and not args.concat_single:
        process_one_dataset(
            in_tokens_dir=Path(args.input_tokens),
            in_counts_dir=Path(args.input_counts) if args.input_counts else None,
            out_base=Path(args.out_dir),
            trunc_mod=int(args.truncate_mod or 0),
            shard_size_opt=int(args.shard_size or 0),
            loader_workers=int(args.loader_workers or 8),
            loader_prefetch=int(args.loader_prefetch or 8),
        )
    elif args.concat_single:
        process_one_dataset_concat_single(
            in_tokens_dir=Path(args.input_tokens),
            in_counts_dir=Path(args.input_counts) if args.input_counts else None,
            out_base=Path(args.out_dir),
            trunc_mod=int(args.truncate_mod or 0),
            shard_size_opt=int(args.shard_size or 0),
        )
    else:
        process_one_dataset_symlink(
            in_tokens_dir=Path(args.input_tokens),
            in_counts_dir=Path(args.input_counts) if args.input_counts else None,
            out_base=Path(args.out_dir),
            trunc_mod=int(args.truncate_mod or 0),
            shard_size_opt=int(args.shard_size or 0),
        )


if __name__ == '__main__':
    main()



