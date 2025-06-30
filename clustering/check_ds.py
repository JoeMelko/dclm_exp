import argparse
import gzip
import json
import tarfile
from pathlib import Path
from typing import Any

TOKEN_ID = 50256  # token ID to look for


def count_token(obj: Any, target: int) -> int:
    """Recursively count how many times *target* appears in *obj* (which may be nested)."""
    if isinstance(obj, list):
        return sum(count_token(item, target) for item in obj)
    if isinstance(obj, dict):
        return sum(count_token(v, target) for v in obj.values())
    return int(obj == target)


def count_in_tar(tar_path: Path, token: int) -> int:
    """Return the total occurrences of *token* in all *.json.gz* members of *tar_path*."""
    total = 0
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json.gz")]
        for member in members:
            extracted = tar.extractfile(member)
            if extracted is None:
                continue  # skip if we cannot read member
            # Wrap the extracted file object with gzip
            try:
                with gzip.open(extracted, "rt", encoding="utf-8") as g:
                    try:
                        data = json.load(g)
                    except json.JSONDecodeError:
                        # Skip files that are not valid JSON arrays/dicts
                        continue
                    total += count_token(data, token)
            finally:
                extracted.close()
    return total


def main():
    parser = argparse.ArgumentParser(description="Count occurrences of a token ID (default 50256) in the first tar shard of a directory.")
    parser.add_argument("directory", type=Path, help="Path to the directory that contains shard_XXXXX.tar files")
    parser.add_argument("--token", type=int, default=TOKEN_ID, help="Token ID to count (default: 50256)")
    parser.add_argument("--max-shards", type=int, default=100, help="Maximum number of shard .tar files to process (default: 100)")
    args = parser.parse_args()

    tar_files = sorted(args.directory.glob("*.tar"))[: args.max_shards]
    if not tar_files:
        print(f"No .tar files found in {args.directory}")
        return

    total_count = 0
    for idx, tar_path in enumerate(tar_files, 1):
        print(f"[{idx}/{len(tar_files)}] Processing shard: {tar_path.name}")
        shard_count = count_in_tar(tar_path, args.token)
        total_count += shard_count
        print(f"    Occurrences in this shard: {shard_count}")

    print("-" * 60)
    print(f"Processed {len(tar_files)} shard(s)")
    print(f"Total occurrences of token {args.token}: {total_count}")


if __name__ == "__main__":
    main()
