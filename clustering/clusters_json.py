import argparse
import json
from pathlib import Path
from typing import List, Dict
import sys


def read_shard_lengths(path: Path) -> List[int]:
    """Read a JSON file that contains a list of shard lengths (ints)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON in {path}: {e}") from e

    if not isinstance(data, list) or not all(isinstance(x, int) for x in data):
        raise ValueError(f"Expected {path} to contain a JSON array of integers (shard lengths).")

    return data


def read_tsv_column(path: Path, column_idx: int = 1) -> List[int]:
    """Read the specified column (0-based index) from a TSV file and return it as a list of integers."""
    values: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            # Strip newline and split on tab
            parts = line.rstrip("\n").split("\t")
            if column_idx >= len(parts):
                raise ValueError(
                    f"TSV {path} line {line_num} does not have column index {column_idx}."
                )
            cell = parts[column_idx]
            try:
                values.append(int(cell))
            except ValueError as exc:
                raise ValueError(
                    f"Expected integer in column {column_idx} at line {line_num} of {path}, got '{cell}'."
                ) from exc
    return values


def build_shard_mapping(shard_lengths: List[int], column_values: List[int]) -> Dict[str, List[int]]:
    """Build the mapping {shard_name: [int]} using the provided shard lengths and TSV column values."""
    expected_total = sum(shard_lengths)
    if expected_total != len(column_values):
        raise ValueError(
            f"Sum of shard lengths ({expected_total}) does not match number of TSV rows ({len(column_values)})."
        )

    mapping: Dict[str, List[int]] = {}
    offset = 0
    for idx, shard_len in enumerate(shard_lengths):
        shard_key = f"shard_{idx:08d}_processed.jsonl.zstd"
        mapping[shard_key] = column_values[offset : offset + shard_len]
        offset += shard_len

    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a JSON mapping from shard filenames to arrays of TSV column values "
            "based on shard length metadata."
        )
    )
    parser.add_argument(
        "shard_lengths_json",
        type=Path,
        help="Path to JSON file that contains an array of shard lengths.",
    )
    parser.add_argument(
        "tsv_file",
        type=Path,
        help="Path to TSV file whose second column values will be partitioned.",
    )
    parser.add_argument(
        "output_json",
        type=Path,
        help="Path where the resulting JSON should be written.",
    )
    parser.add_argument(
        "--column-index",
        type=int,
        default=1,
        help="0-based index of the TSV column to extract (default: 1 for second column).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for pretty-printing the output JSON (default: 2).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Detect common mistake where the positional arguments are supplied in the wrong order
    # (TSV first, JSON second). We use file extensions as a heuristic.
    if args.shard_lengths_json.suffix.lower() in {".tsv", ".csv"} and args.tsv_file.suffix.lower() == ".json":
        print(
            "[warning] Detected that the positional arguments might be reversed (TSV given where JSON was "
            "expected). Swapping the first two arguments.",
            file=sys.stderr,
        )
        # Swap
        args.shard_lengths_json, args.tsv_file = args.tsv_file, args.shard_lengths_json

    shard_lengths = read_shard_lengths(args.shard_lengths_json)
    column_values = read_tsv_column(args.tsv_file, args.column_index)
    mapping = build_shard_mapping(shard_lengths, column_values)

    # Ensure output directory exists
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=args.indent, ensure_ascii=False)

    print(f"Wrote shard mapping JSON to {args.output_json} (shards: {len(mapping)})")


if __name__ == "__main__":
    main()
