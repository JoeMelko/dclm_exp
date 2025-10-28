import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Convert counts to normalized per-dataset ratios (sum to 1) for GPU schedule")
    p.add_argument("--counts", type=Path, required=True, help="Path to counts JSON file")
    p.add_argument("--num-groups", type=int, default=10_000, help="Number of datasets (iterates 0..num_groups-1)")
    p.add_argument("--out", type=Path, required=True, help="Output path for ratios JSON file")

    args = p.parse_args(argv)

    with args.counts.open("r", encoding="utf-8") as f:
        counts = json.load(f)
    if not isinstance(counts, dict):
        raise SystemExit("Counts file must be a JSON object mapping dataset keys to numeric counts")

    keys = [f"dataset{i}" for i in range(args.num_groups)]
    values = [float(counts.get(k, 0.0)) for k in keys]
    total = sum(values)

    if total <= 0.0:
        ratios = {k: 0.0 for k in keys}
    else:
        ratios = {k: (v / total) for k, v in zip(keys, values)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(ratios, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


