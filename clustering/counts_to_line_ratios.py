import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Convert counts to per-dataset line ratios for line_ratio_resampler.py")
    p.add_argument("--counts0", type=Path, required=True, help="Path to baseline/original counts JSON file")
    p.add_argument("--counts-cur", dest="counts_cur", type=Path, required=True, help="Path to current counts JSON file")
    p.add_argument("--num-groups", type=int, default=10_000, help="Number of datasets (iterates 0..num_groups-1)")
    p.add_argument("--keep-ratio", dest="keep_ratio", type=float, default=1.0, help="Scalar to multiply all ratios by")
    p.add_argument("--out", type=Path, required=True, help="Output path for ratios JSON file")

    args = p.parse_args(argv)

    with args.counts0.open("r", encoding="utf-8") as f:
        c0 = json.load(f)
    with args.counts_cur.open("r", encoding="utf-8") as f:
        ccur = json.load(f)
    if not isinstance(c0, dict) or not isinstance(ccur, dict):
        raise SystemExit("Counts files must be JSON objects mapping dataset keys to numeric counts")

    ratios = {}
    for i in range(args.num_groups):
        key = f"dataset{i}"
        base = float(c0.get(key, 0.0))
        cur = float(ccur.get(key, 0.0))
        ratios[key] = (0.0 if base <= 0.0 else (cur / base)) * args.keep_ratio

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(ratios, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
