from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifer_training.utils import load_records, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine weak4 train/validation prompts and split them into shard JSONLs.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be at least 1.")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    expected_files = [output_dir / f"shard{i}.jsonl" for i in range(args.num_shards)]
    if not args.overwrite and summary_path.exists() and all(path.exists() for path in expected_files):
        print(f"[skip] weak4 shard dataset already exists: {output_dir}")
        return

    split_paths = []
    for split_name in ("train", "validation"):
        split_path = input_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing weak4 split file: {split_path}")
        split_paths.append(split_path)

    rows = []
    split_counts: dict[str, int] = {}
    for split_path in split_paths:
        split_rows = [dict(row) for row in load_records(split_path)]
        split_name = split_path.stem
        split_counts[split_name] = len(split_rows)
        rows.extend(split_rows)

    shards: list[list[dict]] = [[] for _ in range(args.num_shards)]
    for idx, row in enumerate(rows):
        shards[idx % args.num_shards].append(row)

    write_jsonl(output_dir / "all.jsonl", rows)
    for shard_idx, shard_rows in enumerate(shards):
        write_jsonl(output_dir / f"shard{shard_idx}.jsonl", shard_rows)

    summary = {
        "dataset_name": "dapo_math_17k_weak4",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_rows_total": len(rows),
        "split_counts": split_counts,
        "num_shards": args.num_shards,
        "shard_sizes": [len(shard_rows) for shard_rows in shards],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
