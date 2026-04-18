from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _chunk_evenly(rows: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(num_shards)]
    for idx, row in enumerate(rows):
        shards[idx % num_shards].append(row)
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert IFBench test JSONL into normalized prompt dataset files and shard files.")
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shard_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", default="ifbench_test")
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = _load_jsonl(args.input_path.expanduser().resolve())
    if args.max_examples is not None:
        raw_rows = raw_rows[: args.max_examples]

    normalized_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        prompt = str(row["prompt"])
        normalized_rows.append(
            {
                "dataset_name": str(args.dataset_name),
                "task_id": str(row["key"]),
                "split": "test",
                "user_input": prompt,
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "instruction_id_list": row.get("instruction_id_list", []),
                "kwargs": row.get("kwargs", []),
            }
        )

    output_dir = args.output_dir.expanduser().resolve()
    shard_dir = args.shard_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_dir / "test.jsonl", normalized_rows)

    shards = _chunk_evenly(normalized_rows, max(int(args.num_shards), 1))
    for shard_idx, shard_rows in enumerate(shards):
        _write_jsonl(shard_dir / f"shard{shard_idx}.jsonl", shard_rows)
    _write_jsonl(shard_dir / "all.jsonl", normalized_rows)

    summary = {
        "input_path": str(args.input_path.expanduser().resolve()),
        "output_dir": str(output_dir),
        "shard_dir": str(shard_dir),
        "dataset_name": str(args.dataset_name),
        "num_examples": int(len(normalized_rows)),
        "num_shards": int(len(shards)),
        "shard_sizes": [int(len(shard)) for shard in shards],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (shard_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
