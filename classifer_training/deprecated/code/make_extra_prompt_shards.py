from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create additional DAPO prompt shards from a full source JSONL.")
    parser.add_argument("--source_jsonl", type=Path, required=True)
    parser.add_argument("--exclude_jsonls", nargs="*", default=[])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_prompts", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _task_id(row: dict) -> str:
    task_id = row.get("task_id")
    if task_id is not None:
        return str(task_id)
    extra_info = row.get("extra_info") or {}
    for key in ("index",):
        if key in extra_info:
            return str(extra_info[key])
    raise KeyError("Could not find task_id in row.")


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.source_jsonl.expanduser().resolve())

    excluded: set[str] = set()
    for path_str in args.exclude_jsonls:
        path = Path(path_str).expanduser().resolve()
        for row in _load_jsonl(path):
            excluded.add(_task_id(row))

    available = [row for row in rows if _task_id(row) not in excluded]
    available.sort(key=_task_id)

    if len(available) < args.num_prompts:
        raise ValueError(f"Requested {args.num_prompts} prompts but only {len(available)} are available.")

    import random

    rng = random.Random(args.seed)
    chosen = available[:]
    rng.shuffle(chosen)
    chosen = chosen[: args.num_prompts]
    chosen.sort(key=_task_id)

    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    shard_size = (len(chosen) + args.num_shards - 1) // args.num_shards
    for shard_idx in range(args.num_shards):
        shard_rows = chosen[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        shard_path = args.output_dir.expanduser().resolve() / f"extra_shard{shard_idx}.jsonl"
        with shard_path.open("w", encoding="utf-8") as f:
            for row in shard_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_jsonl": str(args.source_jsonl.expanduser().resolve()),
        "num_source_rows": len(rows),
        "num_excluded_task_ids": len(excluded),
        "num_available_rows": len(available),
        "num_selected_rows": len(chosen),
        "num_shards": args.num_shards,
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
