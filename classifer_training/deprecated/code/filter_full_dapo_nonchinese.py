from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from classifer_training.utils import write_jsonl

_CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def _row_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("prompt", "question", "problem", "instruction", "user_input"):
        value = row.get(key)
        if value is not None:
            parts.append(str(value))
    user_input = row.get("user_input")
    if user_input is not None:
        parts.append(str(user_input))
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
    return "\n".join(parts)


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter full DAPO-Math-17K to non-Chinese prompts and shard it.")
    parser.add_argument("--hf_dataset_id", default="open-r1/DAPO-Math-17k-Processed")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--dataset_name", default="dapo_math_17k_full_nonzh")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(summary_path.read_text(encoding="utf-8"))
        return

    dataset = load_dataset(args.hf_dataset_id, split=args.hf_split)
    kept: list[dict[str, Any]] = []
    num_cjk = 0

    for row in dataset:
        record = dict(row)
        text = _row_text(record)
        if _contains_cjk(text):
            num_cjk += 1
            continue
        record["dataset_name"] = args.dataset_name
        record["split"] = "full"
        kept.append(record)

    write_jsonl(outdir / "all_nonchinese.jsonl", kept)

    shards: list[list[dict[str, Any]]] = [[] for _ in range(args.num_shards)]
    for idx, row in enumerate(kept):
        shards[idx % args.num_shards].append(row)
    for shard_idx, shard_rows in enumerate(shards):
        write_jsonl(outdir / f"shard{shard_idx}.jsonl", shard_rows)

    summary = {
        "hf_dataset_id": args.hf_dataset_id,
        "hf_split": args.hf_split,
        "dataset_name": args.dataset_name,
        "num_total_rows": int(len(dataset)),
        "num_cjk_filtered": int(num_cjk),
        "num_nonchinese_rows": int(len(kept)),
        "num_shards": int(args.num_shards),
        "shard_sizes": [int(len(rows)) for rows in shards],
        "output_dir": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
