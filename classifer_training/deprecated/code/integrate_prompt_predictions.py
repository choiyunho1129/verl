from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from classifer_training.utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach prompt-only predictions to the original DAPO dataset.")
    parser.add_argument("--predictions_path", type=Path, required=True)
    parser.add_argument("--hf_dataset_id", default="open-r1/DAPO-Math-17k-Processed")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--output_full_jsonl", type=Path, required=True)
    parser.add_argument("--output_tags_jsonl", type=Path, required=True)
    parser.add_argument("--output_full_parquet", type=Path, default=None)
    return parser.parse_args()


def _task_id_from_row(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    task_id = extra_info.get("index") or row.get("task_id") or row.get("id")
    if task_id is None:
        raise KeyError("Could not resolve task_id from dataset row.")
    return str(task_id)


def main() -> None:
    args = parse_args()

    pred_by_task_id: dict[str, dict[str, Any]] = {}
    with args.predictions_path.expanduser().resolve().open() as f:
        for line in f:
            row = json.loads(line)
            pred_by_task_id[str(row["task_id"])] = row

    dataset = load_dataset(args.hf_dataset_id, split=args.hf_split)
    full_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []

    missing = 0
    for raw_row in dataset:
        row = dict(raw_row)
        task_id = _task_id_from_row(row)
        pred_row = pred_by_task_id.get(task_id)
        if pred_row is None:
            missing += 1
            continue
        predicted_difficulty = float(pred_row["predicted_difficulty"])
        predicted_value = float(1.0 - predicted_difficulty)
        merged = dict(row)
        merged["task_id"] = task_id
        merged["predicted_difficulty"] = predicted_difficulty
        merged["predicted_value"] = predicted_value
        merged["probe"] = str(pred_row.get("probe", "prompt_only_probe"))
        full_rows.append(merged)
        tag_rows.append(
            {
                "task_id": task_id,
                "predicted_difficulty": predicted_difficulty,
                "predicted_value": predicted_value,
                "probe": str(pred_row.get("probe", "prompt_only_probe")),
            }
        )

    args.output_full_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_tags_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_full_jsonl.expanduser().resolve(), full_rows)
    write_jsonl(args.output_tags_jsonl.expanduser().resolve(), tag_rows)

    if args.output_full_parquet is not None:
        args.output_full_parquet.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(full_rows).to_parquet(str(args.output_full_parquet.expanduser().resolve()))

    print(
        json.dumps(
            {
                "num_predictions": len(pred_by_task_id),
                "num_full_rows": len(full_rows),
                "num_tag_rows": len(tag_rows),
                "missing_rows": missing,
                "output_full_jsonl": str(args.output_full_jsonl),
                "output_tags_jsonl": str(args.output_tags_jsonl),
                "output_full_parquet": str(args.output_full_parquet) if args.output_full_parquet is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
