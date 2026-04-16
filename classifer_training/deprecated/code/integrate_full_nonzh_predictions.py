from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from classifer_training.utils import load_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach full non-Chinese DAPO predictions to source rows.")
    parser.add_argument("--shard_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--predictions_path", type=Path, required=True)
    parser.add_argument("--output_full_jsonl", type=Path, required=True)
    parser.add_argument("--output_tags_jsonl", type=Path, required=True)
    parser.add_argument("--output_full_parquet", type=Path, default=None)
    parser.add_argument("--legacy_compatible", action="store_true")
    return parser.parse_args()


def _parse_shard_id(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("shard"):
        raise ValueError(f"Could not parse shard id from {path}")
    return int(stem.replace("shard", ""))


def _resolve_original_task_id(row: dict) -> str:
    extra_info = row.get("extra_info") or {}
    task_id = extra_info.get("index") or row.get("task_id") or row.get("id")
    if task_id is None:
        raise KeyError("Could not resolve original task_id from row.")
    return str(task_id)


def _build_legacy_row(row: dict, task_id: str, predicted_difficulty: float, predicted_value: float, probe: str) -> dict:
    # Match the older integrated format and key order.
    return {
        "prompt": row.get("prompt"),
        "solution": row.get("solution"),
        "data_source": row.get("data_source"),
        "source_prompt": row.get("source_prompt"),
        "ability": row.get("ability"),
        "reward_model": row.get("reward_model"),
        "extra_info": row.get("extra_info"),
        "task_id": task_id,
        "predicted_difficulty": predicted_difficulty,
        "predicted_value": predicted_value,
        "probe": probe,
    }


def main() -> None:
    args = parse_args()

    pred_by_task_id: dict[str, dict] = {}
    for row in load_records(args.predictions_path.expanduser().resolve()):
        pred_by_task_id[str(row["task_id"])] = row

    full_rows: list[dict] = []
    tag_rows: list[dict] = []
    missing = 0

    for shard_path in sorted(args.shard_paths):
        shard_id = _parse_shard_id(shard_path)
        rows = load_records(shard_path.expanduser().resolve())
        for local_idx, row in enumerate(rows):
            synthetic_task_id = f"shard{shard_id}:{local_idx}"
            pred = pred_by_task_id.get(synthetic_task_id)
            if pred is None:
                missing += 1
                continue
            predicted_difficulty = float(pred["predicted_difficulty"])
            predicted_value = float(pred.get("predicted_value", 1.0 - predicted_difficulty))
            output_task_id = _resolve_original_task_id(row) if args.legacy_compatible else synthetic_task_id
            probe = str(pred.get("probe", "two_rollout_think_cascade_decomposition_best"))
            if args.legacy_compatible:
                merged = _build_legacy_row(row, output_task_id, predicted_difficulty, predicted_value, probe)
            else:
                merged = dict(row)
                merged["task_id"] = output_task_id
                merged["predicted_difficulty"] = predicted_difficulty
                merged["predicted_value"] = predicted_value
                merged["probe"] = probe
            full_rows.append(merged)
            tag_rows.append(
                {
                    "task_id": output_task_id,
                    "predicted_difficulty": predicted_difficulty,
                    "predicted_value": predicted_value,
                    "probe": probe,
                }
            )

    output_full_jsonl = args.output_full_jsonl.expanduser().resolve()
    output_tags_jsonl = args.output_tags_jsonl.expanduser().resolve()
    write_jsonl(output_full_jsonl, full_rows)
    write_jsonl(output_tags_jsonl, tag_rows)

    if args.output_full_parquet is not None:
        output_full_parquet = args.output_full_parquet.expanduser().resolve()
        output_full_parquet.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(full_rows).to_parquet(str(output_full_parquet))

    print(
        json.dumps(
            {
                "num_predictions": len(pred_by_task_id),
                "num_full_rows": len(full_rows),
                "num_tag_rows": len(tag_rows),
                "missing_rows": missing,
                "output_full_jsonl": str(output_full_jsonl),
                "output_tags_jsonl": str(output_tags_jsonl),
                "output_full_parquet": str(args.output_full_parquet.expanduser().resolve())
                if args.output_full_parquet is not None
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
