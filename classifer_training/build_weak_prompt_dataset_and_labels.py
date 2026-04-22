from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.utils import load_records, write_jsonl


def _stable_split(task_id: str, val_ratio: float) -> str:
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if value < val_ratio else "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak-label prompt dataset splits and aggregated labels from existing pseudo run dirs.")
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--ignore_existing_split",
        action="store_true",
        help="Recompute train/validation assignments from task_id hashes instead of preserving split values already stored in the run rows.",
    )
    parser.add_argument(
        "--train_run_dir_names",
        nargs="*",
        default=[],
        help="Run directory basenames whose prompts should be assigned to the train split.",
    )
    parser.add_argument(
        "--validation_run_dir_names",
        nargs="*",
        default=[],
        help="Run directory basenames whose prompts should be assigned to the validation split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_run_dir_names = set(args.train_run_dir_names)
    validation_run_dir_names = set(args.validation_run_dir_names)
    overlapping_run_dir_names = sorted(train_run_dir_names & validation_run_dir_names)
    if overlapping_run_dir_names:
        raise ValueError(f"Run dirs cannot be both train and validation: {overlapping_run_dir_names}")

    prompt_rows: dict[str, dict[str, Any]] = {}
    label_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    inputs_summary: list[dict[str, Any]] = []

    for run_dir in [path.expanduser().resolve() for path in args.run_dirs]:
        experiments_path = run_dir / "all_experiments.jsonl"
        evaluation_path = run_dir / "evaluation_results.jsonl"
        experiments = load_records(experiments_path)
        evaluations = load_records(evaluation_path)
        correctness = list(evaluations[-1]["correctness"])
        usable = min(len(experiments), len(correctness))
        experiments = experiments[:usable]
        correctness = correctness[:usable]
        inputs_summary.append(
            {
                "run_dir": str(run_dir),
                "num_examples": int(usable),
                "num_prompts": int(len({str(row["task_id"]) for row in experiments})),
            }
        )
        run_dir_split = None
        if run_dir.name in train_run_dir_names:
            run_dir_split = "train"
        elif run_dir.name in validation_run_dir_names:
            run_dir_split = "validation"

        for row_idx, (row, correct) in enumerate(zip(experiments, correctness)):
            dataset_name = str(row.get("dataset_name", ""))
            task_id = str(row["task_id"])
            if run_dir_split is not None:
                split = run_dir_split
            elif args.ignore_existing_split:
                split = _stable_split(task_id, float(args.val_ratio))
            else:
                split = str(row.get("split") or _stable_split(task_id, float(args.val_ratio)))
            user_input = str(row.get("user_input", ""))

            existing_prompt_row = prompt_rows.get(task_id)
            if existing_prompt_row is not None and existing_prompt_row["split"] != split:
                raise ValueError(
                    f"Conflicting split assignment for task_id={task_id}: "
                    f"{existing_prompt_row['split']} vs {split}"
                )
            prompt_rows.setdefault(
                task_id,
                {
                    "dataset_name": dataset_name,
                    "task_id": task_id,
                    "split": split,
                    "user_input": user_input,
                    "prompt": user_input,
                    "messages": [{"role": "user", "content": user_input}],
                },
            )

            bucket = label_buckets.setdefault(
                (dataset_name, task_id),
                {
                    "dataset_name": dataset_name,
                    "task_id": task_id,
                    "user_input": user_input,
                    "correctness": [],
                    "temperatures": [],
                    "feature_values": defaultdict(list),
                    "source_run_dirs": set(),
                },
            )
            bucket["correctness"].append(int(correct))
            temperature = row.get("config", {}).get("temperature") if isinstance(row.get("config"), dict) else None
            if temperature is not None:
                bucket["temperatures"].append(float(temperature))
            bucket["source_run_dirs"].add(str(run_dir))
            for feature_name, feature_value in extract_rollout_numeric_features(row).items():
                bucket["feature_values"][feature_name].append(float(feature_value))

            if (row_idx + 1) % 5000 == 0:
                print(json.dumps({"stage": "accumulate", "run_dir": str(run_dir), "processed_examples": row_idx + 1}), flush=True)

    prompt_dataset_dir = args.prompt_dataset_dir.expanduser().resolve()
    prompt_dataset_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows_sorted = sorted(prompt_rows.values(), key=lambda row: row["task_id"])
    train_rows = [row for row in prompt_rows_sorted if row["split"] == "train"]
    val_rows = [row for row in prompt_rows_sorted if row["split"] == "validation"]
    write_jsonl(prompt_dataset_dir / "train.jsonl", train_rows)
    write_jsonl(prompt_dataset_dir / "validation.jsonl", val_rows)

    label_rows: list[dict[str, Any]] = []
    for bucket in label_buckets.values():
        correctness = np.asarray(bucket["correctness"], dtype=np.float32)
        aggregated_features: dict[str, float] = {}
        for feature_name, values in sorted(bucket["feature_values"].items()):
            values_array = np.asarray(values, dtype=np.float32)
            aggregated_features[f"{feature_name}_mean"] = float(values_array.mean())
            aggregated_features[f"{feature_name}_std"] = float(values_array.std(ddof=0))
            aggregated_features[f"{feature_name}_min"] = float(values_array.min())
            aggregated_features[f"{feature_name}_max"] = float(values_array.max())
        label_rows.append(
            {
                "dataset_name": bucket["dataset_name"],
                "task_id": bucket["task_id"],
                "user_input": bucket["user_input"],
                "num_runs": int(len(correctness)),
                "correct_count": int(correctness.sum()),
                "wrong_count": int(len(correctness) - correctness.sum()),
                "sampling_accuracy": float(correctness.mean()) if len(correctness) else 0.0,
                "difficulty": float(1.0 - correctness.mean()) if len(correctness) else 1.0,
                "temperatures": sorted({round(temp, 8) for temp in bucket["temperatures"]}),
                "aggregated_features": aggregated_features,
                "source_run_dirs": sorted(bucket["source_run_dirs"]),
            }
        )
    label_rows.sort(key=lambda row: (str(row["dataset_name"]), str(row["task_id"])))
    write_jsonl(args.labels_path.expanduser().resolve(), label_rows)

    summary = {
        "num_run_dirs": int(len(args.run_dirs)),
        "num_prompts_total": int(len(prompt_rows_sorted)),
        "num_prompts_train": int(len(train_rows)),
        "num_prompts_validation": int(len(val_rows)),
        "num_label_rows": int(len(label_rows)),
        "val_ratio": float(args.val_ratio),
        "ignore_existing_split": bool(args.ignore_existing_split),
        "train_run_dir_names": sorted(train_run_dir_names),
        "validation_run_dir_names": sorted(validation_run_dir_names),
        "prompt_dataset_dir": str(prompt_dataset_dir),
        "labels_path": str(args.labels_path.expanduser().resolve()),
        "inputs": inputs_summary,
    }
    args.summary_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.expanduser().resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
