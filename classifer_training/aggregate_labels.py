from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.utils import (
    coerce_float,
    get_nested_value,
    load_records,
    write_jsonl,
)


def load_run_examples(run_dir: Path, extra_numeric_fields: list[str]) -> list[dict[str, Any]]:
    experiments_path = run_dir / "all_experiments.jsonl"
    evaluations_path = run_dir / "evaluation_results.jsonl"
    experiment_rows = load_records(experiments_path)
    evaluation_rows = load_records(evaluations_path)
    if not evaluation_rows:
        raise ValueError(f"No evaluation rows found in {evaluations_path}.")

    correctness = evaluation_rows[-1]["correctness"]
    total_rows = min(len(experiment_rows), len(correctness))
    examples: list[dict[str, Any]] = []
    for row_idx in range(total_rows):
        experiment = experiment_rows[row_idx]
        examples.append(
            {
                "dataset_name": str(experiment.get("dataset_name", "")),
                "task_id": str(experiment.get("task_id", row_idx)),
                "user_input": experiment.get("user_input"),
                "temperature": coerce_float(get_nested_value(experiment, "config.temperature", default=None)),
                "correct": int(correctness[row_idx]),
                "numeric_features": extract_rollout_numeric_features(experiment, extra_numeric_fields),
            }
        )
    return examples


def build_aggregated_label_records(
    run_dirs: list[Path],
    extra_numeric_fields: list[str],
) -> list[dict[str, Any]]:
    aggregated: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    for run_dir in run_dirs:
        for example in load_run_examples(run_dir, extra_numeric_fields=extra_numeric_fields):
            key = (example["dataset_name"], example["task_id"])
            bucket = aggregated.setdefault(
                key,
                {
                    "dataset_name": example["dataset_name"],
                    "task_id": example["task_id"],
                    "user_input": example.get("user_input"),
                    "correctness": [],
                    "temperatures": [],
                    "feature_values": {},
                    "source_run_dirs": [],
                },
            )
            bucket["correctness"].append(int(example["correct"]))
            if example["temperature"] is not None:
                bucket["temperatures"].append(float(example["temperature"]))
            bucket["source_run_dirs"].append(str(run_dir))
            if bucket.get("user_input") is None and example.get("user_input") is not None:
                bucket["user_input"] = example["user_input"]

            for feature_name, feature_value in example["numeric_features"].items():
                bucket["feature_values"].setdefault(feature_name, []).append(float(feature_value))

    records: list[dict[str, Any]] = []
    for bucket in aggregated.values():
        correctness = np.asarray(bucket["correctness"], dtype=np.float32)
        aggregated_features: dict[str, float] = {}
        for feature_name, values in sorted(bucket["feature_values"].items()):
            values_array = np.asarray(values, dtype=np.float32)
            aggregated_features[f"{feature_name}_mean"] = float(values_array.mean())
            aggregated_features[f"{feature_name}_std"] = float(values_array.std(ddof=0))
            aggregated_features[f"{feature_name}_min"] = float(values_array.min())
            aggregated_features[f"{feature_name}_max"] = float(values_array.max())

        records.append(
            {
                "dataset_name": bucket["dataset_name"],
                "task_id": bucket["task_id"],
                "user_input": bucket.get("user_input"),
                "num_runs": int(len(correctness)),
                "correct_count": int(correctness.sum()),
                "wrong_count": int(len(correctness) - correctness.sum()),
                "sampling_accuracy": float(correctness.mean()) if len(correctness) else 0.0,
                "difficulty": float(1.0 - correctness.mean()) if len(correctness) else 1.0,
                "temperatures": sorted({round(temp, 8) for temp in bucket["temperatures"]}),
                "aggregated_features": aggregated_features,
                "source_run_dirs": sorted(set(bucket["source_run_dirs"])),
            }
        )
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiple sampled inference runs into per-prompt temperature-sampling "
            "accuracy labels."
        )
    )
    parser.add_argument(
        "--run_dirs",
        nargs="*",
        default=[],
        help="Inference run directories that each contain all_experiments.jsonl and evaluation_results.jsonl.",
    )
    parser.add_argument(
        "--run_glob",
        type=str,
        default=None,
        help="Optional glob such as 'data/inference_runs/temp07_seed*/' to collect run directories.",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--extra_numeric_fields",
        nargs="*",
        default=[],
        help="Extra dotted numeric fields to aggregate from all_experiments.jsonl records.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    if args.run_glob:
        run_dirs.extend(sorted(Path().glob(args.run_glob)))
    run_dirs = sorted({path.resolve() for path in run_dirs})

    if not run_dirs:
        raise ValueError("At least one run directory is required.")

    records = build_aggregated_label_records(
        run_dirs=run_dirs,
        extra_numeric_fields=args.extra_numeric_fields,
    )
    write_jsonl(args.output_path.expanduser().resolve(), records)
    print(f"Wrote {len(records)} label records to {args.output_path}")


if __name__ == "__main__":
    main()
