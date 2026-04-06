from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from classifer_training.utils import coerce_float, load_records, write_jsonl


DEFAULT_AGGREGATIONS = ("mean", "std", "min", "max", "p25", "p50", "p75")


def _aggregate(values: list[float], aggregations: tuple[str, ...]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    result: dict[str, float] = {}
    if array.size == 0:
        for name in aggregations:
            result[name] = 0.0
        return result

    for name in aggregations:
        if name == "mean":
            result[name] = float(np.mean(array))
        elif name == "std":
            result[name] = float(np.std(array))
        elif name == "min":
            result[name] = float(np.min(array))
        elif name == "max":
            result[name] = float(np.max(array))
        elif name == "p25":
            result[name] = float(np.percentile(array, 25))
        elif name == "p50":
            result[name] = float(np.percentile(array, 50))
        elif name == "p75":
            result[name] = float(np.percentile(array, 75))
        else:
            raise ValueError(f"Unsupported aggregation: {name}")
    return result


def build_augmented_label_rows(
    *,
    label_rows: list[dict],
    rollout_rows: list[dict],
    prefix: str,
    aggregations: tuple[str, ...],
) -> list[dict]:
    grouped_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rollout_rows:
        dataset_name = str(row.get("dataset_name", ""))
        task_id = str(row["task_id"])
        rollout_features = row.get("rollout_features", {})
        if not isinstance(rollout_features, dict):
            continue
        grouped = grouped_values[(dataset_name, task_id)]
        for feature_name, raw_value in rollout_features.items():
            numeric = coerce_float(raw_value)
            if numeric is None:
                continue
            grouped[str(feature_name)].append(numeric)

    augmented_rows: list[dict] = []
    for row in label_rows:
        dataset_name = str(row.get("dataset_name", ""))
        task_id = str(row["task_id"])
        grouped = grouped_values.get((dataset_name, task_id)) or grouped_values.get(("", task_id)) or {}

        extra_features: dict[str, float] = {}
        for feature_name, values in grouped.items():
            stats = _aggregate(values, aggregations)
            for agg_name, agg_value in stats.items():
                extra_features[f"{prefix}_{feature_name}_{agg_name}"] = agg_value

        updated = dict(row)
        aggregated = dict(updated.get("aggregated_features", {}))
        aggregated.update(extra_features)
        updated["aggregated_features"] = aggregated
        updated["augmented_rollout_feature_count"] = len(extra_features)
        augmented_rows.append(updated)
    return augmented_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment prompt-level label rows with aggregated numeric rollout features."
    )
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--rollout_index_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--prefix", default="enriched")
    parser.add_argument("--aggregations", nargs="+", default=list(DEFAULT_AGGREGATIONS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    label_rows = load_records(args.labels_path.expanduser().resolve())
    rollout_rows = load_records(args.rollout_index_path.expanduser().resolve())
    augmented = build_augmented_label_rows(
        label_rows=label_rows,
        rollout_rows=rollout_rows,
        prefix=args.prefix,
        aggregations=tuple(args.aggregations),
    )
    output_path = args.output_path.expanduser().resolve()
    write_jsonl(output_path, augmented)
    print(
        json.dumps(
            {
                "labels_path": str(args.labels_path.expanduser().resolve()),
                "rollout_index_path": str(args.rollout_index_path.expanduser().resolve()),
                "output_path": str(output_path),
                "num_rows": len(augmented),
                "aggregations": list(args.aggregations),
                "prefix": args.prefix,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
