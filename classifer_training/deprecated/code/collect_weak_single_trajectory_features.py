from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.single_rollout_hidden_utils import (
    build_rollout_hidden_lookup,
    build_rollout_index_lookup,
    build_split_lookup,
    group_weak_rollouts,
    load_labels_by_task,
    load_prompt_hidden_lookup,
    select_single_rollout,
)
from classifer_training.utils import write_jsonl
from verl.utils.single_trajectory_estimator import extract_derived_rollout_features


DEFAULT_RESPONSE_SCALAR_KEYS = [
    "output_length",
    "think_tokens",
    "answer_tokens",
    "has_complete_answer",
    "has_reasoning_content",
    "output_mean_token_entropy",
    "reasoning_mean_token_entropy",
    "answer_mean_token_entropy",
    "output_unique_token_ratio",
    "answer_unique_token_ratio",
    "output_repetition_ratio",
    "reasoning_repetition_ratio",
    "duplicate_line_ratio",
]

DEFAULT_DERIVED_RESPONSE_SCALAR_KEYS = [
    "think_ratio",
    "answer_ratio",
    "entropy_gap_reasoning_answer",
    "unique_gap_reasoning_output",
    "repetition_gap_reasoning_output",
    "reasoning_x_log_output_length",
    "answer_entropy_gap_vs_output",
]


def _label_to_difficulty(label_row: dict[str, object]) -> float:
    return float(label_row["difficulty"])


def _label_to_value(label_row: dict[str, object]) -> float:
    return 1.0 - _label_to_difficulty(label_row)


def _build_response_feature_map(
    index_row: dict[str, object],
    *,
    scalar_keys: list[str],
    derived_keys: list[str],
) -> dict[str, float]:
    feature_map = extract_rollout_numeric_features(index_row)
    feature_map.update(extract_derived_rollout_features(feature_map))
    output: dict[str, float] = {}
    for key in list(scalar_keys) + list(derived_keys):
        if key not in feature_map:
            raise KeyError(f"Missing rollout feature {key!r} in rollout index row.")
        output[key] = float(feature_map[key])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect real weak single-trajectory feature rows from prompt hidden + real rollout artifacts."
    )
    parser.add_argument("--weak_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--weak_labels_path", type=Path, required=True)
    parser.add_argument("--weak_prompt_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_rollout_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_rollout_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_layer_index", type=int, default=26)
    parser.add_argument("--response_component", type=str, default="think_end_hidden")
    parser.add_argument("--response_pool_mode", choices=["mean", "last", "first", "flatten"], default="mean")
    parser.add_argument("--single_rollout_strategy", choices=["first", "all"], default="first")
    parser.add_argument("--response_scalar_keys", nargs="*", default=list(DEFAULT_RESPONSE_SCALAR_KEYS))
    parser.add_argument("--derived_response_scalar_keys", nargs="*", default=list(DEFAULT_DERIVED_RESPONSE_SCALAR_KEYS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_by_task = load_labels_by_task(args.weak_labels_path.expanduser().resolve())
    split_lookup = build_split_lookup(args.weak_prompt_dataset_dir.expanduser().resolve())
    prompt_lookup = load_prompt_hidden_lookup(
        [path.expanduser().resolve() for path in args.weak_prompt_hidden_paths],
        [path.expanduser().resolve() for path in args.weak_prompt_index_paths],
        layer_index=int(args.prompt_layer_index),
    )
    rollout_hidden_lookup = build_rollout_hidden_lookup(
        [path.expanduser().resolve() for path in args.weak_rollout_hidden_paths],
        [path.expanduser().resolve() for path in args.weak_rollout_index_paths],
        component_name=args.response_component,
        layer_index=0,
        pool_mode=args.response_pool_mode,
    )
    rollout_index_lookup = build_rollout_index_lookup(
        [path.expanduser().resolve() for path in args.weak_rollout_index_paths]
    )

    grouped = group_weak_rollouts(
        weak_run_dirs=[path.expanduser().resolve() for path in args.weak_run_dirs],
        split_lookup=split_lookup,
        labels_by_task=labels_by_task,
        rollout_hidden_lookup=rollout_hidden_lookup,
        rollout_index_lookup=rollout_index_lookup,
        rollout_scalar_keys=list(args.response_scalar_keys),
        derived_rollout_scalar_keys=list(args.derived_response_scalar_keys),
        extra_rollout_scalar_field_paths=[],
    )
    rows = select_single_rollout(grouped, args.single_rollout_strategy)

    metadata_rows: list[dict[str, object]] = []
    prompt_vectors: list[np.ndarray] = []
    response_vectors: list[np.ndarray] = []

    ordered_feature_names = list(args.response_scalar_keys) + list(args.derived_response_scalar_keys)
    for row_idx, row in enumerate(rows):
        task_id = str(row["task_id"])
        prompt_hidden = prompt_lookup.get(task_id)
        if prompt_hidden is None:
            continue

        run_dir = str(row.get("run_dir", ""))
        rollout_row_index = int(row["rollout_row_index"])
        index_row = rollout_index_lookup.get((run_dir, rollout_row_index))
        if index_row is None:
            continue

        label_row = labels_by_task[task_id]
        response_feature_map = _build_response_feature_map(
            index_row,
            scalar_keys=list(args.response_scalar_keys),
            derived_keys=list(args.derived_response_scalar_keys),
        )
        prompt_vectors.append(np.asarray(prompt_hidden, dtype=np.float32).reshape(-1))
        response_vectors.append(np.asarray(row["rollout_hidden_vec"], dtype=np.float32).reshape(-1))
        metadata_rows.append(
            {
                "row_index": row_idx,
                "task_id": task_id,
                "split": str(row["split"]),
                "difficulty": _label_to_difficulty(label_row),
                "value": _label_to_value(label_row),
                "run_dir": run_dir,
                "rollout_row_index": rollout_row_index,
                "response_features": response_feature_map,
            }
        )

    if not metadata_rows:
        raise RuntimeError("No feature rows were collected.")

    prompt_array = np.stack(prompt_vectors, axis=0).astype(np.float32)
    response_array = np.stack(response_vectors, axis=0).astype(np.float32)
    np.save(output_dir / "prompt_hidden.npy", prompt_array)
    np.save(output_dir / "response_hidden.npy", response_array)
    write_jsonl(output_dir / "metadata.jsonl", metadata_rows)

    summary = {
        "setting": "weak_single_trajectory_feature_collection",
        "num_rows": int(len(metadata_rows)),
        "num_train_rows": int(sum(row["split"] == "train" for row in metadata_rows)),
        "num_validation_rows": int(sum(row["split"] == "validation" for row in metadata_rows)),
        "num_prompts": int(len({str(row["task_id"]) for row in metadata_rows})),
        "prompt_hidden_dim": int(prompt_array.shape[1]),
        "response_hidden_dim": int(response_array.shape[1]),
        "response_component": args.response_component,
        "response_pool_mode": args.response_pool_mode,
        "single_rollout_strategy": args.single_rollout_strategy,
        "response_feature_names": ordered_feature_names,
        "output_files": {
            "metadata_jsonl": str(output_dir / "metadata.jsonl"),
            "prompt_hidden_npy": str(output_dir / "prompt_hidden.npy"),
            "response_hidden_npy": str(output_dir / "response_hidden.npy"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
