from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from classifer_training.data import load_hidden_rows
from classifer_training.enrich_rollout_index import _single_run_features
from classifer_training.prompt_only_experiments import _hidden_relation_features, _prompt_features
from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.train_prompt_two_trajectory_promptsearch import build_matrix, build_pair_rows
from classifer_training.utils import load_records, write_jsonl


def _iter_hidden_rows_any(hidden_path: Path, index_path: Path) -> list[dict[str, Any]]:
    return load_hidden_rows(hidden_path, index_path=index_path, dataset_name="dapo_math_17k")


def _build_prompt_lookup_from_paths(
    raw_hidden_paths: list[Path],
    raw_index_paths: list[Path],
    last6_hidden_paths: list[Path],
    last6_index_paths: list[Path],
) -> dict[str, dict[str, np.ndarray]]:
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for hidden_path, index_path in zip(raw_hidden_paths, raw_index_paths):
        for row in _iter_hidden_rows_any(hidden_path, index_path):
            task_id = str(row["task_id"])
            hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
            index_row = row["index_row"]
            user_input = str(index_row.get("user_input", ""))
            input_length = int(index_row.get("input_length", 0))
            lookup[task_id] = {
                "user_input": user_input,
                "prompt_feats": _prompt_features(user_input, input_length),
                "rel_last": _hidden_relation_features(hidden_layers),
                "last_l22": hidden_layers[22],
                "last_l23": hidden_layers[23],
                "last_l17": hidden_layers[17],
                "last_l24": hidden_layers[24],
                "last_l25": hidden_layers[25],
                "last_l26": hidden_layers[26],
                "last_l35": hidden_layers[35],
                "last_mean": np.stack(hidden_layers, axis=0).mean(axis=0).astype(np.float32),
            }
    for hidden_path, index_path in zip(last6_hidden_paths, last6_index_paths):
        for row in _iter_hidden_rows_any(hidden_path, index_path):
            task_id = str(row["task_id"])
            if task_id not in lookup:
                continue
            hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
            lookup[task_id].update(
                {
                    "rel_l10": _hidden_relation_features(hidden_layers),
                    "l10_l22": hidden_layers[22],
                    "l10_l23": hidden_layers[23],
                    "l10_l17": hidden_layers[17],
                    "l10_l24": hidden_layers[24],
                    "l10_l25": hidden_layers[25],
                    "l10_l26": hidden_layers[26],
                    "l10_l35": hidden_layers[35],
                    "l10_mean": np.stack(hidden_layers, axis=0).mean(axis=0).astype(np.float32),
                }
            )
    return lookup


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_records(path)


def _build_training_labels(run_root: Path) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for seed in range(1, 17):
        run_dir = run_root / f"temp0.7_seed{seed}"
        experiment_rows = _load_jsonl(run_dir / "all_experiments.jsonl")
        evaluation_rows = _load_jsonl(run_dir / "evaluation_results.jsonl")
        correctness = evaluation_rows[-1]["correctness"]
        total = min(len(experiment_rows), len(correctness))
        for idx in range(total):
            row = experiment_rows[idx]
            task_id = str(row["task_id"])
            bucket = buckets.setdefault(task_id, {"split": str(row.get("split", "")), "correct": []})
            bucket["correct"].append(int(correctness[idx]))
    return buckets


def _group_training_rollouts(rollout_index_path: Path, label_buckets: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _load_jsonl(rollout_index_path)
    feature_keys = sorted(rows[0]["rollout_features"].keys())
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in label_buckets:
            continue
        label_bucket = label_buckets[task_id]
        stats_vec = np.asarray([float(row["rollout_features"].get(key, 0.0)) for key in feature_keys], dtype=np.float32)
        group = grouped.setdefault(
            task_id,
            {
                "task_id": task_id,
                "split": str(row.get("split", label_bucket["split"])),
                "y_true": float(1.0 - (sum(label_bucket["correct"]) / len(label_bucket["correct"]))),
                "rollouts": [],
            },
        )
        group["rollouts"].append(
            {
                "rollout_row_index": int(row.get("rollout_row_index", len(group["rollouts"]))),
                "stats_vec": stats_vec,
            }
        )
    return [grouped[key] for key in sorted(grouped.keys())], feature_keys


def _group_target_rollouts(run_dirs: list[Path], feature_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        rows = _load_jsonl(run_dir / "all_experiments.jsonl")
        for row_idx, row in enumerate(rows):
            task_id = str(row["task_id"])
            feats = {}
            feats.update(extract_rollout_numeric_features(row))
            feats.update(_single_run_features(row))
            stats_vec = np.asarray([float(feats.get(key, 0.0)) for key in feature_keys], dtype=np.float32)
            group = grouped.setdefault(
                task_id,
                {"task_id": task_id, "split": "full", "y_true": 0.0, "rollouts": []},
            )
            rollout_row_index = row.get("sample_index")
            if rollout_row_index is None:
                rollout_row_index = row_idx
            group["rollouts"].append(
                {
                    "rollout_row_index": int(rollout_row_index),
                    "stats_vec": stats_vec,
                }
            )
    return [grouped[key] for key in sorted(grouped.keys())]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the best 2-rollout ET probe on finished16 and score new prompt shards.")
    parser.add_argument("--repo_root", type=Path, default=Path("/home/jongwonlim/verl/yoonho/verl"))
    parser.add_argument("--target_run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--target_raw_hidden_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--target_raw_index_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--target_last6_hidden_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--target_last6_index_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--max_features", type=float, default=0.7)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=24)
    parser.add_argument("--model_cache_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()

    if len(args.target_raw_hidden_paths) != len(args.target_raw_index_paths):
        raise ValueError("Raw hidden/index path lists must have the same length.")
    if len(args.target_last6_hidden_paths) != len(args.target_last6_index_paths):
        raise ValueError("Last6 hidden/index path lists must have the same length.")

    train_run_root = repo / "classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507"
    train_rollout_index_path = (
        repo
        / "classifer_training/artifacts/rollout_index/dapo_math_17k/qwen3_4b_instruct_2507_promptonly_finished16/finished16_promptonly_rollout_index_compact.jsonl"
    )
    train_raw_hidden_dir = repo / "classifer_training/artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507"
    train_raw_index_dir = repo / "classifer_training/artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507"
    train_last6_hidden_path = repo / "classifer_training/artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507_last6mean/hidden_states.pt"
    train_last6_index_path = repo / "classifer_training/artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507_last6mean/index.jsonl"

    label_buckets = _build_training_labels(train_run_root)
    grouped_train_rows, feature_keys = _group_training_rollouts(train_rollout_index_path, label_buckets)
    train_prompt_lookup = _build_prompt_lookup_from_paths(
        raw_hidden_paths=[
            train_raw_hidden_dir / "hidden_states_train.pt",
            train_raw_hidden_dir / "hidden_states_validation.pt",
            train_raw_hidden_dir / "hidden_states_test.pt",
        ],
        raw_index_paths=[
            train_raw_index_dir / "index_train.jsonl",
            train_raw_index_dir / "index_validation.jsonl",
            train_raw_index_dir / "index_test.jsonl",
        ],
        last6_hidden_paths=[train_last6_hidden_path],
        last6_index_paths=[train_last6_index_path],
    )
    train_pair_rows = build_pair_rows(
        grouped_rows=grouped_train_rows,
        feature_keys=feature_keys,
        train_splits={"train", "validation", "test"},
        test_splits=set(),
        train_pairs_per_prompt=4,
        test_pairs_per_prompt=0,
        random_seed=args.random_seed,
    )
    X_train, y_train, _, _ = build_matrix(train_pair_rows, train_prompt_lookup, "l10_l26")
    model_cache_path = args.model_cache_path.expanduser().resolve() if args.model_cache_path else None
    if model_cache_path is not None and model_cache_path.exists():
        model = joblib.load(model_cache_path)
    else:
        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
        model.fit(X_train, y_train)
        if model_cache_path is not None:
            model_cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_cache_path)

    target_prompt_lookup = _build_prompt_lookup_from_paths(
        raw_hidden_paths=[path.expanduser().resolve() for path in args.target_raw_hidden_paths],
        raw_index_paths=[path.expanduser().resolve() for path in args.target_raw_index_paths],
        last6_hidden_paths=[path.expanduser().resolve() for path in args.target_last6_hidden_paths],
        last6_index_paths=[path.expanduser().resolve() for path in args.target_last6_index_paths],
    )
    target_grouped_rows = _group_target_rollouts(
        run_dirs=[path.expanduser().resolve() for path in args.target_run_dirs],
        feature_keys=feature_keys,
    )
    target_pair_rows = build_pair_rows(
        grouped_rows=target_grouped_rows,
        feature_keys=feature_keys,
        train_splits=set(),
        test_splits={"full"},
        train_pairs_per_prompt=0,
        test_pairs_per_prompt=10,
        random_seed=args.random_seed,
    )
    X_target, _, _, metadata_rows = build_matrix(target_pair_rows, target_prompt_lookup, "l10_l26")
    pred = np.clip(np.asarray(model.predict(X_target), dtype=np.float32).reshape(-1), 0.0, 1.0)

    prompt_groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"preds": [], "user_input": ""})
    for meta, pred_val in zip(metadata_rows, pred.tolist()):
        task_id = str(meta["task_id"])
        prompt_groups[task_id]["preds"].append(float(pred_val))
        prompt_groups[task_id]["user_input"] = str(target_prompt_lookup.get(task_id, {}).get("user_input", ""))

    rows = []
    for task_id in sorted(prompt_groups):
        row = {
            "task_id": task_id,
            "user_input": prompt_groups[task_id]["user_input"],
            "predicted_difficulty": float(np.mean(prompt_groups[task_id]["preds"])),
            "predicted_value": float(1.0 - np.mean(prompt_groups[task_id]["preds"])),
            "probe": "two_rollout_last6mean_layer26_extratrees_refit_finished16_all",
            "num_pair_predictions": int(len(prompt_groups[task_id]["preds"])),
        }
        rows.append(row)

    output_path = args.output_path.expanduser().resolve()
    write_jsonl(output_path, rows)
    summary = {
        "output_path": str(output_path),
        "num_target_prompts": int(len(rows)),
        "num_target_pair_rows": int(len(metadata_rows)),
        "train_num_rows": int(X_train.shape[0]),
        "feature_dim": int(X_train.shape[1]),
        "params": {
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "random_seed": args.random_seed,
        },
        "model_cache_path": str(model_cache_path) if model_cache_path is not None else None,
    }
    (output_path.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
