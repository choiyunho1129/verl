from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.prompt_only_experiments import _prompt_features
from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.utils import load_records, write_jsonl


BASE_ROLLOUT_FEATURE_KEYS = [
    "input_length",
    "output_length",
    "generation_time",
    "think_tokens",
    "answer_tokens",
    "has_complete_answer",
    "has_reasoning_content",
    "output_text_entropy",
    "reasoning_text_entropy",
    "answer_text_entropy",
    "output_unique_token_ratio",
    "reasoning_unique_token_ratio",
    "answer_unique_token_ratio",
    "output_repetition_ratio",
    "reasoning_repetition_ratio",
    "answer_repetition_ratio",
    "output_repeated_bigram_ratio",
    "output_repeated_trigram_ratio",
    "reasoning_repeated_bigram_ratio",
    "reasoning_repeated_trigram_ratio",
    "duplicate_line_ratio",
    "answer_terminal_punctuation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weak-label 2-rollout transfer baselines and evaluate on the original clean test split.")
    parser.add_argument("--weak_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--weak_labels_path", type=Path, required=True)
    parser.add_argument("--clean_rollout_index_path", type=Path, required=True)
    parser.add_argument("--clean_labels_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--weak_pairs_per_prompt", type=int, default=6)
    parser.add_argument("--clean_test_pairs_per_prompt", type=int, default=10)
    return parser.parse_args()


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _stable_prompt_feats(user_input: str) -> np.ndarray:
    text = str(user_input or "")
    approx_input_length = len(text.split())
    return _prompt_features(text, approx_input_length)


def _build_split_lookup(prompt_dataset_dir: Path) -> dict[str, str]:
    split_lookup: dict[str, str] = {}
    for split_name in ("train", "validation"):
        path = prompt_dataset_dir / f"{split_name}.jsonl"
        for row in load_records(path):
            split_lookup[str(row["task_id"])] = split_name
    return split_lookup


def _build_prompt_lookup_from_labels(label_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in label_rows:
        task_id = str(row["task_id"])
        user_input = str(row.get("user_input", ""))
        lookup[task_id] = {
            "user_input": user_input,
            "prompt_feats": _stable_prompt_feats(user_input),
        }
    return lookup


def _weak_row_features(row: dict[str, Any]) -> dict[str, float]:
    return extract_rollout_numeric_features(row)


def _group_weak_rollouts(
    weak_run_dirs: list[Path],
    feature_keys: list[str],
    weak_labels_by_task: dict[str, dict[str, Any]],
    split_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run_dir in weak_run_dirs:
        rows = load_records(run_dir / "all_experiments.jsonl")
        for row_idx, row in enumerate(rows):
            task_id = str(row["task_id"])
            label_row = weak_labels_by_task.get(task_id)
            if label_row is None:
                continue
            split = split_lookup.get(task_id, str(row.get("split", "train")))
            feature_map = _weak_row_features(row)
            stats_vec = np.asarray([float(feature_map.get(key, 0.0)) for key in feature_keys], dtype=np.float32)
            group = grouped.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "split": split,
                    "y_true": float(label_row["difficulty"]),
                    "rollouts": [],
                },
            )
            sample_index = row.get("sample_index")
            if sample_index is None:
                sample_index = row_idx
            group["rollouts"].append(
                {
                    "rollout_row_index": int(sample_index),
                    "stats_vec": stats_vec,
                }
            )
    return [grouped[key] for key in sorted(grouped.keys())]


def _group_clean_rollouts(
    clean_rows: list[dict[str, Any]],
    feature_keys: list[str],
    clean_labels_by_task: dict[str, dict[str, Any]],
    allowed_splits: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in clean_rows:
        split = str(row.get("split", ""))
        if split not in allowed_splits:
            continue
        task_id = str(row["task_id"])
        label_row = clean_labels_by_task.get(task_id)
        if label_row is None:
            continue
        rollout_features = row.get("rollout_features") or {}
        stats_vec = np.asarray([float(rollout_features.get(key, 0.0)) for key in feature_keys], dtype=np.float32)
        group = grouped.setdefault(
            task_id,
            {
                "task_id": task_id,
                "split": split,
                "y_true": float(label_row["difficulty"]),
                "rollouts": [],
            },
        )
        group["rollouts"].append(
            {
                "rollout_row_index": int(row.get("rollout_row_index", len(group["rollouts"]))),
                "stats_vec": stats_vec,
            }
        )
    return [grouped[key] for key in sorted(grouped.keys())]


def _order_score(stats_vec: np.ndarray, feature_keys: list[str]) -> tuple[float, ...]:
    candidates = ["output_length", "reasoning_text_entropy", "answer_tokens", "output_text_entropy"]
    values = []
    for key in candidates:
        if key in feature_keys:
            values.append(float(stats_vec[feature_keys.index(key)]))
    if not values:
        values.append(float(np.sum(stats_vec)))
    return tuple(values)


def build_pair_rows(
    grouped_rows: list[dict[str, Any]],
    feature_keys: list[str],
    split_to_budget: dict[str, int],
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    pair_rows: list[dict[str, Any]] = []
    for group in grouped_rows:
        split = str(group["split"])
        pair_budget = int(split_to_budget.get(split, 0))
        rollouts = group["rollouts"]
        if pair_budget <= 0 or len(rollouts) < 2:
            continue
        all_pairs = list(itertools.combinations(range(len(rollouts)), 2))
        if pair_budget >= len(all_pairs):
            selected_pairs = all_pairs
        else:
            selected_idx = rng.choice(len(all_pairs), size=pair_budget, replace=False)
            selected_pairs = [all_pairs[int(idx)] for idx in np.sort(selected_idx)]
        for left_idx, right_idx in selected_pairs:
            left = rollouts[left_idx]
            right = rollouts[right_idx]
            left_vec = np.asarray(left["stats_vec"], dtype=np.float32)
            right_vec = np.asarray(right["stats_vec"], dtype=np.float32)
            if _order_score(left_vec, feature_keys) > _order_score(right_vec, feature_keys):
                left, right = right, left
                left_vec, right_vec = right_vec, left_vec
            pair_mean = (left_vec + right_vec) / 2.0
            pair_absdiff = np.abs(left_vec - right_vec)
            pair_min = np.minimum(left_vec, right_vec)
            pair_max = np.maximum(left_vec, right_vec)
            denom = np.maximum(pair_max, 1e-6)
            pair_rel_diff = pair_absdiff / denom
            cosine_num = float(np.dot(left_vec, right_vec))
            cosine_den = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec)) + 1e-8
            cosine = np.asarray([cosine_num / cosine_den], dtype=np.float32)
            l2 = np.asarray([float(np.linalg.norm(left_vec - right_vec))], dtype=np.float32)
            pair_rows.append(
                {
                    "task_id": str(group["task_id"]),
                    "split": split,
                    "y_true": float(group["y_true"]),
                    "pair_rollout_row_indices": [int(left["rollout_row_index"]), int(right["rollout_row_index"])],
                    "left_vec": left_vec,
                    "right_vec": right_vec,
                    "pair_mean": pair_mean,
                    "pair_absdiff": pair_absdiff,
                    "pair_min": pair_min,
                    "pair_max": pair_max,
                    "pair_rel_diff": pair_rel_diff,
                    "cosine": cosine,
                    "l2": l2,
                }
            )
    return pair_rows


def build_matrix(
    pair_rows: list[dict[str, Any]],
    prompt_lookup: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    split_rows: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for row in pair_rows:
        task_id = str(row["task_id"])
        prompt = prompt_lookup.get(task_id)
        if prompt is None:
            continue
        feature_row = np.concatenate(
            [
                np.asarray(prompt["prompt_feats"], dtype=np.float32),
                row["left_vec"],
                row["right_vec"],
                row["pair_mean"],
                row["pair_absdiff"],
                row["pair_min"],
                row["pair_max"],
                row["pair_rel_diff"],
                row["cosine"],
                row["l2"],
            ],
            axis=0,
        ).astype(np.float32)
        X_rows.append(feature_row)
        y_rows.append(float(row["y_true"]))
        split_rows.append(str(row["split"]))
        metadata_rows.append(
            {
                "task_id": task_id,
                "split": str(row["split"]),
                "pair_rollout_row_indices": row["pair_rollout_row_indices"],
                "y_true": float(row["y_true"]),
            }
        )
    return np.stack(X_rows), np.asarray(y_rows, dtype=np.float32), np.asarray(split_rows), metadata_rows


def _prompt_mean_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    metadata_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for meta, pred_val in zip(metadata_rows, pred.tolist()):
        task_id = str(meta["task_id"])
        groups[task_id]["y_true"].append(float(meta["y_true"]))
        groups[task_id]["y_pred"].append(float(pred_val))
    prompt_rows = []
    for task_id in sorted(groups):
        prompt_rows.append(
            {
                "task_id": task_id,
                "y_true": float(np.mean(groups[task_id]["y_true"])),
                "y_pred": float(np.mean(groups[task_id]["y_pred"])),
                "num_pairs": int(len(groups[task_id]["y_pred"])),
            }
        )
    prompt_true = np.asarray([row["y_true"] for row in prompt_rows], dtype=np.float32)
    prompt_pred = np.asarray([row["y_pred"] for row in prompt_rows], dtype=np.float32)
    return _reg_metrics(prompt_true, prompt_pred), prompt_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    weak_labels = load_records(args.weak_labels_path.expanduser().resolve())
    clean_labels = load_records(args.clean_labels_path.expanduser().resolve())
    weak_labels_by_task = {str(row["task_id"]): row for row in weak_labels}
    clean_labels_by_task = {str(row["task_id"]): row for row in clean_labels}
    split_lookup = _build_split_lookup(args.weak_prompt_dataset_dir.expanduser().resolve())

    clean_index_rows = load_records(args.clean_rollout_index_path.expanduser().resolve())
    feature_keys = [key for key in BASE_ROLLOUT_FEATURE_KEYS if any(key in (row.get("rollout_features") or {}) for row in clean_index_rows)]
    print(
        json.dumps(
            {
                "stage": "loaded_inputs",
                "num_weak_labels": len(weak_labels),
                "num_clean_labels": len(clean_labels),
                "num_clean_rollout_rows": len(clean_index_rows),
                "num_feature_keys": len(feature_keys),
            }
        ),
        flush=True,
    )

    weak_grouped = _group_weak_rollouts(
        weak_run_dirs=[path.expanduser().resolve() for path in args.weak_run_dirs],
        feature_keys=feature_keys,
        weak_labels_by_task=weak_labels_by_task,
        split_lookup=split_lookup,
    )
    clean_test_grouped = _group_clean_rollouts(
        clean_rows=clean_index_rows,
        feature_keys=feature_keys,
        clean_labels_by_task=clean_labels_by_task,
        allowed_splits={"test"},
    )
    print(
        json.dumps(
            {
                "stage": "grouped_rollouts",
                "num_weak_grouped_prompts": len(weak_grouped),
                "num_clean_test_grouped_prompts": len(clean_test_grouped),
            }
        ),
        flush=True,
    )

    weak_pair_rows = build_pair_rows(
        grouped_rows=weak_grouped,
        feature_keys=feature_keys,
        split_to_budget={"train": args.weak_pairs_per_prompt, "validation": args.weak_pairs_per_prompt},
        random_seed=args.random_seed,
    )
    clean_test_pair_rows = build_pair_rows(
        grouped_rows=clean_test_grouped,
        feature_keys=feature_keys,
        split_to_budget={"test": args.clean_test_pairs_per_prompt},
        random_seed=args.random_seed,
    )
    print(
        json.dumps(
            {
                "stage": "built_pair_rows",
                "num_weak_pair_rows": len(weak_pair_rows),
                "num_clean_test_pair_rows": len(clean_test_pair_rows),
            }
        ),
        flush=True,
    )

    prompt_lookup = {}
    prompt_lookup.update(_build_prompt_lookup_from_labels(weak_labels))
    prompt_lookup.update(_build_prompt_lookup_from_labels(clean_labels))

    weak_X, weak_y, weak_splits, weak_meta = build_matrix(weak_pair_rows, prompt_lookup)
    clean_test_X, clean_test_y, _, clean_test_meta = build_matrix(clean_test_pair_rows, prompt_lookup)

    train_mask = weak_splits == "train"
    val_mask = weak_splits == "validation"
    X_train, y_train = weak_X[train_mask], weak_y[train_mask]
    X_val, y_val = weak_X[val_mask], weak_y[val_mask]
    val_meta = [weak_meta[idx] for idx, keep in enumerate(val_mask.tolist()) if keep]
    print(
        json.dumps(
            {
                "stage": "built_matrices",
                "feature_dim": int(X_train.shape[1]),
                "num_train_rows": int(X_train.shape[0]),
                "num_val_rows": int(X_val.shape[0]),
                "num_clean_test_rows": int(clean_test_X.shape[0]),
            }
        ),
        flush=True,
    )

    candidates: list[tuple[str, Any]] = []
    for alpha in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0):
        candidates.append(
            (
                f"ridge_a{alpha:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", Ridge(alpha=alpha, random_state=args.random_seed)),
                    ]
                ),
            )
        )
    for n_estimators in (500, 1000, 2000):
        for min_samples_leaf in (3, 5, 7):
            for max_features in (0.3, 0.5, 0.7):
                candidates.append(
                    (
                        f"et_n{n_estimators}_l{min_samples_leaf}_mf{max_features}",
                        ExtraTreesRegressor(
                            n_estimators=n_estimators,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            random_state=args.random_seed,
                            n_jobs=8,
                        ),
                    )
                )

    results = []
    best_name = None
    best_model = None
    best_val_r2 = -1e18

    for name, model in candidates:
        model.fit(X_train, y_train)
        val_pred = np.clip(np.asarray(model.predict(X_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
        val_row_metrics = _reg_metrics(y_val, val_pred)
        val_prompt_metrics, _ = _prompt_mean_metrics(y_val, val_pred, val_meta)
        result = {
            "name": name,
            "val_row_metrics": val_row_metrics,
            "val_prompt_mean_metrics": val_prompt_metrics,
        }
        results.append(result)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(json.dumps({"candidate": name, "val_prompt_r2": val_prompt_metrics["r2"]}), flush=True)
        if val_prompt_metrics["r2"] > best_val_r2:
            best_val_r2 = val_prompt_metrics["r2"]
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None
    test_pred = np.clip(np.asarray(best_model.predict(clean_test_X), dtype=np.float32).reshape(-1), 0.0, 1.0)
    test_row_metrics = _reg_metrics(clean_test_y, test_pred)
    test_prompt_metrics, prompt_rows = _prompt_mean_metrics(clean_test_y, test_pred, clean_test_meta)

    predictions = []
    prompt_pred_by_task = {row["task_id"]: row["y_pred"] for row in prompt_rows}
    for row in prompt_rows:
        label_row = clean_labels_by_task[str(row["task_id"])]
        predictions.append(
            {
                "task_id": str(row["task_id"]),
                "user_input": str(label_row.get("user_input", "")),
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
                "num_pairs": int(row["num_pairs"]),
            }
        )
    write_jsonl(args.output_dir / "predictions_test.jsonl", predictions)

    summary = {
        "setting": "weak_train_clean_test_text_pair_transfer",
        "weak_prompt_dataset_dir": str(args.weak_prompt_dataset_dir.expanduser().resolve()),
        "weak_labels_path": str(args.weak_labels_path.expanduser().resolve()),
        "clean_rollout_index_path": str(args.clean_rollout_index_path.expanduser().resolve()),
        "clean_labels_path": str(args.clean_labels_path.expanduser().resolve()),
        "num_weak_train_rows": int(X_train.shape[0]),
        "num_weak_val_rows": int(X_val.shape[0]),
        "num_clean_test_rows": int(clean_test_X.shape[0]),
        "num_clean_test_prompts": int(len(prompt_rows)),
        "feature_dim": int(X_train.shape[1]),
        "best_model": best_name,
        "best_val_prompt_mean_metrics": next(row["val_prompt_mean_metrics"] for row in results if row["name"] == best_name),
        "test_row_metrics": test_row_metrics,
        "test_prompt_mean_metrics": test_prompt_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
