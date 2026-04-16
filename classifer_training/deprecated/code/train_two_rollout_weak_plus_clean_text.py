from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.train_two_rollout_weak_transfer_text import (
    BASE_ROLLOUT_FEATURE_KEYS,
    _build_prompt_lookup_from_labels,
    _build_split_lookup,
    _group_clean_rollouts,
    _group_weak_rollouts,
    _prompt_mean_metrics,
    _reg_metrics,
    build_matrix,
    build_pair_rows,
)
from classifer_training.utils import load_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with weak + clean train/validation data, evaluate on the original clean test split.")
    parser.add_argument("--weak_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--weak_labels_path", type=Path, required=True)
    parser.add_argument("--clean_rollout_index_path", type=Path, required=True)
    parser.add_argument("--clean_labels_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--weak_pairs_per_prompt", type=int, default=6)
    parser.add_argument("--clean_train_pairs_per_prompt", type=int, default=4)
    parser.add_argument("--clean_validation_pairs_per_prompt", type=int, default=4)
    parser.add_argument("--clean_test_pairs_per_prompt", type=int, default=10)
    return parser.parse_args()


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
    clean_grouped = _group_clean_rollouts(
        clean_rows=clean_index_rows,
        feature_keys=feature_keys,
        clean_labels_by_task=clean_labels_by_task,
        allowed_splits={"train", "validation", "test"},
    )
    print(
        json.dumps(
            {
                "stage": "grouped_rollouts",
                "num_weak_grouped_prompts": len(weak_grouped),
                "num_clean_grouped_prompts": len(clean_grouped),
            }
        ),
        flush=True,
    )

    weak_pair_rows = build_pair_rows(
        grouped_rows=weak_grouped,
        feature_keys=feature_keys,
        split_to_budget={"train": args.weak_pairs_per_prompt},
        random_seed=args.random_seed,
    )
    clean_pair_rows = build_pair_rows(
        grouped_rows=clean_grouped,
        feature_keys=feature_keys,
        split_to_budget={
            "train": args.clean_train_pairs_per_prompt,
            "validation": args.clean_validation_pairs_per_prompt,
            "test": args.clean_test_pairs_per_prompt,
        },
        random_seed=args.random_seed,
    )
    print(
        json.dumps(
            {
                "stage": "built_pair_rows",
                "num_weak_pair_rows": len(weak_pair_rows),
                "num_clean_pair_rows": len(clean_pair_rows),
            }
        ),
        flush=True,
    )

    prompt_lookup = {}
    prompt_lookup.update(_build_prompt_lookup_from_labels(weak_labels))
    prompt_lookup.update(_build_prompt_lookup_from_labels(clean_labels))

    weak_X, weak_y, weak_splits, weak_meta = build_matrix(weak_pair_rows, prompt_lookup)
    clean_X, clean_y, clean_splits, clean_meta = build_matrix(clean_pair_rows, prompt_lookup)

    train_mask_clean = clean_splits == "train"
    val_mask_clean = clean_splits == "validation"
    test_mask_clean = clean_splits == "test"

    X_train = np.concatenate([weak_X, clean_X[train_mask_clean]], axis=0)
    y_train = np.concatenate([weak_y, clean_y[train_mask_clean]], axis=0)
    X_val = clean_X[val_mask_clean]
    y_val = clean_y[val_mask_clean]
    X_test = clean_X[test_mask_clean]
    y_test = clean_y[test_mask_clean]
    val_meta = [clean_meta[idx] for idx, keep in enumerate(val_mask_clean.tolist()) if keep]
    test_meta = [clean_meta[idx] for idx, keep in enumerate(test_mask_clean.tolist()) if keep]
    print(
        json.dumps(
            {
                "stage": "built_matrices",
                "feature_dim": int(X_train.shape[1]),
                "num_train_rows": int(X_train.shape[0]),
                "num_val_rows": int(X_val.shape[0]),
                "num_test_rows": int(X_test.shape[0]),
            }
        ),
        flush=True,
    )

    candidates: list[tuple[str, object]] = []
    for alpha in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
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
        for min_samples_leaf in (5, 7):
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

    best_name = None
    best_model = None
    best_val_r2 = -1e18
    results = []
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
    test_pred = np.clip(np.asarray(best_model.predict(X_test), dtype=np.float32).reshape(-1), 0.0, 1.0)
    test_row_metrics = _reg_metrics(y_test, test_pred)
    test_prompt_metrics, prompt_rows = _prompt_mean_metrics(y_test, test_pred, test_meta)

    predictions = []
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
        "setting": "weak_plus_clean_trainval_text_pair_transfer",
        "num_train_rows": int(X_train.shape[0]),
        "num_val_rows": int(X_val.shape[0]),
        "num_test_rows": int(X_test.shape[0]),
        "num_test_prompts": int(len(prompt_rows)),
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
