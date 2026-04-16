from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from classifer_training.search_two_rollout_finished16_focus import (
    _build_label_buckets,
    _load_grouped_rows,
    _parse_prompt_config,
)
from classifer_training.train_prompt_two_trajectory_promptsearch import (
    build_matrix,
    build_pair_rows,
    build_prompt_lookup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the finished16 2-rollout probe on all labeled prompts.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--rollout_index_path", type=Path, required=True)
    parser.add_argument("--prompt_hidden_dir", type=Path, required=True)
    parser.add_argument("--prompt_index_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_config", type=str, default="last6:l10_l26")
    parser.add_argument("--train_pairs_per_prompt", type=int, default=4)
    parser.add_argument("--test_pairs_per_prompt", type=int, default=10)
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--max_features", type=float, default=0.7)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=12)
    return parser.parse_args()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _make_groups_with_split(grouped_rows: list[dict], test_task_ids: set[str]) -> list[dict]:
    updated: list[dict] = []
    for row in grouped_rows:
        new_row = dict(row)
        new_row["rollouts"] = row["rollouts"]
        new_row["split"] = "test" if str(row["task_id"]) in test_task_ids else "train"
        updated.append(new_row)
    return updated


def _fit_prompt_predictions(
    grouped_rows: list[dict],
    feature_keys: list[str],
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    prompt_mode: str,
    train_pairs_per_prompt: int,
    test_pairs_per_prompt: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: float,
    random_seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pair_rows = build_pair_rows(
        grouped_rows=grouped_rows,
        feature_keys=feature_keys,
        train_splits={"train"},
        test_splits={"test"},
        train_pairs_per_prompt=train_pairs_per_prompt,
        test_pairs_per_prompt=test_pairs_per_prompt,
        random_seed=random_seed,
    )
    X, y, splits, metas = build_matrix(pair_rows, prompt_lookup, prompt_mode)
    train_mask = splits == "train"
    test_mask = splits == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[test_mask]
    test_metas = [metas[idx] for idx, keep in enumerate(test_mask.tolist()) if keep]

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    pred_test = np.clip(np.asarray(model.predict(X_test), dtype=np.float32).reshape(-1), 0.0, 1.0)

    prompt_true: dict[str, list[float]] = {}
    prompt_pred: dict[str, list[float]] = {}
    for meta, pred in zip(test_metas, pred_test.tolist()):
        task_id = str(meta["task_id"])
        prompt_true.setdefault(task_id, []).append(float(meta["y_true"]))
        prompt_pred.setdefault(task_id, []).append(float(pred))
    task_ids = sorted(prompt_true.keys())
    y_true = np.asarray([float(np.mean(prompt_true[task_id])) for task_id in task_ids], dtype=np.float32)
    y_pred = np.asarray([float(np.mean(prompt_pred[task_id])) for task_id in task_ids], dtype=np.float32)
    return y_true, y_pred, task_ids


def _fit_full_model(
    grouped_rows: list[dict],
    feature_keys: list[str],
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    prompt_mode: str,
    train_pairs_per_prompt: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: float,
    random_seed: int,
    n_jobs: int,
) -> tuple[ExtraTreesRegressor, int, int]:
    updated_rows = []
    for row in grouped_rows:
        new_row = dict(row)
        new_row["rollouts"] = row["rollouts"]
        new_row["split"] = "train"
        updated_rows.append(new_row)
    pair_rows = build_pair_rows(
        grouped_rows=updated_rows,
        feature_keys=feature_keys,
        train_splits={"train"},
        test_splits=set(),
        train_pairs_per_prompt=train_pairs_per_prompt,
        test_pairs_per_prompt=0,
        random_seed=random_seed,
    )
    X, y, _, _ = build_matrix(pair_rows, prompt_lookup, prompt_mode)
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    model.fit(X, y)
    return model, int(X.shape[0]), int(X.shape[1])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = args.run_root.expanduser().resolve().parents[4]
    _, pooled_hidden_dir, pooled_index_dir, prompt_mode = _parse_prompt_config(args.prompt_config, repo_root)

    label_buckets = _build_label_buckets(args.run_root.expanduser().resolve())
    grouped_rows, feature_keys = _load_grouped_rows(args.rollout_index_path.expanduser().resolve(), label_buckets)
    prompt_lookup = build_prompt_lookup(
        args.prompt_hidden_dir.expanduser().resolve(),
        args.prompt_index_dir.expanduser().resolve(),
        pooled_hidden_dir.expanduser().resolve(),
        pooled_index_dir.expanduser().resolve(),
    )

    task_ids = np.asarray([str(row["task_id"]) for row in grouped_rows])
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)

    oof_pred: dict[str, float] = {}
    oof_true: dict[str, float] = {}
    fold_results = []

    for fold_idx, (_, test_idx) in enumerate(kf.split(task_ids), start=1):
        test_task_ids = {str(task_ids[idx]) for idx in test_idx.tolist()}
        fold_rows = _make_groups_with_split(grouped_rows, test_task_ids)
        y_true, y_pred, fold_task_ids = _fit_prompt_predictions(
            grouped_rows=fold_rows,
            feature_keys=feature_keys,
            prompt_lookup=prompt_lookup,
            prompt_mode=prompt_mode,
            train_pairs_per_prompt=args.train_pairs_per_prompt,
            test_pairs_per_prompt=args.test_pairs_per_prompt,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
        )
        for task_id, target, pred in zip(fold_task_ids, y_true.tolist(), y_pred.tolist()):
            oof_true[task_id] = float(target)
            oof_pred[task_id] = float(pred)
        fold_results.append(
            {
                "fold": fold_idx,
                "num_test_prompts": len(fold_task_ids),
                "metrics": _metrics(y_true, y_pred),
            }
        )

    ordered_task_ids = sorted(oof_true.keys())
    oof_true_arr = np.asarray([oof_true[task_id] for task_id in ordered_task_ids], dtype=np.float32)
    oof_pred_arr = np.asarray([oof_pred[task_id] for task_id in ordered_task_ids], dtype=np.float32)
    cv_metrics = _metrics(oof_true_arr, oof_pred_arr)

    model, num_train_rows, feature_dim = _fit_full_model(
        grouped_rows=grouped_rows,
        feature_keys=feature_keys,
        prompt_lookup=prompt_lookup,
        prompt_mode=prompt_mode,
        train_pairs_per_prompt=args.train_pairs_per_prompt,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )

    summary = {
        "setting": "two_rollout_finished16_full_data",
        "prompt_config": args.prompt_config,
        "prompt_mode": prompt_mode,
        "params": {
            "train_pairs_per_prompt": args.train_pairs_per_prompt,
            "test_pairs_per_prompt": args.test_pairs_per_prompt,
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "random_seed": args.random_seed,
            "cv_folds": args.cv_folds,
        },
        "oof_cv_metrics": cv_metrics,
        "fold_results": fold_results,
        "full_fit": {
            "num_prompts": len(grouped_rows),
            "num_train_rows": num_train_rows,
            "feature_dim": feature_dim,
        },
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "oof_predictions.jsonl").open("w", encoding="utf-8") as f:
        for task_id in ordered_task_ids:
            f.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "y_true": oof_true[task_id],
                        "y_pred": oof_pred[task_id],
                    }
                )
                + "\n"
            )
    with (args.output_dir / "model.pkl").open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "prompt_config": args.prompt_config,
                "prompt_mode": prompt_mode,
                "feature_keys": feature_keys,
                "params": summary["params"],
            },
            f,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
