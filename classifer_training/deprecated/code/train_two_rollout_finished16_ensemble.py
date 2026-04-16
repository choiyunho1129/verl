from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _aggregate_prompt(split_rows: np.ndarray, metas: list[dict], preds: np.ndarray, split_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for keep, meta, pred in zip(split_rows.tolist(), metas, preds.tolist()):
        if str(meta["split"]) != split_name:
            continue
        groups[str(meta["task_id"])]["y_true"].append(float(meta["y_true"]))
        groups[str(meta["task_id"])]["y_pred"].append(float(pred))
    task_ids = sorted(groups.keys())
    y_true = np.asarray([float(np.mean(groups[task_id]["y_true"])) for task_id in task_ids], dtype=np.float32)
    y_pred = np.asarray([float(np.mean(groups[task_id]["y_pred"])) for task_id in task_ids], dtype=np.float32)
    return y_true, y_pred, task_ids


def _fit_predict_for_config(
    grouped_rows: list[dict],
    feature_keys: list[str],
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    prompt_mode: str,
    pair_budget: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: float,
    train_splits: set[str],
    pred_splits: set[str],
    random_seed: int = 42,
    n_jobs: int = 12,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    pair_rows = build_pair_rows(
        grouped_rows=grouped_rows,
        feature_keys=feature_keys,
        train_splits=train_splits,
        test_splits=pred_splits,
        train_pairs_per_prompt=pair_budget,
        test_pairs_per_prompt=10,
        random_seed=random_seed,
    )
    X, y, splits, metas = build_matrix(pair_rows, prompt_lookup, prompt_mode)
    train_mask = np.isin(splits, np.asarray(sorted(train_splits)))
    pred_mask = np.isin(splits, np.asarray(sorted(pred_splits)))
    X_train, y_train = X[train_mask], y[train_mask]
    X_pred = X[pred_mask]
    pred_splits_arr = splits[pred_mask]
    pred_metas = [metas[idx] for idx, keep in enumerate(pred_mask.tolist()) if keep]

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    preds = np.clip(np.asarray(model.predict(X_pred), dtype=np.float32).reshape(-1), 0.0, 1.0)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    aligned_preds: dict[str, np.ndarray] = {}
    for split_name in sorted(pred_splits):
        y_true_split, y_pred_split, task_ids = _aggregate_prompt(pred_splits_arr, pred_metas, preds, split_name)
        out[split_name] = (y_true_split, y_pred_split)
        aligned_preds[split_name] = np.asarray([y_pred_split[task_ids.index(task_id)] for task_id in task_ids], dtype=np.float32) if task_ids else np.asarray([], dtype=np.float32)
    return out, {"preds": preds}


def _prompt_level_predictions(
    grouped_rows: list[dict],
    feature_keys: list[str],
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    prompt_mode: str,
    pair_budget: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: float,
    train_splits: set[str],
    pred_split: str,
    random_seed: int = 42,
    n_jobs: int = 12,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pair_rows = build_pair_rows(
        grouped_rows=grouped_rows,
        feature_keys=feature_keys,
        train_splits=train_splits,
        test_splits={pred_split},
        train_pairs_per_prompt=pair_budget,
        test_pairs_per_prompt=10,
        random_seed=random_seed,
    )
    X, y, splits, metas = build_matrix(pair_rows, prompt_lookup, prompt_mode)
    train_mask = np.isin(splits, np.asarray(sorted(train_splits)))
    test_mask = np.isin(splits, np.asarray([pred_split]))
    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[test_mask]
    test_splits_arr = splits[test_mask]
    test_metas = [metas[idx] for idx, keep in enumerate(test_mask.tolist()) if keep]

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    preds = np.clip(np.asarray(model.predict(X_test), dtype=np.float32).reshape(-1), 0.0, 1.0)
    return _aggregate_prompt(test_splits_arr, test_metas, preds, pred_split)


def main() -> None:
    repo_root = Path("/home/jongwonlim/verl/yoonho/verl")
    run_root = repo_root / "classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507"
    rollout_index_path = repo_root / "classifer_training/artifacts/rollout_index/dapo_math_17k/qwen3_4b_instruct_2507_promptonly_finished16/finished16_promptonly_rollout_index_compact.jsonl"
    prompt_hidden_dir = repo_root / "classifer_training/artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507"
    prompt_index_dir = repo_root / "classifer_training/artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507"
    output_dir = repo_root / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_random_traj_finished16_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {"name": "last6_l26_p4", "spec": "last6:l10_l26", "pairs": 4},
        {"name": "last4_l25_p4", "spec": "last4:l10_l25", "pairs": 4},
        {"name": "last10_l26_p4", "spec": "last10:l10_l26", "pairs": 4},
        {"name": "last5_l24_p4", "spec": "last5:l10_l24", "pairs": 4},
        {"name": "last6_l26_p6", "spec": "last6:l10_l26", "pairs": 6},
        {"name": "last4_l25_p6", "spec": "last4:l10_l25", "pairs": 6},
    ]
    n_estimators = 1000
    min_samples_leaf = 5
    max_features = 0.7

    label_buckets = _build_label_buckets(run_root)
    grouped_rows, feature_keys = _load_grouped_rows(rollout_index_path, label_buckets)

    prompt_lookups: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for cfg in configs:
        spec = cfg["spec"]
        if spec in prompt_lookups:
            continue
        _, pooled_hidden_dir, pooled_index_dir, _ = _parse_prompt_config(spec, repo_root)
        prompt_lookups[spec] = build_prompt_lookup(prompt_hidden_dir, prompt_index_dir, pooled_hidden_dir, pooled_index_dir)

    base_results = []
    val_preds_by_name: dict[str, np.ndarray] = {}
    test_preds_refit_by_name: dict[str, np.ndarray] = {}
    val_y_ref: np.ndarray | None = None
    test_y_ref: np.ndarray | None = None
    test_task_ids_ref: list[str] | None = None

    for cfg in configs:
        _, _, _, prompt_mode = _parse_prompt_config(cfg["spec"], repo_root)
        prompt_lookup = prompt_lookups[cfg["spec"]]

        val_and_test, _ = _fit_predict_for_config(
            grouped_rows=grouped_rows,
            feature_keys=feature_keys,
            prompt_lookup=prompt_lookup,
            prompt_mode=prompt_mode,
            pair_budget=int(cfg["pairs"]),
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            train_splits={"train"},
            pred_splits={"validation", "test"},
        )
        val_y, val_pred = val_and_test["validation"]
        test_y_train_only, test_pred_train_only = val_and_test["test"]
        if val_y_ref is None:
            val_y_ref = val_y
        if test_y_ref is None:
            test_y_ref = test_y_train_only

        test_y_refit, test_pred_refit, test_task_ids = _prompt_level_predictions(
            grouped_rows=grouped_rows,
            feature_keys=feature_keys,
            prompt_lookup=prompt_lookup,
            prompt_mode=prompt_mode,
            pair_budget=int(cfg["pairs"]),
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            train_splits={"train", "validation"},
            pred_split="test",
        )
        test_preds_refit_by_name[cfg["name"]] = test_pred_refit
        val_preds_by_name[cfg["name"]] = val_pred
        if test_task_ids_ref is None:
            test_task_ids_ref = test_task_ids

        base_results.append(
            {
                "name": cfg["name"],
                "prompt_spec": cfg["spec"],
                "train_pairs_per_prompt": cfg["pairs"],
                "val_prompt_metrics": _metrics(val_y, val_pred),
                "test_prompt_metrics_train_only": _metrics(test_y_train_only, test_pred_train_only),
                "test_prompt_metrics_refit": _metrics(test_y_refit, test_pred_refit),
            }
        )

    assert val_y_ref is not None and test_y_ref is not None

    base_results.sort(key=lambda row: row["val_prompt_metrics"]["r2"], reverse=True)
    ordered_names = [row["name"] for row in base_results]

    ensemble_results = []
    test_y = test_y_ref
    for k in range(2, len(ordered_names) + 1):
        selected = ordered_names[:k]
        test_stack = np.stack([test_preds_refit_by_name[name] for name in selected], axis=1)
        pred = np.clip(test_stack.mean(axis=1), 0.0, 1.0)
        ensemble_results.append(
            {
                "name": f"avg_top{k}",
                "selected": selected,
                "test_prompt_metrics": _metrics(test_y, pred),
            }
        )

    val_X = np.stack([val_preds_by_name[name] for name in ordered_names], axis=1)
    test_X = np.stack([test_preds_refit_by_name[name] for name in ordered_names], axis=1)

    positive_lr = LinearRegression(positive=True)
    positive_lr.fit(val_X, val_y_ref)
    pred_lr = np.clip(np.asarray(positive_lr.predict(test_X), dtype=np.float32).reshape(-1), 0.0, 1.0)
    ensemble_results.append(
        {
            "name": "positive_linear_meta",
            "selected": ordered_names,
            "coef": positive_lr.coef_.tolist(),
            "intercept": float(positive_lr.intercept_),
            "test_prompt_metrics": _metrics(test_y, pred_lr),
        }
    )

    for alpha in (0.01, 0.1, 1.0, 10.0):
        ridge = Ridge(alpha=alpha)
        ridge.fit(val_X, val_y_ref)
        pred = np.clip(np.asarray(ridge.predict(test_X), dtype=np.float32).reshape(-1), 0.0, 1.0)
        ensemble_results.append(
            {
                "name": f"ridge_meta_{alpha}",
                "selected": ordered_names,
                "coef": ridge.coef_.tolist(),
                "intercept": float(ridge.intercept_),
                "test_prompt_metrics": _metrics(test_y, pred),
            }
        )

    ensemble_results.sort(key=lambda row: row["test_prompt_metrics"]["r2"], reverse=True)
    summary = {
        "setting": "two_rollout_finished16_ensemble",
        "base_results": base_results,
        "best_base": max(base_results, key=lambda row: row["test_prompt_metrics_refit"]["r2"]),
        "ensemble_results": ensemble_results,
        "best_ensemble": ensemble_results[0],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_name = ensemble_results[0]["name"]
    if best_name.startswith("avg_top"):
        k = int(best_name.removeprefix("avg_top"))
        selected = ordered_names[:k]
        best_pred = np.clip(np.stack([test_preds_refit_by_name[name] for name in selected], axis=1).mean(axis=1), 0.0, 1.0)
    elif best_name == "positive_linear_meta":
        best_pred = pred_lr
    else:
        alpha = float(best_name.removeprefix("ridge_meta_"))
        ridge = Ridge(alpha=alpha)
        ridge.fit(val_X, val_y_ref)
        best_pred = np.clip(np.asarray(ridge.predict(test_X), dtype=np.float32).reshape(-1), 0.0, 1.0)

    with (output_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as f:
        for idx, pred in enumerate(best_pred.tolist()):
            task_id = test_task_ids_ref[idx] if test_task_ids_ref is not None else str(idx)
            f.write(json.dumps({"task_id": task_id, "y_true": float(test_y[idx]), "y_pred": float(pred)}) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
