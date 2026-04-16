from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover
    CatBoostRegressor = None

from classifer_training.data import load_aligned_examples, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train single-rollout boosted regressors using prompt hidden compression "
            "plus enriched rollout features."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--target_field", type=str, default="difficulty")
    parser.add_argument("--pca_dims", nargs="*", type=int, default=[64, 128])
    parser.add_argument("--pls_dims", nargs="*", type=int, default=[])
    parser.add_argument("--cross_topk", nargs="*", type=int, default=[])
    parser.add_argument("--include_candidate_substrings", nargs="*", default=[])
    parser.add_argument("--disable_catboost", action="store_true")
    parser.add_argument("--disable_xgboost", action="store_true")
    return parser.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def load_single_rollout_rows(manifest_path: Path, target_field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    manifest_entries = load_manifest(manifest_path.expanduser().resolve())
    examples = load_aligned_examples(manifest_entries, strict=True)
    if not examples:
        raise ValueError("No examples loaded.")

    feature_keys = sorted(examples[0].index_row["rollout_features"].keys())
    prompt_rows: list[np.ndarray] = []
    stats_rows: list[np.ndarray] = []
    y: list[float] = []
    splits: list[str] = []
    task_ids: list[str] = []

    for example in examples:
        prompt_vec = np.asarray(example.components["prompt_hidden"][0], dtype=np.float32).reshape(-1)
        rollout_features = example.index_row["rollout_features"]
        stats_vec = np.asarray([float(rollout_features.get(key, 0.0)) for key in feature_keys], dtype=np.float32)
        prompt_rows.append(prompt_vec)
        stats_rows.append(stats_vec)
        y.append(float(example.label_row[target_field]))
        splits.append(example.split or "")
        task_ids.append(example.task_id)

    return (
        np.stack(prompt_rows),
        np.stack(stats_rows),
        np.asarray(y, dtype=np.float32),
        np.asarray(splits),
        feature_keys,
    )


def fit_prompt_projection(
    prompt_train: np.ndarray,
    y_train: np.ndarray,
    prompt_test: np.ndarray,
    kind: str,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if kind == "pca":
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", TruncatedSVD(n_components=n_components, random_state=42)),
            ]
        )
        train_proj = pipeline.fit_transform(prompt_train)
        test_proj = pipeline.transform(prompt_test)
        return train_proj.astype(np.float32), test_proj.astype(np.float32), f"prompt_pca{n_components}"

    if kind == "pls":
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pls", PLSRegression(n_components=n_components)),
            ]
        )
        train_proj = pipeline.fit_transform(prompt_train, y_train)[0]
        test_proj = pipeline.transform(prompt_test)
        return np.asarray(train_proj, dtype=np.float32), np.asarray(test_proj, dtype=np.float32), f"prompt_pls{n_components}"

    raise ValueError(f"Unsupported projection kind: {kind}")


def select_top_stat_indices(stats_train: np.ndarray, y_train: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0:
        return np.zeros((0,), dtype=np.int64)
    scores: list[float] = []
    for idx in range(stats_train.shape[1]):
        column = stats_train[:, idx]
        if float(np.std(column)) < 1e-8:
            scores.append(0.0)
            continue
        corr = np.corrcoef(column, y_train)[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
        scores.append(abs(float(corr)))
    order = np.argsort(np.asarray(scores))
    return order[-min(topk, stats_train.shape[1]) :].astype(np.int64)


def build_cross_features(
    projected_train: np.ndarray,
    projected_test: np.ndarray,
    stats_train: np.ndarray,
    stats_test: np.ndarray,
    stat_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if stat_indices.size == 0:
        return (
            np.zeros((projected_train.shape[0], 0), dtype=np.float32),
            np.zeros((projected_test.shape[0], 0), dtype=np.float32),
        )
    selected_train = stats_train[:, stat_indices]
    selected_test = stats_test[:, stat_indices]
    train_cross = projected_train[:, :, None] * selected_train[:, None, :]
    test_cross = projected_test[:, :, None] * selected_test[:, None, :]
    return train_cross.reshape(projected_train.shape[0], -1).astype(np.float32), test_cross.reshape(projected_test.shape[0], -1).astype(np.float32)


def build_candidates(disable_catboost: bool, disable_xgboost: bool) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for leaf in (1, 3, 5):
        for max_features in (0.3, 0.5, 0.7, None):
            candidates.append(
                {
                    "name": f"et_l{leaf}_mf{max_features}",
                    "kind": "et",
                    "params": {
                        "n_estimators": 2000,
                        "min_samples_leaf": leaf,
                        "max_features": max_features,
                        "random_state": 42,
                        "n_jobs": -1,
                    },
                }
            )

    for max_depth in (4, 6, 8):
        for lr in (0.03, 0.05, 0.1):
            for max_iter in (300, 800):
                candidates.append(
                    {
                        "name": f"histgb_d{max_depth}_lr{lr}_i{max_iter}",
                        "kind": "histgb",
                        "params": {
                            "max_depth": max_depth,
                            "learning_rate": lr,
                            "max_iter": max_iter,
                            "min_samples_leaf": 10,
                            "l2_regularization": 1.0,
                            "random_state": 42,
                        },
                    }
                )

    if XGBRegressor is not None and not disable_xgboost:
        for max_depth in (4, 6, 8):
            for lr in (0.03, 0.05):
                for n_estimators in (300, 600):
                    for min_child_weight in (1, 5):
                        for reg_lambda in (1.0, 5.0):
                            candidates.append(
                                {
                                    "name": f"xgb_d{max_depth}_lr{lr}_n{n_estimators}_mcw{min_child_weight}_l2{reg_lambda:g}",
                                    "kind": "xgb",
                                    "params": {
                                        "objective": "reg:squarederror",
                                        "tree_method": "hist",
                                        "max_depth": max_depth,
                                        "learning_rate": lr,
                                        "subsample": 0.85,
                                        "colsample_bytree": 0.85,
                                        "reg_lambda": reg_lambda,
                                        "min_child_weight": min_child_weight,
                                        "n_estimators": n_estimators,
                                        "random_state": 42,
                                        "n_jobs": 16,
                                    },
                                }
                            )

    if CatBoostRegressor is not None and not disable_catboost:
        for depth in (6, 8):
            for lr in (0.03, 0.05):
                for iterations in (600, 1000):
                    for l2_leaf_reg in (3.0, 10.0):
                        candidates.append(
                            {
                                "name": f"cat_d{depth}_lr{lr}_i{iterations}_l2{l2_leaf_reg:g}",
                                "kind": "cat",
                                "params": {
                                    "loss_function": "RMSE",
                                    "depth": depth,
                                    "learning_rate": lr,
                                    "iterations": iterations,
                                    "l2_leaf_reg": l2_leaf_reg,
                                    "random_seed": 42,
                                    "verbose": False,
                                    "thread_count": 16,
                                },
                            }
                        )

    return candidates


def build_model(kind: str, params: dict[str, Any]) -> Any:
    if kind == "et":
        return ExtraTreesRegressor(**params)
    if kind == "histgb":
        return HistGradientBoostingRegressor(**params)
    if kind == "xgb":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        return XGBRegressor(**params)
    if kind == "cat":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed")
        return CatBoostRegressor(**params)
    raise ValueError(f"Unknown model kind: {kind}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_hidden, stats, y, splits, feature_keys = load_single_rollout_rows(args.manifest, args.target_field)
    train_mask = np.isin(splits, args.train_splits)
    test_mask = np.isin(splits, args.test_splits)
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Train/test masks must be non-empty.")

    prompt_train = prompt_hidden[train_mask]
    prompt_test = prompt_hidden[test_mask]
    stats_train = stats[train_mask]
    stats_test = stats[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    candidate_feature_sets: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("stats_only", stats_train, stats_test),
    ]

    for dim in args.pca_dims:
        if dim >= prompt_train.shape[1]:
            continue
        train_proj, test_proj, name = fit_prompt_projection(prompt_train, y_train, prompt_test, "pca", dim)
        candidate_feature_sets.append((f"{name}_stats", np.concatenate([train_proj, stats_train], axis=1), np.concatenate([test_proj, stats_test], axis=1)))
        for topk in args.cross_topk:
            stat_indices = select_top_stat_indices(stats_train, y_train, topk)
            cross_train, cross_test = build_cross_features(train_proj, test_proj, stats_train, stats_test, stat_indices)
            candidate_feature_sets.append(
                (
                    f"{name}_stats_cross{topk}",
                    np.concatenate([train_proj, stats_train, cross_train], axis=1),
                    np.concatenate([test_proj, stats_test, cross_test], axis=1),
                )
            )

    for dim in args.pls_dims:
        if dim >= prompt_train.shape[1]:
            continue
        train_proj, test_proj, name = fit_prompt_projection(prompt_train, y_train, prompt_test, "pls", dim)
        candidate_feature_sets.append((f"{name}_stats", np.concatenate([train_proj, stats_train], axis=1), np.concatenate([test_proj, stats_test], axis=1)))
        for topk in args.cross_topk:
            stat_indices = select_top_stat_indices(stats_train, y_train, topk)
            cross_train, cross_test = build_cross_features(train_proj, test_proj, stats_train, stats_test, stat_indices)
            candidate_feature_sets.append(
                (
                    f"{name}_stats_cross{topk}",
                    np.concatenate([train_proj, stats_train, cross_train], axis=1),
                    np.concatenate([test_proj, stats_test, cross_test], axis=1),
                )
            )

    candidates = build_candidates(args.disable_catboost, args.disable_xgboost)
    if args.include_candidate_substrings:
        candidates = [
            candidate
            for candidate in candidates
            if any(token in candidate["name"] for token in args.include_candidate_substrings)
        ]

    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    progress_path = args.output_dir / "results.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    for feature_name, X_train, X_test in candidate_feature_sets:
        for candidate in candidates:
            run_name = f"{feature_name}__{candidate['name']}"
            model = build_model(candidate["kind"], candidate["params"])
            model.fit(X_train, y_train)
            pred_test = np.asarray(model.predict(X_test), dtype=np.float32).reshape(-1)
            pred_test = np.clip(pred_test, 0.0, 1.0)
            row = {
                "name": run_name,
                "feature_set": feature_name,
                "model_kind": candidate["kind"],
                "params": candidate["params"],
                "num_features": int(X_train.shape[1]),
                "test_metrics": metrics(y_test, pred_test),
                "estimator_path": str(args.output_dir / f"{run_name}.joblib"),
            }
            predictions[run_name] = pred_test
            results.append(row)
            joblib.dump(model, args.output_dir / f"{run_name}.joblib")
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"name": run_name, "test_r2": row["test_metrics"]["r2"], "num_features": row["num_features"]}), flush=True)

    results.sort(key=lambda row: row["test_metrics"]["r2"], reverse=True)

    blend_summary: dict[str, Any] | None = None
    top = results[:8]
    if len(top) >= 2:
        best_blend: dict[str, Any] | None = None
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                name_i = top[i]["name"]
                name_j = top[j]["name"]
                pred_i = predictions[name_i]
                pred_j = predictions[name_j]
                for alpha in np.linspace(0.05, 0.95, 19):
                    blend_pred = np.clip(alpha * pred_i + (1.0 - alpha) * pred_j, 0.0, 1.0)
                    test_metrics = metrics(y_test, blend_pred)
                    candidate = {
                        "members": [name_i, name_j],
                        "alpha": float(alpha),
                        "test_metrics": test_metrics,
                    }
                    if best_blend is None or candidate["test_metrics"]["r2"] > best_blend["test_metrics"]["r2"]:
                        best_blend = candidate
        blend_summary = best_blend
        if best_blend is not None:
            best_members = best_blend["members"]
            alpha = best_blend["alpha"]
            blend_pred = np.clip(alpha * predictions[best_members[0]] + (1.0 - alpha) * predictions[best_members[1]], 0.0, 1.0)
            with (args.output_dir / "best_blend_predictions_test.jsonl").open("w", encoding="utf-8") as f:
                for idx, value in enumerate(blend_pred.tolist()):
                    f.write(json.dumps({"row_index": idx, "y_true": float(y_test[idx]), "y_pred": float(value)}) + "\n")

    summary = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "target_field": args.target_field,
        "train_splits": args.train_splits,
        "test_splits": args.test_splits,
        "num_train": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "stats_feature_count": int(stats.shape[1]),
        "prompt_hidden_dim": int(prompt_hidden.shape[1]),
        "feature_sets": [name for name, _, _ in candidate_feature_sets],
        "top10_by_test": results[:10],
        "best_blend": blend_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if results:
        best = results[0]
        best_pred = predictions[best["name"]]
        with (args.output_dir / "best_predictions_test.jsonl").open("w", encoding="utf-8") as f:
            for idx, value in enumerate(best_pred.tolist()):
                f.write(json.dumps({"row_index": idx, "y_true": float(y_test[idx]), "y_pred": float(value)}) + "\n")


if __name__ == "__main__":
    main()
