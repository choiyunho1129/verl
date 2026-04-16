from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed prompt-level model search.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_splits", nargs="+", default=["train"])
    parser.add_argument("--eval_splits", nargs="+", default=["validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    return parser.parse_args()


def load_rows(manifest_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    manifest = json.loads(manifest_path.read_text())
    labels: dict[str, dict[str, Any]] = {}
    with labels_path.open() as f:
        for line in f:
            row = json.loads(line)
            labels[str(row["task_id"])] = row

    feature_keys = sorted(next(iter(labels.values()))["aggregated_features"].keys())
    X: list[list[float]] = []
    y: list[float] = []
    splits: list[str] = []
    task_ids: list[str] = []
    for entry in manifest:
        split_name = Path(entry["index_path"]).stem.replace("index_", "")
        with Path(entry["index_path"]).open() as f:
            for line in f:
                index_row = json.loads(line)
                label_row = labels[str(index_row["task_id"])]
                X.append([float(label_row["aggregated_features"].get(k, 0.0)) for k in feature_keys])
                y.append(float(label_row["difficulty"]))
                splits.append(split_name)
                task_ids.append(str(index_row["task_id"]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(splits), task_ids


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def hard_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.8) -> dict[str, float]:
    mask = y_true >= threshold
    if not np.any(mask):
        return {"hard_mae": float("nan"), "hard_mean_pred": float("nan")}
    return {
        "hard_mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "hard_mean_pred": float(y_pred[mask].mean()),
    }


def candidate_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for leaf in [1, 2, 3, 5, 8]:
        for max_features in [0.3, 0.5, 0.7, None]:
            configs.append(
                {
                    "kind": "et",
                    "name": f"et_leaf{leaf}_mf{max_features}",
                    "params": {
                        "n_estimators": 3000,
                        "min_samples_leaf": leaf,
                        "max_features": max_features,
                        "bootstrap": False,
                        "random_state": 42,
                        "n_jobs": -1,
                    },
                }
            )
            configs.append(
                {
                    "kind": "et_weighted",
                    "name": f"et_weighted_leaf{leaf}_mf{max_features}",
                    "params": {
                        "n_estimators": 3000,
                        "min_samples_leaf": leaf,
                        "max_features": max_features,
                        "bootstrap": False,
                        "random_state": 42,
                        "n_jobs": -1,
                    },
                }
            )
    for leaf in [1, 2, 5]:
        for max_features in ["sqrt", 0.3, 0.5]:
            configs.append(
                {
                    "kind": "rf",
                    "name": f"rf_leaf{leaf}_mf{max_features}",
                    "params": {
                        "n_estimators": 2000,
                        "min_samples_leaf": leaf,
                        "max_features": max_features,
                        "bootstrap": True,
                        "random_state": 42,
                        "n_jobs": -1,
                    },
                }
            )
    for depth in [4, 6, 8]:
        for lr in [0.03, 0.05, 0.1]:
            configs.append(
                {
                    "kind": "histgb",
                    "name": f"histgb_depth{depth}_lr{lr}",
                    "params": {
                        "max_depth": depth,
                        "learning_rate": lr,
                        "max_iter": 600,
                        "min_samples_leaf": 10,
                        "l2_regularization": 1.0,
                        "random_state": 42,
                    },
                }
            )
    if XGBRegressor is not None:
        for depth in [4, 6, 8]:
            for lr in [0.03, 0.05]:
                configs.append(
                    {
                        "kind": "xgb",
                        "name": f"xgb_depth{depth}_lr{lr}",
                        "params": {
                            "objective": "reg:squarederror",
                            "tree_method": "hist",
                            "max_depth": depth,
                            "learning_rate": lr,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                            "n_estimators": 800,
                            "random_state": 42,
                            "n_jobs": 16,
                        },
                    }
                )
    return configs


def build_model(kind: str, params: dict[str, Any]) -> Any:
    if kind in {"et", "et_weighted"}:
        return ExtraTreesRegressor(**params)
    if kind == "rf":
        return RandomForestRegressor(**params)
    if kind == "histgb":
        return HistGradientBoostingRegressor(**params)
    if kind == "xgb":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        return XGBRegressor(**params)
    raise ValueError(f"Unknown kind: {kind}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y, splits, task_ids = load_rows(args.manifest, args.labels_path)

    train_mask = np.isin(splits, args.train_splits)
    eval_mask = np.isin(splits, args.eval_splits)
    test_mask = np.isin(splits, args.test_splits)
    trainval_mask = np.isin(splits, args.train_splits + args.eval_splits)

    train_idx = np.where(train_mask)[0]
    eval_idx = np.where(eval_mask)[0]
    test_idx = np.where(test_mask)[0]
    trainval_idx = np.where(trainval_mask)[0]

    progress_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    best_refit_path = args.output_dir / "best_trainval_refit.json"
    pred_path = args.output_dir / "predictions_test.jsonl"

    seen_names: set[str] = set()
    if progress_path.exists():
        with progress_path.open() as f:
            for line in f:
                try:
                    seen_names.add(json.loads(line)["name"])
                except Exception:
                    continue

    results: list[dict[str, Any]] = []
    if progress_path.exists():
        with progress_path.open() as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except Exception:
                    continue

    for cfg in candidate_configs():
        if cfg["name"] in seen_names:
            continue
        model = build_model(cfg["kind"], cfg["params"])
        fit_kwargs: dict[str, Any] = {}
        if cfg["kind"] == "et_weighted":
            weights = np.where(y[train_idx] >= 0.8, 3.5, np.where(y[train_idx] >= 0.6, 2.0, 1.0))
            fit_kwargs["sample_weight"] = weights
        if cfg["kind"] == "xgb":
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[eval_idx], y[eval_idx])], verbose=False, **fit_kwargs)
        else:
            model.fit(X[train_idx], y[train_idx], **fit_kwargs)
        val_pred = np.clip(model.predict(X[eval_idx]), 0.0, 1.0)
        test_pred = np.clip(model.predict(X[test_idx]), 0.0, 1.0)
        row = {
            "name": cfg["name"],
            "kind": cfg["kind"],
            "params": cfg["params"],
            "validation": metrics(y[eval_idx], val_pred),
            "test": {
                **metrics(y[test_idx], test_pred),
                **hard_metrics(y[test_idx], test_pred),
            },
        }
        results.append(row)
        with progress_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        best_val = max(r["validation"]["r2"] for r in results)
        print(f"done {cfg['name']} | best_val_r2={best_val:.6f}", flush=True)

    results.sort(key=lambda r: r["validation"]["r2"], reverse=True)
    summary_path.write_text(json.dumps({"num_features": int(X.shape[1]), "results": results[:50]}, indent=2))

    best = results[0]
    best_model = build_model(best["kind"], best["params"])
    fit_kwargs = {}
    if best["kind"] == "et_weighted":
        fit_kwargs["sample_weight"] = np.where(y[trainval_idx] >= 0.8, 3.5, np.where(y[trainval_idx] >= 0.6, 2.0, 1.0))
    if best["kind"] == "xgb":
        best_model.fit(X[trainval_idx], y[trainval_idx], verbose=False, **fit_kwargs)
    else:
        best_model.fit(X[trainval_idx], y[trainval_idx], **fit_kwargs)
    final_pred = np.clip(best_model.predict(X[test_idx]), 0.0, 1.0)
    best_refit = {
        "selected_by_validation": best,
        "trainval_refit_test": {
            **metrics(y[test_idx], final_pred),
            **hard_metrics(y[test_idx], final_pred),
        },
    }
    best_refit_path.write_text(json.dumps(best_refit, indent=2))
    with pred_path.open("w") as f:
        for idx, pred in zip(test_idx, final_pred):
            f.write(
                json.dumps(
                    {
                        "task_id": task_ids[idx],
                        "y_true": float(y[idx]),
                        "y_pred": float(pred),
                        "abs_error": float(abs(pred - y[idx])),
                    }
                )
                + "\n"
            )
    print(json.dumps(best_refit, indent=2))


if __name__ == "__main__":
    main()
