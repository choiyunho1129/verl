from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _load_dataset_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "x_train": data["x_train"],
            "y_train": data["y_train"],
            "train_task_ids": data["train_task_ids"].tolist(),
            "x_val": data["x_val"],
            "y_val": data["y_val"],
            "val_task_ids": data["val_task_ids"].tolist(),
            "x_test": data["x_test"],
            "y_test": data["y_test"],
            "test_task_ids": data["test_task_ids"].tolist(),
            "feature_dim": int(data["feature_dim"][0]),
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def aggregate_prompt(task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped_true: dict[str, list[float]] = defaultdict(list)
    grouped_pred: dict[str, list[float]] = defaultdict(list)
    for task_id, y_val, pred_val in zip(task_ids, y_true.tolist(), y_pred.tolist()):
        grouped_true[str(task_id)].append(float(y_val))
        grouped_pred[str(task_id)].append(float(pred_val))
    ordered = sorted(grouped_true)
    yt = np.asarray([np.mean(grouped_true[t]) for t in ordered], dtype=np.float32)
    yp = np.asarray([np.mean(grouped_pred[t]) for t in ordered], dtype=np.float32)
    return ordered, yt, yp


def evaluate(task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray, hard_threshold: float = 0.8) -> dict[str, float]:
    ordered, yt, yp = aggregate_prompt(task_ids, y_true, y_pred)
    hard_mask = yt >= hard_threshold
    return {
        "row_r2": float(r2_score(y_true, y_pred)),
        "prompt_mean_r2": float(r2_score(yt, yp)),
        "prompt_mean_mae": float(mean_absolute_error(yt, yp)),
        "hard_prompt_mae": float(mean_absolute_error(yt[hard_mask], yp[hard_mask])) if hard_mask.any() else float("nan"),
        "num_prompts": int(len(ordered)),
        "num_hard_prompts": int(hard_mask.sum()),
    }


def _fit_ridge(alpha: float, x_train: np.ndarray, y_train: np.ndarray):
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=42))])
    model.fit(x_train, y_train)
    return model


def _fit_detector(c_val: float, hard_threshold: float, x_train: np.ndarray, y_train: np.ndarray):
    y_bin = (y_train >= hard_threshold).astype(np.int32)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_val,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_bin)
    return model


def _fit_specialist(spec: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray):
    subset_mask = y_train >= spec["subset_threshold"]
    x_sub = x_train[subset_mask]
    y_sub = y_train[subset_mask]
    if spec["family"] == "ridge":
        return _fit_ridge(spec["alpha"], x_sub, y_sub)
    return ExtraTreesRegressor(
        n_estimators=spec["n_estimators"],
        min_samples_leaf=spec["min_samples_leaf"],
        max_features=spec["max_features"],
        n_jobs=12,
        random_state=42,
    ).fit(x_sub, y_sub)


def _specialist_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for subset_threshold in (0.6, 0.7, 0.8):
        for alpha in (300.0, 1000.0, 3000.0):
            specs.append({"family": "ridge", "subset_threshold": subset_threshold, "alpha": alpha})
        for n_estimators in (500, 1000):
            for min_samples_leaf in (3, 5):
                for max_features in (0.5, 0.7):
                    specs.append(
                        {
                            "family": "et",
                            "subset_threshold": subset_threshold,
                            "n_estimators": n_estimators,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                        }
                    )
    return specs


def _cascade_predictions(base_pred: np.ndarray, specialist_pred: np.ndarray, p_hard: np.ndarray, p0: float, gamma: float) -> np.ndarray:
    weight = np.clip((p_hard - p0) / max(1.0 - p0, 1e-6), 0.0, 1.0) ** gamma
    pred = (1.0 - weight) * base_pred + weight * specialist_pred
    return np.clip(pred.astype(np.float32), 0.0, 1.0)


def _save_prompt_predictions(path: Path, task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    ordered, yt, yp = aggregate_prompt(task_ids, y_true, y_pred)
    with path.open("w", encoding="utf-8") as handle:
        for task_id, y_val, pred_val in zip(ordered, yt.tolist(), yp.tolist()):
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "y_true_difficulty": y_val,
                        "predicted_difficulty": pred_val,
                    }
                )
                + "\n"
            )


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    cache_dir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_cascade_search"
    outdir.mkdir(parents=True, exist_ok=True)

    cache_paths = sorted(cache_dir.glob("*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No dataset caches found under {cache_dir}")

    best: dict[str, Any] | None = None
    for cache_path in cache_paths:
        dataset_key = cache_path.stem
        data = _load_dataset_cache(cache_path)
        x_train = np.asarray(data["x_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.float32)
        x_val = np.asarray(data["x_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.float32)
        val_task_ids = list(data["val_task_ids"])
        print(json.dumps({"stage": "dataset", "dataset_key": dataset_key}), flush=True)

        for alpha in (3000.0, 10000.0, 30000.0):
            base_model = _fit_ridge(alpha, x_train, y_train)
            base_val_pred = np.clip(np.asarray(base_model.predict(x_val), dtype=np.float32), 0.0, 1.0)
            base_val_metric = evaluate(val_task_ids, y_val, base_val_pred)
            for hard_threshold in (0.8, 0.9):
                for c_val in (0.3, 1.0, 3.0):
                    detector = _fit_detector(c_val, hard_threshold, x_train, y_train)
                    p_hard = np.asarray(detector.predict_proba(x_val)[:, 1], dtype=np.float32)
                    for spec in _specialist_specs():
                        if int((y_train >= spec["subset_threshold"]).sum()) < 100:
                            continue
                        specialist = _fit_specialist(spec, x_train, y_train)
                        specialist_val_pred = np.clip(np.asarray(specialist.predict(x_val), dtype=np.float32), 0.0, 1.0)
                        for p0 in (0.2, 0.3, 0.4, 0.5):
                            for gamma in (1.0, 2.0, 3.0):
                                cascade_val_pred = _cascade_predictions(base_val_pred, specialist_val_pred, p_hard, p0, gamma)
                                metric = evaluate(val_task_ids, y_val, cascade_val_pred)
                                row = {
                                    "dataset_key": dataset_key,
                                    "base": {"family": "ridge", "alpha": alpha, "val": base_val_metric},
                                    "detector": {"family": "logistic", "hard_threshold": hard_threshold, "C": c_val},
                                    "specialist": spec,
                                    "routing": {"p0": p0, "gamma": gamma},
                                    "val": metric,
                                }
                                score = (metric["prompt_mean_r2"], -metric["hard_prompt_mae"])
                                best_score = None if best is None else (best["val"]["prompt_mean_r2"], -best["val"]["hard_prompt_mae"])
                                if best is None or score > best_score:
                                    best = row
                                    _write_json(outdir / "progress.json", best)

    if best is None:
        raise RuntimeError("No cascade candidate was evaluated")

    data = _load_dataset_cache(cache_dir / f"{best['dataset_key']}.npz")
    x_trainval = np.concatenate([np.asarray(data["x_train"]), np.asarray(data["x_val"])], axis=0)
    y_trainval = np.concatenate([np.asarray(data["y_train"]), np.asarray(data["y_val"])], axis=0)
    x_test = np.asarray(data["x_test"])
    y_test = np.asarray(data["y_test"])
    test_task_ids = list(data["test_task_ids"])

    base_model = _fit_ridge(best["base"]["alpha"], x_trainval, y_trainval)
    detector = _fit_detector(best["detector"]["C"], best["detector"]["hard_threshold"], x_trainval, y_trainval)
    specialist = _fit_specialist(best["specialist"], x_trainval, y_trainval)

    base_test_pred = np.clip(np.asarray(base_model.predict(x_test), dtype=np.float32), 0.0, 1.0)
    p_hard_test = np.asarray(detector.predict_proba(x_test)[:, 1], dtype=np.float32)
    specialist_test_pred = np.clip(np.asarray(specialist.predict(x_test), dtype=np.float32), 0.0, 1.0)
    cascade_test_pred = _cascade_predictions(base_test_pred, specialist_test_pred, p_hard_test, best["routing"]["p0"], best["routing"]["gamma"])

    summary = {
        "setting": "two_rollout_think_cascade",
        "best": best,
        "test": {
            "base": evaluate(test_task_ids, y_test, base_test_pred),
            "cascade": evaluate(test_task_ids, y_test, cascade_test_pred),
        },
    }
    _save_prompt_predictions(outdir / "base_predictions_test.jsonl", test_task_ids, y_test, base_test_pred)
    _save_prompt_predictions(outdir / "cascade_predictions_test.jsonl", test_task_ids, y_test, cascade_test_pred)
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
