from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


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
        }


def aggregate_prompt(task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped_true: dict[str, list[float]] = defaultdict(list)
    grouped_pred: dict[str, list[float]] = defaultdict(list)
    for task_id, yt, yp in zip(task_ids, y_true.tolist(), y_pred.tolist()):
        grouped_true[str(task_id)].append(float(yt))
        grouped_pred[str(task_id)].append(float(yp))
    ordered = sorted(grouped_true)
    yt = np.asarray([np.mean(grouped_true[k]) for k in ordered], dtype=np.float32)
    yp = np.asarray([np.mean(grouped_pred[k]) for k in ordered], dtype=np.float32)
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


def _fit_ridge(alpha: float, x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=42))])
    model.fit(x_train, y_train)
    return model


def _fit_detector(c_val: float, hard_threshold: float, x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    labels = (y_train >= hard_threshold).astype(np.int32)
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
    model.fit(x_train, labels)
    return model


def _fit_specialist(spec: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray):
    subset_mask = y_train >= spec["subset_threshold"]
    x_sub = x_train[subset_mask]
    y_sub = y_train[subset_mask]
    if spec["family"] == "ridge":
        return _fit_ridge(float(spec["alpha"]), x_sub, y_sub)
    model = ExtraTreesRegressor(
        n_estimators=int(spec["n_estimators"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        max_features=float(spec["max_features"]),
        n_jobs=12,
        random_state=42,
    )
    model.fit(x_sub, y_sub)
    return model


def _cascade_predictions(base_pred: np.ndarray, specialist_pred: np.ndarray, p_hard: np.ndarray, p0: float, gamma: float) -> np.ndarray:
    weight = np.clip((p_hard - p0) / max(1.0 - p0, 1e-6), 0.0, 1.0) ** gamma
    pred = (1.0 - weight) * base_pred + weight * specialist_pred
    return np.clip(pred.astype(np.float32), 0.0, 1.0)


def _save_prompt_predictions(path: Path, task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    ordered, yt, yp = aggregate_prompt(task_ids, y_true, y_pred)
    with path.open("w", encoding="utf-8") as handle:
        for task_id, y_val, pred_val in zip(ordered, yt.tolist(), yp.tolist()):
            handle.write(json.dumps({"task_id": task_id, "y_true_difficulty": y_val, "predicted_difficulty": pred_val}) + "\n")


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    progress_path = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_cascade_search/progress.json"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_cascade_search_eval"
    outdir.mkdir(parents=True, exist_ok=True)

    best = json.loads(progress_path.read_text(encoding="utf-8"))
    cache_path = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache" / f"{best['dataset_key']}.npz"
    data = _load_dataset_cache(cache_path)

    x_trainval = np.concatenate([np.asarray(data["x_train"]), np.asarray(data["x_val"])], axis=0)
    y_trainval = np.concatenate([np.asarray(data["y_train"]), np.asarray(data["y_val"])], axis=0)
    x_test = np.asarray(data["x_test"])
    y_test = np.asarray(data["y_test"])
    test_task_ids = list(data["test_task_ids"])

    base_model = _fit_ridge(float(best["base"]["alpha"]), x_trainval, y_trainval)
    detector = _fit_detector(float(best["detector"]["C"]), float(best["detector"]["hard_threshold"]), x_trainval, y_trainval)
    specialist = _fit_specialist(best["specialist"], x_trainval, y_trainval)

    base_test_pred = np.clip(np.asarray(base_model.predict(x_test), dtype=np.float32), 0.0, 1.0)
    p_hard_test = np.asarray(detector.predict_proba(x_test)[:, 1], dtype=np.float32)
    specialist_test_pred = np.clip(np.asarray(specialist.predict(x_test), dtype=np.float32), 0.0, 1.0)
    cascade_test_pred = _cascade_predictions(base_test_pred, specialist_test_pred, p_hard_test, float(best["routing"]["p0"]), float(best["routing"]["gamma"]))

    summary = {
        "setting": "two_rollout_think_cascade_eval_from_progress",
        "best_from_val": best,
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
