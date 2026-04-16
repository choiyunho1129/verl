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
    g_true: dict[str, list[float]] = defaultdict(list)
    g_pred: dict[str, list[float]] = defaultdict(list)
    for task_id, yt, yp in zip(task_ids, y_true.tolist(), y_pred.tolist()):
        g_true[str(task_id)].append(float(yt))
        g_pred[str(task_id)].append(float(yp))
    ordered = sorted(g_true)
    true_prompt = np.asarray([np.mean(g_true[k]) for k in ordered], dtype=np.float32)
    pred_prompt = np.asarray([np.mean(g_pred[k]) for k in ordered], dtype=np.float32)
    return ordered, true_prompt, pred_prompt


def evaluate(task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    _, yt, yp = aggregate_prompt(task_ids, y_true, y_pred)
    hard_mask = yt >= 0.8
    vhard_mask = yt >= 0.9
    easy_mask = yt <= 0.1
    return {
        "row_r2": float(r2_score(y_true, y_pred)),
        "prompt_mean_r2": float(r2_score(yt, yp)),
        "prompt_mean_mae": float(mean_absolute_error(yt, yp)),
        "easy_prompt_mae": float(mean_absolute_error(yt[easy_mask], yp[easy_mask])) if easy_mask.any() else float("nan"),
        "hard_prompt_mae": float(mean_absolute_error(yt[hard_mask], yp[hard_mask])) if hard_mask.any() else float("nan"),
        "very_hard_prompt_mae": float(mean_absolute_error(yt[vhard_mask], yp[vhard_mask])) if vhard_mask.any() else float("nan"),
        "num_prompts": int(len(yt)),
        "num_easy_prompts": int(easy_mask.sum()),
        "num_hard_prompts": int(hard_mask.sum()),
        "num_very_hard_prompts": int(vhard_mask.sum()),
    }


def _fit_ridge(alpha: float, x: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=42))])
    model.fit(x, y)
    return model


def _fit_logistic(c_val: float, x: np.ndarray, y_bin: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_val,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x, y_bin)
    return model


def _fit_et(n_estimators: int, min_samples_leaf: int, max_features: float, x: np.ndarray, y: np.ndarray) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=12,
        random_state=42,
    )
    model.fit(x, y)
    return model


def _clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32).reshape(-1), 0.0, 1.0)


def _soft_gate(prob: np.ndarray, threshold: float, gamma: float) -> np.ndarray:
    return np.clip((prob - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0) ** gamma


def _compose_mid(
    base: np.ndarray,
    hard: np.ndarray,
    vhard: np.ndarray,
    p80: np.ndarray,
    p90: np.ndarray,
    p100: np.ndarray,
    t80: float,
    t90: float,
    t100: float,
    g80: float,
    g90: float,
    g100: float,
    beta80: float,
    beta90: float,
    beta100: float,
) -> np.ndarray:
    w80 = _soft_gate(p80, t80, g80)
    w90 = np.minimum(w80, _soft_gate(p90, t90, g90))
    w100 = np.minimum(w90, _soft_gate(p100, t100, g100))
    pred = (1.0 - w80) * base + (w80 - w90) * hard + (w90 - w100) * vhard + w100 * 1.0
    pred = pred + beta80 * (p80 - 0.5) + beta90 * (p90 - 0.5) + beta100 * (p100 - 0.5)
    return _clip(pred)


def _compose_with_easy(mid: np.ndarray, p10: np.ndarray, t10: float, g10: float, beta10: float) -> np.ndarray:
    w10 = _soft_gate(p10, t10, g10)
    pred = (1.0 - w10) * mid
    pred = pred - beta10 * np.clip(p10 - 0.5, 0.0, 1.0)
    return _clip(pred)


def _save_prompt_predictions(path: Path, task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    ordered, yt, yp = aggregate_prompt(task_ids, y_true, y_pred)
    with path.open("w", encoding="utf-8") as handle:
        for task_id, yv, pv in zip(ordered, yt.tolist(), yp.tolist()):
            handle.write(json.dumps({"task_id": task_id, "y_true_difficulty": yv, "predicted_difficulty": pv}) + "\n")


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    src_summary = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_cascade_decomp_search/summary.json"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_easy_head_search"
    outdir.mkdir(parents=True, exist_ok=True)

    src = json.loads(src_summary.read_text())
    best_src = src["best"]
    dataset_key = str(best_src["dataset_key"])
    cache_path = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache" / f"{dataset_key}.npz"
    data = _load_dataset_cache(cache_path)

    x_train = np.asarray(data["x_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32)
    x_val = np.asarray(data["x_val"], dtype=np.float32)
    y_val = np.asarray(data["y_val"], dtype=np.float32)
    x_trainval = np.concatenate([x_train, x_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    x_test = np.asarray(data["x_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float32)
    val_task_ids = list(data["val_task_ids"])
    test_task_ids = list(data["test_task_ids"])

    # Refit current best decomp components on train split for validation search.
    base_alpha = 3000.0 if "a3000" in best_src["base_key"] else 10000.0
    base_model = _fit_ridge(base_alpha, x_train, y_train)

    def fit_specialist_from_key(key: str, x: np.ndarray, y: np.ndarray):
        import re
        m = re.search(r"_sub([0-9.]+)_", key)
        subset_thr = float(m.group(1))
        mask = y >= subset_thr
        x_sub = x[mask]
        y_sub = y[mask]
        if key.startswith("ridge_"):
            alpha = float(key.split("_a")[-1])
            return _fit_ridge(alpha, x_sub, y_sub)
        parts = key.split("_")
        n_estimators = int(parts[2][1:])
        min_samples_leaf = int(parts[3][1:])
        max_features = float(parts[4][2:])
        return _fit_et(n_estimators, min_samples_leaf, max_features, x_sub, y_sub)

    hard_model = fit_specialist_from_key(best_src["hard_key"], x_train, y_train)
    vhard_model = fit_specialist_from_key(best_src["vhard_key"], x_train, y_train)

    def fit_detector_from_key(key: str, x: np.ndarray, y: np.ndarray):
        target_name, cpart = key.split("_c")
        c_val = float(cpart)
        if target_name == "p80":
            labels = (y >= 0.8).astype(np.int32)
        elif target_name == "p90":
            labels = (y >= 0.9).astype(np.int32)
        else:
            labels = (y == 1.0).astype(np.int32)
        return _fit_logistic(c_val, x, labels)

    p80_model = fit_detector_from_key(best_src["detectors"]["p80"], x_train, y_train)
    p90_model = fit_detector_from_key(best_src["detectors"]["p90"], x_train, y_train)
    p100_model = fit_detector_from_key(best_src["detectors"]["p100"], x_train, y_train)

    base_val = _clip(base_model.predict(x_val))
    hard_val = _clip(hard_model.predict(x_val))
    vhard_val = _clip(vhard_model.predict(x_val))
    p80_val = _clip(p80_model.predict_proba(x_val)[:, 1])
    p90_val = _clip(p90_model.predict_proba(x_val)[:, 1])
    p100_val = _clip(p100_model.predict_proba(x_val)[:, 1])
    mid_val = _compose_mid(
        base_val,
        hard_val,
        vhard_val,
        p80_val,
        p90_val,
        p100_val,
        **best_src["routing"],
    )
    base_val_metric = evaluate(val_task_ids, y_val, mid_val)

    best: dict[str, Any] | None = None
    for c10 in (0.3, 1.0, 3.0, 10.0):
        p10_model = _fit_logistic(c10, x_train, (y_train <= 0.1).astype(np.int32))
        p10_val = _clip(p10_model.predict_proba(x_val)[:, 1])
        for t10 in (0.3, 0.5, 0.7):
            for g10 in (1.0, 2.0, 3.0):
                for beta10 in (0.0, 0.05, 0.1):
                    pred = _compose_with_easy(mid_val, p10_val, t10, g10, beta10)
                    metric = evaluate(val_task_ids, y_val, pred)
                    row = {
                        "dataset_key": dataset_key,
                        "base_decomp": best_src,
                        "base_val": base_val_metric,
                        "easy_detector": f"p10_c{c10}",
                        "easy_routing": {"t10": t10, "g10": g10, "beta10": beta10},
                        "val": metric,
                    }
                    score = (
                        metric["prompt_mean_r2"],
                        -metric["easy_prompt_mae"],
                        -metric["hard_prompt_mae"],
                    )
                    best_score = None if best is None else (
                        best["val"]["prompt_mean_r2"],
                        -best["val"]["easy_prompt_mae"],
                        -best["val"]["hard_prompt_mae"],
                    )
                    if best is None or score > best_score:
                        best = row
                        _write_json(outdir / "progress.json", best)

    if best is None:
        raise RuntimeError("No easy-head candidate evaluated")

    # Refit on train+val and evaluate on test.
    base_model = _fit_ridge(base_alpha, x_trainval, y_trainval)
    hard_model = fit_specialist_from_key(best_src["hard_key"], x_trainval, y_trainval)
    vhard_model = fit_specialist_from_key(best_src["vhard_key"], x_trainval, y_trainval)
    p80_model = fit_detector_from_key(best_src["detectors"]["p80"], x_trainval, y_trainval)
    p90_model = fit_detector_from_key(best_src["detectors"]["p90"], x_trainval, y_trainval)
    p100_model = fit_detector_from_key(best_src["detectors"]["p100"], x_trainval, y_trainval)
    c10 = float(str(best["easy_detector"]).split("_c")[-1])
    p10_model = _fit_logistic(c10, x_trainval, (y_trainval <= 0.1).astype(np.int32))

    base_test = _clip(base_model.predict(x_test))
    hard_test = _clip(hard_model.predict(x_test))
    vhard_test = _clip(vhard_model.predict(x_test))
    p80_test = _clip(p80_model.predict_proba(x_test)[:, 1])
    p90_test = _clip(p90_model.predict_proba(x_test)[:, 1])
    p100_test = _clip(p100_model.predict_proba(x_test)[:, 1])
    mid_test = _compose_mid(
        base_test,
        hard_test,
        vhard_test,
        p80_test,
        p90_test,
        p100_test,
        **best_src["routing"],
    )
    p10_test = _clip(p10_model.predict_proba(x_test)[:, 1])
    pred_test = _compose_with_easy(mid_test, p10_test, **best["easy_routing"])

    summary = {
        "setting": "two_rollout_think_easy_head_on_best_decomp",
        "best": best,
        "test": {
            "base_decomp": evaluate(test_task_ids, y_test, mid_test),
            "easy_head": evaluate(test_task_ids, y_test, pred_test),
        },
    }
    _save_prompt_predictions(outdir / "base_decomp_predictions_test.jsonl", test_task_ids, y_test, mid_test)
    _save_prompt_predictions(outdir / "easy_head_predictions_test.jsonl", test_task_ids, y_test, pred_test)
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
