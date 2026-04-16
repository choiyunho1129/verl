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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


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
            "feature_dim": int(data["feature_dim"][0]),
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
    return {
        "row_r2": float(r2_score(y_true, y_pred)),
        "prompt_mean_r2": float(r2_score(yt, yp)),
        "prompt_mean_mae": float(mean_absolute_error(yt, yp)),
        "hard_prompt_mae": float(mean_absolute_error(yt[hard_mask], yp[hard_mask])) if hard_mask.any() else float("nan"),
        "very_hard_prompt_mae": float(mean_absolute_error(yt[vhard_mask], yp[vhard_mask])) if vhard_mask.any() else float("nan"),
        "num_prompts": int(len(yt)),
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


def _compose_prediction(
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


def _hard_specialist_specs() -> list[str]:
    return [
        "ridge_sub0.7_a1000",
        "ridge_sub0.7_a3000",
        "ridge_sub0.8_a1000",
        "et_sub0.7_n500_l5_mf0.5",
        "et_sub0.8_n500_l5_mf0.5",
    ]


def _vhard_specialist_specs() -> list[str]:
    return [
        "ridge_sub0.9_a300",
        "ridge_sub0.9_a1000",
        "et_sub0.9_n500_l5_mf0.5",
    ]


def _beta_profiles() -> list[tuple[float, float, float]]:
    return [
        (0.0, 0.0, 0.0),
        (0.05, 0.05, 0.05),
        (0.0, 0.05, 0.1),
    ]


def _save_prompt_predictions(path: Path, task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    ordered, yt, yp = aggregate_prompt(task_ids, y_true, y_pred)
    with path.open("w", encoding="utf-8") as handle:
        for task_id, yv, pv in zip(ordered, yt.tolist(), yp.tolist()):
            handle.write(json.dumps({"task_id": task_id, "y_true_difficulty": yv, "predicted_difficulty": pv}) + "\n")


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    cache_dir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_cascade_decomp_search"
    outdir.mkdir(parents=True, exist_ok=True)

    best: dict[str, Any] | None = None
    for cache_path in sorted(cache_dir.glob("*.npz")):
        dataset_key = cache_path.stem
        print(json.dumps({"stage": "dataset", "dataset_key": dataset_key}), flush=True)
        data = _load_dataset_cache(cache_path)
        x_train = np.asarray(data["x_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.float32)
        x_val = np.asarray(data["x_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.float32)
        val_task_ids = list(data["val_task_ids"])

        base_preds_val: dict[str, np.ndarray] = {}
        base_preds_test: dict[str, np.ndarray] = {}
        base_models: dict[str, Any] = {}
        for alpha in (3000.0, 10000.0):
            key = f"ridge_a{int(alpha)}"
            model = _fit_ridge(alpha, x_train, y_train)
            base_models[key] = ("ridge", alpha, model)
            base_preds_val[key] = _clip(model.predict(x_val))
            base_preds_test[key] = _clip(model.predict(np.asarray(data["x_test"], dtype=np.float32)))

        detector_val: dict[str, np.ndarray] = {}
        detector_test: dict[str, np.ndarray] = {}
        detector_models: dict[str, Any] = {}
        for target_name, thr, labels in (
            ("p80", 0.8, (y_train >= 0.8).astype(np.int32)),
            ("p90", 0.9, (y_train >= 0.9).astype(np.int32)),
            ("p100", 1.0, (y_train == 1.0).astype(np.int32)),
        ):
            for c_val in (0.3, 1.0, 3.0):
                key = f"{target_name}_c{c_val}"
                model = _fit_logistic(c_val, x_train, labels)
                detector_models[key] = model
                detector_val[key] = _clip(model.predict_proba(x_val)[:, 1])
                detector_test[key] = _clip(model.predict_proba(np.asarray(data["x_test"], dtype=np.float32))[:, 1])

        specialist_val: dict[str, np.ndarray] = {}
        specialist_test: dict[str, np.ndarray] = {}
        specialist_models: dict[str, Any] = {}
        for subset_threshold in (0.7, 0.8, 0.9):
            mask = y_train >= subset_threshold
            if int(mask.sum()) < 100:
                continue
            x_sub = x_train[mask]
            y_sub = y_train[mask]
            for alpha in (300.0, 1000.0, 3000.0):
                key = f"ridge_sub{subset_threshold:.1f}_a{int(alpha)}"
                model = _fit_ridge(alpha, x_sub, y_sub)
                specialist_models[key] = model
                specialist_val[key] = _clip(model.predict(x_val))
                specialist_test[key] = _clip(model.predict(np.asarray(data["x_test"], dtype=np.float32)))
            for n_estimators in (500,):
                for min_samples_leaf in (3, 5):
                    for max_features in (0.5, 0.7):
                        key = f"et_sub{subset_threshold:.1f}_n{n_estimators}_l{min_samples_leaf}_mf{max_features}"
                        model = _fit_et(n_estimators, min_samples_leaf, max_features, x_sub, y_sub)
                        specialist_models[key] = model
                        specialist_val[key] = _clip(model.predict(x_val))
                        specialist_test[key] = _clip(model.predict(np.asarray(data["x_test"], dtype=np.float32)))

        allowed_hard = set(_hard_specialist_specs())
        allowed_vhard = set(_vhard_specialist_specs())
        beta_profiles = _beta_profiles()
        for base_key, base_val in base_preds_val.items():
            base_val_metric = evaluate(val_task_ids, y_val, base_val)
            for hard_key, hard_val in specialist_val.items():
                if hard_key not in allowed_hard:
                    continue
                for vhard_key, vhard_val in specialist_val.items():
                    if vhard_key not in allowed_vhard:
                        continue
                    for p80_key in [k for k in detector_val if k.startswith("p80_")]:
                        for p90_key in [k for k in detector_val if k.startswith("p90_")]:
                            for p100_key in [k for k in detector_val if k.startswith("p100_")]:
                                p80 = detector_val[p80_key]
                                p90 = detector_val[p90_key]
                                p100 = detector_val[p100_key]
                                for t80 in (0.3, 0.5):
                                    for t90 in (0.3, 0.5):
                                        for t100 in (0.3, 0.5):
                                            for g80 in (1.0, 2.0):
                                                for g90 in (1.0, 2.0):
                                                    for g100 in (1.0, 2.0):
                                                        for beta80, beta90, beta100 in beta_profiles:
                                                                    pred = _compose_prediction(
                                                                        base_val,
                                                                        hard_val,
                                                                        vhard_val,
                                                                        p80,
                                                                        p90,
                                                                        p100,
                                                                        t80,
                                                                        t90,
                                                                        t100,
                                                                        g80,
                                                                        g90,
                                                                        g100,
                                                                        beta80,
                                                                        beta90,
                                                                        beta100,
                                                                    )
                                                                    metric = evaluate(val_task_ids, y_val, pred)
                                                                    row = {
                                                                        "dataset_key": dataset_key,
                                                                        "base_key": base_key,
                                                                        "base_val": base_val_metric,
                                                                        "hard_key": hard_key,
                                                                        "vhard_key": vhard_key,
                                                                        "detectors": {"p80": p80_key, "p90": p90_key, "p100": p100_key},
                                                                        "routing": {
                                                                            "t80": t80,
                                                                            "t90": t90,
                                                                            "t100": t100,
                                                                            "g80": g80,
                                                                            "g90": g90,
                                                                            "g100": g100,
                                                                            "beta80": beta80,
                                                                            "beta90": beta90,
                                                                            "beta100": beta100,
                                                                        },
                                                                        "val": metric,
                                                                    }
                                                                    score = (
                                                                        metric["prompt_mean_r2"],
                                                                        -metric["hard_prompt_mae"],
                                                                        -metric["very_hard_prompt_mae"],
                                                                    )
                                                                    best_score = None if best is None else (
                                                                        best["val"]["prompt_mean_r2"],
                                                                        -best["val"]["hard_prompt_mae"],
                                                                        -best["val"]["very_hard_prompt_mae"],
                                                                    )
                                                                    if best is None or score > best_score:
                                                                        best = row
                                                                        _write_json(outdir / "progress.json", best)

        print(json.dumps({"stage": "dataset_done", "dataset_key": dataset_key}), flush=True)

    if best is None:
        raise RuntimeError("No cascade-decomposition candidate was evaluated")

    data = _load_dataset_cache(cache_dir / f"{best['dataset_key']}.npz")
    x_trainval = np.concatenate([np.asarray(data["x_train"]), np.asarray(data["x_val"])], axis=0)
    y_trainval = np.concatenate([np.asarray(data["y_train"]), np.asarray(data["y_val"])], axis=0)
    x_test = np.asarray(data["x_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float32)
    test_task_ids = list(data["test_task_ids"])

    base_alpha = 3000.0 if "a3000" in best["base_key"] else 10000.0
    base_model = _fit_ridge(base_alpha, x_trainval, y_trainval)

    def fit_specialist_from_key(key: str):
        import re
        m = re.search(r"_sub([0-9.]+)_", key)
        subset_thr = float(m.group(1))
        mask = y_trainval >= subset_thr
        x_sub = x_trainval[mask]
        y_sub = y_trainval[mask]
        if key.startswith("ridge_"):
            alpha = float(key.split("_a")[-1])
            return _fit_ridge(alpha, x_sub, y_sub)
        parts = key.split("_")
        n_estimators = int(parts[2][1:])
        min_samples_leaf = int(parts[3][1:])
        max_features = float(parts[4][2:])
        return _fit_et(n_estimators, min_samples_leaf, max_features, x_sub, y_sub)

    hard_model = fit_specialist_from_key(best["hard_key"])
    vhard_model = fit_specialist_from_key(best["vhard_key"])

    def fit_detector_from_key(key: str):
        target_name, cpart = key.split("_c")
        c_val = float(cpart)
        if target_name == "p80":
            labels = (y_trainval >= 0.8).astype(np.int32)
        elif target_name == "p90":
            labels = (y_trainval >= 0.9).astype(np.int32)
        else:
            labels = (y_trainval == 1.0).astype(np.int32)
        return _fit_logistic(c_val, x_trainval, labels)

    p80_model = fit_detector_from_key(best["detectors"]["p80"])
    p90_model = fit_detector_from_key(best["detectors"]["p90"])
    p100_model = fit_detector_from_key(best["detectors"]["p100"])

    base_test = _clip(base_model.predict(x_test))
    hard_test = _clip(hard_model.predict(x_test))
    vhard_test = _clip(vhard_model.predict(x_test))
    p80_test = _clip(p80_model.predict_proba(x_test)[:, 1])
    p90_test = _clip(p90_model.predict_proba(x_test)[:, 1])
    p100_test = _clip(p100_model.predict_proba(x_test)[:, 1])
    cascade_test = _compose_prediction(
        base_test,
        hard_test,
        vhard_test,
        p80_test,
        p90_test,
        p100_test,
        **best["routing"],
    )

    summary = {
        "setting": "two_rollout_think_cascade_decomposition",
        "best": best,
        "test": {
            "base": evaluate(test_task_ids, y_test, base_test),
            "cascade": evaluate(test_task_ids, y_test, cascade_test),
        },
    }
    _save_prompt_predictions(outdir / "base_predictions_test.jsonl", test_task_ids, y_test, base_test)
    _save_prompt_predictions(outdir / "cascade_predictions_test.jsonl", test_task_ids, y_test, cascade_test)
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
