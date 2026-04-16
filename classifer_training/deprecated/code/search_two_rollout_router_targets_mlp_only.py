from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def aggregate_xy(task_ids: list[str], x: np.ndarray, y: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    ys: dict[str, list[float]] = defaultdict(list)
    for task_id, xi, yi in zip(task_ids, x, y):
        key = str(task_id)
        if key not in sums:
            sums[key] = np.asarray(xi, dtype=np.float64).copy()
            counts[key] = 1
        else:
            sums[key] += np.asarray(xi, dtype=np.float64)
            counts[key] += 1
        ys[key].append(float(yi))
    ordered = sorted(sums)
    x_prompt = np.stack([sums[k] / counts[k] for k in ordered]).astype(np.float32)
    y_prompt = np.asarray([float(np.mean(ys[k])) for k in ordered], dtype=np.float32)
    return ordered, x_prompt, y_prompt


def _fit_base(alpha: float, x: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=42))])
    model.fit(x, y)
    return model


def _balanced_resample(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return x, y
    if len(pos_idx) < len(neg_idx):
        extra = rng.choice(pos_idx, size=len(neg_idx) - len(pos_idx), replace=True)
        keep = np.concatenate([np.arange(len(y)), extra])
    elif len(neg_idx) < len(pos_idx):
        extra = rng.choice(neg_idx, size=len(pos_idx) - len(neg_idx), replace=True)
        keep = np.concatenate([np.arange(len(y)), extra])
    else:
        keep = np.arange(len(y))
    rng.shuffle(keep)
    return x[keep], y[keep]


def _metrics(y_bin: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(np.int32)
    return {
        "positive_rate": float(np.mean(y_bin)),
        "pred_positive_rate": float(np.mean(pred)),
        "precision": float(precision_score(y_bin, pred, zero_division=0)),
        "recall": float(recall_score(y_bin, pred, zero_division=0)),
        "f1": float(f1_score(y_bin, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_bin, prob)),
        "ap": float(average_precision_score(y_bin, prob)),
        "brier": float(brier_score_loss(y_bin, prob)),
        "threshold": float(threshold),
    }


def _best_threshold(y_bin: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for thr in np.linspace(0.05, 0.95, 19):
        metric = _metrics(y_bin, prob, float(thr))
        if best is None or metric["f1"] > best["f1"]:
            best = metric
    assert best is not None
    return best


def _make_mlp(hidden_layer_sizes: tuple[int, ...], alpha: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    batch_size=128,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                    max_iter=500,
                    random_state=seed,
                ),
            ),
        ]
    )


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    cache_path = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache/think_end_hidden:mean.npz"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_router_target_search_mlp_exact"
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(cache_path, allow_pickle=True) as data:
        _, x_train_p, y_train_p = aggregate_xy(data["train_task_ids"].tolist(), data["x_train"], data["y_train"])
        _, x_val_p, y_val_p = aggregate_xy(data["val_task_ids"].tolist(), data["x_val"], data["y_val"])
        _, x_test_p, y_test_p = aggregate_xy(data["test_task_ids"].tolist(), data["x_test"], data["y_test"])

    base = _fit_base(10000.0, x_train_p, y_train_p)
    base_train_pred = np.clip(np.asarray(base.predict(x_train_p), dtype=np.float32), 0.0, 1.0)
    base_val_pred = np.clip(np.asarray(base.predict(x_val_p), dtype=np.float32), 0.0, 1.0)
    base_test_pred = np.clip(np.asarray(base.predict(x_test_p), dtype=np.float32), 0.0, 1.0)

    targets = {
        "true_ge_085": {
            "train": (y_train_p >= 0.85).astype(np.int32),
            "val": (y_val_p >= 0.85).astype(np.int32),
            "test": (y_test_p >= 0.85).astype(np.int32),
        },
        "miss85": {
            "train": ((y_train_p >= 0.85) & (base_train_pred < 0.85)).astype(np.int32),
            "val": ((y_val_p >= 0.85) & (base_val_pred < 0.85)).astype(np.int32),
            "test": ((y_test_p >= 0.85) & (base_test_pred < 0.85)).astype(np.int32),
        },
    }

    specs = [
        {"hidden_layer_sizes": (256, 128), "alpha": 1e-4, "seed": 1},
        {"hidden_layer_sizes": (256, 128), "alpha": 1e-4, "seed": 2},
        {"hidden_layer_sizes": (256, 128), "alpha": 1e-4, "seed": 3},
        {"hidden_layer_sizes": (256, 128), "alpha": 5e-4, "seed": 1},
        {"hidden_layer_sizes": (256, 128), "alpha": 5e-4, "seed": 2},
        {"hidden_layer_sizes": (256, 128), "alpha": 5e-4, "seed": 3},
        {"hidden_layer_sizes": (512, 256, 128), "alpha": 1e-4, "seed": 1},
        {"hidden_layer_sizes": (512, 256, 128), "alpha": 1e-4, "seed": 2},
        {"hidden_layer_sizes": (512, 256, 128), "alpha": 1e-4, "seed": 3},
        {"hidden_layer_sizes": (512, 256, 128), "alpha": 5e-4, "seed": 1},
        {"hidden_layer_sizes": (512, 256, 128), "alpha": 5e-4, "seed": 2},
        {"hidden_layer_sizes": (512, 256, 128), "alpha": 5e-4, "seed": 3},
    ]

    summary: dict[str, Any] = {"dataset_key": "think_end_hidden:mean", "targets": {}}
    for target_name, labels in targets.items():
        print(json.dumps({"stage": "target_start", "target": target_name}), flush=True)
        best: dict[str, Any] | None = None
        progress_path = outdir / f"{target_name}_progress.json"
        for spec in specs:
            print(json.dumps({"stage": "fit", "target": target_name, "spec": spec}), flush=True)
            fit_x, fit_y = _balanced_resample(x_train_p, labels["train"], seed=int(spec["seed"]))
            model = _make_mlp(tuple(spec["hidden_layer_sizes"]), float(spec["alpha"]), int(spec["seed"]))
            model.fit(fit_x, fit_y)
            val_prob = np.asarray(model.predict_proba(x_val_p)[:, 1], dtype=np.float32)
            row = {
                "best_mlp": spec,
                "val": {
                    "threshold_0.5": _metrics(labels["val"], val_prob, 0.5),
                    "best_f1": _best_threshold(labels["val"], val_prob),
                },
            }
            score = (
                row["val"]["best_f1"]["f1"],
                row["val"]["threshold_0.5"]["ap"],
                row["val"]["threshold_0.5"]["auc"],
            )
            best_score = None if best is None else (
                best["val"]["best_f1"]["f1"],
                best["val"]["threshold_0.5"]["ap"],
                best["val"]["threshold_0.5"]["auc"],
            )
            if best is None or score > best_score:
                best = row
                _write_json(progress_path, best)

        assert best is not None
        spec = best["best_mlp"]
        fit_x, fit_y = _balanced_resample(
            np.concatenate([x_train_p, x_val_p], axis=0),
            np.concatenate([labels["train"], labels["val"]], axis=0),
            seed=int(spec["seed"]),
        )
        final_model = _make_mlp(tuple(spec["hidden_layer_sizes"]), float(spec["alpha"]), int(spec["seed"]))
        final_model.fit(fit_x, fit_y)
        test_prob = np.asarray(final_model.predict_proba(x_test_p)[:, 1], dtype=np.float32)
        result = {
            "best_mlp": best["best_mlp"],
            "val": best["val"],
            "test": {
                "threshold_0.5": _metrics(labels["test"], test_prob, 0.5),
                "threshold_from_val_best_f1": _metrics(labels["test"], test_prob, best["val"]["best_f1"]["threshold"]),
            },
        }
        summary["targets"][target_name] = result
        _write_json(outdir / f"{target_name}_result.json", result)

    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
