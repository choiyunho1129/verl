from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
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


def _base_prompt_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oof = np.zeros(len(y_train), dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fit_idx, pred_idx in kf.split(x_train):
        model = _fit_base(alpha, x_train[fit_idx], y_train[fit_idx])
        oof[pred_idx] = np.clip(np.asarray(model.predict(x_train[pred_idx]), dtype=np.float32), 0.0, 1.0)
    full = _fit_base(alpha, x_train, y_train)
    val_pred = np.clip(np.asarray(full.predict(x_val), dtype=np.float32), 0.0, 1.0)
    test_pred = np.clip(np.asarray(full.predict(x_test), dtype=np.float32), 0.0, 1.0)
    return oof, val_pred, test_pred


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


def _best_threshold_by_f1(y_bin: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for thr in np.linspace(0.05, 0.95, 19):
        metric = _metrics(y_bin, prob, float(thr))
        if best is None or metric["f1"] > best["f1"]:
            best = metric
    assert best is not None
    return best


def _plot_pr(path: Path, y_bin: np.ndarray, prob: np.ndarray, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_bin, prob)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(recall, precision, color="tab:blue", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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


def build_specs() -> list[tuple[str, dict[str, Any], Any]]:
    specs: list[tuple[str, dict[str, Any], Any]] = []
    for c_val in (0.1, 0.3, 1.0, 3.0, 10.0):
        specs.append(
            (
                "logistic",
                {"C": c_val},
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                C=c_val,
                                class_weight="balanced",
                                solver="liblinear",
                                dual=True,
                                max_iter=3000,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            )
        )
    for n_estimators in (500,):
        for min_samples_leaf in (5, 7):
            for max_features in (0.3, 0.5, 0.7):
                specs.append(
                    (
                        "extratrees",
                        {
                            "n_estimators": n_estimators,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                        },
                        ExtraTreesClassifier(
                            n_estimators=n_estimators,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            n_jobs=16,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    )
                )
    for hidden_layer_sizes in ((256, 128), (512, 256, 128)):
        for alpha in (1e-4, 5e-4):
            for seed in (1, 2, 3):
                specs.append(
                    (
                        "mlp",
                        {
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "alpha": alpha,
                            "seed": seed,
                        },
                        Pipeline(
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
                        ),
                    )
                )
    return specs


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    cache_path = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache/think_end_hidden:mean.npz"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_router_target_search_mlp"
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_dataset_cache(cache_path)
    _, x_train_p, y_train_p = aggregate_xy(data["train_task_ids"], np.asarray(data["x_train"], dtype=np.float32), np.asarray(data["y_train"], dtype=np.float32))
    _, x_val_p, y_val_p = aggregate_xy(data["val_task_ids"], np.asarray(data["x_val"], dtype=np.float32), np.asarray(data["y_val"], dtype=np.float32))
    test_task_ids, x_test_p, y_test_p = aggregate_xy(data["test_task_ids"], np.asarray(data["x_test"], dtype=np.float32), np.asarray(data["y_test"], dtype=np.float32))

    base_train_oof, base_val_pred, base_test_pred = _base_prompt_predictions(
        x_train_p,
        y_train_p,
        x_val_p,
        x_test_p,
        alpha=10000.0,
    )

    targets = {
        "true_ge_085": {
            "train": (y_train_p >= 0.85).astype(np.int32),
            "val": (y_val_p >= 0.85).astype(np.int32),
            "test": (y_test_p >= 0.85).astype(np.int32),
        },
        "miss85": {
            "train": ((y_train_p >= 0.85) & (base_train_oof < 0.85)).astype(np.int32),
            "val": ((y_val_p >= 0.85) & (base_val_pred < 0.85)).astype(np.int32),
            "test": ((y_test_p >= 0.85) & (base_test_pred < 0.85)).astype(np.int32),
        },
    }

    _write_json(
        outdir / "target_prevalence.json",
        {
            key: {
                "train_positive_rate": float(np.mean(spec["train"])),
                "val_positive_rate": float(np.mean(spec["val"])),
                "test_positive_rate": float(np.mean(spec["test"])),
                "train_count": int(len(spec["train"])),
                "val_count": int(len(spec["val"])),
                "test_count": int(len(spec["test"])),
            }
            for key, spec in targets.items()
        },
    )

    specs = build_specs()
    summary: dict[str, Any] = {"dataset_key": "think_end_hidden:mean", "base_alpha": 10000.0, "targets": {}}

    for target_name, labels in targets.items():
        print(json.dumps({"stage": "target_start", "target": target_name}), flush=True)
        best: dict[str, Any] | None = None
        progress_path = outdir / f"{target_name}_progress.json"
        for family, params, estimator in specs:
            print(json.dumps({"stage": "fit", "target": target_name, "family": family, "params": params}), flush=True)
            model = clone(estimator)
            fit_x = x_train_p
            fit_y = labels["train"]
            if family == "mlp":
                fit_x, fit_y = _balanced_resample(x_train_p, labels["train"], seed=int(params["seed"]))
            model.fit(fit_x, fit_y)
            val_prob = np.asarray(model.predict_proba(x_val_p)[:, 1], dtype=np.float32)
            val_metrics_05 = _metrics(labels["val"], val_prob, 0.5)
            val_best = _best_threshold_by_f1(labels["val"], val_prob)
            row = {
                "family": family,
                "params": params,
                "val": {
                    "threshold_0.5": val_metrics_05,
                    "best_f1": val_best,
                },
            }
            score = (row["val"]["best_f1"]["f1"], row["val"]["threshold_0.5"]["ap"], row["val"]["threshold_0.5"]["auc"])
            best_score = None if best is None else (
                best["val"]["best_f1"]["f1"],
                best["val"]["threshold_0.5"]["ap"],
                best["val"]["threshold_0.5"]["auc"],
            )
            if best is None or score > best_score:
                best = row
                _write_json(progress_path, best)
                print(json.dumps({"stage": "best_update", "target": target_name, "family": family, "params": params, "val_best_f1": row["val"]["best_f1"]["f1"], "val_ap": row["val"]["threshold_0.5"]["ap"]}), flush=True)

        assert best is not None
        final_model = clone(
            next(est for fam, params, est in specs if fam == best["family"] and params == best["params"])
        )
        x_trainval = np.concatenate([x_train_p, x_val_p], axis=0)
        y_trainval_bin = np.concatenate([labels["train"], labels["val"]], axis=0)
        final_model.fit(x_trainval, y_trainval_bin)
        test_prob = np.asarray(final_model.predict_proba(x_test_p)[:, 1], dtype=np.float32)
        chosen_thr = float(best["val"]["best_f1"]["threshold"])
        test_threshold05 = _metrics(labels["test"], test_prob, 0.5)
        test_threshold_chosen = _metrics(labels["test"], test_prob, chosen_thr)
        target_summary = {
            "best_model": best["family"],
            "best_params": best["params"],
            "val": best["val"],
            "test": {
                "threshold_0.5": test_threshold05,
                "threshold_from_val_best_f1": test_threshold_chosen,
            },
        }
        summary["targets"][target_name] = target_summary
        _write_json(outdir / f"{target_name}_result.json", target_summary)
        _plot_pr(outdir / f"{target_name}_pr_curve.png", labels["test"], test_prob, f"{target_name} PR Curve")
        with (outdir / f"{target_name}_predictions_test.jsonl").open("w", encoding="utf-8") as handle:
            for task_id, yv, prob in zip(test_task_ids, labels["test"].tolist(), test_prob.tolist()):
                handle.write(json.dumps({"task_id": task_id, "y_true": int(yv), "p_pos": float(prob)}) + "\n")

    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
