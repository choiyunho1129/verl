from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train prompt-level ordinal threshold ET models from aggregated label features, "
            "optionally blended with a baseline ET regressor."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_splits", nargs="+", default=["train"])
    parser.add_argument("--eval_splits", nargs="+", default=["validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--max_features", type=float, default=0.5)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument(
        "--blend_alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Pred = (1-alpha) * baseline + alpha * ordinal",
    )
    return parser.parse_args(argv)


def _load_rows(manifest_path: Path, labels_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text())
    labels = {}
    with labels_path.open() as f:
        for line in f:
            row = json.loads(line)
            labels[str(row["task_id"])] = row

    sample_row = next(iter(labels.values()))
    feature_keys = sorted(sample_row["aggregated_features"].keys())

    task_ids: list[str] = []
    splits: list[str] = []
    features: list[list[float]] = []
    difficulty: list[float] = []

    for entry in manifest:
        split_name = Path(entry["index_path"]).stem.replace("index_", "")
        with Path(entry["index_path"]).open() as f:
            for line in f:
                index_row = json.loads(line)
                label_row = labels[str(index_row["task_id"])]
                task_ids.append(str(index_row["task_id"]))
                splits.append(split_name)
                features.append([float(label_row["aggregated_features"].get(key, 0.0)) for key in feature_keys])
                difficulty.append(float(label_row["difficulty"]))

    return (
        task_ids,
        np.asarray(splits),
        np.asarray(features, dtype=np.float32),
        np.asarray(difficulty, dtype=np.float32),
    )


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _bin_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | int | list[float]]]:
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    results = []
    for lo, hi in bins:
        mask = (y_true >= lo) & (y_true < hi)
        if not np.any(mask):
            continue
        results.append(
            {
                "bin": [lo, hi],
                "n": int(mask.sum()),
                "mae": float(np.abs(y_pred[mask] - y_true[mask]).mean()),
                "rmse": float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2))),
                "mean_true": float(y_true[mask].mean()),
                "mean_pred": float(y_pred[mask].mean()),
            }
        )
    return results


def _ordinal_bucket_edges(thresholds: list[float]) -> list[float]:
    ordered = sorted(float(t) for t in thresholds)
    return [0.0] + ordered + [1.01]


def _bucket_means(y: np.ndarray, thresholds: list[float]) -> np.ndarray:
    edges = _ordinal_bucket_edges(thresholds)
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y >= lo) & (y < hi)
        if np.any(mask):
            means.append(float(y[mask].mean()))
        else:
            means.append(float((lo + min(hi, 1.0)) / 2.0))
    return np.asarray(means, dtype=np.float32)


def _ordinal_expected(probs_ge: list[np.ndarray], bucket_means: np.ndarray) -> np.ndarray:
    if not probs_ge:
        raise ValueError("At least one threshold probability array is required.")
    n = probs_ge[0].shape[0]
    ge = [np.asarray(p, dtype=np.float32) for p in probs_ge]
    masses = [1.0 - ge[0]]
    for prev, nxt in zip(ge[:-1], ge[1:]):
        masses.append(np.clip(prev - nxt, 0.0, 1.0))
    masses.append(np.clip(ge[-1], 0.0, 1.0))
    mass_matrix = np.stack(masses, axis=1)
    row_sums = mass_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    mass_matrix = mass_matrix / row_sums
    return np.clip(mass_matrix @ bucket_means, 0.0, 1.0)


def _plot_predictions(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    ordinal_pred: np.ndarray,
    blend_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    order = np.argsort(y_true)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(y_true[order], label="true", color="#111111", linewidth=2)
    axes[0].plot(baseline_pred[order], label="baseline", linewidth=1.2)
    axes[0].plot(ordinal_pred[order], label="ordinal", linewidth=1.2)
    axes[0].plot(blend_pred[order], label="blend", linewidth=1.2)
    axes[0].set_title("Sorted Alignment")
    axes[0].legend(frameon=False)

    axes[1].scatter(y_true, ordinal_pred, s=10, alpha=0.35, label="ordinal", color="#4c78a8")
    axes[1].scatter(y_true, blend_pred, s=10, alpha=0.35, label="blend", color="#e45756")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1].set_xlabel("True difficulty")
    axes[1].set_ylabel("Predicted difficulty")
    axes[1].set_title("True vs Alternative Predictions")
    axes[1].legend(frameon=False)

    hard_mask = y_true >= 0.8
    axes[2].hist(np.abs(baseline_pred[hard_mask] - y_true[hard_mask]), bins=25, alpha=0.6, label="baseline", color="#72b7b2")
    axes[2].hist(np.abs(blend_pred[hard_mask] - y_true[hard_mask]), bins=25, alpha=0.6, label="blend", color="#f58518")
    axes[2].set_title("Hard-Tail Absolute Error")
    axes[2].set_xlabel("|pred - true| on difficulty >= 0.8")
    axes[2].set_ylabel("Count")
    axes[2].legend(frameon=False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task_ids, splits, X, y = _load_rows(args.manifest, args.labels_path)

    train_idx = np.where(np.isin(splits, args.train_splits))[0]
    eval_idx = np.where(np.isin(splits, args.eval_splits))[0]
    test_idx = np.where(np.isin(splits, args.test_splits))[0]
    trainval_idx = np.where(np.isin(splits, args.train_splits + args.eval_splits))[0]
    thresholds = sorted(args.thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_val = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=42,
        n_jobs=-1,
    )
    baseline_val.fit(X[train_idx], y[train_idx])
    baseline_eval_pred = np.clip(baseline_val.predict(X[eval_idx]), 0.0, 1.0)

    bucket_means_train = _bucket_means(y[train_idx], thresholds)
    clf_eval_outputs: list[dict[str, float]] = []
    probs_eval: list[np.ndarray] = []
    for offset, threshold in enumerate(thresholds):
        clf = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            random_state=42 + offset,
            n_jobs=-1,
        )
        target = (y[train_idx] >= threshold).astype(int)
        clf.fit(X[train_idx], target)
        prob = clf.predict_proba(X[eval_idx])[:, 1]
        probs_eval.append(prob)
        clf_eval_outputs.append(
            {
                "threshold": threshold,
                "eval_auc": float(roc_auc_score((y[eval_idx] >= threshold).astype(int), prob)),
            }
        )
    ordinal_eval_pred = _ordinal_expected(probs_eval, bucket_means_train)

    blend_eval_metrics = {}
    for alpha in args.blend_alphas:
        pred = np.clip((1.0 - alpha) * baseline_eval_pred + alpha * ordinal_eval_pred, 0.0, 1.0)
        blend_eval_metrics[str(alpha)] = {
            **_reg_metrics(y[eval_idx], pred),
            "bins": _bin_metrics(y[eval_idx], pred),
        }
    best_alpha = max(blend_eval_metrics.items(), key=lambda kv: kv[1]["r2"])[0]
    best_alpha_float = float(best_alpha)

    baseline_tv = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=42,
        n_jobs=-1,
    )
    baseline_tv.fit(X[trainval_idx], y[trainval_idx])
    baseline_test = np.clip(baseline_tv.predict(X[test_idx]), 0.0, 1.0)

    bucket_means_tv = _bucket_means(y[trainval_idx], thresholds)
    probs_test: list[np.ndarray] = []
    clf_test_outputs: list[dict[str, float]] = []
    for offset, threshold in enumerate(thresholds):
        clf = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            random_state=42 + offset,
            n_jobs=-1,
        )
        target = (y[trainval_idx] >= threshold).astype(int)
        clf.fit(X[trainval_idx], target)
        prob = clf.predict_proba(X[test_idx])[:, 1]
        probs_test.append(prob)
        clf_test_outputs.append(
            {
                "threshold": threshold,
                "test_auc": float(roc_auc_score((y[test_idx] >= threshold).astype(int), prob)),
            }
        )

    ordinal_test = _ordinal_expected(probs_test, bucket_means_tv)
    blend_test = np.clip((1.0 - best_alpha_float) * baseline_test + best_alpha_float * ordinal_test, 0.0, 1.0)

    summary = {
        "thresholds": thresholds,
        "bucket_means_train": bucket_means_train.tolist(),
        "bucket_means_trainval": bucket_means_tv.tolist(),
        "classifier_eval": clf_eval_outputs,
        "classifier_test": clf_test_outputs,
        "baseline_test": {
            **_reg_metrics(y[test_idx], baseline_test),
            "bins": _bin_metrics(y[test_idx], baseline_test),
        },
        "ordinal_test": {
            **_reg_metrics(y[test_idx], ordinal_test),
            "bins": _bin_metrics(y[test_idx], ordinal_test),
        },
        "blend_eval": blend_eval_metrics,
        "best_blend_alpha": best_alpha_float,
        "blend_test": {
            **_reg_metrics(y[test_idx], blend_test),
            "bins": _bin_metrics(y[test_idx], blend_test),
        },
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as f:
        for local_pos, idx in enumerate(test_idx):
            row = {
                "task_id": task_ids[idx],
                "y_true": float(y[idx]),
                "baseline_pred": float(baseline_test[local_pos]),
                "ordinal_pred": float(ordinal_test[local_pos]),
                "blend_pred": float(blend_test[local_pos]),
            }
            for threshold, probs in zip(thresholds, probs_test):
                row[f"p_ge_{threshold:.1f}"] = float(probs[local_pos])
            f.write(json.dumps(row) + "\n")

    _plot_predictions(
        y[test_idx],
        baseline_test,
        ordinal_test,
        blend_test,
        args.output_dir / "comparison.png",
        f"Prompt-level ordinal ET | blend alpha={best_alpha_float:.2f}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
