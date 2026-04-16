from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train prompt-level ExtraTrees baselines from aggregated label features. "
            "Supports both single-stage regression and a two-stage zero-vs-positive gate."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("single_stage", "two_stage", "extreme_filters"),
        default="single_stage",
    )
    parser.add_argument("--train_splits", nargs="+", default=["train"])
    parser.add_argument("--eval_splits", nargs="+", default=["validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--max_features", type=float, default=0.5)
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

    splits: list[str] = []
    task_ids: list[str] = []
    features: list[list[float]] = []
    sampling_accuracy: list[float] = []
    difficulty: list[float] = []

    for entry in manifest:
        split_name = Path(entry["index_path"]).stem.replace("index_", "")
        with Path(entry["index_path"]).open() as f:
            for line in f:
                index_row = json.loads(line)
                label_row = labels[str(index_row["task_id"])]
                splits.append(split_name)
                task_ids.append(str(index_row["task_id"]))
                features.append([float(label_row["aggregated_features"].get(key, 0.0)) for key in feature_keys])
                sampling_accuracy.append(float(label_row["sampling_accuracy"]))
                difficulty.append(float(label_row["difficulty"]))

    return (
        task_ids,
        np.asarray(splits),
        np.asarray(features, dtype=np.float32),
        np.asarray(sampling_accuracy, dtype=np.float32),
        np.asarray(difficulty, dtype=np.float32),
    )


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    order = np.argsort(y_true)
    true_sorted = y_true[order]
    pred_sorted = y_pred[order]
    abs_err_sorted = np.abs(pred_sorted - true_sorted)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(y_true, y_pred, s=12, alpha=0.45, color="#2457a7")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "--", color="#d04a02", linewidth=1.5)
    axes[0].set_title("True vs Predicted Difficulty")
    axes[0].set_xlabel("True difficulty")
    axes[0].set_ylabel("Predicted difficulty")

    axes[1].plot(true_sorted, label="true", color="#111111", linewidth=2)
    axes[1].plot(pred_sorted, label="pred", color="#7a3db8", linewidth=1.5)
    axes[1].set_title("Alignment After Sorting by True Difficulty")
    axes[1].set_xlabel("Test prompts sorted by true difficulty")
    axes[1].set_ylabel("Difficulty")
    axes[1].legend(frameon=False)

    axes[2].plot(abs_err_sorted, color="#2a9d8f", linewidth=1)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Test prompts sorted by true difficulty")
    axes[2].set_ylabel("|pred - true|")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def _plot_extreme_filters(
    y_acc: np.ndarray,
    p_zero: np.ndarray,
    p_one: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    is_zero = y_acc == 0.0
    is_one = y_acc == 1.0
    is_mid = (~is_zero) & (~is_one)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(p_zero[~is_zero], bins=30, alpha=0.65, label="true != 0", color="#4c78a8")
    axes[0].hist(p_zero[is_zero], bins=30, alpha=0.65, label="true = 0", color="#e45756")
    axes[0].set_title("Zero Detector")
    axes[0].set_xlabel("Predicted P(acc = 0)")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)

    axes[1].hist(p_one[~is_one], bins=30, alpha=0.65, label="true != 1", color="#72b7b2")
    axes[1].hist(p_one[is_one], bins=30, alpha=0.65, label="true = 1", color="#f58518")
    axes[1].set_title("One Detector")
    axes[1].set_xlabel("Predicted P(acc = 1)")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False)

    axes[2].scatter(p_zero[is_mid], p_one[is_mid], s=10, alpha=0.35, label="mid", color="#9c755f")
    axes[2].scatter(p_zero[is_zero], p_one[is_zero], s=12, alpha=0.6, label="zero", color="#e45756")
    axes[2].scatter(p_zero[is_one], p_one[is_one], s=12, alpha=0.6, label="one", color="#f58518")
    axes[2].set_title("Extreme-Head Separation")
    axes[2].set_xlabel("P(acc = 0)")
    axes[2].set_ylabel("P(acc = 1)")
    axes[2].legend(frameon=False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task_ids, splits, X, y_acc, y_diff = _load_rows(args.manifest, args.labels_path)

    train_idx = np.where(np.isin(splits, args.train_splits))[0]
    eval_idx = np.where(np.isin(splits, args.eval_splits))[0]
    test_idx = np.where(np.isin(splits, args.test_splits))[0]
    trainval_idx = np.where(np.isin(splits, args.train_splits + args.eval_splits))[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single_stage":
        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y_diff[train_idx])
        val_pred = model.predict(X[eval_idx])
        test_pred = model.predict(X[test_idx])
        summary = {
            "mode": "single_stage",
            "num_features": int(X.shape[1]),
            "validation": {
                "r2": float(r2_score(y_diff[eval_idx], val_pred)),
                "mae": float(mean_absolute_error(y_diff[eval_idx], val_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_diff[eval_idx], val_pred))),
            },
            "test": {
                "r2": float(r2_score(y_diff[test_idx], test_pred)),
                "mae": float(mean_absolute_error(y_diff[test_idx], test_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_diff[test_idx], test_pred))),
            },
        }
        pred_rows = [
            {
                "task_id": task_ids[idx],
                "y_true": float(y_diff[idx]),
                "y_pred": float(pred),
                "sampling_accuracy": float(y_acc[idx]),
                "abs_error": float(abs(pred - y_diff[idx])),
            }
            for idx, pred in zip(test_idx, test_pred)
        ]
        plot_title = f"Prompt-level ET | test R2={summary['test']['r2']:.3f}"
    elif args.mode == "two_stage":
        gate = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        gate.fit(X[trainval_idx], (y_acc[trainval_idx] > 0).astype(int))
        p_pos = gate.predict_proba(X[test_idx])[:, 1]
        gate_pred = (p_pos >= 0.5).astype(int)

        pos_train_idx = trainval_idx[y_acc[trainval_idx] > 0]
        pos_test_idx = test_idx[y_acc[test_idx] > 0]
        reg = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=42,
            n_jobs=-1,
        )
        reg.fit(X[pos_train_idx], y_diff[pos_train_idx])
        pos_pred_on_test = np.clip(reg.predict(X[test_idx]), 0.0, 1.0)
        expected_diff = np.clip((1.0 - p_pos) + p_pos * pos_pred_on_test, 0.0, 1.0)

        summary = {
            "mode": "two_stage",
            "num_features": int(X.shape[1]),
            "stage1_zero_vs_positive": {
                "test_accuracy": float(accuracy_score((y_acc[test_idx] > 0).astype(int), gate_pred)),
                "test_f1": float(f1_score((y_acc[test_idx] > 0).astype(int), gate_pred)),
                "test_auc": float(roc_auc_score((y_acc[test_idx] > 0).astype(int), p_pos)),
                "positive_rate_test": float((y_acc[test_idx] > 0).mean()),
            },
            "stage2_positive_only_regression": {
                "n_trainval": int(len(pos_train_idx)),
                "n_test": int(len(pos_test_idx)),
                "test_r2": float(r2_score(y_diff[pos_test_idx], reg.predict(X[pos_test_idx]))),
                "test_mae": float(mean_absolute_error(y_diff[pos_test_idx], reg.predict(X[pos_test_idx]))),
            },
            "overall_expected_difficulty": {
                "test_r2": float(r2_score(y_diff[test_idx], expected_diff)),
                "test_mae": float(mean_absolute_error(y_diff[test_idx], expected_diff)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_diff[test_idx], expected_diff))),
            },
        }
        pred_rows = [
            {
                "task_id": task_ids[idx],
                "y_true": float(y_diff[idx]),
                "y_pred": float(pred),
                "p_positive": float(p),
                "sampling_accuracy": float(y_acc[idx]),
                "abs_error": float(abs(pred - y_diff[idx])),
            }
            for idx, pred, p in zip(test_idx, expected_diff, p_pos)
        ]
        plot_title = f"Prompt-level two-stage ET | test R2={summary['overall_expected_difficulty']['test_r2']:.3f}"
    else:
        zero_clf = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        one_clf = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            random_state=43,
            n_jobs=-1,
        )
        y_zero_train = (y_acc[trainval_idx] == 0.0).astype(int)
        y_one_train = (y_acc[trainval_idx] == 1.0).astype(int)
        y_zero_test = (y_acc[test_idx] == 0.0).astype(int)
        y_one_test = (y_acc[test_idx] == 1.0).astype(int)
        zero_clf.fit(X[trainval_idx], y_zero_train)
        one_clf.fit(X[trainval_idx], y_one_train)
        p_zero = zero_clf.predict_proba(X[test_idx])[:, 1]
        p_one = one_clf.predict_proba(X[test_idx])[:, 1]
        pred_zero = (p_zero >= 0.5).astype(int)
        pred_one = (p_one >= 0.5).astype(int)

        summary = {
            "mode": "extreme_filters",
            "num_features": int(X.shape[1]),
            "zero_detector": {
                "positive_rate_test": float(y_zero_test.mean()),
                "test_accuracy": float(accuracy_score(y_zero_test, pred_zero)),
                "test_f1": float(f1_score(y_zero_test, pred_zero)),
                "test_auc": float(roc_auc_score(y_zero_test, p_zero)),
            },
            "one_detector": {
                "positive_rate_test": float(y_one_test.mean()),
                "test_accuracy": float(accuracy_score(y_one_test, pred_one)),
                "test_f1": float(f1_score(y_one_test, pred_one)),
                "test_auc": float(roc_auc_score(y_one_test, p_one)),
            },
        }
        pred_rows = [
            {
                "task_id": task_ids[idx],
                "sampling_accuracy": float(y_acc[idx]),
                "difficulty": float(y_diff[idx]),
                "p_zero": float(pz),
                "p_one": float(po),
                "is_zero": bool(y_acc[idx] == 0.0),
                "is_one": bool(y_acc[idx] == 1.0),
            }
            for idx, pz, po in zip(test_idx, p_zero, p_one)
        ]
        plot_title = (
            "Prompt-level ET extreme filters | "
            f"zero AUC={summary['zero_detector']['test_auc']:.3f}, "
            f"one AUC={summary['one_detector']['test_auc']:.3f}"
        )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row) + "\n")
    if args.mode == "extreme_filters":
        y_plot_acc = np.asarray([row["sampling_accuracy"] for row in pred_rows], dtype=np.float32)
        p_zero_plot = np.asarray([row["p_zero"] for row in pred_rows], dtype=np.float32)
        p_one_plot = np.asarray([row["p_one"] for row in pred_rows], dtype=np.float32)
        _plot_extreme_filters(
            y_plot_acc,
            p_zero_plot,
            p_one_plot,
            args.output_dir / "prediction_alignment.png",
            plot_title,
        )
    else:
        y_plot_true = np.asarray([row["y_true"] for row in pred_rows], dtype=np.float32)
        y_plot_pred = np.asarray([row["y_pred"] for row in pred_rows], dtype=np.float32)
        _plot_predictions(y_plot_true, y_plot_pred, args.output_dir / "prediction_alignment.png", plot_title)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
