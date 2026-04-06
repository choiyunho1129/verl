from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from classifer_training.data import load_aligned_examples, load_manifest
from classifer_training.features import FeatureExtractionConfig, build_feature_matrix
from classifer_training.models import build_estimator, evaluate_classification, evaluate_regression
from classifer_training.utils import coerce_float, get_nested_value, write_jsonl


def _resolve_target(label_row: dict[str, Any], target_field: str, target_transform: str) -> float:
    value = coerce_float(get_nested_value(label_row, target_field, default=None))
    if value is None:
        raise KeyError(f"Target field {target_field!r} is missing or non-numeric.")

    if target_transform == "identity":
        return value
    if target_transform == "difficulty":
        return 1.0 - value
    raise ValueError(f"Unsupported target transform: {target_transform}")


def _normalize_split_names(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    normalized = [value.strip() for value in values if value.strip()]
    return normalized or None


def _evaluate_split(
    *,
    split_name: str,
    estimator,
    task_type: str,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    metadata_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_pred = estimator.predict(X_eval)
    if task_type == "classification":
        metrics = evaluate_classification(estimator, X_eval, y_eval, y_pred)
    else:
        metrics = evaluate_regression(y_eval, y_pred)

    predictions: list[dict[str, Any]] = []
    for meta, y_true_value, y_pred_value in zip(metadata_rows, y_eval, y_pred):
        predictions.append(
            {
                "split": split_name,
                "dataset_name": meta.get("dataset_name"),
                "task_id": meta.get("task_id"),
                "user_input": meta.get("user_input"),
                "y_true": float(y_true_value) if task_type == "regression" else int(y_true_value),
                "y_pred": float(y_pred_value) if task_type == "regression" else int(y_pred_value),
            }
        )
    return metrics, predictions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a prompt-difficulty predictor from hidden states and optional "
            "auxiliary features."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--task_type",
        choices=("regression", "classification"),
        default="regression",
    )
    parser.add_argument("--target_field", type=str, default="sampling_accuracy")
    parser.add_argument(
        "--target_transform",
        choices=("identity", "difficulty"),
        default="identity",
        help="Optionally convert accuracy into difficulty using 1 - accuracy.",
    )
    parser.add_argument(
        "--classification_threshold",
        type=float,
        default=0.5,
        help="Used only when task_type=classification. y >= threshold becomes class 1.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Regression: linear, ridge, mlp, random_forest, hist_gb. "
            "Classification: logistic, linear_svm, mlp, random_forest, hist_gb."
        ),
    )
    parser.add_argument("--components", nargs="*", default=["hidden"])
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument(
        "--component_pooling",
        choices=("concat", "mean", "max"),
        default="concat",
    )
    parser.add_argument(
        "--engineered_feature_set",
        type=str,
        default="raw",
        help=(
            "Optional rollout-oriented hidden feature construction mode. "
            "Examples: raw, prompt_response, delta_only, prompt_response_delta_prod."
        ),
    )
    parser.add_argument(
        "--extra_features",
        nargs="*",
        default=[],
        help=(
            "Dotted numeric feature paths. Unprefixed paths search labels first, then "
            "the index file. Prefix with label. or index. to force a source."
        ),
    )
    parser.add_argument(
        "--disable_hidden_features",
        action="store_true",
        help="Train on auxiliary features only.",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--strict_alignment", action="store_true")
    parser.add_argument("--train_splits", nargs="*", default=None)
    parser.add_argument("--eval_splits", nargs="*", default=None)
    parser.add_argument("--test_splits", nargs="*", default=None)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--hidden_layer_sizes", type=str, default="256,64")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--class_weight_balanced", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = load_manifest(args.manifest.expanduser().resolve())
    examples = load_aligned_examples(
        manifest_entries=manifest_entries,
        strict=args.strict_alignment,
    )
    if not examples:
        raise ValueError("No aligned examples were loaded from the manifest.")

    feature_config = FeatureExtractionConfig(
        components=args.components,
        layers=args.layers,
        component_pooling=args.component_pooling,
        extra_feature_paths=args.extra_features,
        include_hidden_features=not args.disable_hidden_features,
        engineered_feature_set=args.engineered_feature_set,
    )
    X, feature_names, metadata_rows = build_feature_matrix(examples, feature_config)

    y_continuous = np.asarray(
        [
            _resolve_target(
                example.label_row,
                target_field=args.target_field,
                target_transform=args.target_transform,
            )
            for example in examples
        ],
        dtype=np.float32,
    )

    if args.task_type == "classification":
        y = (y_continuous >= args.classification_threshold).astype(np.int64)
    else:
        y = y_continuous

    train_splits = _normalize_split_names(args.train_splits)
    eval_splits = _normalize_split_names(args.eval_splits)
    test_splits = _normalize_split_names(args.test_splits)

    metrics_by_split: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []

    if train_splits is not None:
        row_splits = [str(meta.get("split", "")) for meta in metadata_rows]
        train_indices = [idx for idx, split in enumerate(row_splits) if split in set(train_splits)]
        if not train_indices:
            raise ValueError(f"No examples matched train_splits={train_splits}.")

        eval_indices = [idx for idx, split in enumerate(row_splits) if eval_splits and split in set(eval_splits)]
        test_indices = [idx for idx, split in enumerate(row_splits) if test_splits and split in set(test_splits)]

        X_train = X[train_indices]
        y_train = y[train_indices]
        split_name = "predefined"
        num_eval = 0
        num_test = 0
    elif not 0.0 < args.test_size < 1.0:
        X_train = X
        y_train = y
        split_name = "train"
        eval_indices = list(range(len(y)))
        test_indices = []
        num_eval = len(eval_indices)
        num_test = 0
    else:
        stratify = y if args.task_type == "classification" and len(np.unique(y)) > 1 else None
        (
            X_train,
            X_eval,
            y_train,
            y_eval,
            _train_indices,
            eval_indices,
        ) = train_test_split(
            X,
            y,
            np.arange(len(y)),
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )
        split_name = "validation"
        test_indices = []
        num_eval = len(eval_indices)
        num_test = 0

    estimator = build_estimator(
        task_type=args.task_type,
        model_name=args.model,
        hidden_layer_sizes=args.hidden_layer_sizes,
        alpha=args.alpha,
        max_iter=args.max_iter,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        class_weight_balanced=args.class_weight_balanced,
    )
    estimator.fit(X_train, y_train)

    if train_splits is not None:
        if eval_indices:
            metrics_by_split["validation"], split_predictions = _evaluate_split(
                split_name="validation",
                estimator=estimator,
                task_type=args.task_type,
                X_eval=X[eval_indices],
                y_eval=y[eval_indices],
                metadata_rows=[metadata_rows[idx] for idx in eval_indices],
            )
            predictions.extend(split_predictions)
            num_eval = len(eval_indices)
        if test_indices:
            metrics_by_split["test"], split_predictions = _evaluate_split(
                split_name="test",
                estimator=estimator,
                task_type=args.task_type,
                X_eval=X[test_indices],
                y_eval=y[test_indices],
                metadata_rows=[metadata_rows[idx] for idx in test_indices],
            )
            predictions.extend(split_predictions)
            num_test = len(test_indices)
        if not metrics_by_split:
            metrics_by_split["train"], split_predictions = _evaluate_split(
                split_name="train",
                estimator=estimator,
                task_type=args.task_type,
                X_eval=X[train_indices],
                y_eval=y[train_indices],
                metadata_rows=[metadata_rows[idx] for idx in train_indices],
            )
            predictions.extend(split_predictions)
            num_eval = len(train_indices)
            num_test = 0
        metrics = metrics_by_split.get("validation") or metrics_by_split.get("test") or next(iter(metrics_by_split.values()))
    else:
        metrics, predictions = _evaluate_split(
            split_name=split_name,
            estimator=estimator,
            task_type=args.task_type,
            X_eval=X[eval_indices] if split_name != "validation" else X_eval,
            y_eval=y[eval_indices] if split_name != "validation" else y_eval,
            metadata_rows=[metadata_rows[int(index)] for index in eval_indices],
        )
        metrics_by_split[split_name] = metrics

    config_payload = {
        "task_type": args.task_type,
        "target_field": args.target_field,
        "target_transform": args.target_transform,
        "classification_threshold": args.classification_threshold,
        "model": args.model,
        "components": args.components,
        "layers": args.layers,
        "component_pooling": args.component_pooling,
        "engineered_feature_set": args.engineered_feature_set,
        "extra_features": args.extra_features,
        "include_hidden_features": not args.disable_hidden_features,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train_splits": train_splits,
        "eval_splits": eval_splits,
        "test_splits": test_splits,
        "num_examples": int(len(X)),
        "num_train": int(len(X_train)),
        "num_eval": int(num_eval),
        "num_test": int(num_test),
        "num_features": int(X.shape[1]),
        "eval_split": split_name,
        "metrics": metrics,
        "metrics_by_split": metrics_by_split,
    }

    joblib.dump(estimator, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (output_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "components": args.components,
                "layers": args.layers,
                "component_pooling": args.component_pooling,
                "engineered_feature_set": args.engineered_feature_set,
                "extra_features": args.extra_features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    print(json.dumps(config_payload, indent=2))


if __name__ == "__main__":
    main()
