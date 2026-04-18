from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.single_rollout_hidden_utils import (
    apply_prompt_hidden_pca,
    apply_rollout_hidden_pca,
    build_matrix,
    build_prompt_scalar_lookup,
    build_rollout_hidden_lookup,
    build_rollout_index_lookup,
    build_split_lookup,
    fit_prompt_hidden_pca,
    fit_rollout_hidden_pca,
    group_weak_rollouts,
    load_labels_by_task,
    load_prompt_hidden_lookup,
    prompt_mean_metrics,
    reg_metrics,
    save_diagnostics_plot,
    select_single_rollout,
    write_predictions,
)


def _build_support_model_config(*, alpha: float | None, feature_dim: int) -> dict[str, float | int | None]:
    return {
        "alpha": float(alpha) if alpha is not None else None,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "feature_dim": int(feature_dim),
    }


def _add_support_compatibility(
    *,
    estimator_config: dict,
    rollout_hidden_pca,
    bundle: dict,
) -> tuple[dict, dict]:
    prompt_projection = dict(estimator_config["prompt"]["hidden_projection"])
    response_projection = dict(estimator_config["response"]["hidden_projection"])
    response_feature_keys = list(estimator_config["response"].get("scalar_keys", []))
    derived_response_feature_keys = list(estimator_config["response"].get("derived_scalar_keys", []))

    original_model_config = dict(estimator_config["model"])
    support_model_config = _build_support_model_config(
        alpha=original_model_config.get("alpha"),
        feature_dim=int(original_model_config["feature_dim"]),
    )

    estimator_config["model_full"] = original_model_config
    estimator_config["model"] = support_model_config
    estimator_config["prompt_hidden_projection"] = prompt_projection
    estimator_config["response_hidden_projection"] = response_projection
    estimator_config["response_feature_keys"] = response_feature_keys
    estimator_config["derived_response_feature_keys"] = derived_response_feature_keys

    bundle["bundle_version"] = 2
    bundle["response_hidden_pca"] = rollout_hidden_pca
    bundle["trajectory_hidden_pca"] = rollout_hidden_pca
    bundle["think_end_hidden_pca"] = rollout_hidden_pca
    return estimator_config, bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a weak-only single-rollout Ridge value estimator."
    )
    parser.add_argument("--weak_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--weak_labels_path", type=Path, required=True)
    parser.add_argument("--weak_prompt_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_rollout_hidden_paths", nargs="+", type=Path)
    parser.add_argument("--weak_rollout_index_paths", nargs="+", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_layer_index", type=int, default=26)
    parser.add_argument("--rollout_component", type=str, default="think_end_last10_hidden")
    parser.add_argument("--rollout_pool_mode", type=str, default="mean")
    parser.add_argument("--feature_mode", choices=["prompt_only", "prompt_plus_rollout"], required=True)
    parser.add_argument("--prompt_feature_keys", nargs="*", default=[])
    parser.add_argument("--rollout_scalar_keys", nargs="*", default=[])
    parser.add_argument("--derived_rollout_scalar_keys", nargs="*", default=[])
    parser.add_argument("--extra_rollout_scalar_field_paths", nargs="*", default=[])
    parser.add_argument("--prompt_hidden_pca_dim", type=int, default=0)
    parser.add_argument("--rollout_hidden_pca_dim", type=int, default=0)
    parser.add_argument("--single_rollout_strategy", choices=["first", "all"], default="first")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--alphas", nargs="+", type=float, default=[100.0, 300.0, 1000.0, 3000.0, 10000.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    weak_labels_by_task = load_labels_by_task(args.weak_labels_path)
    split_lookup = build_split_lookup(args.weak_prompt_dataset_dir.expanduser().resolve())
    prompt_scalar_lookup = build_prompt_scalar_lookup(weak_labels_by_task, list(args.prompt_feature_keys))

    prompt_lookup = load_prompt_hidden_lookup(
        [path.expanduser().resolve() for path in args.weak_prompt_hidden_paths],
        [path.expanduser().resolve() for path in args.weak_prompt_index_paths],
        layer_index=args.prompt_layer_index,
    )
    prompt_hidden_pca = fit_prompt_hidden_pca(prompt_lookup, split_lookup, int(args.prompt_hidden_pca_dim))
    prompt_lookup = apply_prompt_hidden_pca(prompt_lookup, prompt_hidden_pca)

    rollout_hidden_lookup = None
    if args.feature_mode == "prompt_plus_rollout":
        if not args.weak_rollout_hidden_paths or not args.weak_rollout_index_paths:
            raise ValueError("Prompt+rollout mode requires weak rollout hidden/index paths.")
        rollout_hidden_lookup = build_rollout_hidden_lookup(
            [path.expanduser().resolve() for path in args.weak_rollout_hidden_paths],
            [path.expanduser().resolve() for path in args.weak_rollout_index_paths],
            component_name=args.rollout_component,
            layer_index=0,
            pool_mode=args.rollout_pool_mode,
        )

    rollout_index_lookup = None
    if args.weak_rollout_index_paths:
        rollout_index_lookup = build_rollout_index_lookup(
            [path.expanduser().resolve() for path in args.weak_rollout_index_paths]
        )

    weak_grouped = group_weak_rollouts(
        weak_run_dirs=[path.expanduser().resolve() for path in args.weak_run_dirs],
        split_lookup=split_lookup,
        labels_by_task=weak_labels_by_task,
        rollout_hidden_lookup=rollout_hidden_lookup,
        rollout_index_lookup=rollout_index_lookup,
        rollout_scalar_keys=list(args.rollout_scalar_keys),
        derived_rollout_scalar_keys=list(args.derived_rollout_scalar_keys),
        extra_rollout_scalar_field_paths=list(args.extra_rollout_scalar_field_paths),
    )
    weak_rows = select_single_rollout(weak_grouped, args.single_rollout_strategy)
    rollout_hidden_pca = None
    if args.feature_mode == "prompt_plus_rollout":
        rollout_hidden_pca = fit_rollout_hidden_pca(weak_rows, int(args.rollout_hidden_pca_dim))
        weak_rows = apply_rollout_hidden_pca(weak_rows, rollout_hidden_pca)
    weak_x, weak_y, weak_splits, weak_meta = build_matrix(
        weak_rows,
        prompt_lookup,
        prompt_scalar_lookup,
        feature_mode=args.feature_mode,
    )

    weak_train_mask = weak_splits == "train"
    weak_val_mask = weak_splits == "validation"
    if not np.any(weak_train_mask):
        raise ValueError("No weak train rows were built.")
    if not np.any(weak_val_mask):
        raise ValueError("No weak validation rows were built.")

    x_train, y_train = weak_x[weak_train_mask], weak_y[weak_train_mask]
    x_weak_val, y_weak_val = weak_x[weak_val_mask], weak_y[weak_val_mask]
    weak_val_meta = [weak_meta[idx] for idx in np.where(weak_val_mask)[0]]

    best_bundle: dict[str, Any] | None = None
    best_weak_val_r2 = -1e18
    for alpha in args.alphas:
        model_name = f"ridge_a{alpha:g}"
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=alpha, random_state=args.random_seed)),
            ]
        )
        estimator.fit(x_train, y_train)
        weak_val_pred = np.clip(np.asarray(estimator.predict(x_weak_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
        weak_val_row_metrics = reg_metrics(y_weak_val, weak_val_pred)
        weak_val_prompt_metrics, weak_val_prompt_rows = prompt_mean_metrics(y_weak_val, weak_val_pred, weak_val_meta)
        result = {
            "name": model_name,
            "weak_val_row_metrics": weak_val_row_metrics,
            "weak_val_prompt_mean_metrics": weak_val_prompt_metrics,
        }
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        if weak_val_prompt_metrics["r2"] > best_weak_val_r2:
            best_weak_val_r2 = weak_val_prompt_metrics["r2"]
            best_bundle = {
                "name": model_name,
                "estimator": estimator,
                "weak_val_row_metrics": weak_val_row_metrics,
                "weak_val_prompt_mean_metrics": weak_val_prompt_metrics,
                "weak_val_prompt_rows": weak_val_prompt_rows,
                "feature_dim": int(x_train.shape[1]),
                "num_train_rows": int(x_train.shape[0]),
                "num_weak_val_rows": int(x_weak_val.shape[0]),
                "num_weak_val_prompts": int(len(weak_val_prompt_rows)),
            }

    if best_bundle is None:
        raise RuntimeError("Failed to fit any model.")

    estimator_pipeline = best_bundle["estimator"]
    estimator_step = estimator_pipeline.named_steps.get("model", estimator_pipeline)
    estimator_config = {
        "prediction_target": "value",
        "prompt": {
            "hidden_layer_index": int(args.prompt_layer_index),
            "hidden_projection": {
                "type": None if prompt_hidden_pca is None else "pca",
                "input_dim": None if prompt_hidden_pca is None else int(prompt_hidden_pca.n_features_in_),
                "output_dim": None if prompt_hidden_pca is None else int(prompt_hidden_pca.n_components_),
            },
            "prompt_scalar_keys": list(args.prompt_feature_keys),
        },
        "response": {
            "hidden_component": args.rollout_component if args.feature_mode == "prompt_plus_rollout" else None,
            "hidden_pool_mode": args.rollout_pool_mode if args.feature_mode == "prompt_plus_rollout" else None,
            "hidden_projection": {
                "type": None if rollout_hidden_pca is None else "pca",
                "input_dim": None if rollout_hidden_pca is None else int(rollout_hidden_pca.n_features_in_),
                "output_dim": None if rollout_hidden_pca is None else int(rollout_hidden_pca.n_components_),
            },
            "scalar_keys": list(args.rollout_scalar_keys),
            "derived_scalar_keys": list(args.derived_rollout_scalar_keys),
            "extra_scalar_field_paths": list(args.extra_rollout_scalar_field_paths),
        },
        "model": {
            "pipeline": ["standard_scaler", type(estimator_step).__name__.lower()],
            "estimator_class": type(estimator_step).__name__,
            "alpha": float(getattr(estimator_step, "alpha", 0.0)) if hasattr(estimator_step, "alpha") else None,
            "clip_min": 0.0,
            "clip_max": 1.0,
            "best_model_name": best_bundle["name"],
            "feature_dim": int(best_bundle["feature_dim"]),
        },
    }
    bundle = {
        "bundle_type": "single_rollout_value_estimator",
        "bundle_version": 1,
        "config": estimator_config,
        "feature_mode": args.feature_mode,
        "single_rollout_strategy": args.single_rollout_strategy,
        "rollout_component": args.rollout_component if args.feature_mode == "prompt_plus_rollout" else None,
        "rollout_pool_mode": args.rollout_pool_mode if args.feature_mode == "prompt_plus_rollout" else None,
        "estimator": estimator_pipeline,
        "prompt_hidden_pca": prompt_hidden_pca,
        "rollout_hidden_pca": rollout_hidden_pca,
    }
    estimator_config, bundle = _add_support_compatibility(
        estimator_config=estimator_config,
        rollout_hidden_pca=rollout_hidden_pca,
        bundle=bundle,
    )
    joblib.dump(bundle, args.output_dir / "model.joblib")
    (args.output_dir / "estimator_config.json").write_text(json.dumps(estimator_config, indent=2), encoding="utf-8")

    write_predictions(args.output_dir / "predictions_weakval.jsonl", best_bundle["weak_val_prompt_rows"], weak_labels_by_task)
    save_diagnostics_plot(
        args.output_dir / "prediction_diagnostics_weakval.png",
        best_bundle["weak_val_prompt_rows"],
        f"Weak Validation: {best_bundle['name']}",
    )

    summary = {
        "setting": "weak_only_single_rollout_hidden",
        "prediction_target": "value",
        "feature_mode": args.feature_mode,
        "prompt_layer_index": int(args.prompt_layer_index),
        "prompt_hidden_pca_dim": int(args.prompt_hidden_pca_dim),
        "rollout_hidden_pca_dim": int(args.rollout_hidden_pca_dim),
        "prompt_feature_keys": list(args.prompt_feature_keys),
        "rollout_scalar_keys": list(args.rollout_scalar_keys),
        "derived_rollout_scalar_keys": list(args.derived_rollout_scalar_keys),
        "extra_rollout_scalar_field_paths": list(args.extra_rollout_scalar_field_paths),
        "rollout_component": args.rollout_component if args.feature_mode == "prompt_plus_rollout" else None,
        "rollout_pool_mode": args.rollout_pool_mode if args.feature_mode == "prompt_plus_rollout" else None,
        "single_rollout_strategy": args.single_rollout_strategy,
        "alphas": [float(alpha) for alpha in args.alphas],
        "best_model": best_bundle["name"],
        "feature_dim": best_bundle["feature_dim"],
        "num_train_rows": best_bundle["num_train_rows"],
        "num_weak_val_rows": best_bundle["num_weak_val_rows"],
        "num_weak_val_prompts": best_bundle["num_weak_val_prompts"],
        "weak_val_row_metrics": best_bundle["weak_val_row_metrics"],
        "weak_val_prompt_mean_metrics": best_bundle["weak_val_prompt_mean_metrics"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
