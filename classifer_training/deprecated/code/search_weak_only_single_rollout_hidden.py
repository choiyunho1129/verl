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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search weak-only single-rollout Ridge value estimator hyperparameters."
    )
    parser.add_argument("--weak_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--weak_labels_path", type=Path, required=True)
    parser.add_argument("--weak_prompt_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_rollout_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_rollout_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_layer_index", type=int, default=26)
    parser.add_argument("--rollout_components", nargs="+", default=["think_end_hidden", "think_end_last10_hidden"])
    parser.add_argument("--rollout_pool_mode", type=str, default="mean")
    parser.add_argument("--prompt_feature_keys", nargs="*", default=[])
    parser.add_argument("--rollout_scalar_keys", nargs="*", default=[])
    parser.add_argument("--derived_rollout_scalar_keys", nargs="*", default=[])
    parser.add_argument("--extra_rollout_scalar_field_paths", nargs="*", default=[])
    parser.add_argument("--prompt_hidden_pca_dims", nargs="+", type=int, default=[0, 8, 16, 32, 64, 128])
    parser.add_argument("--rollout_hidden_pca_dims", nargs="+", type=int, default=[0, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--single_rollout_strategy", choices=["first", "all"], default="first")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0],
    )
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def _make_estimator(alpha: float, random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=random_seed)),
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    weak_labels_by_task = load_labels_by_task(args.weak_labels_path)
    split_lookup = build_split_lookup(args.weak_prompt_dataset_dir.expanduser().resolve())
    prompt_scalar_lookup = build_prompt_scalar_lookup(weak_labels_by_task, list(args.prompt_feature_keys))

    base_prompt_lookup = load_prompt_hidden_lookup(
        [path.expanduser().resolve() for path in args.weak_prompt_hidden_paths],
        [path.expanduser().resolve() for path in args.weak_prompt_index_paths],
        layer_index=args.prompt_layer_index,
    )
    prompt_lookup_by_dim: dict[int, tuple[dict[str, np.ndarray], Any]] = {}
    for prompt_pca_dim in sorted(set(int(value) for value in args.prompt_hidden_pca_dims)):
        prompt_pca = fit_prompt_hidden_pca(base_prompt_lookup, split_lookup, prompt_pca_dim)
        prompt_lookup_by_dim[prompt_pca_dim] = (apply_prompt_hidden_pca(base_prompt_lookup, prompt_pca), prompt_pca)

    rollout_index_lookup = build_rollout_index_lookup(
        [path.expanduser().resolve() for path in args.weak_rollout_index_paths]
    )

    all_results: list[dict[str, Any]] = []
    best_bundle: dict[str, Any] | None = None
    best_weak_val_r2 = -1e18

    for rollout_component in args.rollout_components:
        rollout_hidden_lookup = build_rollout_hidden_lookup(
            [path.expanduser().resolve() for path in args.weak_rollout_hidden_paths],
            [path.expanduser().resolve() for path in args.weak_rollout_index_paths],
            component_name=rollout_component,
            layer_index=0,
            pool_mode=args.rollout_pool_mode,
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
        base_rows = select_single_rollout(weak_grouped, args.single_rollout_strategy)

        rollout_rows_by_dim: dict[int, tuple[list[dict[str, Any]], Any]] = {}
        for rollout_pca_dim in sorted(set(int(value) for value in args.rollout_hidden_pca_dims)):
            rollout_pca = fit_rollout_hidden_pca(base_rows, rollout_pca_dim)
            rollout_rows_by_dim[rollout_pca_dim] = (apply_rollout_hidden_pca(base_rows, rollout_pca), rollout_pca)

        for prompt_pca_dim, (prompt_lookup, prompt_pca) in prompt_lookup_by_dim.items():
            for rollout_pca_dim, (weak_rows, rollout_pca) in rollout_rows_by_dim.items():
                weak_x, weak_y, weak_splits, weak_meta = build_matrix(
                    weak_rows,
                    prompt_lookup,
                    prompt_scalar_lookup,
                    feature_mode="prompt_plus_rollout",
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

                for alpha in args.alphas:
                    model_name = (
                        f"{rollout_component}"
                        f"_ppca{prompt_pca_dim}"
                        f"_rpca{rollout_pca_dim}"
                        f"_ridge_a{alpha:g}"
                    )
                    estimator = _make_estimator(alpha, args.random_seed)
                    estimator.fit(x_train, y_train)
                    weak_val_pred = np.clip(
                        np.asarray(estimator.predict(x_weak_val), dtype=np.float32).reshape(-1), 0.0, 1.0
                    )
                    weak_val_row_metrics = reg_metrics(y_weak_val, weak_val_pred)
                    weak_val_prompt_metrics, weak_val_prompt_rows = prompt_mean_metrics(
                        y_weak_val, weak_val_pred, weak_val_meta
                    )
                    result = {
                        "name": model_name,
                        "rollout_component": rollout_component,
                        "prompt_hidden_pca_dim": int(prompt_pca_dim),
                        "rollout_hidden_pca_dim": int(rollout_pca_dim),
                        "alpha": float(alpha),
                        "feature_dim": int(x_train.shape[1]),
                        "num_train_rows": int(x_train.shape[0]),
                        "num_weak_val_rows": int(x_weak_val.shape[0]),
                        "num_weak_val_prompts": int(len(weak_val_prompt_rows)),
                        "weak_val_row_metrics": weak_val_row_metrics,
                        "weak_val_prompt_mean_metrics": weak_val_prompt_metrics,
                    }
                    all_results.append(result)
                    with results_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(result) + "\n")

                    if weak_val_prompt_metrics["r2"] > best_weak_val_r2:
                        best_weak_val_r2 = weak_val_prompt_metrics["r2"]
                        best_bundle = {
                            "result": result,
                            "estimator": estimator,
                            "prompt_pca": prompt_pca,
                            "rollout_pca": rollout_pca,
                            "weak_val_prompt_rows": weak_val_prompt_rows,
                        }

    if best_bundle is None:
        raise RuntimeError("Failed to fit any model.")

    best_result = best_bundle["result"]
    estimator_pipeline = best_bundle["estimator"]
    estimator_step = estimator_pipeline.named_steps["model"]
    estimator_config = {
        "prediction_target": "value",
        "prompt": {
            "hidden_layer_index": int(args.prompt_layer_index),
            "hidden_projection": {
                "type": None if best_bundle["prompt_pca"] is None else "pca",
                "input_dim": None if best_bundle["prompt_pca"] is None else int(best_bundle["prompt_pca"].n_features_in_),
                "output_dim": None if best_bundle["prompt_pca"] is None else int(best_bundle["prompt_pca"].n_components_),
            },
            "prompt_scalar_keys": list(args.prompt_feature_keys),
        },
        "response": {
            "hidden_component": str(best_result["rollout_component"]),
            "hidden_pool_mode": args.rollout_pool_mode,
            "hidden_projection": {
                "type": None if best_bundle["rollout_pca"] is None else "pca",
                "input_dim": None if best_bundle["rollout_pca"] is None else int(best_bundle["rollout_pca"].n_features_in_),
                "output_dim": None if best_bundle["rollout_pca"] is None else int(best_bundle["rollout_pca"].n_components_),
            },
            "scalar_keys": list(args.rollout_scalar_keys),
            "derived_scalar_keys": list(args.derived_rollout_scalar_keys),
            "extra_scalar_field_paths": list(args.extra_rollout_scalar_field_paths),
        },
        "model": {
            "pipeline": ["standard_scaler", type(estimator_step).__name__.lower()],
            "estimator_class": type(estimator_step).__name__,
            "alpha": float(estimator_step.alpha),
            "clip_min": 0.0,
            "clip_max": 1.0,
            "best_model_name": str(best_result["name"]),
            "feature_dim": int(best_result["feature_dim"]),
        },
    }
    bundle = {
        "bundle_type": "single_rollout_value_estimator",
        "bundle_version": 1,
        "config": estimator_config,
        "feature_mode": "prompt_plus_rollout",
        "single_rollout_strategy": args.single_rollout_strategy,
        "rollout_component": str(best_result["rollout_component"]),
        "rollout_pool_mode": args.rollout_pool_mode,
        "estimator": estimator_pipeline,
        "prompt_hidden_pca": best_bundle["prompt_pca"],
        "rollout_hidden_pca": best_bundle["rollout_pca"],
    }
    joblib.dump(bundle, args.output_dir / "model.joblib")
    (args.output_dir / "estimator_config.json").write_text(json.dumps(estimator_config, indent=2), encoding="utf-8")

    write_predictions(args.output_dir / "predictions_weakval.jsonl", best_bundle["weak_val_prompt_rows"], weak_labels_by_task)
    save_diagnostics_plot(
        args.output_dir / "prediction_diagnostics_weakval.png",
        best_bundle["weak_val_prompt_rows"],
        f"Weak Validation: {best_result['name']}",
    )

    top_results = sorted(
        all_results,
        key=lambda row: (
            float(row["weak_val_prompt_mean_metrics"]["r2"]),
            -float(row["weak_val_prompt_mean_metrics"]["mae"]),
        ),
        reverse=True,
    )
    top_k = max(int(args.top_k), 1)
    (args.output_dir / "top_results.json").write_text(json.dumps(top_results[:top_k], indent=2), encoding="utf-8")

    summary = {
        "setting": "weak_only_single_rollout_hidden_search",
        "prediction_target": "value",
        "prompt_layer_index": int(args.prompt_layer_index),
        "prompt_feature_keys": list(args.prompt_feature_keys),
        "rollout_scalar_keys": list(args.rollout_scalar_keys),
        "derived_rollout_scalar_keys": list(args.derived_rollout_scalar_keys),
        "extra_rollout_scalar_field_paths": list(args.extra_rollout_scalar_field_paths),
        "rollout_components": list(args.rollout_components),
        "prompt_hidden_pca_dims": [int(value) for value in args.prompt_hidden_pca_dims],
        "rollout_hidden_pca_dims": [int(value) for value in args.rollout_hidden_pca_dims],
        "alphas": [float(alpha) for alpha in args.alphas],
        "best_model": str(best_result["name"]),
        "best_rollout_component": str(best_result["rollout_component"]),
        "best_prompt_hidden_pca_dim": int(best_result["prompt_hidden_pca_dim"]),
        "best_rollout_hidden_pca_dim": int(best_result["rollout_hidden_pca_dim"]),
        "best_alpha": float(best_result["alpha"]),
        "feature_dim": int(best_result["feature_dim"]),
        "num_train_rows": int(best_result["num_train_rows"]),
        "num_weak_val_rows": int(best_result["num_weak_val_rows"]),
        "num_weak_val_prompts": int(best_result["num_weak_val_prompts"]),
        "weak_val_row_metrics": best_result["weak_val_row_metrics"],
        "weak_val_prompt_mean_metrics": best_result["weak_val_prompt_mean_metrics"],
        "num_configs": int(len(all_results)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
