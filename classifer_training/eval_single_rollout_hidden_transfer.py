from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from classifer_training.train_weak_only_single_rollout_hidden import (
    LogitRidgeValueEstimator,
    PromptMiddleGatedRidgeValueEstimator,
    PromptPriorBlendRidgeValueEstimator,
    PromptResidualRidgeValueEstimator,
    PromptTrajectoryMeanRidgeValueEstimator,
    PromptTrajectoryScoreMLPValueEstimator,
    PromptTrajectoryScoreStackRidgeValueEstimator,
    PromptTrajectoryStackedRidgeValueEstimator,
    PromptValuePriorResidualRidgeValueEstimator,
    TwoHeadBinaryValueEstimator,
)
from classifer_training.single_rollout_hidden_utils import (
    apply_prompt_hidden_pca,
    apply_rollout_hidden_pca,
    build_matrix,
    build_prompt_scalar_lookup,
    build_rollout_hidden_lookup,
    group_eval_rollouts,
    load_labels_by_task,
    load_prompt_hidden_lookup,
    prompt_mean_metrics,
    reg_metrics,
    save_diagnostics_plot,
    select_single_rollout,
    write_predictions,
)
from classifer_training.utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained single-rollout value estimator on validation/test splits."
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--prompt_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--prompt_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--eval_rollout_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--eval_rollout_hidden_paths", nargs="+", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allowed_splits", nargs="+", default=["validation", "test"])
    parser.add_argument(
        "--prompt_hidden_component_override",
        type=str,
        default="",
        help="Use this prompt hidden component from eval prompt hidden files instead of the component stored in the bundle.",
    )
    parser.add_argument(
        "--rollout_component_override",
        type=str,
        default="",
        help="Use this hidden component from eval rollout hidden files instead of the component stored in the bundle.",
    )
    parser.add_argument(
        "--allow_missing_entropy_scalars",
        action="store_true",
        help="Fill missing entropy scalar features with 0 instead of failing.",
    )
    return parser.parse_args()


def _write_row_predictions(
    output_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata_rows: list[dict],
    labels_by_task: dict[str, dict],
) -> None:
    rows = []
    for true_value, pred_value, meta in zip(y_true.tolist(), y_pred.tolist(), metadata_rows, strict=True):
        label_row = labels_by_task[str(meta["task_id"])]
        rows.append(
            {
                "task_id": str(meta["task_id"]),
                "user_input": str(label_row.get("user_input", "")),
                "value_true": float(true_value),
                "value_pred": float(pred_value),
                "rollout_row_index": int(meta.get("rollout_row_index", -1)),
                "sample_index": int(meta.get("sample_index", -1)),
                "num_rows": 1,
            }
        )
    write_jsonl(output_path, rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model_path)
    estimator = bundle["estimator"]
    estimator_config = bundle["config"]
    if estimator_config.get("prediction_target") != "value":
        raise ValueError("This evaluation script only supports value estimators.")

    feature_mode = str(bundle["feature_mode"])
    single_rollout_strategy = str(bundle.get("single_rollout_strategy", "first"))
    prompt_hidden_component = str(estimator_config.get("prompt", {}).get("hidden_component", "hidden"))
    if args.prompt_hidden_component_override:
        prompt_hidden_component = str(args.prompt_hidden_component_override)
    prompt_layer_index = int(estimator_config["prompt"]["hidden_layer_index"])
    prompt_feature_keys = list(estimator_config["prompt"].get("prompt_scalar_keys", []))
    response_config = estimator_config["response"]
    rollout_scalar_keys = list(response_config.get("scalar_keys", []))
    derived_rollout_scalar_keys = list(response_config.get("derived_scalar_keys", []))
    extra_rollout_scalar_field_paths = list(response_config.get("extra_scalar_field_paths", []))
    rollout_layer_index = int(response_config.get("hidden_layer_index", 0) or 0)

    labels_by_task = load_labels_by_task(args.labels_path)
    prompt_scalar_lookup = build_prompt_scalar_lookup(labels_by_task, prompt_feature_keys)
    prompt_lookup = load_prompt_hidden_lookup(
        [path.expanduser().resolve() for path in args.prompt_hidden_paths],
        [path.expanduser().resolve() for path in args.prompt_index_paths],
        layer_index=prompt_layer_index,
        component_name=prompt_hidden_component,
    )
    prompt_lookup = apply_prompt_hidden_pca(prompt_lookup, bundle.get("prompt_hidden_pca"))

    rollout_hidden_lookup = None
    effective_rollout_component = str(bundle["rollout_component"])
    if args.rollout_component_override:
        effective_rollout_component = str(args.rollout_component_override)
    if feature_mode == "prompt_plus_rollout":
        if not args.eval_rollout_hidden_paths:
            raise ValueError("Prompt+rollout models require --eval_rollout_hidden_paths.")
        rollout_hidden_lookup = build_rollout_hidden_lookup(
            [path.expanduser().resolve() for path in args.eval_rollout_hidden_paths],
            [path.expanduser().resolve() for path in args.eval_rollout_index_paths],
            component_name=effective_rollout_component,
            layer_index=rollout_layer_index,
            pool_mode=str(bundle.get("rollout_pool_mode", "mean")),
        )

    grouped_rows = group_eval_rollouts(
        labels_by_task=labels_by_task,
        index_paths=[path.expanduser().resolve() for path in args.eval_rollout_index_paths],
        rollout_hidden_lookup=rollout_hidden_lookup,
        rollout_scalar_keys=rollout_scalar_keys,
        derived_rollout_scalar_keys=derived_rollout_scalar_keys,
        extra_rollout_scalar_field_paths=extra_rollout_scalar_field_paths,
        allowed_splits=set(args.allowed_splits),
        strict_missing_entropy=not bool(args.allow_missing_entropy_scalars),
    )
    rows = select_single_rollout(grouped_rows, single_rollout_strategy)
    rows = apply_rollout_hidden_pca(rows, bundle.get("rollout_hidden_pca"))
    x_eval, y_eval, eval_splits, eval_meta = build_matrix(
        rows,
        prompt_lookup,
        prompt_scalar_lookup,
        feature_mode=feature_mode,
    )

    summary = {
        "setting": "single_rollout_hidden_transfer_eval",
        "prediction_target": "value",
        "model_path": str(args.model_path.expanduser().resolve()),
        "feature_mode": feature_mode,
        "evaluated_splits": [],
    }
    for split_name in args.allowed_splits:
        split_mask = eval_splits == split_name
        if not np.any(split_mask):
            continue
        x_split = x_eval[split_mask]
        y_split = y_eval[split_mask]
        split_meta = [eval_meta[idx] for idx in np.where(split_mask)[0]]
        split_pred = np.clip(np.asarray(estimator.predict(x_split), dtype=np.float32).reshape(-1), 0.0, 1.0)
        split_row_metrics = reg_metrics(y_split, split_pred)
        split_prompt_metrics, split_prompt_rows = prompt_mean_metrics(y_split, split_pred, split_meta)

        _write_row_predictions(
            args.output_dir / f"predictions_{split_name}_rows.jsonl",
            y_split,
            split_pred,
            split_meta,
            labels_by_task,
        )
        write_predictions(args.output_dir / f"predictions_{split_name}.jsonl", split_prompt_rows, labels_by_task)
        save_diagnostics_plot(
            args.output_dir / f"prediction_diagnostics_{split_name}.png",
            split_prompt_rows,
            f"{split_name.title()}: {Path(args.model_path).stem}",
        )

        summary["evaluated_splits"].append(split_name)
        summary[f"{split_name}_row_metrics"] = split_row_metrics
        summary[f"{split_name}_prompt_mean_metrics"] = split_prompt_metrics
        summary[f"num_{split_name}_rows"] = int(x_split.shape[0])
        summary[f"num_{split_name}_prompts"] = int(len(split_prompt_rows))

    summary["bundle_rollout_component"] = str(bundle.get("rollout_component", ""))
    summary["effective_prompt_hidden_component"] = prompt_hidden_component
    summary["prompt_hidden_component_override"] = str(args.prompt_hidden_component_override)
    summary["effective_rollout_component"] = effective_rollout_component
    summary["effective_rollout_layer_index"] = int(rollout_layer_index)
    summary["rollout_component_override"] = str(args.rollout_component_override)
    summary["allow_missing_entropy_scalars"] = bool(args.allow_missing_entropy_scalars)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
