from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

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
    return parser.parse_args()


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
    prompt_layer_index = int(estimator_config["prompt"]["hidden_layer_index"])
    prompt_feature_keys = list(estimator_config["prompt"].get("prompt_scalar_keys", []))
    response_config = estimator_config["response"]
    rollout_scalar_keys = list(response_config.get("scalar_keys", []))
    derived_rollout_scalar_keys = list(response_config.get("derived_scalar_keys", []))
    extra_rollout_scalar_field_paths = list(response_config.get("extra_scalar_field_paths", []))

    labels_by_task = load_labels_by_task(args.labels_path)
    prompt_scalar_lookup = build_prompt_scalar_lookup(labels_by_task, prompt_feature_keys)
    prompt_lookup = load_prompt_hidden_lookup(
        [path.expanduser().resolve() for path in args.prompt_hidden_paths],
        [path.expanduser().resolve() for path in args.prompt_index_paths],
        layer_index=prompt_layer_index,
    )
    prompt_lookup = apply_prompt_hidden_pca(prompt_lookup, bundle.get("prompt_hidden_pca"))

    rollout_hidden_lookup = None
    if feature_mode == "prompt_plus_rollout":
        if not args.eval_rollout_hidden_paths:
            raise ValueError("Prompt+rollout models require --eval_rollout_hidden_paths.")
        rollout_hidden_lookup = build_rollout_hidden_lookup(
            [path.expanduser().resolve() for path in args.eval_rollout_hidden_paths],
            [path.expanduser().resolve() for path in args.eval_rollout_index_paths],
            component_name=str(bundle["rollout_component"]),
            layer_index=0,
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

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
