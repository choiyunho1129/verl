from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from classifer_training.data import load_hidden_rows
from classifer_training.single_rollout_hidden_utils import _resolve_layer_position
from classifer_training.utils import load_records, write_jsonl
from verl.utils.reward_score.math_dapo import compute_score as math_dapo_compute_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path("/home/jongwonlim/verl/yoonho/qwen3_4b_simple_structure_other_rollout_2048prompts.joblib")
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "classifer_training/artifacts/eval/"
    "qwen3_4b_simple_structure_other_rollout_2048prompts_spo16_thinkend_dapo"
)
DEFAULT_PROMPT_HIDDEN_GLOB = (
    "classifer_training/artifacts/hidden/spo_temp1_subset0to4_shard*/"
    "qwen3_4b_base_l18_35_last5_10_15mean/hidden_states.pt"
)
DEFAULT_PROMPT_INDEX_GLOB = (
    "classifer_training/artifacts/index/spo_temp1_subset0to4_shard*/"
    "qwen3_4b_base_l18_35_last5_10_15mean/index.jsonl"
)
DEFAULT_ROLLOUT_HIDDEN_PATH = (
    ROOT
    / "classifer_training/artifacts/rollout_hidden/"
    "spo_temp1_subset0to4_thinkendlast10_l19/Qwen_Qwen3-4B/rollout_hidden_states.pt"
)
DEFAULT_ROLLOUT_INDEX_PATH = (
    ROOT
    / "classifer_training/artifacts/rollout_index/"
    "spo_temp1_subset0to4_thinkendlast10_l19/Qwen_Qwen3-4B/rollout_index.jsonl"
)


def _subset_id(row: dict[str, Any]) -> int | None:
    text = str(row.get("run_name") or row.get("run_dir") or row.get("source_jsonl") or "")
    match = re.search(r"subset[_-](\d+)", text)
    return int(match.group(1)) if match else None


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _load_prompt_lookup(
    *,
    hidden_paths: list[Path],
    index_paths: list[Path],
    component_name: str,
    layer_index: int,
) -> dict[str, np.ndarray]:
    if len(hidden_paths) != len(index_paths):
        raise ValueError(f"Prompt hidden/index path counts differ: {len(hidden_paths)} vs {len(index_paths)}")
    lookup: dict[str, np.ndarray] = {}
    for hidden_path, index_path in zip(hidden_paths, index_paths, strict=True):
        rows = load_hidden_rows(hidden_path, index_path=index_path, dataset_name="spo_temp1_subset0to4")
        for row in rows:
            index_row = row["index_row"]
            layers = row["components"][component_name]
            layer_pos = _resolve_layer_position(
                layers,
                layer_index=layer_index,
                index_row=index_row,
                context="prompt",
            )
            lookup[str(row["task_id"])] = np.asarray(layers[layer_pos], dtype=np.float32).reshape(-1)
    return lookup


def _rollout_correctness(row: dict[str, Any], label_source: str) -> float:
    if label_source == "legacy":
        return 1.0 if float(row.get("reward", row.get("score", 0.0)) or 0.0) >= 1.0 else 0.0
    if label_source == "dapo":
        generated_text = str(row.get("generated_text", ""))
        solution = generated_text.split("</think>", maxsplit=1)[0]
        result = math_dapo_compute_score(
            solution_str=solution,
            ground_truth=str(row.get("ground_truth", "")),
            strict_box_verify=True,
        )
        return 1.0 if bool(result["acc"]) else 0.0
    raise ValueError(f"Unsupported label source: {label_source}")


def _pool_rollout_hidden(value: Any, pooling: str) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu().to(torch.float32)
    if tensor.ndim == 1:
        pooled = tensor
    elif tensor.ndim == 2:
        if pooling == "mean":
            pooled = tensor.mean(dim=0)
        elif pooling == "last":
            pooled = tensor[-1]
        elif pooling == "flatten":
            pooled = tensor.reshape(-1)
        else:
            raise ValueError(f"Unsupported rollout pooling: {pooling}")
    else:
        raise ValueError(f"Expected rollout hidden tensor with 1 or 2 dims, got {tuple(tensor.shape)}")
    return pooled.numpy().astype(np.float32, copy=False).reshape(-1)


def _scalar_features(row: dict[str, Any], feature_keys: list[str]) -> dict[str, float]:
    rollout_features = row.get("rollout_features")
    feature_map = dict(rollout_features) if isinstance(rollout_features, dict) else {}
    return {key: float(feature_map.get(key, 0.0) or 0.0) for key in feature_keys}


def _prompt_mean_rows(y_true: np.ndarray, y_pred: np.ndarray, task_ids: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"true": [], "pred": []})
    for task_id, true_value, pred_value in zip(task_ids, y_true.tolist(), y_pred.tolist(), strict=True):
        grouped[task_id]["true"].append(float(true_value))
        grouped[task_id]["pred"].append(float(pred_value))
    return [
        {
            "task_id": task_id,
            "value_true": float(np.mean(values["true"])),
            "value_pred": float(np.mean(values["pred"])),
            "num_rows": int(len(values["pred"])),
        }
        for task_id, values in sorted(grouped.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single_trajectory_estimator on SPO subset 2/3/4 Avg@16.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-hidden-glob", default=DEFAULT_PROMPT_HIDDEN_GLOB)
    parser.add_argument("--prompt-index-glob", default=DEFAULT_PROMPT_INDEX_GLOB)
    parser.add_argument("--rollout-hidden-path", type=Path, default=DEFAULT_ROLLOUT_HIDDEN_PATH)
    parser.add_argument("--rollout-index-path", type=Path, default=DEFAULT_ROLLOUT_INDEX_PATH)
    parser.add_argument("--prompt-component", default="hidden_last10_mean")
    parser.add_argument("--rollout-component", default="think_end_last10_hidden")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--rollout-pooling", choices=("mean", "last", "flatten"), default="mean")
    parser.add_argument("--label-source", choices=("dapo", "legacy"), default="dapo")
    parser.add_argument("--test-subsets", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model_path.expanduser().resolve())
    config = bundle["config"]
    estimator = bundle["estimator"]
    prompt_pca = bundle.get("prompt_hidden_pca")
    response_pca = bundle.get("response_hidden_pca")
    feature_keys = list(config.get("response_feature_keys", []))
    derived_feature_keys = list(config.get("derived_response_feature_keys", []))
    if derived_feature_keys:
        raise ValueError(f"This evaluator does not implement derived feature keys: {derived_feature_keys}")

    prompt_hidden_paths = sorted((ROOT / ".").glob(args.prompt_hidden_glob))
    prompt_index_paths = sorted((ROOT / ".").glob(args.prompt_index_glob))
    print(json.dumps({"event": "load_prompt_start", "num_hidden_paths": len(prompt_hidden_paths)}), flush=True)
    prompt_lookup_raw = _load_prompt_lookup(
        hidden_paths=prompt_hidden_paths,
        index_paths=prompt_index_paths,
        component_name=args.prompt_component,
        layer_index=args.layer_index,
    )
    print(json.dumps({"event": "load_prompt_done", "num_prompts": len(prompt_lookup_raw)}), flush=True)

    prompt_lookup: dict[str, np.ndarray]
    if prompt_pca is None:
        prompt_lookup = prompt_lookup_raw
    else:
        keys = list(prompt_lookup_raw.keys())
        x_prompt = np.stack([prompt_lookup_raw[key] for key in keys], axis=0)
        x_prompt = prompt_pca.transform(x_prompt).astype(np.float32, copy=False)
        prompt_lookup = {key: x_prompt[idx] for idx, key in enumerate(keys)}

    print(json.dumps({"event": "load_rollout_hidden_start", "path": str(args.rollout_hidden_path)}), flush=True)
    rollout_payload = torch.load(args.rollout_hidden_path.expanduser().resolve(), map_location="cpu")
    hidden_examples = rollout_payload["examples"]
    print(json.dumps({"event": "load_rollout_hidden_done", "num_examples": len(hidden_examples)}), flush=True)

    index_rows = load_records(args.rollout_index_path.expanduser().resolve())
    if len(index_rows) != len(hidden_examples):
        raise ValueError(f"Rollout index/hidden count mismatch: {len(index_rows)} vs {len(hidden_examples)}")

    test_subsets = set(int(value) for value in args.test_subsets)
    selected_rows: list[dict[str, Any]] = []
    correctness_by_task: dict[str, list[float]] = defaultdict(list)
    for row_idx, row in enumerate(index_rows):
        subset_id = _subset_id(row)
        if subset_id not in test_subsets:
            continue
        correctness = _rollout_correctness(row, args.label_source)
        selected_rows.append({"row_idx": row_idx, "row": row, "correctness": correctness})
        correctness_by_task[str(row.get("task_id", ""))].append(float(correctness))
        if args.max_rows is not None and len(selected_rows) >= args.max_rows:
            break

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    task_ids: list[str] = []
    prediction_rows: list[dict[str, Any]] = []
    for item in selected_rows:
        row_idx = int(item["row_idx"])
        row = item["row"]
        task_id = str(row.get("task_id", ""))
        prompt_vec = prompt_lookup.get(task_id)
        if prompt_vec is None:
            continue
        hidden_payload = hidden_examples[row_idx][args.rollout_component][0]
        response_vec = _pool_rollout_hidden(hidden_payload, args.rollout_pooling)
        if response_pca is not None:
            response_vec = response_pca.transform(response_vec.reshape(1, -1)).astype(np.float32, copy=False)[0]
        scalar_map = _scalar_features(row, feature_keys)
        scalar_vec = np.asarray([scalar_map[key] for key in feature_keys], dtype=np.float32)
        x_rows.append(np.concatenate([prompt_vec, response_vec, scalar_vec], axis=0).astype(np.float32, copy=False))
        y_true = float(np.mean(correctness_by_task[task_id]))
        y_rows.append(y_true)
        task_ids.append(task_id)
        prediction_rows.append(
            {
                "task_id": task_id,
                "subset_id": _subset_id(row),
                "rollout_row_index": int(row.get("rollout_row_index", -1)),
                "sample_index": int(row.get("sample_index", -1)),
                "rollout_correctness": float(item["correctness"]),
                "value_true": y_true,
            }
        )

    x = np.stack(x_rows, axis=0)
    y = np.asarray(y_rows, dtype=np.float32)
    pred = np.clip(np.asarray(estimator.predict(x), dtype=np.float32), 0.0, 1.0)
    for row, pred_value in zip(prediction_rows, pred.tolist(), strict=True):
        row["value_pred"] = float(pred_value)

    row_metrics = _reg_metrics(y, pred)
    prompt_rows = _prompt_mean_rows(y, pred, task_ids)
    prompt_y = np.asarray([row["value_true"] for row in prompt_rows], dtype=np.float32)
    prompt_pred = np.asarray([row["value_pred"] for row in prompt_rows], dtype=np.float32)
    prompt_metrics = _reg_metrics(prompt_y, prompt_pred)

    summary = {
        "setting": "single_trajectory_estimator_spo16_eval",
        "model_path": str(args.model_path.expanduser().resolve()),
        "rollout_hidden_path": str(args.rollout_hidden_path.expanduser().resolve()),
        "rollout_index_path": str(args.rollout_index_path.expanduser().resolve()),
        "prompt_component": args.prompt_component,
        "rollout_component": args.rollout_component,
        "rollout_pooling": args.rollout_pooling,
        "layer_index": int(args.layer_index),
        "label_source": args.label_source,
        "test_subsets": sorted(test_subsets),
        "feature_keys": feature_keys,
        "feature_dim": int(x.shape[1]),
        "num_rows": int(x.shape[0]),
        "num_prompts": int(len(prompt_rows)),
        "row_metrics": row_metrics,
        "prompt_mean_metrics": prompt_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "predictions_rows.jsonl", prediction_rows)
    write_jsonl(output_dir / "predictions_prompt_mean.jsonl", prompt_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
