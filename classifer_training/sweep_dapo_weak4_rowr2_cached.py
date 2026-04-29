from __future__ import annotations

import copy
import json
import os
from pathlib import Path

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
    label_to_value,
    load_labels_by_task,
    load_prompt_hidden_lookup,
    normalize_run_dir,
    prompt_mean_metrics,
    reg_metrics,
    rollout_to_correctness,
)
from classifer_training.train_weak_only_single_rollout_hidden import (
    _apply_train_target_mode,
    _row_prediction_rows,
    _safe_subset_metrics,
    _selection_score,
    _write_row_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT = ROOT / "classifer_training/artifacts/probe/dapo_math_17k_weak4_simple_ridge_entropy_rowr2_cached_axis_sweep"
DATASET_DIR = ROOT / "classifer_training/artifacts/datasets/dapo_math_17k_weak4"
LABELS_PATH = ROOT / "classifer_training/artifacts/labels/dapo_math_17k/qwen3_4b_instruct_2507/weak4_labels.jsonl"
RUN_DIRS = [
    ROOT / "classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/0",
    ROOT / "classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/1",
]
ROLLOUT_MODEL_DIR = (
    "_data2_sangjunsong__cache_hf_hub_models--Qwen--Qwen3-4B-Instruct-2507"
    "_snapshots_cdbee75f17c01a7cc42f958dc650907174af0554"
)
ROLLOUT_HIDDEN_DIR = ROOT / "classifer_training/artifacts/rollout_hidden/dapo_math_17k_weak4_think_end_l26" / ROLLOUT_MODEL_DIR
ROLLOUT_INDEX_DIR = ROOT / "classifer_training/artifacts/rollout_index/dapo_math_17k_weak4_think_end_l26" / ROLLOUT_MODEL_DIR

SCALARS = [
    "output_mean_token_entropy",
    "reasoning_mean_token_entropy",
    "output_last_token_entropy",
    "output_max_token_entropy",
    "output_min_token_entropy",
    "reasoning_last_token_entropy",
    "reasoning_max_token_entropy",
    "reasoning_min_token_entropy",
    "answer_last_token_entropy",
    "answer_max_token_entropy",
    "answer_mean_token_entropy",
    "answer_min_token_entropy",
    "entropy_gap_reasoning_answer",
    "answer_entropy_gap_vs_output",
    "rollout_features.answer_mean_token_entropy",
]


def _fast_entropy_scalar_vec(record: dict) -> np.ndarray:
    rollout_features = record.get("rollout_features")
    feature_map = dict(rollout_features) if isinstance(rollout_features, dict) else {}
    reasoning_mean = float(feature_map.get("reasoning_mean_token_entropy", 0.0) or 0.0)
    answer_mean = float(feature_map.get("answer_mean_token_entropy", 0.0) or 0.0)
    output_mean = float(feature_map.get("output_mean_token_entropy", 0.0) or 0.0)
    feature_map.setdefault("entropy_gap_reasoning_answer", reasoning_mean - answer_mean)
    feature_map.setdefault("answer_entropy_gap_vs_output", answer_mean - output_mean)
    feature_map.setdefault("rollout_features.answer_mean_token_entropy", answer_mean)
    return np.asarray([float(feature_map.get(key, 0.0) or 0.0) for key in SCALARS], dtype=np.float32)


def _prompt_paths(prompt_slug: str) -> tuple[list[Path], list[Path]]:
    hidden_paths = sorted((ROOT / "classifer_training/artifacts/hidden").glob(f"dapo_math_17k_weak4_shard*/{prompt_slug}/hidden_states.pt"))
    index_paths = sorted((ROOT / "classifer_training/artifacts/index").glob(f"dapo_math_17k_weak4_shard*/{prompt_slug}/index.jsonl"))
    if len(hidden_paths) != 4 or len(index_paths) != 4:
        raise FileNotFoundError(f"Expected 4 prompt hidden/index shards for {prompt_slug}; got {len(hidden_paths)}/{len(index_paths)}")
    return hidden_paths, index_paths


def _fit_one(
    *,
    name: str,
    prompt_lookup_raw: dict[str, np.ndarray],
    split_lookup: dict[str, str],
    prompt_scalar_lookup: dict[str, np.ndarray],
    rows_raw: list[dict],
    labels_by_task: dict[str, dict],
    prompt_slug: str,
    prompt_layer: int,
    rollout_component: str,
    prompt_pca_dim: int,
    rollout_pca_dim: int,
    prompt_projection_cache: dict[tuple[str, int, int], tuple[object, dict[str, np.ndarray]]],
    rollout_projection_cache: dict[tuple[str, int], tuple[object, list[dict]]],
) -> dict:
    output_dir = BASE_OUTPUT / name
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_cache_key = (prompt_slug, int(prompt_layer), int(prompt_pca_dim))
    if prompt_cache_key not in prompt_projection_cache:
        print(json.dumps({"event": "fit_prompt_pca_start", "name": name, "key": list(prompt_cache_key)}), flush=True)
        prompt_pca = fit_prompt_hidden_pca(prompt_lookup_raw, split_lookup, prompt_pca_dim)
        prompt_lookup = apply_prompt_hidden_pca(prompt_lookup_raw, prompt_pca)
        prompt_projection_cache[prompt_cache_key] = (prompt_pca, prompt_lookup)
        print(json.dumps({"event": "fit_prompt_pca_done", "name": name, "key": list(prompt_cache_key)}), flush=True)
    else:
        prompt_pca, prompt_lookup = prompt_projection_cache[prompt_cache_key]

    rollout_cache_key = (rollout_component, int(rollout_pca_dim))
    if rollout_cache_key not in rollout_projection_cache:
        print(json.dumps({"event": "fit_rollout_pca_start", "name": name, "key": list(rollout_cache_key)}), flush=True)
        rows = copy.deepcopy(rows_raw)
        rollout_pca = fit_rollout_hidden_pca(rows, rollout_pca_dim)
        rows = apply_rollout_hidden_pca(rows, rollout_pca)
        rollout_projection_cache[rollout_cache_key] = (rollout_pca, rows)
        print(json.dumps({"event": "fit_rollout_pca_done", "name": name, "key": list(rollout_cache_key)}), flush=True)
    else:
        rollout_pca, rows = rollout_projection_cache[rollout_cache_key]
        print(json.dumps({"event": "reuse_rollout_pca", "name": name, "key": list(rollout_cache_key)}), flush=True)

    print(json.dumps({"event": "build_matrix_start", "name": name}), flush=True)
    x, y, splits, meta = build_matrix(rows, prompt_lookup, prompt_scalar_lookup, feature_mode="prompt_plus_rollout")
    print(json.dumps({"event": "build_matrix_done", "name": name, "shape": list(x.shape)}), flush=True)

    prompt_y = np.asarray([float(row.get("prompt_value_true", row["value_true"])) for row in meta], dtype=np.float32)
    train_mask = splits == "train"
    val_mask = splits == "validation"
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    val_meta = [meta[idx] for idx in np.where(val_mask)[0]]

    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=0.01, random_state=42)),
        ]
    )
    print(json.dumps({"event": "fit_ridge_start", "name": name}), flush=True)
    estimator.fit(x_train, y_train)
    print(json.dumps({"event": "fit_ridge_done", "name": name}), flush=True)
    pred = np.clip(np.asarray(estimator.predict(x_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
    row_metrics = reg_metrics(y_val, pred)
    prompt_metrics, prompt_rows = prompt_mean_metrics(y_val, pred, val_meta)
    row_subset = _safe_subset_metrics(y_val, pred)
    prompt_true = np.asarray([float(row["value_true"]) for row in prompt_rows], dtype=np.float32)
    prompt_pred = np.asarray([float(row["value_pred"]) for row in prompt_rows], dtype=np.float32)
    prompt_subset = _safe_subset_metrics(prompt_true, prompt_pred)
    selection_score = _selection_score(
        "row_r2",
        row_metrics=row_metrics,
        prompt_metrics=prompt_metrics,
        row_subset_metrics=row_subset,
        prompt_subset_metrics=prompt_subset,
    )

    summary = {
        "setting": "dapo_weak4_rowr2_cached_axis_sweep",
        "name": name,
        "model": "StandardScaler -> Ridge(alpha=0.01) -> clip[0,1]",
        "train_target_mode": "other_rollout_correctness",
        "selection_metric": "row_r2",
        "selection_score": float(selection_score),
        "prompt_slug": prompt_slug,
        "prompt_hidden_component": "hidden",
        "prompt_layer_index": int(prompt_layer),
        "prompt_hidden_pca_dim": int(prompt_pca_dim),
        "rollout_component": rollout_component,
        "rollout_layer_index": 26,
        "rollout_pool_mode": "mean",
        "rollout_hidden_pca_dim": int(rollout_pca_dim),
        "rollout_scalar_keys": SCALARS,
        "feature_dim": int(x_train.shape[1]),
        "num_train_rows": int(x_train.shape[0]),
        "num_weak_val_rows": int(x_val.shape[0]),
        "num_weak_val_prompts": int(len(prompt_rows)),
        "weak_val_row_metrics": row_metrics,
        "weak_val_row_subset_metrics": row_subset,
        "weak_val_prompt_mean_metrics": prompt_metrics,
        "weak_val_prompt_subset_metrics": prompt_subset,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_row_predictions(output_dir / "predictions_weakval_rows.jsonl", _row_prediction_rows(y_val, pred, val_meta), labels_by_task)
    bundle = {
        "bundle_type": "single_rollout_value_estimator",
        "bundle_version": 1,
        "config": summary,
        "estimator": estimator,
        "prompt_hidden_pca": prompt_pca,
        "rollout_hidden_pca": rollout_pca,
    }
    joblib.dump(bundle, output_dir / "model.joblib")
    return summary


def main() -> None:
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    smoke_only = os.environ.get("SMOKE_ONLY", "0") == "1"
    print(json.dumps({"event": "load_labels_start"}), flush=True)
    labels_by_task = load_labels_by_task(LABELS_PATH)
    split_lookup = build_split_lookup(DATASET_DIR)
    prompt_scalar_lookup = build_prompt_scalar_lookup(labels_by_task, [])
    rollout_index_paths = sorted(ROLLOUT_INDEX_DIR.glob("rollout_index.shard*.jsonl"))
    rollout_hidden_paths = sorted(ROLLOUT_HIDDEN_DIR.glob("rollout_hidden_states.shard*.pt"))
    print(
        json.dumps(
            {
                "event": "load_labels_done",
                "num_labels": len(labels_by_task),
                "num_split_rows": len(split_lookup),
                "num_rollout_hidden_paths": len(rollout_hidden_paths),
                "num_rollout_index_paths": len(rollout_index_paths),
            }
        ),
        flush=True,
    )
    print(json.dumps({"event": "load_rollout_index_start"}), flush=True)
    rollout_index_lookup = build_rollout_index_lookup(rollout_index_paths)
    print(json.dumps({"event": "load_rollout_index_done", "num_rows": len(rollout_index_lookup)}), flush=True)

    prompt_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def get_prompt(prompt_slug: str, layer: int) -> dict[str, np.ndarray]:
        key = (prompt_slug, int(layer))
        if key not in prompt_cache:
            print(json.dumps({"event": "load_prompt_start", "prompt_slug": prompt_slug, "layer": layer}), flush=True)
            hidden_paths, index_paths = _prompt_paths(prompt_slug)
            prompt_cache[key] = load_prompt_hidden_lookup(
                hidden_paths,
                index_paths,
                layer_index=layer,
                component_name="hidden",
            )
            print(json.dumps({"event": "load_prompt_done", "prompt_slug": prompt_slug, "layer": layer, "num_rows": len(prompt_cache[key])}), flush=True)
        return prompt_cache[key]

    row_cache: dict[str, list[dict]] = {}

    def get_rows(rollout_component: str) -> list[dict]:
        if rollout_component not in row_cache:
            print(json.dumps({"event": "load_rollout_hidden_start", "component": rollout_component}), flush=True)
            hidden_lookup = build_rollout_hidden_lookup(
                rollout_hidden_paths,
                rollout_index_paths,
                component_name=rollout_component,
                layer_index=26,
                pool_mode="mean",
            )
            print(json.dumps({"event": "load_rollout_hidden_done", "component": rollout_component, "num_rows": len(hidden_lookup)}), flush=True)
            print(json.dumps({"event": "group_rows_start", "component": rollout_component}), flush=True)
            rows = []
            for key, index_row in sorted(rollout_index_lookup.items()):
                task_id = str(index_row.get("task_id", ""))
                label_row = labels_by_task.get(task_id)
                if label_row is None:
                    continue
                run_dir, rollout_row_index = key
                rollout_hidden = hidden_lookup.get((run_dir, rollout_row_index))
                if rollout_hidden is None:
                    continue
                split = split_lookup.get(task_id, str(index_row.get("split", "train")))
                if split not in {"train", "validation"}:
                    continue
                scalar_vec = _fast_entropy_scalar_vec(index_row)
                rows.append(
                    {
                        "task_id": task_id,
                        "split": split,
                        "value_true": label_to_value(label_row),
                        "run_dir": normalize_run_dir(run_dir),
                        "rollout_hidden_vec": np.asarray(rollout_hidden, dtype=np.float32).reshape(-1),
                        "rollout_scalar_vec": scalar_vec,
                        "rollout_correctness": rollout_to_correctness(index_row),
                        "rollout_row_index": int(rollout_row_index),
                        "sample_index": int(index_row.get("sample_index", -1)),
                    }
                )
            print(json.dumps({"event": "group_rows_done", "component": rollout_component, "num_rows": len(rows)}), flush=True)
            row_cache[rollout_component], _ = _apply_train_target_mode(rows, "other_rollout_correctness")
            print(json.dumps({"event": "target_done", "component": rollout_component, "num_rows": len(row_cache[rollout_component])}), flush=True)
        return row_cache[rollout_component]

    configs = []
    configs.append(("center_prompt_last6_L26_thinkend_L26_p32_r256", "qwen3_4b_instruct_2507_last6mean", 26, "think_end_hidden", 32, 256))
    if smoke_only:
        configs = configs[:1]
    for layer in [18, 20, 22, 24, 26, 28, 30, 32, 34, 35]:
        configs.append((f"prompt_layer_sweep_last6_L{layer}_thinkend_L26_p32_r256", "qwen3_4b_instruct_2507_last6mean", layer, "think_end_hidden", 32, 256))
    configs.append(("prompt_pool_last_L26_thinkend_L26_p32_r256", "qwen3_4b_instruct_2507", 26, "think_end_hidden", 32, 256))
    configs.append(("rollout_component_thinkend_last10_prompt_last6_L26_p32_r256", "qwen3_4b_instruct_2507_last6mean", 26, "think_end_last10_hidden", 32, 256))
    for prompt_pca in [16, 32, 64, 128]:
        configs.append((f"prompt_pca_sweep_p{prompt_pca}_r256_last6_L26_thinkend_L26", "qwen3_4b_instruct_2507_last6mean", 26, "think_end_hidden", prompt_pca, 256))
    for rollout_pca in [64, 128, 256, 512]:
        configs.append((f"rollout_pca_sweep_p32_r{rollout_pca}_last6_L26_thinkend_L26", "qwen3_4b_instruct_2507_last6mean", 26, "think_end_hidden", 32, rollout_pca))

    seen = set()
    results = []
    prompt_projection_cache: dict[tuple[str, int, int], tuple[object, dict[str, np.ndarray]]] = {}
    rollout_projection_cache: dict[tuple[str, int], tuple[object, list[dict]]] = {}
    for config in configs:
        name, prompt_slug, prompt_layer, rollout_component, prompt_pca_dim, rollout_pca_dim = config
        if name in seen:
            continue
        seen.add(name)
        print(json.dumps({"event": "start", "name": name}), flush=True)
        summary = _fit_one(
            name=name,
            prompt_lookup_raw=get_prompt(prompt_slug, prompt_layer),
            split_lookup=split_lookup,
            prompt_scalar_lookup=prompt_scalar_lookup,
            rows_raw=get_rows(rollout_component),
            labels_by_task=labels_by_task,
            prompt_slug=prompt_slug,
            prompt_layer=prompt_layer,
            rollout_component=rollout_component,
            prompt_pca_dim=prompt_pca_dim,
            rollout_pca_dim=rollout_pca_dim,
            prompt_projection_cache=prompt_projection_cache,
            rollout_projection_cache=rollout_projection_cache,
        )
        results.append(summary)
        print(
            json.dumps(
                {
                    "event": "done",
                    "name": name,
                    "row_r2": summary["weak_val_row_metrics"]["r2"],
                    "prompt_mean_r2": summary["weak_val_prompt_mean_metrics"]["r2"],
                    "feature_dim": summary["feature_dim"],
                }
            ),
            flush=True,
        )

    results.sort(key=lambda row: row["weak_val_row_metrics"]["r2"], reverse=True)
    (BASE_OUTPUT / "axis_sweep_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    with (BASE_OUTPUT / "axis_sweep_summary.md").open("w") as f:
        f.write("| rank | name | row_r2 | prompt_mean_r2 | row_mae | prompt_mean_mae | dim |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for idx, row in enumerate(results, 1):
            f.write(
                f"| {idx} | {row['name']} | {row['weak_val_row_metrics']['r2']:.4f} | "
                f"{row['weak_val_prompt_mean_metrics']['r2']:.4f} | {row['weak_val_row_metrics']['mae']:.4f} | "
                f"{row['weak_val_prompt_mean_metrics']['mae']:.4f} | {row['feature_dim']} |\n"
            )
    print(json.dumps({"event": "finished", "best": results[0]["name"], "row_r2": results[0]["weak_val_row_metrics"]["r2"]}), flush=True)


if __name__ == "__main__":
    main()
