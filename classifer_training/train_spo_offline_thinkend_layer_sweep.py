from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.single_rollout_hidden_utils import normalize_run_dir, rollout_to_correctness
from classifer_training.sweep_spo_base_rowr2_axis import FastPCA, _fit_pca
from classifer_training.utils import load_records, parse_layer_spec


DEFAULT_SCALAR_KEYS = [
    "output_mean_token_entropy",
    "reasoning_mean_token_entropy",
    "answer_mean_token_entropy",
]


def _subset_id(row: dict[str, Any]) -> int | None:
    if row.get("source_subset_id") is not None:
        return int(row["source_subset_id"])
    text = str(row.get("run_name") or row.get("run_dir") or row.get("source_validation_data") or "")
    match = re.search(r"subset[_-](\d+)", text)
    return int(match.group(1)) if match else None


def _resolve_layer_position(values: list[Any], index_row: dict[str, Any], layer_index: int) -> int:
    selected_layers = index_row.get("selected_layers")
    if isinstance(selected_layers, list):
        selected = [int(value) for value in selected_layers]
        if int(layer_index) in selected:
            return selected.index(int(layer_index))
    if 0 <= int(layer_index) < len(values):
        return int(layer_index)
    raise ValueError(
        f"Layer {layer_index} is not present. Stored layers={selected_layers}; component length={len(values)}"
    )


def _component_vector(value: Any, *, pool_window: bool) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu().to(torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    elif tensor.ndim > 1:
        tensor = tensor.mean(dim=0) if pool_window else tensor.reshape(-1, tensor.shape[-1])[-1]
    return tensor.numpy().astype(np.float32, copy=False).reshape(-1)


def _load_payload_examples(hidden_path: Path) -> list[dict[str, Any]]:
    payload = torch.load(hidden_path.expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("examples"), list):
        raise TypeError(f"Expected an examples-list payload in {hidden_path}.")
    return payload["examples"]


def _load_prompt_hidden_by_layer(
    *,
    hidden_paths: list[Path],
    index_paths: list[Path],
    component_name: str,
    layers: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    if len(hidden_paths) != len(index_paths):
        raise ValueError("Prompt hidden/index path counts must match.")
    out: dict[int, dict[str, np.ndarray]] = {int(layer): {} for layer in layers}
    for hidden_path, index_path in zip(hidden_paths, index_paths, strict=True):
        examples = _load_payload_examples(hidden_path)
        index_rows = load_records(index_path.expanduser().resolve())
        if len(examples) != len(index_rows):
            raise ValueError(f"Prompt hidden/index length mismatch for {hidden_path}: {len(examples)} vs {len(index_rows)}")
        for example, index_row in zip(examples, index_rows, strict=True):
            task_id = str(index_row.get("task_id") or example.get("task_id"))
            values = example.get(component_name)
            if not isinstance(values, list):
                raise ValueError(f"Missing prompt component {component_name!r} in {hidden_path}.")
            for layer in layers:
                pos = _resolve_layer_position(values, index_row, int(layer))
                out[int(layer)][task_id] = _component_vector(values[pos], pool_window=False)
        del examples
        gc.collect()
    return out


def _load_rollout_hidden_by_layer(
    *,
    hidden_paths: list[Path],
    index_paths: list[Path],
    component_name: str,
    layers: list[int],
) -> dict[int, dict[tuple[str, int], np.ndarray]]:
    if len(hidden_paths) != len(index_paths):
        raise ValueError("Rollout hidden/index path counts must match.")
    out: dict[int, dict[tuple[str, int], np.ndarray]] = {int(layer): {} for layer in layers}
    for hidden_path, index_path in zip(hidden_paths, index_paths, strict=True):
        examples = _load_payload_examples(hidden_path)
        index_rows = load_records(index_path.expanduser().resolve())
        if len(examples) != len(index_rows):
            raise ValueError(f"Rollout hidden/index length mismatch for {hidden_path}: {len(examples)} vs {len(index_rows)}")
        for example, index_row in zip(examples, index_rows, strict=True):
            run_dir = normalize_run_dir(str(index_row.get("run_dir", "")))
            rollout_row_index = int(index_row.get("rollout_row_index", index_row.get("sample_index", -1)))
            if not run_dir or rollout_row_index < 0:
                continue
            values = example.get(component_name)
            if not isinstance(values, list):
                raise ValueError(f"Missing rollout component {component_name!r} in {hidden_path}.")
            key = (run_dir, rollout_row_index)
            for layer in layers:
                pos = _resolve_layer_position(values, index_row, int(layer))
                out[int(layer)][key] = _component_vector(values[pos], pool_window=True)
        del examples
        gc.collect()
    return out


def _load_rollout_index_lookup(index_paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for index_path in index_paths:
        for row in load_records(index_path.expanduser().resolve()):
            run_dir = normalize_run_dir(str(row.get("run_dir", "")))
            rollout_row_index = int(row.get("rollout_row_index", row.get("sample_index", -1)))
            if run_dir and rollout_row_index >= 0:
                lookup[(run_dir, rollout_row_index)] = row
    return lookup


def _load_prompt_text_by_task(index_paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for index_path in index_paths:
        for row in load_records(index_path.expanduser().resolve()):
            out[str(row.get("task_id", ""))] = str(row.get("user_input", ""))
    return out


def _scalar_vec(record: dict[str, Any], scalar_keys: list[str]) -> np.ndarray:
    feature_map = dict(record.get("rollout_features") or {})
    reasoning_mean = float(feature_map.get("reasoning_mean_token_entropy", 0.0) or 0.0)
    answer_mean = float(feature_map.get("answer_mean_token_entropy", 0.0) or 0.0)
    output_mean = float(feature_map.get("output_mean_token_entropy", 0.0) or 0.0)
    feature_map.setdefault("entropy_gap_reasoning_answer", reasoning_mean - answer_mean)
    feature_map.setdefault("answer_entropy_gap_vs_output", answer_mean - output_mean)
    values = []
    for key in scalar_keys:
        normalized_key = key.removeprefix("rollout_features.")
        values.append(float(feature_map.get(normalized_key, 0.0) or 0.0))
    return np.asarray(values, dtype=np.float32)


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _prompt_mean_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_ids: list[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for task_id, true_value, pred_value in zip(task_ids, y_true.tolist(), y_pred.tolist(), strict=True):
        row = grouped.setdefault(task_id, {"value_true": [], "value_pred": []})
        row["value_true"].append(float(true_value))
        row["value_pred"].append(float(pred_value))
    rows = [
        {
            "task_id": task_id,
            "value_true": float(np.mean(values["value_true"])),
            "value_pred": float(np.mean(values["value_pred"])),
            "num_rows": int(len(values["value_pred"])),
        }
        for task_id, values in sorted(grouped.items())
    ]
    yt = np.asarray([row["value_true"] for row in rows], dtype=np.float32)
    yp = np.asarray([row["value_pred"] for row in rows], dtype=np.float32)
    return _reg_metrics(yt, yp), rows


def _transform_lookup(lookup: dict[str, np.ndarray], pca: FastPCA | None) -> dict[str, np.ndarray]:
    if pca is None:
        return lookup
    keys = list(lookup.keys())
    x = np.stack([np.asarray(lookup[key], dtype=np.float32).reshape(-1) for key in keys], axis=0)
    transformed = pca.transform(x).astype(np.float32, copy=False)
    return {key: transformed[idx] for idx, key in enumerate(keys)}


def _transform_rows(rows: list[dict[str, Any]], pca: FastPCA | None) -> list[dict[str, Any]]:
    if pca is None:
        return rows
    x = np.stack([np.asarray(row["rollout_hidden_vec"], dtype=np.float32).reshape(-1) for row in rows], axis=0)
    projected = pca.transform(x).astype(np.float32, copy=False)
    transformed = []
    for idx, row in enumerate(rows):
        updated = dict(row)
        updated["rollout_hidden_vec"] = projected[idx]
        transformed.append(updated)
    return transformed


def _build_rows(
    *,
    rollout_index_lookup: dict[tuple[str, int], dict[str, Any]],
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray],
    train_subsets: set[int],
    validation_subsets: set[int],
    scalar_keys: list[str],
) -> list[dict[str, Any]]:
    correctness_by_task: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    raw_rows = []
    for key, index_row in sorted(rollout_index_lookup.items()):
        subset_id = _subset_id(index_row)
        if subset_id is None or subset_id not in train_subsets | validation_subsets:
            continue
        hidden_vec = rollout_hidden_lookup.get(key)
        if hidden_vec is None:
            continue
        run_dir, rollout_row_index = key
        task_id = str(index_row.get("task_id", ""))
        correctness = float(rollout_to_correctness(index_row))
        correctness_by_task[task_id].append((int(subset_id), int(rollout_row_index), correctness))
        raw_rows.append(
            {
                "task_id": task_id,
                "subset_id": int(subset_id),
                "run_dir": run_dir,
                "rollout_row_index": int(rollout_row_index),
                "sample_index": int(index_row.get("sample_index", -1)),
                "rollout_hidden_vec": np.asarray(hidden_vec, dtype=np.float32).reshape(-1),
                "rollout_scalar_vec": _scalar_vec(index_row, scalar_keys),
                "rollout_correctness": correctness,
            }
        )

    rows = []
    for row in raw_rows:
        task_values = correctness_by_task.get(row["task_id"], [])
        subset_id = int(row["subset_id"])
        updated = dict(row)
        if subset_id in train_subsets:
            siblings = [
                value
                for sid, rollout_row_index, value in task_values
                if sid in train_subsets and int(rollout_row_index) != int(row["rollout_row_index"])
            ]
            if not siblings:
                continue
            updated["split"] = "train"
            updated["value_true"] = float(np.mean(siblings))
        else:
            validation_values = [value for sid, _, value in task_values if sid in validation_subsets]
            if not validation_values:
                continue
            updated["split"] = "validation"
            updated["value_true"] = float(np.mean(validation_values))
        rows.append(updated)
    return rows


def _matrix(
    rows: list[dict[str, Any]],
    prompt_lookup: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    x_rows, y_rows, task_ids, meta_rows = [], [], [], []
    for row in rows:
        prompt_vec = prompt_lookup.get(str(row["task_id"]))
        if prompt_vec is None:
            continue
        x_rows.append(
            np.concatenate(
                [
                    np.asarray(prompt_vec, dtype=np.float32).reshape(-1),
                    np.asarray(row["rollout_hidden_vec"], dtype=np.float32).reshape(-1),
                    np.asarray(row["rollout_scalar_vec"], dtype=np.float32).reshape(-1),
                ],
                axis=0,
            ).astype(np.float32)
        )
        y_rows.append(float(row["value_true"]))
        task_ids.append(str(row["task_id"]))
        meta_rows.append(row)
    if not x_rows:
        raise ValueError("No matrix rows were built.")
    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.float32), task_ids, meta_rows


def _fit_layer(
    *,
    layer: int,
    output_dir: Path,
    prompt_lookup_raw: dict[str, np.ndarray],
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray],
    rollout_index_lookup: dict[tuple[str, int], dict[str, Any]],
    prompt_text_by_task: dict[str, str],
    train_subsets: set[int],
    validation_subsets: set[int],
    scalar_keys: list[str],
    prompt_pca_dim: int,
    rollout_pca_dim: int,
    prompt_component: str,
    rollout_component: str,
    overwrite: bool,
) -> dict[str, Any]:
    layer_dir = output_dir / f"layer{int(layer):02d}_p{int(prompt_pca_dim)}_r{int(rollout_pca_dim)}_hidden_entropy{len(scalar_keys)}"
    summary_path = layer_dir / "summary.json"
    model_path = layer_dir / "model.joblib"
    if summary_path.exists() and model_path.exists() and not overwrite:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    layer_dir.mkdir(parents=True, exist_ok=True)

    rows_raw = _build_rows(
        rollout_index_lookup=rollout_index_lookup,
        rollout_hidden_lookup=rollout_hidden_lookup,
        train_subsets=train_subsets,
        validation_subsets=validation_subsets,
        scalar_keys=scalar_keys,
    )
    train_rows_raw = [row for row in rows_raw if row["split"] == "train"]
    validation_rows_raw = [row for row in rows_raw if row["split"] == "validation"]
    if not train_rows_raw or not validation_rows_raw:
        raise ValueError(f"Layer {layer}: need train and validation rows, got {len(train_rows_raw)}/{len(validation_rows_raw)}.")

    train_task_ids = {str(row["task_id"]) for row in train_rows_raw}
    prompt_pca = None
    if int(prompt_pca_dim) > 0:
        prompt_pca = _fit_pca(
            [vec for task_id, vec in prompt_lookup_raw.items() if task_id in train_task_ids],
            int(prompt_pca_dim),
        )
    prompt_lookup = _transform_lookup(prompt_lookup_raw, prompt_pca)

    rollout_pca = None
    if int(rollout_pca_dim) > 0:
        rollout_pca = _fit_pca([row["rollout_hidden_vec"] for row in train_rows_raw], int(rollout_pca_dim))
    rows = _transform_rows(rows_raw, rollout_pca)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]

    x_train, y_train, train_task_ids_list, train_meta = _matrix(train_rows, prompt_lookup)
    x_val, y_val, val_task_ids, val_meta = _matrix(validation_rows, prompt_lookup)

    estimator = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=0.01, solver="lsqr"))])
    estimator.fit(x_train, y_train)
    pred_train = np.clip(np.asarray(estimator.predict(x_train), dtype=np.float32), 0.0, 1.0)
    pred_val = np.clip(np.asarray(estimator.predict(x_val), dtype=np.float32), 0.0, 1.0)
    val_prompt_metrics, val_prompt_rows = _prompt_mean_metrics(y_val, pred_val, val_task_ids)

    summary: dict[str, Any] = {
        "setting": "spo_offline_thinkend_last10_layer_sweep",
        "bundle_type": "spo_subset_rowr2_probe",
        "sklearn_version": sklearn_version,
        "model": "StandardScaler -> Ridge(alpha=0.01, solver='lsqr') -> clip[0,1]",
        "train_subsets": sorted(train_subsets),
        "validation_subsets": sorted(validation_subsets),
        "train_target": "other rollout correctness within subset 0/1 prompts",
        "validation_target": "prompt Avg correctness within subset 2/3 prompts",
        "label_source": "math_dapo.compute_score from prepared SPO validation_data",
        "prompt_component": prompt_component,
        "prompt_layer_index": int(layer),
        "prompt_hidden_pca_dim": int(prompt_pca_dim),
        "rollout_component": rollout_component,
        "rollout_layer_index": int(layer),
        "rollout_pool_mode": "mean",
        "rollout_hidden_pca_dim": int(rollout_pca_dim),
        "rollout_scalar_keys": scalar_keys,
        "feature_dim": int(x_train.shape[1]),
        "num_train_rows": int(x_train.shape[0]),
        "num_train_prompts": int(len(set(train_task_ids_list))),
        "num_validation_rows": int(x_val.shape[0]),
        "num_validation_prompts": int(len(set(val_task_ids))),
        "train_row_metrics": _reg_metrics(y_train, pred_train),
        "validation_row_metrics": _reg_metrics(y_val, pred_val),
        "validation_prompt_mean_metrics": val_prompt_metrics,
    }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (layer_dir / "predictions_validation_rows.jsonl").open("w", encoding="utf-8") as f:
        for true_value, pred_value, task_id, meta in zip(y_val.tolist(), pred_val.tolist(), val_task_ids, val_meta, strict=True):
            f.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "user_input": prompt_text_by_task.get(task_id, ""),
                        "subset_id": int(meta["subset_id"]),
                        "rollout_row_index": int(meta["rollout_row_index"]),
                        "value_true": float(true_value),
                        "value_pred": float(pred_value),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (layer_dir / "predictions_validation_prompt_mean.jsonl").open("w", encoding="utf-8") as f:
        for row in val_prompt_rows:
            output_row = dict(row)
            output_row["user_input"] = prompt_text_by_task.get(str(row["task_id"]), "")
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    joblib.dump(
        {
            "bundle_type": "spo_subset_rowr2_probe",
            "config": summary,
            "estimator": estimator,
            "prompt_hidden_pca": prompt_pca,
            "rollout_hidden_pca": rollout_pca,
        },
        model_path,
    )
    return summary


def _parse_int_set(values: list[int]) -> set[int]:
    return {int(value) for value in values}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer sweep for SPO offline prompt-last10 + think_end_last10 Ridge probe.")
    parser.add_argument("--prompt-hidden-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--prompt-index-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--rollout-hidden-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--rollout-index-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-component", default="hidden_last10_mean")
    parser.add_argument("--rollout-component", default="think_end_last10_hidden")
    parser.add_argument("--layers", default="14:27")
    parser.add_argument("--num-model-layers", type=int, default=28)
    parser.add_argument("--train-subsets", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--validation-subsets", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--prompt-pca-dim", type=int, default=32)
    parser.add_argument("--rollout-pca-dim", type=int, default=256)
    parser.add_argument("--scalar-keys", nargs="*", default=DEFAULT_SCALAR_KEYS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layer_spec(args.layers, int(args.num_model_layers))
    train_subsets = _parse_int_set(args.train_subsets)
    validation_subsets = _parse_int_set(args.validation_subsets)

    prompt_hidden_paths = [path.expanduser().resolve() for path in args.prompt_hidden_paths]
    prompt_index_paths = [path.expanduser().resolve() for path in args.prompt_index_paths]
    rollout_hidden_paths = [path.expanduser().resolve() for path in args.rollout_hidden_paths]
    rollout_index_paths = [path.expanduser().resolve() for path in args.rollout_index_paths]

    print(json.dumps({"event": "load_prompt_start", "layers": layers}), flush=True)
    prompt_by_layer = _load_prompt_hidden_by_layer(
        hidden_paths=prompt_hidden_paths,
        index_paths=prompt_index_paths,
        component_name=args.prompt_component,
        layers=layers,
    )
    print(json.dumps({"event": "load_prompt_done", "rows_per_layer": {str(k): len(v) for k, v in prompt_by_layer.items()}}), flush=True)

    print(json.dumps({"event": "load_rollout_index_start"}), flush=True)
    rollout_index_lookup = _load_rollout_index_lookup(rollout_index_paths)
    prompt_text_by_task = _load_prompt_text_by_task(prompt_index_paths)
    print(json.dumps({"event": "load_rollout_index_done", "num_rows": len(rollout_index_lookup)}), flush=True)

    print(json.dumps({"event": "load_rollout_hidden_start", "layers": layers}), flush=True)
    rollout_hidden_by_layer = _load_rollout_hidden_by_layer(
        hidden_paths=rollout_hidden_paths,
        index_paths=rollout_index_paths,
        component_name=args.rollout_component,
        layers=layers,
    )
    print(
        json.dumps(
            {"event": "load_rollout_hidden_done", "rows_per_layer": {str(k): len(v) for k, v in rollout_hidden_by_layer.items()}},
        ),
        flush=True,
    )

    summaries = []
    for layer in layers:
        print(json.dumps({"event": "fit_layer_start", "layer": int(layer)}), flush=True)
        summary = _fit_layer(
            layer=int(layer),
            output_dir=output_dir,
            prompt_lookup_raw=prompt_by_layer[int(layer)],
            rollout_hidden_lookup=rollout_hidden_by_layer[int(layer)],
            rollout_index_lookup=rollout_index_lookup,
            prompt_text_by_task=prompt_text_by_task,
            train_subsets=train_subsets,
            validation_subsets=validation_subsets,
            scalar_keys=list(args.scalar_keys),
            prompt_pca_dim=int(args.prompt_pca_dim),
            rollout_pca_dim=int(args.rollout_pca_dim),
            prompt_component=str(args.prompt_component),
            rollout_component=str(args.rollout_component),
            overwrite=bool(args.overwrite),
        )
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "event": "fit_layer_done",
                    "layer": int(layer),
                    "row_r2": summary["validation_row_metrics"]["r2"],
                    "prompt_mean_r2": summary["validation_prompt_mean_metrics"]["r2"],
                }
            ),
            flush=True,
        )

    summaries.sort(key=lambda row: row["validation_row_metrics"]["r2"], reverse=True)
    (output_dir / "layer_sweep_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "layer_sweep_summary.md").open("w", encoding="utf-8") as f:
        f.write("| rank | layer | row_r2 | prompt_mean_r2 | row_mae | prompt_mean_mae | dim |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for idx, row in enumerate(summaries, 1):
            f.write(
                f"| {idx} | {row['prompt_layer_index']} | {row['validation_row_metrics']['r2']:.4f} | "
                f"{row['validation_prompt_mean_metrics']['r2']:.4f} | {row['validation_row_metrics']['mae']:.4f} | "
                f"{row['validation_prompt_mean_metrics']['mae']:.4f} | {row['feature_dim']} |\n"
            )
    print(json.dumps({"event": "done", "output_dir": str(output_dir), "num_layers": len(summaries)}), flush=True)


if __name__ == "__main__":
    main()
