from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

from classifer_training.single_rollout_hidden_utils import (
    build_rollout_hidden_lookup,
    build_rollout_index_lookup,
    load_prompt_hidden_lookup,
    normalize_run_dir,
    rollout_to_correctness,
)
from classifer_training.utils import load_records


ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT = Path(
    os.environ.get(
        "BASE_OUTPUT",
        str(ROOT / "classifer_training/artifacts/probe/spo_temp1_subset0to4_qwen3_4b_base_rowr2_axis_sweep"),
    )
)
DATASET_DIR = ROOT / "classifer_training/artifacts/datasets/spo_temp1_subset0to4"
PROMPT_SLUG = "qwen3_4b_base_l18_35_last5_10_15mean"
ROLLOUT_SLUG = "Qwen_Qwen3-4B"
ROLLOUT_HIDDEN_DIR = Path(
    os.environ.get(
        "ROLLOUT_HIDDEN_DIR",
        str(ROOT / "classifer_training/artifacts/rollout_hidden/spo_temp1_subset0to4" / ROLLOUT_SLUG),
    )
)
ROLLOUT_INDEX_DIR = Path(
    os.environ.get(
        "ROLLOUT_INDEX_DIR",
        str(ROOT / "classifer_training/artifacts/rollout_index/spo_temp1_subset0to4" / ROLLOUT_SLUG),
    )
)
INCLUDE_PROMPT_HIDDEN = os.environ.get("INCLUDE_PROMPT_HIDDEN", "1") == "1"
INCLUDE_ROLLOUT_HIDDEN = os.environ.get("INCLUDE_ROLLOUT_HIDDEN", "1") == "1"
LABEL_SOURCE = os.environ.get("LABEL_SOURCE", "rollout_score_or_reward")

TRAIN_SUBSETS = {0, 1}
TEST_SUBSETS = {2, 3, 4}

DEFAULT_SCALARS = [
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


def _load_scalar_keys() -> list[str]:
    raw_json = os.environ.get("ROLLOUT_SCALAR_KEYS_JSON", "").strip()
    if raw_json:
        value = json.loads(raw_json)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("ROLLOUT_SCALAR_KEYS_JSON must be a JSON list of strings")
        return [item.strip() for item in value if item.strip()]

    raw_csv = os.environ.get("ROLLOUT_SCALAR_KEYS", "").strip()
    if raw_csv:
        return [item.strip() for item in raw_csv.split(",") if item.strip()]

    return list(DEFAULT_SCALARS)


SCALARS = _load_scalar_keys()


class FastPCA:
    def __init__(self, mean: np.ndarray, components: np.ndarray):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.components_ = np.asarray(components, dtype=np.float32)
        self.n_components_ = int(self.components_.shape[0])
        self.n_features_in_ = int(self.components_.shape[1])

    def transform(self, x: np.ndarray) -> np.ndarray:
        value = np.asarray(x, dtype=np.float32)
        return (value - self.mean_) @ self.components_.T


def _subset_id(row: dict[str, Any]) -> int | None:
    text = str(row.get("run_name") or row.get("run_dir") or row.get("source_jsonl") or "")
    match = re.search(r"subset[_-](\d+)", text)
    return int(match.group(1)) if match else None


def _fast_entropy_scalar_vec(record: dict[str, Any]) -> np.ndarray:
    rollout_features = record.get("rollout_features")
    feature_map = dict(rollout_features) if isinstance(rollout_features, dict) else {}
    reasoning_mean = float(feature_map.get("reasoning_mean_token_entropy", 0.0) or 0.0)
    answer_mean = float(feature_map.get("answer_mean_token_entropy", 0.0) or 0.0)
    output_mean = float(feature_map.get("output_mean_token_entropy", 0.0) or 0.0)
    feature_map.setdefault("entropy_gap_reasoning_answer", reasoning_mean - answer_mean)
    feature_map.setdefault("answer_entropy_gap_vs_output", answer_mean - output_mean)
    feature_map.setdefault("rollout_features.answer_mean_token_entropy", answer_mean)
    return np.asarray([float(feature_map.get(key, 0.0) or 0.0) for key in SCALARS], dtype=np.float32)


def _prompt_paths() -> tuple[list[Path], list[Path]]:
    hidden_paths = sorted((ROOT / "classifer_training/artifacts/hidden").glob(f"spo_temp1_subset0to4_shard*/{PROMPT_SLUG}/hidden_states.pt"))
    index_paths = sorted((ROOT / "classifer_training/artifacts/index").glob(f"spo_temp1_subset0to4_shard*/{PROMPT_SLUG}/index.jsonl"))
    if len(hidden_paths) != 4 or len(index_paths) != 4:
        raise FileNotFoundError(f"Expected 4 prompt hidden/index shards; got {len(hidden_paths)}/{len(index_paths)}")
    return hidden_paths, index_paths


def _load_prompt_text_by_task() -> dict[str, str]:
    rows = []
    for split in ("train", "validation"):
        path = DATASET_DIR / f"{split}.jsonl"
        if path.exists():
            rows.extend(load_records(path))
    return {str(row["task_id"]): str(row.get("user_input", "")) for row in rows}


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _prompt_mean_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_ids: list[str]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for task_id, true_value, pred_value in zip(task_ids, y_true.tolist(), y_pred.tolist(), strict=True):
        row = grouped.setdefault(task_id, {"value_true": [], "value_pred": []})
        row["value_true"].append(float(true_value))
        row["value_pred"].append(float(pred_value))
    prompt_rows = [
        {
            "task_id": task_id,
            "value_true": float(np.mean(values["value_true"])),
            "value_pred": float(np.mean(values["value_pred"])),
            "num_rows": int(len(values["value_pred"])),
        }
        for task_id, values in sorted(grouped.items())
    ]
    yt = np.asarray([row["value_true"] for row in prompt_rows], dtype=np.float32)
    yp = np.asarray([row["value_pred"] for row in prompt_rows], dtype=np.float32)
    return _reg_metrics(yt, yp), prompt_rows


def _fit_pca(vectors: list[np.ndarray], dim: int) -> FastPCA:
    x = np.stack([np.asarray(vec, dtype=np.float32).reshape(-1) for vec in vectors], axis=0)
    effective_dim = min(int(dim), int(x.shape[0]), int(x.shape[1]))
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = x - mean
    _, _, vt = randomized_svd(
        centered,
        n_components=effective_dim,
        n_iter=2,
        power_iteration_normalizer="none",
        random_state=42,
    )
    return FastPCA(mean=mean, components=vt.astype(np.float32, copy=False))


def _transform_lookup(lookup: dict[str, np.ndarray], pca: FastPCA) -> dict[str, np.ndarray]:
    keys = list(lookup.keys())
    x = np.stack([np.asarray(lookup[key], dtype=np.float32).reshape(-1) for key in keys], axis=0)
    transformed = pca.transform(x).astype(np.float32, copy=False)
    return {key: transformed[idx] for idx, key in enumerate(keys)}


def _transform_rows(rows: list[dict[str, Any]], pca: FastPCA) -> list[dict[str, Any]]:
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
    rollout_component: str,
    rollout_index_lookup: dict[tuple[str, int], dict[str, Any]],
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray] | None,
) -> list[dict[str, Any]]:
    correctness_by_task: dict[str, list[tuple[int, int, float]]] = {}
    raw_rows = []
    for key, index_row in sorted(rollout_index_lookup.items()):
        task_id = str(index_row.get("task_id", ""))
        subset_id = _subset_id(index_row)
        if subset_id is None or subset_id not in (TRAIN_SUBSETS | TEST_SUBSETS):
            continue
        run_dir, rollout_row_index = key
        hidden_vec = None if rollout_hidden_lookup is None else rollout_hidden_lookup.get((run_dir, rollout_row_index))
        if INCLUDE_ROLLOUT_HIDDEN and hidden_vec is None:
            continue
        correctness = rollout_to_correctness(index_row)
        correctness_by_task.setdefault(task_id, []).append((subset_id, int(rollout_row_index), float(correctness)))
        raw_rows.append(
            {
                "task_id": task_id,
                "subset_id": int(subset_id),
                "run_dir": normalize_run_dir(run_dir),
                "rollout_row_index": int(rollout_row_index),
                "sample_index": int(index_row.get("sample_index", -1)),
                "rollout_hidden_vec": None if hidden_vec is None else np.asarray(hidden_vec, dtype=np.float32).reshape(-1),
                "rollout_scalar_vec": _fast_entropy_scalar_vec(index_row),
                "rollout_correctness": float(correctness),
            }
        )

    rows = []
    for row in raw_rows:
        task_targets = correctness_by_task.get(row["task_id"], [])
        subset_id = int(row["subset_id"])
        updated = dict(row)
        if subset_id in TRAIN_SUBSETS:
            sibling = [
                value
                for sid, rollout_row_index, value in task_targets
                if sid in TRAIN_SUBSETS and rollout_row_index != int(row["rollout_row_index"])
            ]
            if not sibling:
                continue
            updated["split"] = "train"
            updated["value_true"] = float(np.mean(sibling))
        else:
            test_values = [value for sid, _, value in task_targets if sid in TEST_SUBSETS]
            if not test_values:
                continue
            updated["split"] = "test"
            updated["value_true"] = float(np.mean(test_values))
        rows.append(updated)
    return rows


def _fit_one(
    *,
    name: str,
    prompt_lookup_raw: dict[str, np.ndarray],
    prompt_slug: str,
    prompt_component: str,
    prompt_layer: int,
    rollout_component: str,
    rollout_layer: int,
    prompt_pca_dim: int,
    rollout_pca_dim: int,
    rows_raw: list[dict[str, Any]],
    prompt_projection_cache: dict[tuple[str, int, int], tuple[FastPCA | None, dict[str, np.ndarray]]],
    rollout_projection_cache: dict[tuple[str, int, int], tuple[FastPCA | None, list[dict[str, Any]]]],
    prompt_text_by_task: dict[str, str],
) -> dict[str, Any]:
    output_dir = BASE_OUTPUT / name
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_pca = None
    prompt_lookup = prompt_lookup_raw
    if INCLUDE_PROMPT_HIDDEN:
        train_task_ids = {row["task_id"] for row in rows_raw if row["split"] == "train"}
        prompt_key = (prompt_component, int(prompt_layer), int(prompt_pca_dim))
        if prompt_key not in prompt_projection_cache:
            if int(prompt_pca_dim) <= 0:
                print(json.dumps({"event": "skip_prompt_pca", "name": name, "key": list(prompt_key)}), flush=True)
                prompt_projection_cache[prompt_key] = (None, prompt_lookup_raw)
            else:
                print(json.dumps({"event": "fit_prompt_pca_start", "name": name, "key": list(prompt_key)}), flush=True)
                pca = _fit_pca([vec for task_id, vec in prompt_lookup_raw.items() if task_id in train_task_ids], prompt_pca_dim)
                prompt_projection_cache[prompt_key] = (pca, _transform_lookup(prompt_lookup_raw, pca))
                print(json.dumps({"event": "fit_prompt_pca_done", "name": name, "key": list(prompt_key)}), flush=True)
        prompt_pca, prompt_lookup = prompt_projection_cache[prompt_key]

    rollout_pca = None
    rows = rows_raw
    if INCLUDE_ROLLOUT_HIDDEN:
        rollout_key = (rollout_component, int(rollout_layer), int(rollout_pca_dim))
        if rollout_key not in rollout_projection_cache:
            if int(rollout_pca_dim) <= 0:
                print(json.dumps({"event": "skip_rollout_pca", "name": name, "key": list(rollout_key)}), flush=True)
                rollout_projection_cache[rollout_key] = (None, rows_raw)
            else:
                print(json.dumps({"event": "fit_rollout_pca_start", "name": name, "key": list(rollout_key)}), flush=True)
                train_vectors = [row["rollout_hidden_vec"] for row in rows_raw if row["split"] == "train"]
                pca = _fit_pca(train_vectors, rollout_pca_dim)
                rollout_projection_cache[rollout_key] = (pca, _transform_rows(rows_raw, pca))
                print(json.dumps({"event": "fit_rollout_pca_done", "name": name, "key": list(rollout_key)}), flush=True)
        rollout_pca, rows = rollout_projection_cache[rollout_key]

    x_rows, y_rows, splits, task_ids = [], [], [], []
    for row in rows:
        prompt_vec = prompt_lookup.get(row["task_id"])
        if INCLUDE_PROMPT_HIDDEN and prompt_vec is None:
            continue
        pieces = []
        if INCLUDE_PROMPT_HIDDEN:
            pieces.append(np.asarray(prompt_vec, dtype=np.float32).reshape(-1))
        if INCLUDE_ROLLOUT_HIDDEN:
            pieces.append(np.asarray(row["rollout_hidden_vec"], dtype=np.float32).reshape(-1))
        pieces.append(np.asarray(row["rollout_scalar_vec"], dtype=np.float32).reshape(-1))
        x_rows.append(np.concatenate(pieces, axis=0).astype(np.float32))
        y_rows.append(float(row["value_true"]))
        splits.append(str(row["split"]))
        task_ids.append(str(row["task_id"]))

    x = np.stack(x_rows, axis=0)
    y = np.asarray(y_rows, dtype=np.float32)
    split_arr = np.asarray(splits)
    train_mask = split_arr == "train"
    test_mask = split_arr == "test"
    estimator = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=0.01, solver="lsqr"))])
    print(json.dumps({"event": "fit_ridge_start", "name": name, "shape": list(x.shape)}), flush=True)
    estimator.fit(x[train_mask], y[train_mask])
    pred = np.clip(np.asarray(estimator.predict(x[test_mask]), dtype=np.float32), 0.0, 1.0)
    print(json.dumps({"event": "fit_ridge_done", "name": name}), flush=True)

    test_y = y[test_mask]
    test_task_ids = [task_ids[idx] for idx in np.where(test_mask)[0]]
    row_metrics = _reg_metrics(test_y, pred)
    prompt_metrics, prompt_rows = _prompt_mean_metrics(test_y, pred, test_task_ids)
    summary = {
        "setting": "spo_qwen3_4b_base_rowr2_axis_sweep",
        "name": name,
        "model": "StandardScaler -> Ridge(alpha=0.01, solver='lsqr') -> clip[0,1]",
        "train_subsets": sorted(TRAIN_SUBSETS),
        "test_subsets": sorted(TEST_SUBSETS),
        "label_source": LABEL_SOURCE,
        "rollout_index_dir": str(ROLLOUT_INDEX_DIR),
        "train_target": "other rollout correctness within train prompts from subset 0/1",
        "test_target": "prompt Avg correctness within test prompts from subset 2/3/4",
        "prompt_slug": prompt_slug,
        "prompt_component": prompt_component,
        "prompt_layer_index": int(prompt_layer),
        "prompt_hidden_pca_dim": int(prompt_pca_dim),
        "include_prompt_hidden": bool(INCLUDE_PROMPT_HIDDEN),
        "rollout_component": rollout_component,
        "rollout_layer_index": int(rollout_layer),
        "rollout_hidden_pca_dim": int(rollout_pca_dim),
        "include_rollout_hidden": bool(INCLUDE_ROLLOUT_HIDDEN),
        "rollout_scalar_keys": SCALARS,
        "feature_dim": int(x.shape[1]),
        "num_train_rows": int(train_mask.sum()),
        "num_test_rows": int(test_mask.sum()),
        "num_test_prompts": int(len(prompt_rows)),
        "test_row_metrics": row_metrics,
        "test_prompt_mean_metrics": prompt_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "predictions_test_rows.jsonl").open("w") as f:
        for true_value, pred_value, task_id in zip(test_y.tolist(), pred.tolist(), test_task_ids, strict=True):
            f.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "user_input": prompt_text_by_task.get(task_id, ""),
                        "value_true": float(true_value),
                        "value_pred": float(pred_value),
                    }
                )
                + "\n"
            )
    with (output_dir / "predictions_test_prompt_mean.jsonl").open("w") as f:
        for row in prompt_rows:
            row = dict(row)
            row["user_input"] = prompt_text_by_task.get(str(row["task_id"]), "")
            f.write(json.dumps(row) + "\n")
    joblib.dump(
        {
            "bundle_type": "spo_subset_rowr2_probe",
            "config": summary,
            "estimator": estimator,
            "prompt_hidden_pca": prompt_pca,
            "rollout_hidden_pca": rollout_pca,
        },
        output_dir / "model.joblib",
    )
    return summary


def main() -> None:
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    smoke_only = os.environ.get("SMOKE_ONLY", "0") == "1"
    prompt_text_by_task = _load_prompt_text_by_task()
    rollout_index_paths = sorted(ROLLOUT_INDEX_DIR.glob("rollout_index.shard*.jsonl"))
    rollout_hidden_paths = sorted(ROLLOUT_HIDDEN_DIR.glob("rollout_hidden_states.shard*.pt"))
    if not rollout_index_paths and not rollout_hidden_paths:
        single_index_path = ROLLOUT_INDEX_DIR / "rollout_index.jsonl"
        single_hidden_path = ROLLOUT_HIDDEN_DIR / "rollout_hidden_states.pt"
        if single_index_path.exists() and single_hidden_path.exists():
            rollout_index_paths = [single_index_path]
            rollout_hidden_paths = [single_hidden_path]
    rollout_index_lookup = build_rollout_index_lookup(rollout_index_paths)
    print(json.dumps({"event": "loaded_index", "num_rows": len(rollout_index_lookup)}), flush=True)

    prompt_hidden_paths, prompt_index_paths = _prompt_paths()
    prompt_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    rollout_rows_cache: dict[str, list[dict[str, Any]]] = {}
    prompt_projection_cache: dict[tuple[str, int, int], tuple[FastPCA | None, dict[str, np.ndarray]]] = {}
    rollout_projection_cache: dict[tuple[str, int, int], tuple[FastPCA | None, list[dict[str, Any]]]] = {}

    def get_prompt(component: str, layer: int) -> dict[str, np.ndarray]:
        key = (component, int(layer))
        if key not in prompt_cache:
            print(json.dumps({"event": "load_prompt_start", "component": component, "layer": layer}), flush=True)
            prompt_cache[key] = load_prompt_hidden_lookup(
                prompt_hidden_paths,
                prompt_index_paths,
                layer_index=layer,
                component_name=component,
            )
            print(json.dumps({"event": "load_prompt_done", "component": component, "layer": layer, "num_rows": len(prompt_cache[key])}), flush=True)
        return prompt_cache[key]

    def get_rows(component: str, layer: int) -> list[dict[str, Any]]:
        key = f"{component}_L{int(layer)}"
        if key not in rollout_rows_cache:
            hidden_lookup = None
            if INCLUDE_ROLLOUT_HIDDEN:
                print(json.dumps({"event": "load_rollout_hidden_start", "component": component, "layer": int(layer)}), flush=True)
                hidden_lookup = build_rollout_hidden_lookup(
                    rollout_hidden_paths,
                    rollout_index_paths,
                    component_name=component,
                    layer_index=int(layer),
                    pool_mode="mean",
                )
                print(json.dumps({"event": "load_rollout_hidden_done", "component": component, "layer": int(layer), "num_rows": len(hidden_lookup)}), flush=True)
            rollout_rows_cache[key] = _build_rows(
                rollout_component=component,
                rollout_index_lookup=rollout_index_lookup,
                rollout_hidden_lookup=hidden_lookup,
            )
            print(json.dumps({"event": "build_rows_done", "component": component, "layer": int(layer), "num_rows": len(rollout_rows_cache[key])}), flush=True)
        return rollout_rows_cache[key]

    no_pca_sweep = os.environ.get("NO_PCA_SWEEP", "0") == "1"
    pca_focus_sweep = os.environ.get("PCA_FOCUS_SWEEP", "0") == "1"
    pca_layer_n_sweep = os.environ.get("PCA_LAYER_N_SWEEP", "0") == "1"
    pca_tied_layer_n_sweep = os.environ.get("PCA_TIED_LAYER_N_SWEEP", "0") == "1"
    pca_tied_n_only_sweep = os.environ.get("PCA_TIED_N_ONLY_SWEEP", "0") == "1"
    pca_tied_full_grid = os.environ.get("PCA_TIED_FULL_GRID", "0") == "1"
    single_tied_config = os.environ.get("SINGLE_TIED_CONFIG", "0") == "1"
    if pca_tied_full_grid:
        configs = []
        for n_name, prompt_component, rollout_component in [
            ("last5", "hidden_last5_mean", "response_last5_mean_hidden"),
            ("last10", "hidden_last10_mean", "response_last10_mean_hidden"),
            ("last15", "hidden_last15_mean", "response_last15_mean_hidden"),
        ]:
            for layer in range(18, 36):
                configs.append((f"pca_tied_grid_{n_name}_L{layer}_p32_r256", prompt_component, rollout_component, layer, layer, 32, 256))
    elif single_tied_config:
        n_value = os.environ["TIED_N_VALUE"]
        layer = int(os.environ["TIED_LAYER"])
        n_components = {
            "5": ("last5", "hidden_last5_mean", "response_last5_mean_hidden"),
            "10": ("last10", "hidden_last10_mean", "response_last10_mean_hidden"),
            "15": ("last15", "hidden_last15_mean", "response_last15_mean_hidden"),
        }
        n_name, prompt_component, rollout_component = n_components[n_value]
        rollout_component = os.environ.get("SINGLE_ROLLOUT_COMPONENT", rollout_component)
        name_suffix = os.environ.get("SINGLE_NAME_SUFFIX", "")
        configs = [
            (f"pca_tied_single_{n_name}_L{layer}_p32_r256{name_suffix}", prompt_component, rollout_component, layer, layer, 32, 256),
        ]
    elif pca_tied_n_only_sweep:
        n_components = {
            "5": ("last5", "hidden_last5_mean", "response_last5_mean_hidden"),
            "10": ("last10", "hidden_last10_mean", "response_last10_mean_hidden"),
            "15": ("last15", "hidden_last15_mean", "response_last15_mean_hidden"),
        }
        requested_n_values = [
            value.strip()
            for value in os.environ.get("TIED_N_VALUES", "5,10,15").split(",")
            if value.strip()
        ]
        configs = []
        for requested_n in requested_n_values:
            n_name, prompt_component, rollout_component = n_components[requested_n]
            configs.append((f"pca_tied_n_{n_name}_L35_p32_r256", prompt_component, rollout_component, 35, 35, 32, 256))
    elif pca_tied_layer_n_sweep:
        configs = [
            ("pca_tied_center_last5_L35_p32_r256", "hidden_last5_mean", "response_last5_mean_hidden", 35, 35, 32, 256),
        ]
    elif pca_layer_n_sweep:
        configs = [
            ("pca_layer_n_center_pLast5_L35_rLast5_L26_p32_r256", "hidden_last5_mean", "response_last5_mean_hidden", 35, 26, 32, 256),
        ]
    elif pca_focus_sweep:
        configs = [
            ("focus_pLast5_L35_rLast5_L26_p32_r256", "hidden_last5_mean", "response_last5_mean_hidden", 35, 26, 32, 256),
        ]
    elif no_pca_sweep:
        configs = [
            ("center_no_pca_pLast5_L26_rLast5_L26", "hidden_last5_mean", "response_last5_mean_hidden", 26, 26, 0, 0),
        ]
    else:
        configs = [
            ("center_pLast5_rLast5_L26_p32_r256", "hidden_last5_mean", "response_last5_mean_hidden", 26, 26, 32, 256),
        ]
    if not smoke_only:
        if pca_tied_full_grid or single_tied_config or pca_tied_n_only_sweep:
            pass
        elif pca_tied_layer_n_sweep:
            tied_layers = [18, 22, 26, 30, 34, 35]
            if os.environ.get("DENSE_TIED_LAYERS", "0") == "1":
                tied_layers = [18, 20, 22, 24, 26, 28, 30, 32, 34, 35]
            for layer in tied_layers:
                configs.append((f"pca_tied_layer_last5_L{layer}_p32_r256", "hidden_last5_mean", "response_last5_mean_hidden", layer, layer, 32, 256))
            for n_value, prompt_component, rollout_component in [
                ("last5", "hidden_last5_mean", "response_last5_mean_hidden"),
                ("last10", "hidden_last10_mean", "response_last10_mean_hidden"),
                ("last15", "hidden_last15_mean", "response_last15_mean_hidden"),
            ]:
                configs.append((f"pca_tied_n_{n_value}_L35_p32_r256", prompt_component, rollout_component, 35, 35, 32, 256))
        elif pca_layer_n_sweep:
            for layer in [18, 20, 22, 24, 26, 28, 30, 32, 34, 35]:
                configs.append((f"pca_prompt_layer_pLast5_L{layer}_rLast5_L26_p32_r256", "hidden_last5_mean", "response_last5_mean_hidden", layer, 26, 32, 256))
            for layer in [18, 20, 22, 24, 26, 28, 30, 32, 34, 35]:
                configs.append((f"pca_rollout_layer_pLast5_L35_rLast5_L{layer}_p32_r256", "hidden_last5_mean", "response_last5_mean_hidden", 35, layer, 32, 256))
            for prompt_component in ["hidden_last5_mean", "hidden_last10_mean", "hidden_last15_mean"]:
                n_value = prompt_component.removeprefix("hidden_").removesuffix("_mean")
                configs.append((f"pca_prompt_n_{n_value}_L35_rLast5_L26_p32_r256", prompt_component, "response_last5_mean_hidden", 35, 26, 32, 256))
            for rollout_component in ["response_last5_mean_hidden", "response_last10_mean_hidden", "response_last15_mean_hidden"]:
                n_value = rollout_component.removeprefix("response_").removesuffix("_mean_hidden")
                configs.append((f"pca_rollout_n_pLast5_L35_r{n_value}_L26_p32_r256", "hidden_last5_mean", rollout_component, 35, 26, 32, 256))
        elif pca_focus_sweep:
            for prompt_pca in [8, 16, 32, 64, 128, 256]:
                configs.append((f"focus_prompt_pca_p{prompt_pca}_r256_pLast5_L35_rLast5_L26", "hidden_last5_mean", "response_last5_mean_hidden", 35, 26, prompt_pca, 256))
            for rollout_pca in [32, 64, 128, 256, 512, 1024]:
                configs.append((f"focus_rollout_pca_p32_r{rollout_pca}_pLast5_L35_rLast5_L26", "hidden_last5_mean", "response_last5_mean_hidden", 35, 26, 32, rollout_pca))
        elif no_pca_sweep:
            for layer in [18, 20, 22, 24, 26, 28, 30, 32, 34, 35]:
                configs.append((f"prompt_layer_no_pca_pLast5_L{layer}_rLast5_L26", "hidden_last5_mean", "response_last5_mean_hidden", layer, 26, 0, 0))
            for layer in [18, 20, 22, 24, 26, 28, 30, 32, 34, 35]:
                configs.append((f"rollout_layer_no_pca_pLast5_L26_rLast5_L{layer}", "hidden_last5_mean", "response_last5_mean_hidden", 26, layer, 0, 0))
            for prompt_component in ["hidden_last5_mean", "hidden_last10_mean", "hidden_last15_mean"]:
                configs.append((f"prompt_n_no_pca_{prompt_component}_L26_rLast5_L26", prompt_component, "response_last5_mean_hidden", 26, 26, 0, 0))
            for rollout_component in ["response_last5_mean_hidden", "response_last10_mean_hidden", "response_last15_mean_hidden"]:
                configs.append((f"rollout_n_no_pca_pLast5_L26_{rollout_component}_L26", "hidden_last5_mean", rollout_component, 26, 26, 0, 0))
        else:
            for layer in [18, 20, 22, 24, 26, 28, 30, 32, 34, 35]:
                configs.append((f"prompt_layer_L{layer}_pLast5_rLast5_p32_r256", "hidden_last5_mean", "response_last5_mean_hidden", layer, 26, 32, 256))
            for prompt_component in ["hidden_last5_mean", "hidden_last10_mean", "hidden_last15_mean"]:
                configs.append((f"prompt_n_{prompt_component}_L26_rLast5_p32_r256", prompt_component, "response_last5_mean_hidden", 26, 26, 32, 256))
            for rollout_component in ["response_last5_mean_hidden", "response_last10_mean_hidden", "response_last15_mean_hidden"]:
                configs.append((f"rollout_n_{rollout_component}_L26_pLast5_p32_r256", "hidden_last5_mean", rollout_component, 26, 26, 32, 256))
            for prompt_pca in [16, 32, 64, 128]:
                configs.append((f"prompt_pca_p{prompt_pca}_r256_L26_last5", "hidden_last5_mean", "response_last5_mean_hidden", 26, 26, prompt_pca, 256))
            for rollout_pca in [64, 128, 256, 512]:
                configs.append((f"rollout_pca_p32_r{rollout_pca}_L26_last5", "hidden_last5_mean", "response_last5_mean_hidden", 26, 26, 32, rollout_pca))

    seen = set()
    results = []
    for name, prompt_component, rollout_component, prompt_layer, rollout_layer, prompt_pca, rollout_pca in configs:
        if name in seen:
            continue
        seen.add(name)
        print(json.dumps({"event": "start", "name": name}), flush=True)
        summary = _fit_one(
            name=name,
            prompt_lookup_raw=get_prompt(prompt_component, prompt_layer),
            prompt_slug=PROMPT_SLUG,
            prompt_component=prompt_component,
            prompt_layer=prompt_layer,
            rollout_component=rollout_component,
            rollout_layer=rollout_layer,
            prompt_pca_dim=prompt_pca,
            rollout_pca_dim=rollout_pca,
            rows_raw=get_rows(rollout_component, rollout_layer),
            prompt_projection_cache=prompt_projection_cache,
            rollout_projection_cache=rollout_projection_cache,
            prompt_text_by_task=prompt_text_by_task,
        )
        results.append(summary)
        print(
            json.dumps(
                {
                    "event": "done",
                    "name": name,
                    "row_r2": summary["test_row_metrics"]["r2"],
                    "prompt_mean_r2": summary["test_prompt_mean_metrics"]["r2"],
                    "feature_dim": summary["feature_dim"],
                }
            ),
            flush=True,
        )

    results.sort(key=lambda row: row["test_row_metrics"]["r2"], reverse=True)
    (BASE_OUTPUT / "axis_sweep_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    with (BASE_OUTPUT / "axis_sweep_summary.md").open("w") as f:
        f.write("| rank | name | row_r2 | prompt_mean_r2 | row_mae | prompt_mean_mae | dim |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for idx, row in enumerate(results, 1):
            f.write(
                f"| {idx} | {row['name']} | {row['test_row_metrics']['r2']:.4f} | "
                f"{row['test_prompt_mean_metrics']['r2']:.4f} | {row['test_row_metrics']['mae']:.4f} | "
                f"{row['test_prompt_mean_metrics']['mae']:.4f} | {row['feature_dim']} |\n"
            )
    print(json.dumps({"event": "finished", "best": results[0]["name"], "row_r2": results[0]["test_row_metrics"]["r2"]}), flush=True)


if __name__ == "__main__":
    main()
