from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from classifer_training.data import load_hidden_rows
from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.utils import load_records, write_jsonl
from verl.utils.single_trajectory_estimator import (
    REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS,
    extract_derived_rollout_features,
)

PROMPT_FEATURE_NAMES = [
    "input_length",
    "char_count",
    "word_count",
    "line_count",
    "digit_count",
    "digit_ratio",
    "latex_command_count",
    "dollar_count",
    "backslash_count",
    "equals_count",
    "caret_count",
    "slash_count",
    "paren_count",
    "bracket_count",
    "brace_count",
    "number_literal_count",
    "comma_count",
    "colon_count",
    "question_count",
    "sqrt_count",
    "frac_count",
    "geometry_keyword_count",
    "algebra_keyword_count",
]


def normalize_run_dir(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


def load_labels_by_task(labels_path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in load_records(labels_path.expanduser().resolve())}


def label_to_value(label_row: dict[str, Any]) -> float:
    if "value" in label_row:
        return float(label_row["value"])
    return 1.0 - float(label_row["difficulty"])


def build_split_lookup(prompt_dataset_dir: Path) -> dict[str, str]:
    split_lookup: dict[str, str] = {}
    for split_name in ("train", "validation"):
        path = prompt_dataset_dir / f"{split_name}.jsonl"
        for row in load_records(path):
            split_lookup[str(row["task_id"])] = split_name
    return split_lookup


def _prompt_features(text: str, input_length: int) -> np.ndarray:
    text = text or ""
    char_count = len(text)
    word_count = len(text.split())
    line_count = text.count("\n") + 1
    digit_count = sum(ch.isdigit() for ch in text)
    digit_ratio = digit_count / max(char_count, 1)
    latex_commands = re.findall(r"\\[A-Za-z]+", text)
    number_literals = re.findall(r"-?\d+(?:\.\d+)?", text)
    geometry_keywords = re.findall(
        r"\b(triangle|rectangle|circle|angle|polygon|segment|perpendicular|parallel|isosceles|equilateral)\b",
        text.lower(),
    )
    algebra_keywords = re.findall(
        r"\b(equation|polynomial|integer|prime|factor|divisible|sequence|series|probability|matrix)\b",
        text.lower(),
    )
    return np.asarray(
        [
            float(input_length),
            float(char_count),
            float(word_count),
            float(line_count),
            float(digit_count),
            float(digit_ratio),
            float(len(latex_commands)),
            float(text.count("$")),
            float(text.count("\\")),
            float(text.count("=")),
            float(text.count("^")),
            float(text.count("/")),
            float(sum(text.count(ch) for ch in "()")),
            float(sum(text.count(ch) for ch in "[]")),
            float(sum(text.count(ch) for ch in "{}")),
            float(len(number_literals)),
            float(text.count(",")),
            float(text.count(":")),
            float(text.count("?")),
            float(sum(1 for command in latex_commands if command == "\\sqrt")),
            float(sum(1 for command in latex_commands if command == "\\frac")),
            float(len(geometry_keywords)),
            float(len(algebra_keywords)),
        ],
        dtype=np.float32,
    )


def build_prompt_scalar_lookup(
    labels_by_task: dict[str, dict[str, Any]],
    feature_keys: list[str],
) -> dict[str, np.ndarray]:
    feature_index = {name: idx for idx, name in enumerate(PROMPT_FEATURE_NAMES)}
    missing = [key for key in feature_keys if key not in feature_index]
    if missing:
        raise ValueError(f"Unsupported prompt feature keys: {missing}")

    lookup: dict[str, np.ndarray] = {}
    for task_id, row in labels_by_task.items():
        user_input = str(row.get("user_input", ""))
        input_length = len(user_input.split())
        all_features = _prompt_features(user_input, input_length)
        lookup[task_id] = np.asarray([all_features[feature_index[key]] for key in feature_keys], dtype=np.float32)
    return lookup


def load_prompt_hidden_lookup(
    hidden_paths: list[Path],
    index_paths: list[Path],
    *,
    layer_index: int,
) -> dict[str, np.ndarray]:
    if len(hidden_paths) != len(index_paths):
        raise ValueError("Prompt hidden/index path counts must match.")

    lookup: dict[str, np.ndarray] = {}
    for hidden_path, index_path in zip(hidden_paths, index_paths, strict=True):
        rows = load_hidden_rows(
            hidden_path.expanduser().resolve(),
            index_path=index_path.expanduser().resolve(),
            dataset_name="dapo_math_17k",
            default_component_name="hidden",
        )
        for row in rows:
            layers = row["components"]["hidden"]
            if layer_index >= len(layers):
                raise ValueError(
                    f"Requested prompt layer index {layer_index}, but only {len(layers)} layers are present."
                )
            lookup[str(row["task_id"])] = np.asarray(layers[layer_index], dtype=np.float32).reshape(-1)
    return lookup


def build_rollout_hidden_lookup(
    hidden_paths: list[Path],
    index_paths: list[Path],
    *,
    component_name: str,
    layer_index: int,
    pool_mode: str = "mean",
) -> dict[tuple[str, int], np.ndarray]:
    if len(hidden_paths) != len(index_paths):
        raise ValueError("Rollout hidden/index path counts must match.")

    lookup: dict[tuple[str, int], np.ndarray] = {}
    for hidden_path, index_path in zip(hidden_paths, index_paths, strict=True):
        rows = load_hidden_rows(
            hidden_path.expanduser().resolve(),
            index_path=index_path.expanduser().resolve(),
            dataset_name="dapo_math_17k",
            default_component_name=component_name,
        )
        for row in rows:
            index_row = row["index_row"]
            run_dir = normalize_run_dir(str(index_row.get("run_dir", "")))
            rollout_row_index = int(index_row.get("rollout_row_index", -1))
            if rollout_row_index < 0 or not run_dir:
                continue
            layers = row["components"][component_name]
            if layer_index >= len(layers):
                raise ValueError(
                    f"Requested rollout layer index {layer_index}, but only {len(layers)} layers are present."
                )
            value = np.asarray(layers[layer_index], dtype=np.float32)
            if value.ndim > 1:
                if pool_mode == "mean":
                    value = value.mean(axis=0)
                elif pool_mode == "last":
                    value = value[-1]
                elif pool_mode == "first":
                    value = value[0]
                elif pool_mode == "flatten":
                    value = value.reshape(-1)
                else:
                    raise ValueError(f"Unsupported rollout hidden pool mode: {pool_mode}")
            lookup[(run_dir, rollout_row_index)] = value.astype(np.float32, copy=False)
    return lookup


def build_rollout_index_lookup(index_paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for index_path in index_paths:
        for row in load_records(index_path.expanduser().resolve()):
            run_dir = normalize_run_dir(str(row.get("run_dir", "")))
            rollout_row_index = int(row.get("rollout_row_index", row.get("sample_index", -1)))
            if rollout_row_index < 0:
                continue
            lookup[(run_dir, rollout_row_index)] = row
    return lookup


def candidate_rollout_row_indices(row: dict[str, Any], row_idx: int) -> list[int]:
    values: list[int] = []
    for raw_value in (row.get("rollout_row_index"), row_idx, row.get("sample_index")):
        if raw_value is None:
            continue
        value = int(raw_value)
        if value < 0 or value in values:
            continue
        values.append(value)
    return values


def extract_rollout_scalar_vec(
    record: dict[str, Any],
    feature_keys: list[str],
    derived_feature_keys: list[str],
    extra_field_paths: list[str],
) -> np.ndarray:
    feature_map = extract_rollout_numeric_features(record, extra_numeric_fields=extra_field_paths)
    missing_entropy_keys = sorted(REQUIRED_ACTUAL_TOKEN_ENTROPY_KEYS.intersection(feature_keys) - feature_map.keys())
    if missing_entropy_keys:
        raise ValueError(
            f"Missing actual token entropy features {missing_entropy_keys} in rollout record. "
            "Re-extract rollout index with the updated extractor."
        )
    feature_map.update(extract_derived_rollout_features(feature_map))
    ordered_keys = list(feature_keys) + list(derived_feature_keys)
    ordered_keys.extend(path.replace(".", "_") for path in extra_field_paths)
    return np.asarray([float(feature_map.get(key, 0.0)) for key in ordered_keys], dtype=np.float32)


def fit_prompt_hidden_pca(
    prompt_lookup: dict[str, np.ndarray],
    split_lookup: dict[str, str],
    pca_dim: int,
) -> PCA | None:
    if pca_dim <= 0:
        return None

    train_vectors = [
        np.asarray(vec, dtype=np.float32).reshape(-1)
        for task_id, vec in prompt_lookup.items()
        if split_lookup.get(task_id) == "train"
    ]
    if not train_vectors:
        raise ValueError("No weak-train prompt vectors available for PCA fit.")

    x_train = np.stack(train_vectors, axis=0)
    effective_dim = min(int(pca_dim), int(x_train.shape[0]), int(x_train.shape[1]))
    if effective_dim <= 0:
        raise ValueError(f"Invalid effective PCA dim computed from requested={pca_dim}, shape={x_train.shape}.")

    pca = PCA(n_components=effective_dim, svd_solver="randomized", random_state=42)
    pca.fit(x_train)
    return pca


def apply_prompt_hidden_pca(
    prompt_lookup: dict[str, np.ndarray],
    pca: PCA | None,
) -> dict[str, np.ndarray]:
    if pca is None:
        return prompt_lookup

    transformed: dict[str, np.ndarray] = {}
    for task_id, vec in prompt_lookup.items():
        value = pca.transform(np.asarray(vec, dtype=np.float32).reshape(1, -1))[0].astype(np.float32, copy=False)
        transformed[task_id] = value
    return transformed


def fit_rollout_hidden_pca(
    rows: list[dict[str, Any]],
    pca_dim: int,
) -> PCA | None:
    if pca_dim <= 0:
        return None

    train_vectors = [
        np.asarray(row["rollout_hidden_vec"], dtype=np.float32).reshape(-1)
        for row in rows
        if row.get("split") == "train" and row.get("rollout_hidden_vec") is not None
    ]
    if not train_vectors:
        raise ValueError("No weak-train rollout vectors available for PCA fit.")

    x_train = np.stack(train_vectors, axis=0)
    effective_dim = min(int(pca_dim), int(x_train.shape[0]), int(x_train.shape[1]))
    if effective_dim <= 0:
        raise ValueError(f"Invalid effective rollout PCA dim computed from requested={pca_dim}, shape={x_train.shape}.")

    pca = PCA(n_components=effective_dim, svd_solver="randomized", random_state=42)
    pca.fit(x_train)
    return pca


def apply_rollout_hidden_pca(
    rows: list[dict[str, Any]],
    pca: PCA | None,
) -> list[dict[str, Any]]:
    if pca is None:
        return rows

    transformed: list[dict[str, Any]] = []
    for row in rows:
        rollout_vec = row.get("rollout_hidden_vec")
        if rollout_vec is None:
            transformed.append(dict(row))
            continue
        value = pca.transform(np.asarray(rollout_vec, dtype=np.float32).reshape(1, -1))[0].astype(np.float32, copy=False)
        updated = dict(row)
        updated["rollout_hidden_vec"] = value
        transformed.append(updated)
    return transformed


def group_weak_rollouts(
    *,
    weak_run_dirs: list[Path],
    split_lookup: dict[str, str],
    labels_by_task: dict[str, dict[str, Any]],
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray] | None,
    rollout_index_lookup: dict[tuple[str, int], dict[str, Any]] | None,
    rollout_scalar_keys: list[str],
    derived_rollout_scalar_keys: list[str],
    extra_rollout_scalar_field_paths: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run_dir in weak_run_dirs:
        rows = load_records(run_dir.expanduser().resolve() / "all_experiments.jsonl")
        run_dir_str = normalize_run_dir(str(run_dir.expanduser().resolve()))
        for row_idx, row in enumerate(rows):
            task_id = str(row["task_id"])
            label_row = labels_by_task.get(task_id)
            if label_row is None:
                continue
            group = grouped.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "split": split_lookup.get(task_id, str(row.get("split", "train"))),
                    "value_true": label_to_value(label_row),
                    "rollouts": [],
                },
            )
            sample_index = int(row.get("sample_index", -1))
            candidate_indices = candidate_rollout_row_indices(row, row_idx)
            if not candidate_indices:
                continue
            rollout_row_index = candidate_indices[0]
            rollout_hidden = None
            if rollout_hidden_lookup is not None:
                for candidate_index in candidate_indices:
                    rollout_hidden = rollout_hidden_lookup.get((run_dir_str, candidate_index))
                    if rollout_hidden is not None:
                        rollout_row_index = candidate_index
                        break
                if rollout_hidden is None:
                    continue
            scalar_source = row
            if rollout_index_lookup is not None:
                for candidate_index in candidate_indices:
                    matched = rollout_index_lookup.get((run_dir_str, candidate_index))
                    if matched is not None:
                        scalar_source = matched
                        rollout_row_index = candidate_index
                        break
            rollout_scalar_vec = (
                extract_rollout_scalar_vec(
                    scalar_source,
                    rollout_scalar_keys,
                    derived_rollout_scalar_keys,
                    extra_rollout_scalar_field_paths,
                )
                if (rollout_scalar_keys or derived_rollout_scalar_keys or extra_rollout_scalar_field_paths)
                else None
            )
            group["rollouts"].append(
                {
                    "run_dir": run_dir_str,
                    "rollout_row_index": rollout_row_index,
                    "sample_index": sample_index,
                    "rollout_hidden_vec": None
                    if rollout_hidden is None
                    else np.asarray(rollout_hidden, dtype=np.float32).reshape(-1),
                    "rollout_scalar_vec": rollout_scalar_vec,
                }
            )
    return [grouped[key] for key in sorted(grouped.keys())]


def group_eval_rollouts(
    *,
    labels_by_task: dict[str, dict[str, Any]],
    index_paths: list[Path],
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray] | None,
    rollout_scalar_keys: list[str],
    derived_rollout_scalar_keys: list[str],
    extra_rollout_scalar_field_paths: list[str],
    allowed_splits: set[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    allowed = allowed_splits or {"validation", "test"}
    for index_path in index_paths:
        for row in load_records(index_path.expanduser().resolve()):
            task_id = str(row.get("task_id", ""))
            label_row = labels_by_task.get(task_id)
            if label_row is None:
                continue
            split = str(row.get("split", ""))
            if split not in allowed:
                continue
            run_dir = normalize_run_dir(str(row.get("run_dir", "")))
            rollout_row_index = int(row.get("rollout_row_index", row.get("sample_index", -1)))
            rollout_hidden = None
            if rollout_hidden_lookup is not None:
                rollout_hidden = rollout_hidden_lookup.get((run_dir, rollout_row_index))
                if rollout_hidden is None:
                    continue
            rollout_scalar_vec = (
                extract_rollout_scalar_vec(
                    row,
                    rollout_scalar_keys,
                    derived_rollout_scalar_keys,
                    extra_rollout_scalar_field_paths,
                )
                if (rollout_scalar_keys or derived_rollout_scalar_keys or extra_rollout_scalar_field_paths)
                else None
            )
            group = grouped.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "split": split,
                    "value_true": label_to_value(label_row),
                    "rollouts": [],
                },
            )
            group["rollouts"].append(
                {
                    "rollout_row_index": rollout_row_index,
                    "rollout_hidden_vec": None
                    if rollout_hidden is None
                    else np.asarray(rollout_hidden, dtype=np.float32).reshape(-1),
                    "rollout_scalar_vec": rollout_scalar_vec,
                }
            )
    return [grouped[key] for key in sorted(grouped.keys())]


def select_single_rollout(grouped_rows: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in grouped_rows:
        rollouts = group["rollouts"]
        if not rollouts:
            continue
        if strategy == "first":
            selected_rollouts = [min(rollouts, key=lambda row: int(row["rollout_row_index"]))]
        elif strategy == "all":
            selected_rollouts = sorted(rollouts, key=lambda row: int(row["rollout_row_index"]))
        else:
            raise ValueError(f"Unsupported single rollout strategy: {strategy}")
        for chosen in selected_rollouts:
            rows.append(
                {
                    "task_id": str(group["task_id"]),
                    "split": str(group["split"]),
                    "value_true": float(group["value_true"]),
                    "run_dir": str(chosen.get("run_dir", "")),
                    "rollout_hidden_vec": chosen.get("rollout_hidden_vec"),
                    "rollout_scalar_vec": chosen.get("rollout_scalar_vec"),
                    "rollout_row_index": int(chosen["rollout_row_index"]),
                    "sample_index": int(chosen.get("sample_index", -1)),
                }
            )
    return rows


def build_matrix(
    rows: list[dict[str, Any]],
    prompt_lookup: dict[str, np.ndarray],
    prompt_scalar_lookup: dict[str, np.ndarray],
    *,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    split_rows: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row["task_id"])
        prompt_vec = prompt_lookup.get(task_id)
        if prompt_vec is None:
            continue
        pieces = [np.asarray(prompt_vec, dtype=np.float32).reshape(-1)]
        prompt_scalar_vec = prompt_scalar_lookup.get(task_id)
        if prompt_scalar_vec is not None:
            pieces.append(np.asarray(prompt_scalar_vec, dtype=np.float32).reshape(-1))
        if feature_mode == "prompt_plus_rollout":
            rollout_vec = row.get("rollout_hidden_vec")
            if rollout_vec is None:
                continue
            pieces.append(np.asarray(rollout_vec, dtype=np.float32).reshape(-1))
        rollout_scalar_vec = row.get("rollout_scalar_vec")
        if rollout_scalar_vec is not None:
            pieces.append(np.asarray(rollout_scalar_vec, dtype=np.float32).reshape(-1))
        x_rows.append(np.concatenate(pieces, axis=0).astype(np.float32))
        y_rows.append(float(row["value_true"]))
        split_rows.append(str(row["split"]))
        metadata_rows.append(
            {
                "task_id": task_id,
                "split": str(row["split"]),
                "value_true": float(row["value_true"]),
                "rollout_row_index": int(row.get("rollout_row_index", -1)),
                "sample_index": int(row.get("sample_index", -1)),
            }
        )
    return np.stack(x_rows), np.asarray(y_rows, dtype=np.float32), np.asarray(split_rows), metadata_rows


def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
    }


def prompt_mean_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    metadata_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"value_true": [], "value_pred": []})
    for meta, pred_val in zip(metadata_rows, pred.tolist(), strict=True):
        task_id = str(meta["task_id"])
        groups[task_id]["value_true"].append(float(meta["value_true"]))
        groups[task_id]["value_pred"].append(float(pred_val))

    prompt_rows: list[dict[str, Any]] = []
    for task_id in sorted(groups):
        prompt_rows.append(
            {
                "task_id": task_id,
                "value_true": float(np.mean(groups[task_id]["value_true"])),
                "value_pred": float(np.mean(groups[task_id]["value_pred"])),
                "num_rows": int(len(groups[task_id]["value_pred"])),
            }
        )
    prompt_true = np.asarray([row["value_true"] for row in prompt_rows], dtype=np.float32)
    prompt_pred = np.asarray([row["value_pred"] for row in prompt_rows], dtype=np.float32)
    return reg_metrics(prompt_true, prompt_pred), prompt_rows


def write_predictions(output_path: Path, prompt_rows: list[dict[str, Any]], labels_by_task: dict[str, dict[str, Any]]) -> None:
    rows = []
    for row in prompt_rows:
        label_row = labels_by_task[str(row["task_id"])]
        rows.append(
            {
                "task_id": str(row["task_id"]),
                "user_input": str(label_row.get("user_input", "")),
                "value_true": float(row["value_true"]),
                "value_pred": float(row["value_pred"]),
                "num_rows": int(row["num_rows"]),
            }
        )
    write_jsonl(output_path, rows)


def save_diagnostics_plot(path: Path, prompt_rows: list[dict[str, Any]], title: str) -> None:
    y_true = np.asarray([float(row["value_true"]) for row in prompt_rows], dtype=np.float32)
    y_pred = np.asarray([float(row["value_pred"]) for row in prompt_rows], dtype=np.float32)
    order = np.argsort(y_true)
    true_sorted = y_true[order]
    pred_sorted = y_pred[order]
    abs_err_sorted = np.abs(pred_sorted - true_sorted)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hb = axes[0].hexbin(y_true, y_pred, gridsize=32, cmap="viridis", bins="log", mincnt=1)
    axes[0].plot([0, 1], [0, 1], "--", color="tab:red", lw=1.5)
    axes[0].set_xlabel("True value")
    axes[0].set_ylabel("Predicted value")
    axes[0].set_title("GT vs Pred")
    fig.colorbar(hb, ax=axes[0], label="log count")

    axes[1].plot(true_sorted, color="black", lw=2, label="true")
    axes[1].plot(pred_sorted, color="tab:purple", lw=1.5, label="pred")
    axes[1].set_title("Sorted Alignment")
    axes[1].set_xlabel("Prompts sorted by true value")
    axes[1].legend(frameon=False)

    axes[2].plot(abs_err_sorted, color="teal", lw=1.2)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Prompts sorted by true value")
    axes[2].set_ylabel("|pred-true|")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
