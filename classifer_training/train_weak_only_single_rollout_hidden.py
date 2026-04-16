from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def _normalize_run_dir(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


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


def _build_split_lookup(prompt_dataset_dir: Path) -> dict[str, str]:
    split_lookup: dict[str, str] = {}
    for split_name in ("train", "validation"):
        path = prompt_dataset_dir / f"{split_name}.jsonl"
        for row in load_records(path):
            split_lookup[str(row["task_id"])] = split_name
    return split_lookup


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
    }


def _prompt_mean_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    metadata_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for meta, pred_val in zip(metadata_rows, pred.tolist(), strict=True):
        task_id = str(meta["task_id"])
        groups[task_id]["y_true"].append(float(meta["y_true"]))
        groups[task_id]["y_pred"].append(float(pred_val))
    prompt_rows: list[dict[str, Any]] = []
    for task_id in sorted(groups):
        prompt_rows.append(
            {
                "task_id": task_id,
                "y_true": float(np.mean(groups[task_id]["y_true"])),
                "y_pred": float(np.mean(groups[task_id]["y_pred"])),
                "num_pairs": int(len(groups[task_id]["y_pred"])),
            }
        )
    prompt_true = np.asarray([row["y_true"] for row in prompt_rows], dtype=np.float32)
    prompt_pred = np.asarray([row["y_pred"] for row in prompt_rows], dtype=np.float32)
    return _reg_metrics(prompt_true, prompt_pred), prompt_rows


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
            run_dir = _normalize_run_dir(str(index_row.get("run_dir", "")))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train weak-only prompt/single-rollout hidden regressors, select on weak validation, and report clean transfer."
    )
    parser.add_argument("--weak_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--weak_labels_path", type=Path, required=True)
    parser.add_argument("--weak_prompt_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_rollout_hidden_paths", nargs="+", type=Path)
    parser.add_argument("--weak_rollout_index_paths", nargs="+", type=Path)
    parser.add_argument("--clean_labels_path", type=Path, required=True)
    parser.add_argument("--clean_prompt_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--clean_prompt_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--clean_rollout_hidden_paths", nargs="+", type=Path)
    parser.add_argument("--clean_rollout_index_paths", nargs="+", type=Path)
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
    parser.add_argument("--single_rollout_strategy", choices=["first"], default="first")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--alphas", nargs="+", type=float, default=[100.0, 300.0, 1000.0, 3000.0, 10000.0])
    return parser.parse_args()


def _build_prompt_scalar_lookup(
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


def _load_prompt_hidden_lookup(
    hidden_paths: list[Path],
    index_paths: list[Path],
    *,
    layer_index: int,
) -> dict[str, np.ndarray]:
    if len(hidden_paths) != len(index_paths):
        raise ValueError("Prompt hidden/index path counts must match.")
    lookup: dict[str, np.ndarray] = {}
    for hidden_path, index_path in zip(hidden_paths, index_paths):
        rows = load_hidden_rows(
            hidden_path.expanduser().resolve(),
            index_path=index_path.expanduser().resolve(),
            dataset_name="dapo_math_17k",
            default_component_name="hidden",
        )
        for row in rows:
            layers = row["components"]["hidden"]
            if layer_index >= len(layers):
                raise ValueError(f"Requested prompt layer index {layer_index}, but only {len(layers)} layers are present.")
            lookup[str(row["task_id"])] = np.asarray(layers[layer_index], dtype=np.float32).reshape(-1)
    return lookup


def _build_rollout_index_lookup(index_paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for index_path in index_paths:
        for row in load_records(index_path.expanduser().resolve()):
            run_dir = _normalize_run_dir(str(row.get("run_dir", "")))
            rollout_row_index = int(row.get("rollout_row_index", row.get("sample_index", -1)))
            if rollout_row_index < 0:
                continue
            lookup[(run_dir, rollout_row_index)] = row
    return lookup


def _extract_rollout_scalar_vec(
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


def _fit_prompt_hidden_pca(
    weak_prompt_lookup: dict[str, np.ndarray],
    split_lookup: dict[str, str],
    pca_dim: int,
) -> PCA | None:
    if pca_dim <= 0:
        return None
    train_vectors = [np.asarray(vec, dtype=np.float32).reshape(-1) for task_id, vec in weak_prompt_lookup.items() if split_lookup.get(task_id) == "train"]
    if not train_vectors:
        raise ValueError("No weak-train prompt vectors available for PCA fit.")
    x_train = np.stack(train_vectors, axis=0)
    effective_dim = min(int(pca_dim), int(x_train.shape[0]), int(x_train.shape[1]))
    if effective_dim <= 0:
        raise ValueError(f"Invalid effective PCA dim computed from requested={pca_dim}, shape={x_train.shape}.")
    pca = PCA(n_components=effective_dim, svd_solver="randomized", random_state=42)
    pca.fit(x_train)
    return pca


def _apply_prompt_hidden_pca(
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


def _group_weak_rollouts(
    *,
    weak_run_dirs: list[Path],
    split_lookup: dict[str, str],
    weak_labels_by_task: dict[str, dict[str, Any]],
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray] | None,
    rollout_index_lookup: dict[tuple[str, int], dict[str, Any]] | None,
    rollout_scalar_keys: list[str],
    derived_rollout_scalar_keys: list[str],
    extra_rollout_scalar_field_paths: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run_dir in weak_run_dirs:
        rows = load_records(run_dir.expanduser().resolve() / "all_experiments.jsonl")
        run_dir_str = _normalize_run_dir(str(run_dir.expanduser().resolve()))
        for row_idx, row in enumerate(rows):
            task_id = str(row["task_id"])
            label_row = weak_labels_by_task.get(task_id)
            if label_row is None:
                continue
            group = grouped.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "split": split_lookup.get(task_id, str(row.get("split", "train"))),
                    "y_true": float(label_row["difficulty"]),
                    "rollouts": [],
                },
            )
            rollout_row_index = int(row.get("sample_index", row_idx))
            rollout_hidden = None
            if rollout_hidden_lookup is not None:
                rollout_hidden = rollout_hidden_lookup.get((run_dir_str, rollout_row_index))
                if rollout_hidden is None:
                    continue
            needs_scalar = bool(rollout_scalar_keys or derived_rollout_scalar_keys)
            scalar_source = row
            if rollout_index_lookup is not None:
                scalar_source = rollout_index_lookup.get((run_dir_str, rollout_row_index), row)
            rollout_scalar_vec = (
                _extract_rollout_scalar_vec(
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
                    "rollout_row_index": rollout_row_index,
                    "rollout_hidden_vec": None if rollout_hidden is None else np.asarray(rollout_hidden, dtype=np.float32).reshape(-1),
                    "rollout_scalar_vec": rollout_scalar_vec,
                }
            )
    return [grouped[key] for key in sorted(grouped.keys())]


def _group_clean_rollouts(
    *,
    clean_labels_by_task: dict[str, dict[str, Any]],
    hidden_paths: list[Path],
    index_paths: list[Path],
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray] | None,
    rollout_scalar_keys: list[str],
    derived_rollout_scalar_keys: list[str],
    extra_rollout_scalar_field_paths: list[str],
) -> list[dict[str, Any]]:
    if len(hidden_paths) != len(index_paths):
        raise ValueError("Clean rollout hidden/index path counts must match.")
    grouped: dict[str, dict[str, Any]] = {}
    for hidden_path, index_path in zip(hidden_paths, index_paths):
        rows = load_hidden_rows(
            hidden_path.expanduser().resolve(),
            index_path=index_path.expanduser().resolve(),
            dataset_name="dapo_math_17k",
            default_component_name="prompt_hidden" if rollout_hidden_lookup is None else "think_end_hidden",
        )
        for row in rows:
            index_row = row["index_row"]
            task_id = str(row["task_id"])
            label_row = clean_labels_by_task.get(task_id)
            if label_row is None:
                continue
            split = str(index_row.get("split", ""))
            if split not in {"validation", "test"}:
                continue
            run_dir = _normalize_run_dir(str(index_row.get("run_dir", "")))
            rollout_row_index = int(index_row.get("rollout_row_index", -1))
            rollout_hidden = None
            if rollout_hidden_lookup is not None:
                rollout_hidden = rollout_hidden_lookup.get((run_dir, rollout_row_index))
                if rollout_hidden is None:
                    continue
            needs_scalar = bool(rollout_scalar_keys or derived_rollout_scalar_keys or extra_rollout_scalar_field_paths)
            rollout_scalar_vec = (
                _extract_rollout_scalar_vec(index_row, rollout_scalar_keys, derived_rollout_scalar_keys, extra_rollout_scalar_field_paths)
                if needs_scalar
                else None
            )
            group = grouped.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "split": split,
                    "y_true": float(label_row["difficulty"]),
                    "rollouts": [],
                },
            )
            group["rollouts"].append(
                {
                    "rollout_row_index": rollout_row_index,
                    "rollout_hidden_vec": None if rollout_hidden is None else np.asarray(rollout_hidden, dtype=np.float32).reshape(-1),
                    "rollout_scalar_vec": rollout_scalar_vec,
                }
            )
    return [grouped[key] for key in sorted(grouped.keys())]


def _select_single_rollout(grouped_rows: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in grouped_rows:
        rollouts = group["rollouts"]
        if not rollouts:
            continue
        if strategy == "first":
            chosen = min(rollouts, key=lambda row: int(row["rollout_row_index"]))
        else:
            raise ValueError(f"Unsupported single rollout strategy: {strategy}")
        rows.append(
            {
                "task_id": str(group["task_id"]),
                "split": str(group["split"]),
                "y_true": float(group["y_true"]),
                "rollout_hidden_vec": chosen.get("rollout_hidden_vec"),
                "rollout_scalar_vec": chosen.get("rollout_scalar_vec"),
                "rollout_row_index": int(chosen["rollout_row_index"]),
            }
        )
    return rows


def _build_matrix(
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
        y_rows.append(float(row["y_true"]))
        split_rows.append(str(row["split"]))
        metadata_rows.append(
            {
                "task_id": task_id,
                "split": str(row["split"]),
                "y_true": float(row["y_true"]),
                "rollout_row_index": int(row.get("rollout_row_index", -1)),
            }
        )
    return np.stack(x_rows), np.asarray(y_rows, dtype=np.float32), np.asarray(split_rows), metadata_rows


def _write_predictions(output_path: Path, prompt_rows: list[dict[str, Any]], labels_by_task: dict[str, dict[str, Any]]) -> None:
    rows = []
    for row in prompt_rows:
        label_row = labels_by_task[str(row["task_id"])]
        rows.append(
            {
                "task_id": str(row["task_id"]),
                "user_input": str(label_row.get("user_input", "")),
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
                "num_pairs": 1,
            }
        )
    write_jsonl(output_path, rows)


def _save_diagnostics_plot(path: Path, prompt_rows: list[dict[str, Any]], title: str) -> None:
    y_true = np.asarray([float(row["y_true"]) for row in prompt_rows], dtype=np.float32)
    y_pred = np.asarray([float(row["y_pred"]) for row in prompt_rows], dtype=np.float32)
    order = np.argsort(y_true)
    true_sorted = y_true[order]
    pred_sorted = y_pred[order]
    abs_err_sorted = np.abs(pred_sorted - true_sorted)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hb = axes[0].hexbin(y_true, y_pred, gridsize=32, cmap="viridis", bins="log", mincnt=1)
    axes[0].plot([0, 1], [0, 1], "--", color="tab:red", lw=1.5)
    axes[0].set_xlabel("True difficulty")
    axes[0].set_ylabel("Predicted difficulty")
    axes[0].set_title("GT vs Pred")
    fig.colorbar(hb, ax=axes[0], label="log count")

    axes[1].plot(true_sorted, color="black", lw=2, label="true")
    axes[1].plot(pred_sorted, color="tab:purple", lw=1.5, label="pred")
    axes[1].set_title("Sorted Alignment")
    axes[1].set_xlabel("Prompts sorted by true difficulty")
    axes[1].legend(frameon=False)

    axes[2].plot(abs_err_sorted, color="teal", lw=1.2)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Prompts sorted by true difficulty")
    axes[2].set_ylabel("|pred-true|")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    weak_labels = load_records(args.weak_labels_path.expanduser().resolve())
    clean_labels = load_records(args.clean_labels_path.expanduser().resolve())
    weak_labels_by_task = {str(row["task_id"]): row for row in weak_labels}
    clean_labels_by_task = {str(row["task_id"]): row for row in clean_labels}
    split_lookup = _build_split_lookup(args.weak_prompt_dataset_dir.expanduser().resolve())
    prompt_scalar_lookup = {}
    prompt_scalar_lookup.update(_build_prompt_scalar_lookup(weak_labels_by_task, list(args.prompt_feature_keys)))
    prompt_scalar_lookup.update(_build_prompt_scalar_lookup(clean_labels_by_task, list(args.prompt_feature_keys)))

    weak_prompt_lookup = _load_prompt_hidden_lookup(
        [path.expanduser().resolve() for path in args.weak_prompt_hidden_paths],
        [path.expanduser().resolve() for path in args.weak_prompt_index_paths],
        layer_index=args.prompt_layer_index,
    )
    clean_prompt_lookup = _load_prompt_hidden_lookup(
        [path.expanduser().resolve() for path in args.clean_prompt_hidden_paths],
        [path.expanduser().resolve() for path in args.clean_prompt_index_paths],
        layer_index=args.prompt_layer_index,
    )
    prompt_lookup = {}
    prompt_lookup.update(weak_prompt_lookup)
    prompt_lookup.update(clean_prompt_lookup)
    prompt_hidden_pca = _fit_prompt_hidden_pca(weak_prompt_lookup, split_lookup, int(args.prompt_hidden_pca_dim))
    prompt_lookup = _apply_prompt_hidden_pca(prompt_lookup, prompt_hidden_pca)

    weak_rollout_hidden_lookup = None
    clean_rollout_hidden_lookup = None
    weak_rollout_index_lookup = None
    if args.feature_mode == "prompt_plus_rollout":
        if not args.weak_rollout_hidden_paths or not args.weak_rollout_index_paths:
            raise ValueError("Prompt+rollout mode requires weak rollout hidden/index paths.")
        if not args.clean_rollout_hidden_paths or not args.clean_rollout_index_paths:
            raise ValueError("Prompt+rollout mode requires clean rollout hidden/index paths.")
        weak_rollout_hidden_lookup = build_rollout_hidden_lookup(
            [path.expanduser().resolve() for path in args.weak_rollout_hidden_paths],
            [path.expanduser().resolve() for path in args.weak_rollout_index_paths],
            component_name=args.rollout_component,
            layer_index=0,
            pool_mode=args.rollout_pool_mode,
        )
        clean_rollout_hidden_lookup = build_rollout_hidden_lookup(
            [path.expanduser().resolve() for path in args.clean_rollout_hidden_paths],
            [path.expanduser().resolve() for path in args.clean_rollout_index_paths],
            component_name=args.rollout_component,
            layer_index=0,
            pool_mode=args.rollout_pool_mode,
        )
    if args.weak_rollout_index_paths:
        weak_rollout_index_lookup = _build_rollout_index_lookup(
            [path.expanduser().resolve() for path in args.weak_rollout_index_paths]
        )

    weak_grouped = _group_weak_rollouts(
        weak_run_dirs=[path.expanduser().resolve() for path in args.weak_run_dirs],
        split_lookup=split_lookup,
        weak_labels_by_task=weak_labels_by_task,
        rollout_hidden_lookup=weak_rollout_hidden_lookup,
        rollout_index_lookup=weak_rollout_index_lookup,
        rollout_scalar_keys=list(args.rollout_scalar_keys),
        derived_rollout_scalar_keys=list(args.derived_rollout_scalar_keys),
        extra_rollout_scalar_field_paths=list(args.extra_rollout_scalar_field_paths),
    )
    clean_grouped = _group_clean_rollouts(
        clean_labels_by_task=clean_labels_by_task,
        hidden_paths=[path.expanduser().resolve() for path in (args.clean_rollout_hidden_paths or args.clean_prompt_hidden_paths)],
        index_paths=[path.expanduser().resolve() for path in (args.clean_rollout_index_paths or args.clean_prompt_index_paths)],
        rollout_hidden_lookup=clean_rollout_hidden_lookup,
        rollout_scalar_keys=list(args.rollout_scalar_keys),
        derived_rollout_scalar_keys=list(args.derived_rollout_scalar_keys),
        extra_rollout_scalar_field_paths=list(args.extra_rollout_scalar_field_paths),
    )

    weak_rows = _select_single_rollout(weak_grouped, args.single_rollout_strategy)
    clean_rows = _select_single_rollout(clean_grouped, args.single_rollout_strategy)

    weak_X, weak_y, weak_splits, weak_meta = _build_matrix(
        weak_rows, prompt_lookup, prompt_scalar_lookup, feature_mode=args.feature_mode
    )
    clean_X, clean_y, clean_splits, clean_meta = _build_matrix(
        clean_rows, prompt_lookup, prompt_scalar_lookup, feature_mode=args.feature_mode
    )

    weak_train_mask = weak_splits == "train"
    weak_val_mask = weak_splits == "validation"
    clean_val_mask = clean_splits == "validation"
    clean_test_mask = clean_splits == "test"

    X_train, y_train = weak_X[weak_train_mask], weak_y[weak_train_mask]
    X_weak_val, y_weak_val = weak_X[weak_val_mask], weak_y[weak_val_mask]
    X_clean_val, y_clean_val = clean_X[clean_val_mask], clean_y[clean_val_mask]
    X_test, y_test = clean_X[clean_test_mask], clean_y[clean_test_mask]
    weak_val_meta = [weak_meta[idx] for idx in np.where(weak_val_mask)[0]]
    clean_val_meta = [clean_meta[idx] for idx in np.where(clean_val_mask)[0]]
    test_meta = [clean_meta[idx] for idx in np.where(clean_test_mask)[0]]

    best_bundle: dict[str, Any] | None = None
    best_weak_val_r2 = -1e18
    for alpha in args.alphas:
        name = f"ridge_a{alpha:g}"
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=alpha, random_state=args.random_seed)),
            ]
        )
        model.fit(X_train, y_train)
        weak_val_pred = np.clip(np.asarray(model.predict(X_weak_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
        weak_val_row_metrics = _reg_metrics(y_weak_val, weak_val_pred)
        weak_val_prompt_metrics, weak_val_prompt_rows = _prompt_mean_metrics(y_weak_val, weak_val_pred, weak_val_meta)
        clean_val_pred = np.clip(np.asarray(model.predict(X_clean_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
        clean_val_row_metrics = _reg_metrics(y_clean_val, clean_val_pred)
        clean_val_prompt_metrics, clean_val_prompt_rows = _prompt_mean_metrics(y_clean_val, clean_val_pred, clean_val_meta)
        result = {
            "name": name,
            "weak_val_row_metrics": weak_val_row_metrics,
            "weak_val_prompt_mean_metrics": weak_val_prompt_metrics,
            "clean_val_row_metrics": clean_val_row_metrics,
            "clean_val_prompt_mean_metrics": clean_val_prompt_metrics,
        }
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        if weak_val_prompt_metrics["r2"] > best_weak_val_r2:
            test_pred = np.clip(np.asarray(model.predict(X_test), dtype=np.float32).reshape(-1), 0.0, 1.0)
            test_row_metrics = _reg_metrics(y_test, test_pred)
            test_prompt_metrics, prompt_rows = _prompt_mean_metrics(y_test, test_pred, test_meta)
            best_weak_val_r2 = weak_val_prompt_metrics["r2"]
            best_bundle = {
                "name": name,
                "estimator": model,
                "weak_val_row_metrics": weak_val_row_metrics,
                "weak_val_prompt_mean_metrics": weak_val_prompt_metrics,
                "weak_val_prompt_rows": weak_val_prompt_rows,
                "clean_val_row_metrics": clean_val_row_metrics,
                "clean_val_prompt_mean_metrics": clean_val_prompt_metrics,
                "clean_val_prompt_rows": clean_val_prompt_rows,
                "test_row_metrics": test_row_metrics,
                "test_prompt_mean_metrics": test_prompt_metrics,
                "prompt_rows": prompt_rows,
                "feature_dim": int(X_train.shape[1]),
                "num_train_rows": int(X_train.shape[0]),
                "num_weak_val_rows": int(X_weak_val.shape[0]),
                "num_clean_val_rows": int(X_clean_val.shape[0]),
                "num_test_rows": int(X_test.shape[0]),
            }

    assert best_bundle is not None
    estimator_pipeline = best_bundle["estimator"]
    estimator_step = estimator_pipeline.named_steps.get("model", estimator_pipeline)
    estimator_config = {
        "prediction_target": "difficulty",
        "prompt": {
            "hidden_layer_index": int(args.prompt_layer_index),
            "hidden_projection": {
                "type": None if prompt_hidden_pca is None else "pca",
                "input_dim": None if prompt_hidden_pca is None else int(prompt_hidden_pca.n_features_in_),
                "output_dim": None if prompt_hidden_pca is None else int(prompt_hidden_pca.n_components_),
            },
            "prompt_scalar_keys": list(args.prompt_feature_keys),
        },
        "trajectory": {
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
        "bundle_type": "single_rollout_difficulty_classifier",
        "bundle_version": 1,
        "config": estimator_config,
        "feature_mode": args.feature_mode,
        "single_rollout_strategy": args.single_rollout_strategy,
        "rollout_component": args.rollout_component if args.feature_mode == "prompt_plus_rollout" else None,
        "rollout_pool_mode": args.rollout_pool_mode if args.feature_mode == "prompt_plus_rollout" else None,
        "estimator": best_bundle["estimator"],
        "prompt_hidden_pca": prompt_hidden_pca,
    }
    joblib.dump(bundle, args.output_dir / "model.joblib")
    (args.output_dir / "estimator_config.json").write_text(json.dumps(estimator_config, indent=2), encoding="utf-8")
    _write_predictions(args.output_dir / "predictions_weakval.jsonl", best_bundle["weak_val_prompt_rows"], weak_labels_by_task)
    _save_diagnostics_plot(
        args.output_dir / "prediction_diagnostics_weakval.png",
        best_bundle["weak_val_prompt_rows"],
        f"Weak Val: {best_bundle['name']}",
    )
    _write_predictions(args.output_dir / "predictions_cleanval.jsonl", best_bundle["clean_val_prompt_rows"], clean_labels_by_task)
    _save_diagnostics_plot(
        args.output_dir / "prediction_diagnostics_cleanval.png",
        best_bundle["clean_val_prompt_rows"],
        f"Clean Val: {best_bundle['name']}",
    )
    _write_predictions(args.output_dir / "predictions_test.jsonl", best_bundle["prompt_rows"], clean_labels_by_task)
    _save_diagnostics_plot(
        args.output_dir / "prediction_diagnostics_test.png",
        best_bundle["prompt_rows"],
        f"Test: {best_bundle['name']}",
    )
    summary = {
        "setting": "weak_only_single_rollout_hidden",
        "feature_mode": args.feature_mode,
        "prompt_layer_index": int(args.prompt_layer_index),
        "prompt_hidden_pca_dim": int(args.prompt_hidden_pca_dim),
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
        "num_clean_val_rows": best_bundle["num_clean_val_rows"],
        "num_test_rows": best_bundle["num_test_rows"],
        "num_test_prompts": int(len(best_bundle["prompt_rows"])),
        "weak_val_row_metrics": best_bundle["weak_val_row_metrics"],
        "weak_val_prompt_mean_metrics": best_bundle["weak_val_prompt_mean_metrics"],
        "clean_val_row_metrics": best_bundle["clean_val_row_metrics"],
        "clean_val_prompt_mean_metrics": best_bundle["clean_val_prompt_mean_metrics"],
        "test_row_metrics": best_bundle["test_row_metrics"],
        "test_prompt_mean_metrics": best_bundle["test_prompt_mean_metrics"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
