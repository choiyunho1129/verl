from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.single_rollout_hidden_utils import (
    build_rollout_hidden_lookup,
    build_rollout_index_lookup,
    load_prompt_hidden_lookup,
    normalize_run_dir,
    rollout_to_correctness,
)
from classifer_training.sweep_spo_base_rowr2_axis import FastPCA, _fit_pca


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "classifer_training/artifacts/probe/"
    "spo_offline_subset0_1_qwen3_4b_base_L19_promptlast10_thinkendlast10_entropy3_dapo"
)
SCALAR_KEYS = [
    "output_mean_token_entropy",
    "reasoning_mean_token_entropy",
    "answer_mean_token_entropy",
]


def _scalar_vec(record: dict[str, Any]) -> np.ndarray:
    rollout_features = record.get("rollout_features")
    feature_map = dict(rollout_features) if isinstance(rollout_features, dict) else {}
    return np.asarray([float(feature_map.get(key, 0.0) or 0.0) for key in SCALAR_KEYS], dtype=np.float32)


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _transform_prompt_lookup(lookup: dict[str, np.ndarray], pca: FastPCA) -> dict[str, np.ndarray]:
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


def _build_training_rows(
    *,
    rollout_index_paths: list[Path],
    rollout_hidden_paths: list[Path],
    rollout_component: str,
    layer_index: int,
) -> list[dict[str, Any]]:
    rollout_index_lookup = build_rollout_index_lookup(rollout_index_paths)
    rollout_hidden_lookup = build_rollout_hidden_lookup(
        rollout_hidden_paths,
        rollout_index_paths,
        component_name=rollout_component,
        layer_index=layer_index,
        pool_mode="mean",
    )

    raw_rows = []
    correctness_by_task: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for key, index_row in sorted(rollout_index_lookup.items()):
        run_dir, rollout_row_index = key
        hidden_vec = rollout_hidden_lookup.get((run_dir, rollout_row_index))
        if hidden_vec is None:
            continue
        task_id = str(index_row.get("task_id", ""))
        correctness = float(rollout_to_correctness(index_row))
        correctness_by_task[task_id].append((normalize_run_dir(run_dir), int(rollout_row_index), correctness))
        raw_rows.append(
            {
                "task_id": task_id,
                "run_dir": normalize_run_dir(run_dir),
                "rollout_row_index": int(rollout_row_index),
                "sample_index": int(index_row.get("sample_index", -1)),
                "rollout_hidden_vec": np.asarray(hidden_vec, dtype=np.float32).reshape(-1),
                "rollout_scalar_vec": _scalar_vec(index_row),
                "rollout_correctness": correctness,
            }
        )

    rows = []
    for row in raw_rows:
        siblings = [
            value
            for run_dir, rollout_row_index, value in correctness_by_task[row["task_id"]]
            if not (run_dir == row["run_dir"] and int(rollout_row_index) == int(row["rollout_row_index"]))
        ]
        if not siblings:
            continue
        updated = dict(row)
        updated["split"] = "train"
        updated["value_true"] = float(np.mean(siblings))
        rows.append(updated)
    return rows


def _matrix(rows: list[dict[str, Any]], prompt_lookup: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_rows, y_rows, meta_rows = [], [], []
    for row in rows:
        prompt_vec = prompt_lookup.get(row["task_id"])
        if prompt_vec is None:
            continue
        pieces = [
            np.asarray(prompt_vec, dtype=np.float32).reshape(-1),
            np.asarray(row["rollout_hidden_vec"], dtype=np.float32).reshape(-1),
            np.asarray(row["rollout_scalar_vec"], dtype=np.float32).reshape(-1),
        ]
        x_rows.append(np.concatenate(pieces, axis=0).astype(np.float32))
        y_rows.append(float(row["value_true"]))
        meta_rows.append(row)
    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.float32), meta_rows


def _split_rows(rows: list[dict[str, Any]], holdout_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if holdout_fraction <= 0:
        return rows, []
    task_ids = sorted({str(row["task_id"]) for row in rows})
    rng = np.random.default_rng(seed)
    shuffled = list(task_ids)
    rng.shuffle(shuffled)
    holdout_count = max(1, int(round(len(shuffled) * holdout_fraction)))
    holdout_tasks = set(shuffled[:holdout_count])
    train_rows = [row for row in rows if str(row["task_id"]) not in holdout_tasks]
    holdout_rows = [row for row in rows if str(row["task_id"]) in holdout_tasks]
    return train_rows, holdout_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train L19 prompt-last10 + think_end_last10 SPO offline value estimator.")
    parser.add_argument("--prompt-hidden-path", type=Path, required=True)
    parser.add_argument("--prompt-index-path", type=Path, required=True)
    parser.add_argument("--rollout-hidden-path", type=Path, required=True)
    parser.add_argument("--rollout-index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-component", default="hidden_last10_mean")
    parser.add_argument("--rollout-component", default="think_end_last10_hidden")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--prompt-pca-dim", type=int, default=32)
    parser.add_argument("--rollout-pca-dim", type=int, default=256)
    parser.add_argument("--holdout-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (output_dir / "model.joblib").exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already has model.joblib. Pass --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_lookup_raw = load_prompt_hidden_lookup(
        [args.prompt_hidden_path.expanduser().resolve()],
        [args.prompt_index_path.expanduser().resolve()],
        layer_index=args.layer_index,
        component_name=args.prompt_component,
    )
    rows_raw = _build_training_rows(
        rollout_index_paths=[args.rollout_index_path.expanduser().resolve()],
        rollout_hidden_paths=[args.rollout_hidden_path.expanduser().resolve()],
        rollout_component=args.rollout_component,
        layer_index=args.layer_index,
    )
    train_rows_raw, holdout_rows_raw = _split_rows(rows_raw, args.holdout_fraction, args.seed)
    if not train_rows_raw:
        raise ValueError("No training rows were built.")

    train_task_ids = {row["task_id"] for row in train_rows_raw}
    prompt_pca = _fit_pca(
        [vec for task_id, vec in prompt_lookup_raw.items() if task_id in train_task_ids],
        args.prompt_pca_dim,
    )
    prompt_lookup = _transform_prompt_lookup(prompt_lookup_raw, prompt_pca)

    rollout_pca = _fit_pca([row["rollout_hidden_vec"] for row in train_rows_raw], args.rollout_pca_dim)
    train_rows = _transform_rows(train_rows_raw, rollout_pca)
    holdout_rows = _transform_rows(holdout_rows_raw, rollout_pca) if holdout_rows_raw else []

    x_train, y_train, train_meta = _matrix(train_rows, prompt_lookup)
    estimator = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=0.01, solver="lsqr"))])
    estimator.fit(x_train, y_train)
    pred_train = np.clip(np.asarray(estimator.predict(x_train), dtype=np.float32), 0.0, 1.0)

    summary: dict[str, Any] = {
        "setting": "spo_offline_subset0_1_l19_promptlast10_thinkendlast10_entropy3",
        "bundle_type": "spo_subset_rowr2_probe",
        "model": "StandardScaler -> Ridge(alpha=0.01, solver='lsqr') -> clip[0,1]",
        "label_source": "math_dapo.compute_score from prepared validation_data unless --keep-source-labels was used",
        "train_target": "other rollout correctness within each prompt",
        "prompt_component": args.prompt_component,
        "prompt_layer_index": int(args.layer_index),
        "prompt_hidden_pca_dim": int(args.prompt_pca_dim),
        "rollout_component": args.rollout_component,
        "rollout_layer_index": int(args.layer_index),
        "rollout_pool_mode": "mean",
        "rollout_hidden_pca_dim": int(args.rollout_pca_dim),
        "rollout_scalar_keys": SCALAR_KEYS,
        "feature_dim": int(x_train.shape[1]),
        "num_train_rows": int(x_train.shape[0]),
        "num_train_prompts": int(len({row["task_id"] for row in train_meta})),
        "train_row_metrics": _reg_metrics(y_train, pred_train),
    }

    if holdout_rows:
        x_holdout, y_holdout, holdout_meta = _matrix(holdout_rows, prompt_lookup)
        pred_holdout = np.clip(np.asarray(estimator.predict(x_holdout), dtype=np.float32), 0.0, 1.0)
        summary["num_holdout_rows"] = int(x_holdout.shape[0])
        summary["num_holdout_prompts"] = int(len({row["task_id"] for row in holdout_meta}))
        summary["holdout_row_metrics"] = _reg_metrics(y_holdout, pred_holdout)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
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
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
