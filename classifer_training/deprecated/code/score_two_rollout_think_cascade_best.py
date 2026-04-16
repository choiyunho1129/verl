from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.data import load_hidden_rows
from classifer_training.enrich_rollout_index import _single_run_features
from classifer_training.prompt_only_experiments import _hidden_relation_features, _prompt_features
from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.train_two_rollout_reasoning_probe import (
    build_feature_matrix,
    build_pair_rows,
    build_rollout_hidden_lookup,
)
from classifer_training.utils import load_records, write_jsonl


def _load_dataset_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "x_train": data["x_train"],
            "y_train": data["y_train"],
            "train_task_ids": data["train_task_ids"].tolist(),
            "x_val": data["x_val"],
            "y_val": data["y_val"],
            "val_task_ids": data["val_task_ids"].tolist(),
            "x_test": data["x_test"],
            "y_test": data["y_test"],
            "test_task_ids": data["test_task_ids"].tolist(),
            "feature_dim": int(data["feature_dim"][0]),
        }


def _load_feature_keys(rollout_index_path: Path) -> list[str]:
    rows = load_records(rollout_index_path)
    if not rows:
        raise ValueError(f"No rollout rows found in {rollout_index_path}")
    return sorted(rows[0]["rollout_features"].keys())


def _iter_hidden_rows_any(hidden_path: Path, index_path: Path) -> list[dict[str, Any]]:
    return load_hidden_rows(hidden_path, index_path=index_path, dataset_name="dapo_math_17k")


def _extract_prompt_text(index_row: dict[str, Any]) -> str:
    for key in ("user_input", "prompt", "question", "problem", "instruction"):
        value = index_row.get(key)
        if value:
            return str(value)
    for key in ("messages", "source_prompt"):
        messages = index_row.get(key)
        if isinstance(messages, list) and messages:
            parts: list[str] = []
            for message in messages:
                if isinstance(message, dict):
                    content = str(message.get("content", "")).strip()
                    if content:
                        parts.append(content)
            if parts:
                return "\n\n".join(parts)
    return ""


def _parse_shard_id(text: str) -> int | None:
    match = re.search(r"_shard(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def _parse_shard_chunk(text: str) -> tuple[int, int] | None:
    match = re.search(r"_shard(\d+)_chunk(\d+)", text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _load_chunk_offsets(chunk_root: Path) -> dict[tuple[int, int], int]:
    offsets: dict[tuple[int, int], int] = {}
    for shard_dir in sorted(chunk_root.glob("shard*")):
        match = re.search(r"shard(\d+)$", shard_dir.name)
        if match is None:
            continue
        shard = int(match.group(1))
        manifest_path = shard_dir / "chunk_manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        running = 0
        for chunk_idx, chunk in enumerate(payload.get("chunks", [])):
            offsets[(shard, chunk_idx)] = running
            running += int(chunk.get("num_rows", 0))
    return offsets


def _build_target_prompt_lookup(last6_hidden_paths: list[Path], last6_index_paths: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    if len(last6_hidden_paths) != len(last6_index_paths):
        raise ValueError("target last6 hidden/index path counts must match")
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for hidden_path, index_path in zip(last6_hidden_paths, last6_index_paths):
        shard_id = _parse_shard_id(str(hidden_path)) or _parse_shard_id(str(index_path))
        for row in _iter_hidden_rows_any(hidden_path.expanduser().resolve(), index_path.expanduser().resolve()):
            raw_task_id = str(row["task_id"])
            task_id = f"shard{shard_id}:{raw_task_id}" if shard_id is not None else raw_task_id
            hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
            index_row = row["index_row"]
            user_input = _extract_prompt_text(index_row)
            input_length = int(index_row.get("input_length", 0))
            lookup[task_id] = {
                "prompt_feats": _prompt_features(user_input, input_length),
                "rel_l10": _hidden_relation_features(hidden_layers),
                "l10_l22": hidden_layers[22],
                "l10_l23": hidden_layers[23],
                "l10_l24": hidden_layers[24],
                "l10_l25": hidden_layers[25],
                "l10_l26": hidden_layers[26],
                "l10_l35": hidden_layers[35],
                "user_input": user_input,
            }
    return lookup


def _group_target_rollouts(
    run_dirs: list[Path],
    feature_keys: list[str],
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray],
    chunk_offsets: dict[tuple[int, int], int],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        rows = load_records(run_dir.expanduser().resolve() / "all_experiments.jsonl")
        parsed = _parse_shard_chunk(str(run_dir))
        sample_count_default = int(rows[0].get("sample_count", 1)) if rows else 1
        for row_idx, row in enumerate(rows):
            task_id = str(row.get("task_id", "")).strip()
            if not task_id and parsed is not None:
                shard, chunk = parsed
                chunk_offset = int(chunk_offsets.get((shard, chunk), 0))
                sample_count = int(row.get("sample_count", sample_count_default) or sample_count_default or 1)
                prompt_index_in_chunk = int(row_idx // max(sample_count, 1))
                task_id = f"shard{shard}:{chunk_offset + prompt_index_in_chunk}"
            run_key = (str(run_dir.expanduser().resolve()), int(row_idx))
            rollout_hidden_vec = rollout_hidden_lookup.get(run_key)
            if rollout_hidden_vec is None:
                continue
            feats: dict[str, float] = {}
            feats.update(extract_rollout_numeric_features(row))
            feats.update(_single_run_features(row))
            stats_vec = np.asarray([float(feats.get(key, 0.0)) for key in feature_keys], dtype=np.float32)
            group = grouped.setdefault(
                task_id,
                {"task_id": task_id, "split": "full", "y_true": 0.0, "rollouts": []},
            )
            group["rollouts"].append(
                {
                    "rollout_row_index": int(row_idx),
                    "run_dir": str(run_dir.expanduser().resolve()),
                    "stats_vec": stats_vec,
                    "rollout_hidden_vec": rollout_hidden_vec,
                }
            )
    return [grouped[key] for key in sorted(grouped.keys())]


def _fit_ridge(alpha: float, x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=42))])
    model.fit(x_train, y_train)
    return model


def _fit_logistic(c_val: float, x_train: np.ndarray, y_bin: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_val,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_bin)
    return model


def _fit_et(n_estimators: int, min_samples_leaf: int, max_features: float, x_train: np.ndarray, y_train: np.ndarray) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=24,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def _clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32).reshape(-1), 0.0, 1.0)


def _soft_gate(prob: np.ndarray, threshold: float, gamma: float) -> np.ndarray:
    return np.clip((prob - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0) ** gamma


def _compose_prediction(
    base: np.ndarray,
    hard: np.ndarray,
    vhard: np.ndarray,
    p80: np.ndarray,
    p90: np.ndarray,
    p100: np.ndarray,
    t80: float,
    t90: float,
    t100: float,
    g80: float,
    g90: float,
    g100: float,
    beta80: float,
    beta90: float,
    beta100: float,
) -> np.ndarray:
    w80 = _soft_gate(p80, t80, g80)
    w90 = np.minimum(w80, _soft_gate(p90, t90, g90))
    w100 = np.minimum(w90, _soft_gate(p100, t100, g100))
    pred = (1.0 - w80) * base + (w80 - w90) * hard + (w90 - w100) * vhard + w100 * 1.0
    pred = pred + beta80 * (p80 - 0.5) + beta90 * (p90 - 0.5) + beta100 * (p100 - 0.5)
    return _clip(pred)


def _fit_specialist_from_key(key: str, x_train: np.ndarray, y_train: np.ndarray):
    import re

    match = re.search(r"_sub([0-9.]+)_", key)
    if match is None:
        raise ValueError(f"Could not parse subset threshold from key: {key}")
    subset_thr = float(match.group(1))
    mask = y_train >= subset_thr
    x_sub = x_train[mask]
    y_sub = y_train[mask]
    if key.startswith("ridge_"):
        alpha = float(key.split("_a")[-1])
        return _fit_ridge(alpha, x_sub, y_sub)
    parts = key.split("_")
    n_estimators = int(parts[2][1:])
    min_samples_leaf = int(parts[3][1:])
    max_features = float(parts[4][2:])
    return _fit_et(n_estimators, min_samples_leaf, max_features, x_sub, y_sub)


def _fit_detector_from_key(key: str, x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    target_name, c_part = key.split("_c")
    c_val = float(c_part)
    if target_name == "p80":
        labels = (y_train >= 0.8).astype(np.int32)
    elif target_name == "p90":
        labels = (y_train >= 0.9).astype(np.int32)
    else:
        labels = (y_train == 1.0).astype(np.int32)
    return _fit_logistic(c_val, x_train, labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score full DAPO prompts with the best confirmed 2-rollout think cascade.")
    parser.add_argument("--repo_root", type=Path, default=Path("/home/jongwonlim/verl/yoonho/verl"))
    parser.add_argument("--target_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--target_last6_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--target_last6_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--target_rollout_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--target_rollout_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--test_pairs_per_prompt", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()

    best_summary_path = (
        repo
        / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_cascade_decomp_search/summary.json"
    )
    best_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
    best = best_summary["best"]

    train_dataset_path = (
        repo
        / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache/think_end_hidden:mean.npz"
    )
    train_dataset = _load_dataset_cache(train_dataset_path)
    x_trainval = np.concatenate([np.asarray(train_dataset["x_train"]), np.asarray(train_dataset["x_val"])], axis=0)
    y_trainval = np.concatenate([np.asarray(train_dataset["y_train"]), np.asarray(train_dataset["y_val"])], axis=0)

    base_alpha = 3000.0 if "a3000" in best["base_key"] else 10000.0
    base_model = _fit_ridge(base_alpha, x_trainval, y_trainval)
    hard_model = _fit_specialist_from_key(best["hard_key"], x_trainval, y_trainval)
    vhard_model = _fit_specialist_from_key(best["vhard_key"], x_trainval, y_trainval)
    p80_model = _fit_detector_from_key(best["detectors"]["p80"], x_trainval, y_trainval)
    p90_model = _fit_detector_from_key(best["detectors"]["p90"], x_trainval, y_trainval)
    p100_model = _fit_detector_from_key(best["detectors"]["p100"], x_trainval, y_trainval)

    feature_keys = _load_feature_keys(
        repo
        / "classifer_training/artifacts/rollout_index/dapo_math_17k/qwen3_4b_instruct_2507_promptonly_finished16/finished16_promptonly_rollout_index_compact.jsonl"
    )
    chunk_offsets = _load_chunk_offsets(
        repo / "classifer_training/artifacts/datasets/dapo_math_17k_full_nonzh_chunks"
    )

    target_prompt_lookup = _build_target_prompt_lookup(
        [path.expanduser().resolve() for path in args.target_last6_hidden_paths],
        [path.expanduser().resolve() for path in args.target_last6_index_paths],
    )
    rollout_hidden_lookup = build_rollout_hidden_lookup(
        [path.expanduser().resolve() for path in args.target_rollout_hidden_paths],
        [path.expanduser().resolve() for path in args.target_rollout_index_paths],
        component_name="think_end_hidden",
        layer_index=0,
        pool_mode="mean",
    )
    target_grouped_rows = _group_target_rollouts(
        [path.expanduser().resolve() for path in args.target_run_dirs],
        feature_keys,
        rollout_hidden_lookup,
        chunk_offsets,
    )
    pair_rows = build_pair_rows(
        target_grouped_rows,
        feature_keys,
        train_splits=set(),
        test_splits={"full"},
        train_pairs_per_prompt=0,
        test_pairs_per_prompt=args.test_pairs_per_prompt,
        random_seed=42,
    )
    x_target, _, _, task_ids = build_feature_matrix(pair_rows, target_prompt_lookup, "l10_l26")

    base_pred = _clip(base_model.predict(x_target))
    hard_pred = _clip(hard_model.predict(x_target))
    vhard_pred = _clip(vhard_model.predict(x_target))
    p80 = _clip(p80_model.predict_proba(x_target)[:, 1])
    p90 = _clip(p90_model.predict_proba(x_target)[:, 1])
    p100 = _clip(p100_model.predict_proba(x_target)[:, 1])
    cascade_pred = _compose_prediction(
        base_pred,
        hard_pred,
        vhard_pred,
        p80,
        p90,
        p100,
        **best["routing"],
    )

    prompt_groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"preds": [], "user_input": ""})
    for task_id, pred_val in zip(task_ids, cascade_pred.tolist()):
        task_id = str(task_id)
        prompt_groups[task_id]["preds"].append(float(pred_val))
        prompt_groups[task_id]["user_input"] = str(target_prompt_lookup.get(task_id, {}).get("user_input", ""))

    rows = []
    for task_id in sorted(prompt_groups):
        pred_mean = float(np.mean(prompt_groups[task_id]["preds"]))
        rows.append(
            {
                "task_id": task_id,
                "user_input": prompt_groups[task_id]["user_input"],
                "predicted_difficulty": pred_mean,
                "predicted_value": float(1.0 - pred_mean),
                "probe": "two_rollout_think_cascade_decomposition_best",
                "num_pair_predictions": int(len(prompt_groups[task_id]["preds"])),
            }
        )

    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    summary = {
        "output_path": str(output_path),
        "num_target_prompts": int(len(rows)),
        "num_target_pair_rows": int(len(task_ids)),
        "train_feature_dim": int(x_trainval.shape[1]),
        "best_summary_path": str(best_summary_path),
        "best_from_val": best,
    }
    (output_path.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
