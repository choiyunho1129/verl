from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from classifer_training.data import load_hidden_rows, load_aligned_examples, load_manifest
from classifer_training.prompt_only_experiments import _hidden_relation_features, _prompt_features
from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.enrich_rollout_index import _single_run_features


def _normalize_run_dir(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2-rollout probe with pooled reasoning/think activations plus existing pair features."
    )
    parser.add_argument("--rollout_manifest", type=Path, required=True)
    parser.add_argument("--prompt_hidden_dir", type=Path, required=True)
    parser.add_argument("--prompt_index_dir", type=Path, required=True)
    parser.add_argument("--rollout_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--rollout_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--train_pairs_per_prompt", type=int, default=4)
    parser.add_argument("--test_pairs_per_prompt", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--max_features", type=float, default=0.7)
    parser.add_argument("--n_jobs", type=int, default=12)
    parser.add_argument(
        "--prompt_mode",
        default="l10_l26",
        choices=("l10_l22", "l10_l23", "l10_l24", "l10_l25", "l10_l26", "l10_l35"),
    )
    parser.add_argument(
        "--rollout_component",
        default="think_end_hidden",
        choices=(
            "reasoning_mean_hidden",
            "think_end_hidden",
            "think_end_last10_hidden",
            "reasoning_hidden",
            "answer_hidden",
            "response_hidden",
        ),
    )
    parser.add_argument("--rollout_layer_index", type=int, default=0)
    parser.add_argument("--rollout_pool", default="mean", choices=("mean", "last", "first", "flatten"))
    return parser.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _iter_hidden_rows(hidden_dir: Path, index_dir: Path):
    combined_hidden = hidden_dir / "hidden_states.pt"
    combined_index = index_dir / "index.jsonl"
    if combined_hidden.exists() and combined_index.exists():
        for row in load_hidden_rows(combined_hidden, index_path=combined_index, dataset_name="dapo_math_17k"):
            yield row
        return
    for split in ("train", "validation", "test"):
        hidden_path = hidden_dir / f"hidden_states_{split}.pt"
        index_path = index_dir / f"index_{split}.jsonl"
        if not hidden_path.exists() or not index_path.exists():
            continue
        for row in load_hidden_rows(hidden_path, index_path=index_path, dataset_name="dapo_math_17k"):
            yield row


def build_prompt_lookup(prompt_hidden_dir: Path, prompt_index_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for row in _iter_hidden_rows(prompt_hidden_dir, prompt_index_dir):
        task_id = str(row["task_id"])
        hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
        index_row = row["index_row"]
        user_input = str(index_row.get("user_input", ""))
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
        }
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
        raise ValueError("rollout hidden/index path counts must match")
    lookup: dict[tuple[str, int], np.ndarray] = {}
    for hidden_path, index_path in zip(hidden_paths, index_paths):
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


def load_grouped_rollouts(
    manifest_path: Path,
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray],
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_entries = load_manifest(manifest_path.expanduser().resolve())
    examples = load_aligned_examples(manifest_entries, strict=True)
    for example in examples:
        index_row = example.index_row
        if "rollout_features" not in index_row:
            rollout_features = dict(index_row.get("rollout_features") or {})
            rollout_features.update(extract_rollout_numeric_features(index_row))
            rollout_features.update(_single_run_features(index_row))
            index_row["rollout_features"] = rollout_features
    feature_keys = sorted(examples[0].index_row["rollout_features"].keys())
    grouped: dict[str, dict[str, Any]] = {}
    skipped_missing_rollout_hidden = 0
    for example in examples:
        task_id = example.task_id
        group = grouped.setdefault(
            task_id,
            {
                "task_id": task_id,
                "split": example.split or "",
                "y_true": float(example.label_row["difficulty"]),
                "rollouts": [],
            },
        )
        key = (_normalize_run_dir(str(example.index_row.get("run_dir", ""))), int(example.index_row.get("rollout_row_index", -1)))
        rollout_hidden_vec = rollout_hidden_lookup.get(key)
        if rollout_hidden_vec is None:
            skipped_missing_rollout_hidden += 1
            continue
        stats_vec = np.asarray(
            [float(example.index_row["rollout_features"].get(feature_key, 0.0)) for feature_key in feature_keys],
            dtype=np.float32,
        )
        group["rollouts"].append(
            {
                "rollout_row_index": int(example.index_row.get("rollout_row_index", len(group["rollouts"]))),
                "run_dir": str(example.index_row.get("run_dir", "")),
                "stats_vec": stats_vec,
                "rollout_hidden_vec": rollout_hidden_vec,
            }
        )
    return [group for _, group in sorted(grouped.items(), key=lambda item: item[0])], feature_keys


def _get_order_score(stats_vec: np.ndarray, feature_keys: list[str]) -> tuple[float, ...]:
    candidates = ["output_length", "reasoning_text_entropy", "answer_tokens", "output_text_entropy"]
    values = []
    for key in candidates:
        if key in feature_keys:
            values.append(float(stats_vec[feature_keys.index(key)]))
    if not values:
        values.append(float(np.sum(stats_vec)))
    return tuple(values)


def build_pair_rows(
    grouped_rows: list[dict[str, Any]],
    feature_keys: list[str],
    train_splits: set[str],
    test_splits: set[str],
    train_pairs_per_prompt: int,
    test_pairs_per_prompt: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    pair_rows: list[dict[str, Any]] = []

    for group in grouped_rows:
        task_id = str(group["task_id"])
        split = str(group["split"])
        if split in train_splits:
            pair_budget = train_pairs_per_prompt
        elif split in test_splits:
            pair_budget = test_pairs_per_prompt
        else:
            continue
        rollouts = group["rollouts"]
        if len(rollouts) < 2 or pair_budget <= 0:
            continue
        all_pairs = list(itertools.combinations(range(len(rollouts)), 2))
        if pair_budget >= len(all_pairs):
            selected_pairs = all_pairs
        else:
            selected_indices = rng.choice(len(all_pairs), size=pair_budget, replace=False)
            selected_pairs = [all_pairs[int(idx)] for idx in np.sort(selected_indices)]

        for left_idx, right_idx in selected_pairs:
            left = rollouts[left_idx]
            right = rollouts[right_idx]
            left_vec = np.asarray(left["stats_vec"], dtype=np.float32)
            right_vec = np.asarray(right["stats_vec"], dtype=np.float32)
            left_reasoning = np.asarray(left["rollout_hidden_vec"], dtype=np.float32)
            right_reasoning = np.asarray(right["rollout_hidden_vec"], dtype=np.float32)
            if _get_order_score(left_vec, feature_keys) > _get_order_score(right_vec, feature_keys):
                left, right = right, left
                left_vec, right_vec = right_vec, left_vec
                left_reasoning, right_reasoning = right_reasoning, left_reasoning

            pair_mean = (left_vec + right_vec) / 2.0
            pair_absdiff = np.abs(left_vec - right_vec)
            pair_min = np.minimum(left_vec, right_vec)
            pair_max = np.maximum(left_vec, right_vec)
            denom = np.maximum(pair_max, 1e-6)
            pair_rel_diff = pair_absdiff / denom
            cosine_num = float(np.dot(left_vec, right_vec))
            cosine_den = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec)) + 1e-8
            cosine = np.asarray([cosine_num / cosine_den], dtype=np.float32)
            l2 = np.asarray([float(np.linalg.norm(left_vec - right_vec))], dtype=np.float32)

            reasoning_mean = (left_reasoning + right_reasoning) / 2.0
            reasoning_absdiff = np.abs(left_reasoning - right_reasoning)
            reasoning_cos_num = float(np.dot(left_reasoning, right_reasoning))
            reasoning_cos_den = float(np.linalg.norm(left_reasoning) * np.linalg.norm(right_reasoning)) + 1e-8
            reasoning_cosine = np.asarray([reasoning_cos_num / reasoning_cos_den], dtype=np.float32)
            reasoning_l2 = np.asarray([float(np.linalg.norm(left_reasoning - right_reasoning))], dtype=np.float32)

            pair_rows.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "y_true": float(group["y_true"]),
                    "left_vec": left_vec,
                    "right_vec": right_vec,
                    "pair_mean": pair_mean,
                    "pair_absdiff": pair_absdiff,
                    "pair_min": pair_min,
                    "pair_max": pair_max,
                    "pair_rel_diff": pair_rel_diff,
                    "cosine": cosine,
                    "l2": l2,
                    "left_reasoning": left_reasoning,
                    "right_reasoning": right_reasoning,
                    "reasoning_mean": reasoning_mean,
                    "reasoning_absdiff": reasoning_absdiff,
                    "reasoning_cosine": reasoning_cosine,
                    "reasoning_l2": reasoning_l2,
                }
            )
    return pair_rows


def build_feature_matrix(
    pair_rows: list[dict[str, Any]],
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    prompt_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    split_rows: list[str] = []
    task_ids: list[str] = []
    for row in pair_rows:
        task_id = str(row["task_id"])
        prompt_features = prompt_lookup.get(task_id)
        if prompt_features is None:
            continue
        prompt_vector = np.asarray(prompt_features[prompt_mode], dtype=np.float32)
        prompt_feats = np.asarray(prompt_features["prompt_feats"], dtype=np.float32)
        prompt_rel = np.asarray(prompt_features["rel_l10"], dtype=np.float32)
        x_rows.append(
            np.concatenate(
                [
                    prompt_vector,
                    prompt_feats,
                    prompt_rel,
                    row["left_vec"],
                    row["right_vec"],
                    row["pair_mean"],
                    row["pair_absdiff"],
                    row["pair_min"],
                    row["pair_max"],
                    row["pair_rel_diff"],
                    row["cosine"],
                    row["l2"],
                    row["left_reasoning"],
                    row["right_reasoning"],
                    row["reasoning_mean"],
                    row["reasoning_absdiff"],
                    row["reasoning_cosine"],
                    row["reasoning_l2"],
                ],
                axis=0,
            )
        )
        y_rows.append(float(row["y_true"]))
        split_rows.append(str(row["split"]))
        task_ids.append(task_id)
    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.float32), split_rows, task_ids


def _prompt_mean_metrics(task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], list[dict[str, float]]]:
    grouped_true: dict[str, list[float]] = defaultdict(list)
    grouped_pred: dict[str, list[float]] = defaultdict(list)
    for task_id, target, prediction in zip(task_ids, y_true.tolist(), y_pred.tolist()):
        grouped_true[task_id].append(float(target))
        grouped_pred[task_id].append(float(prediction))
    sorted_task_ids = sorted(grouped_true)
    prompt_targets = np.asarray([np.mean(grouped_true[task_id]) for task_id in sorted_task_ids], dtype=np.float32)
    prompt_predictions = np.asarray([np.mean(grouped_pred[task_id]) for task_id in sorted_task_ids], dtype=np.float32)
    prompt_rows = [
        {
            "task_id": task_id,
            "y_true": float(target),
            "y_pred": float(prediction),
        }
        for task_id, target, prediction in zip(sorted_task_ids, prompt_targets, prompt_predictions)
    ]
    return metrics(prompt_targets, prompt_predictions), prompt_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    prompt_lookup = build_prompt_lookup(args.prompt_hidden_dir, args.prompt_index_dir)
    rollout_hidden_lookup = build_rollout_hidden_lookup(
        args.rollout_hidden_paths,
        args.rollout_index_paths,
        component_name=args.rollout_component,
        layer_index=args.rollout_layer_index,
        pool_mode=args.rollout_pool,
    )
    grouped_rows, feature_keys = load_grouped_rollouts(args.rollout_manifest, rollout_hidden_lookup)
    pair_rows = build_pair_rows(
        grouped_rows,
        feature_keys,
        set(args.train_splits),
        set(args.test_splits),
        args.train_pairs_per_prompt,
        args.test_pairs_per_prompt,
        args.random_seed,
    )
    x, y, splits, task_ids = build_feature_matrix(pair_rows, prompt_lookup, args.prompt_mode)
    train_mask = np.asarray([split in set(args.train_splits) for split in splits], dtype=bool)
    test_mask = np.asarray([split in set(args.test_splits) for split in splits], dtype=bool)

    model = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        random_state=args.random_seed,
    )
    model.fit(x[train_mask], y[train_mask])
    test_pred = model.predict(x[test_mask]).astype(np.float32)
    row_metrics = metrics(y[test_mask], test_pred)
    prompt_metrics, prompt_rows = _prompt_mean_metrics(
        [task_ids[idx] for idx in np.where(test_mask)[0]],
        y[test_mask],
        test_pred,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "prompt_mode": args.prompt_mode,
        "rollout_component": args.rollout_component,
        "rollout_layer_index": args.rollout_layer_index,
        "rollout_pool": args.rollout_pool,
        "train_pairs_per_prompt": args.train_pairs_per_prompt,
        "test_pairs_per_prompt": args.test_pairs_per_prompt,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "feature_dim": int(x.shape[1]),
        "num_train_rows": int(train_mask.sum()),
        "num_test_rows": int(test_mask.sum()),
        "row_metrics": row_metrics,
        "prompt_metrics": prompt_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(
        args.output_dir / "predictions_test.jsonl",
        [
            {
                "task_id": task_ids[idx],
                "split": splits[idx],
                "y_true": float(y[idx]),
                "y_pred": float(test_pred[test_pos]),
            }
            for test_pos, idx in enumerate(np.where(test_mask)[0])
        ],
    )
    write_jsonl(args.output_dir / "prompt_predictions_test.jsonl", prompt_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
