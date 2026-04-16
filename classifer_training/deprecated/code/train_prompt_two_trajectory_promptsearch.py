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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search improved prompt representations for the 2-rollout setting.")
    parser.add_argument("--rollout_manifest", type=Path, required=True)
    parser.add_argument("--prompt_hidden_dir", type=Path, required=True)
    parser.add_argument("--prompt_index_dir", type=Path, required=True)
    parser.add_argument("--prompt_last10_hidden_dir", type=Path, required=True)
    parser.add_argument("--prompt_last10_index_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--train_pairs_per_prompt", type=int, default=10)
    parser.add_argument("--test_pairs_per_prompt", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--min_samples_leaf", type=int, default=3)
    parser.add_argument("--max_features", type=float, default=0.5)
    parser.add_argument("--n_jobs", type=int, default=12)
    parser.add_argument(
        "--prompt_modes",
        nargs="+",
        default=["raw_last35", "l10_l17", "l10_l17_l24", "mid_pair", "means_pair"],
    )
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
        for row in load_hidden_rows(hidden_path, index_path=index_path, dataset_name="dapo_math_17k"):
            yield row


def build_prompt_lookup(prompt_hidden_dir: Path, prompt_index_dir: Path, prompt_last10_hidden_dir: Path, prompt_last10_index_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for row in _iter_hidden_rows(prompt_hidden_dir, prompt_index_dir):
        task_id = str(row["task_id"])
        hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
        index_row = row["index_row"]
        user_input = str(index_row.get("user_input", ""))
        input_length = int(index_row.get("input_length", 0))
        lookup[task_id] = {
            "prompt_feats": _prompt_features(user_input, input_length),
            "rel_last": _hidden_relation_features(hidden_layers),
            "last_l22": hidden_layers[22],
            "last_l23": hidden_layers[23],
            "last_l17": hidden_layers[17],
            "last_l24": hidden_layers[24],
            "last_l25": hidden_layers[25],
            "last_l26": hidden_layers[26],
            "last_l35": hidden_layers[35],
            "last_mean": np.stack(hidden_layers, axis=0).mean(axis=0).astype(np.float32),
        }
    for row in _iter_hidden_rows(prompt_last10_hidden_dir, prompt_last10_index_dir):
        task_id = str(row["task_id"])
        hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
        if task_id not in lookup:
            continue
        lookup[task_id].update(
            {
                "rel_l10": _hidden_relation_features(hidden_layers),
                "l10_l22": hidden_layers[22],
                "l10_l23": hidden_layers[23],
                "l10_l17": hidden_layers[17],
                "l10_l24": hidden_layers[24],
                "l10_l25": hidden_layers[25],
                "l10_l26": hidden_layers[26],
                "l10_l35": hidden_layers[35],
                "l10_mean": np.stack(hidden_layers, axis=0).mean(axis=0).astype(np.float32),
            }
        )
    return lookup


def load_grouped_rollouts(manifest_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
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
        stats_vec = np.asarray(
            [float(example.index_row["rollout_features"].get(key, 0.0)) for key in feature_keys],
            dtype=np.float32,
        )
        group["rollouts"].append(
            {
                "rollout_row_index": int(example.index_row.get("rollout_row_index", len(group["rollouts"]))),
                "stats_vec": stats_vec,
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


def _prompt_vector(features: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "raw_last22":
        return features["last_l22"]
    if mode == "raw_last23":
        return features["last_l23"]
    if mode == "raw_last17":
        return features["last_l17"]
    if mode == "raw_last24":
        return features["last_l24"]
    if mode == "raw_last25":
        return features["last_l25"]
    if mode == "raw_last26":
        return features["last_l26"]
    if mode == "raw_last35":
        return features["last_l35"]
    if mode == "l10_l22":
        return features["l10_l22"]
    if mode == "l10_l23":
        return features["l10_l23"]
    if mode == "l10_l17":
        return features["l10_l17"]
    if mode == "l10_l24":
        return features["l10_l24"]
    if mode == "l10_l25":
        return features["l10_l25"]
    if mode == "l10_l26":
        return features["l10_l26"]
    if mode == "l10_l35":
        return features["l10_l35"]
    if mode == "l10_l17_l24":
        return np.concatenate([features["l10_l17"], features["l10_l24"]], axis=0)
    if mode == "mid_pair":
        return np.concatenate([features["last_l17"], features["last_l24"], features["l10_l17"], features["l10_l24"]], axis=0)
    if mode == "means_pair":
        return np.concatenate([features["last_mean"], features["l10_mean"]], axis=0)
    raise ValueError(mode)


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
            if _get_order_score(left_vec, feature_keys) > _get_order_score(right_vec, feature_keys):
                left, right = right, left
                left_vec, right_vec = right_vec, left_vec

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

            pair_rows.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "y_true": float(group["y_true"]),
                    "pair_rollout_row_indices": [int(left["rollout_row_index"]), int(right["rollout_row_index"])],
                    "left_vec": left_vec,
                    "right_vec": right_vec,
                    "pair_mean": pair_mean,
                    "pair_absdiff": pair_absdiff,
                    "pair_min": pair_min,
                    "pair_max": pair_max,
                    "pair_rel_diff": pair_rel_diff,
                    "cosine": cosine,
                    "l2": l2,
                }
            )

    return pair_rows


def build_matrix(
    pair_rows: list[dict[str, Any]],
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    prompt_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    split_rows: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    for row in pair_rows:
        task_id = str(row["task_id"])
        if task_id not in prompt_lookup:
            continue
        prompt_feats = prompt_lookup[task_id]["prompt_feats"]
        rel_both = np.concatenate([prompt_lookup[task_id]["rel_last"], prompt_lookup[task_id]["rel_l10"]], axis=0)
        prompt_vec = _prompt_vector(prompt_lookup[task_id], prompt_mode)
        feature_row = np.concatenate(
            [
                prompt_vec,
                prompt_feats,
                rel_both,
                row["left_vec"],
                row["right_vec"],
                row["pair_mean"],
                row["pair_absdiff"],
                row["pair_min"],
                row["pair_max"],
                row["pair_rel_diff"],
                row["cosine"],
                row["l2"],
            ],
            axis=0,
        ).astype(np.float32)
        X_rows.append(feature_row)
        y_rows.append(float(row["y_true"]))
        split_rows.append(str(row["split"]))
        metadata_rows.append(
            {
                "task_id": task_id,
                "split": str(row["split"]),
                "pair_rollout_row_indices": row["pair_rollout_row_indices"],
                "y_true": float(row["y_true"]),
                "prompt_mode": prompt_mode,
            }
        )

    return np.stack(X_rows), np.asarray(y_rows, dtype=np.float32), np.asarray(split_rows), metadata_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    prompt_lookup = build_prompt_lookup(
        args.prompt_hidden_dir.expanduser().resolve(),
        args.prompt_index_dir.expanduser().resolve(),
        args.prompt_last10_hidden_dir.expanduser().resolve(),
        args.prompt_last10_index_dir.expanduser().resolve(),
    )
    grouped_rows, feature_keys = load_grouped_rollouts(args.rollout_manifest)

    prompt_modes = list(args.prompt_modes)
    pair_rows = build_pair_rows(
        grouped_rows=grouped_rows,
        feature_keys=feature_keys,
        train_splits=set(args.train_splits),
        test_splits=set(args.test_splits),
        train_pairs_per_prompt=args.train_pairs_per_prompt,
        test_pairs_per_prompt=args.test_pairs_per_prompt,
        random_seed=args.random_seed,
    )
    results = []

    for prompt_mode in prompt_modes:
        X, y, splits, metadata_rows = build_matrix(
            pair_rows=pair_rows,
            prompt_lookup=prompt_lookup,
            prompt_mode=prompt_mode,
        )
        train_mask = np.isin(splits, np.asarray(args.train_splits))
        test_mask = np.isin(splits, np.asarray(args.test_splits))
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_meta = [metadata_rows[idx] for idx, keep in enumerate(test_mask.tolist()) if keep]

        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
        model.fit(X_train, y_train)
        pred_test = np.clip(np.asarray(model.predict(X_test), dtype=np.float32).reshape(-1), 0.0, 1.0)

        prompt_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for meta, pred in zip(test_meta, pred_test.tolist()):
            prompt_groups[str(meta["task_id"])]["y_true"].append(float(meta["y_true"]))
            prompt_groups[str(meta["task_id"])]["y_pred"].append(float(pred))
        prompt_true = np.asarray([float(np.mean(group["y_true"])) for group in prompt_groups.values()], dtype=np.float32)
        prompt_pred = np.asarray([float(np.mean(group["y_pred"])) for group in prompt_groups.values()], dtype=np.float32)

        results.append(
            {
                "prompt_mode": prompt_mode,
                "num_train_rows": int(X_train.shape[0]),
                "num_test_rows": int(X_test.shape[0]),
                "pair_feature_dim": int(X_train.shape[1]),
                "test_metrics": metrics(y_test, pred_test),
                "prompt_mean_test_metrics": metrics(prompt_true, prompt_pred),
                "num_test_prompts": int(len(prompt_groups)),
            }
        )
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(results[-1]) + "\n")
        print(
            json.dumps(
                {
                    "prompt_mode": results[-1]["prompt_mode"],
                    "row_r2": results[-1]["test_metrics"]["r2"],
                    "prompt_mean_r2": results[-1]["prompt_mean_test_metrics"]["r2"],
                }
            ),
            flush=True,
        )

    results.sort(key=lambda row: row["prompt_mean_test_metrics"]["r2"], reverse=True)
    summary = {
        "setting": "two_rollout_prompt_search",
        "train_pairs_per_prompt": args.train_pairs_per_prompt,
        "test_pairs_per_prompt": args.test_pairs_per_prompt,
        "params": {
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "random_seed": args.random_seed,
        },
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
