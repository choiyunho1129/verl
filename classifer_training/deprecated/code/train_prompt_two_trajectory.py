from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover
    CatBoostRegressor = None

from classifer_training.data import load_aligned_examples, load_manifest
from classifer_training.enrich_rollout_index import _single_run_features
from classifer_training.rollout_utils import extract_rollout_numeric_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a prompt-level difficulty model using prompt hidden states plus "
            "exactly two sampled rollout trajectories per prompt."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--target_field", type=str, default="difficulty")
    parser.add_argument("--train_pairs_per_prompt", type=int, default=4)
    parser.add_argument("--test_pairs_per_prompt", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--model", choices=["et", "cat", "xgb", "histgb"], default="et")
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--max_features", type=float, default=0.5)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    return parser.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _get_order_score(stats_vec: np.ndarray, feature_keys: list[str]) -> tuple[float, ...]:
    candidates = [
        "output_length",
        "reasoning_text_entropy",
        "answer_tokens",
        "output_text_entropy",
    ]
    values: list[float] = []
    for key in candidates:
        if key in feature_keys:
            values.append(float(stats_vec[feature_keys.index(key)]))
    if not values:
        values.append(float(np.sum(stats_vec)))
    return tuple(values)


def load_grouped_rollouts(
    manifest_path: Path,
    target_field: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_entries = load_manifest(manifest_path.expanduser().resolve())
    examples = load_aligned_examples(manifest_entries, strict=True)
    if not examples:
        raise ValueError("No examples loaded.")

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
                "y_true": float(example.label_row[target_field]),
                "prompt_hidden": np.asarray(example.components["prompt_hidden"][0], dtype=np.float32).reshape(-1),
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

    rows = [group for _, group in sorted(grouped.items(), key=lambda item: item[0])]
    return rows, feature_keys


def sample_pairs(
    grouped_rows: list[dict[str, Any]],
    feature_keys: list[str],
    train_splits: set[str],
    test_splits: set[str],
    train_pairs_per_prompt: int,
    test_pairs_per_prompt: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(random_seed)
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    split_rows: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    for group in grouped_rows:
        split = str(group["split"])
        if split in train_splits:
            pair_budget = train_pairs_per_prompt
        elif split in test_splits:
            pair_budget = test_pairs_per_prompt
        else:
            continue

        rollouts = group["rollouts"]
        if len(rollouts) < 2:
            continue

        all_pairs = list(itertools.combinations(range(len(rollouts)), 2))
        if pair_budget <= 0:
            continue
        if pair_budget >= len(all_pairs):
            selected_pairs = all_pairs
        else:
            selected_indices = rng.choice(len(all_pairs), size=pair_budget, replace=False)
            selected_pairs = [all_pairs[int(idx)] for idx in np.sort(selected_indices)]

        prompt_hidden = np.asarray(group["prompt_hidden"], dtype=np.float32)
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

            feature_row = np.concatenate(
                [
                    prompt_hidden,
                    left_vec,
                    right_vec,
                    pair_mean,
                    pair_absdiff,
                    pair_min,
                    pair_max,
                    pair_rel_diff,
                    cosine,
                    l2,
                ],
                axis=0,
            ).astype(np.float32)

            X_rows.append(feature_row)
            y_rows.append(float(group["y_true"]))
            split_rows.append(split)
            metadata_rows.append(
                {
                    "task_id": group["task_id"],
                    "split": split,
                    "pair_rollout_row_indices": [int(left["rollout_row_index"]), int(right["rollout_row_index"])],
                    "y_true": float(group["y_true"]),
                }
            )

    return (
        np.stack(X_rows),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(split_rows),
        metadata_rows,
    )


def build_model(args: argparse.Namespace) -> Any:
    if args.model == "et":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
        )
    if args.model == "histgb":
        return HistGradientBoostingRegressor(
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            max_iter=args.n_estimators,
            min_samples_leaf=max(args.min_samples_leaf, 1),
            l2_regularization=1.0,
            random_state=args.random_seed,
        )
    if args.model == "xgb":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed")
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            min_child_weight=max(args.min_samples_leaf, 1),
            n_estimators=args.n_estimators,
            random_state=args.random_seed,
            n_jobs=args.n_jobs if args.n_jobs and args.n_jobs > 0 else 16,
        )
    if args.model == "cat":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed")
        return CatBoostRegressor(
            loss_function="RMSE",
            depth=args.max_depth,
            learning_rate=args.learning_rate,
            iterations=args.n_estimators,
            l2_leaf_reg=args.l2_leaf_reg,
            random_seed=args.random_seed,
            verbose=False,
            thread_count=args.n_jobs if args.n_jobs and args.n_jobs > 0 else 16,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows, feature_keys = load_grouped_rollouts(args.manifest, args.target_field)
    X, y, splits, metadata_rows = sample_pairs(
        grouped_rows=grouped_rows,
        feature_keys=feature_keys,
        train_splits=set(args.train_splits),
        test_splits=set(args.test_splits),
        train_pairs_per_prompt=args.train_pairs_per_prompt,
        test_pairs_per_prompt=args.test_pairs_per_prompt,
        random_seed=args.random_seed,
    )

    train_mask = np.isin(splits, np.asarray(args.train_splits))
    test_mask = np.isin(splits, np.asarray(args.test_splits))
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_meta = [metadata_rows[idx] for idx, keep in enumerate(test_mask.tolist()) if keep]

    model = build_model(args)
    model.fit(X_train, y_train)
    pred_test = np.asarray(model.predict(X_test), dtype=np.float32).reshape(-1)
    pred_test = np.clip(pred_test, 0.0, 1.0)

    prompt_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for meta, pred in zip(test_meta, pred_test.tolist()):
        prompt_groups[str(meta["task_id"])]["y_true"].append(float(meta["y_true"]))
        prompt_groups[str(meta["task_id"])]["y_pred"].append(float(pred))
    prompt_true = np.asarray([float(np.mean(group["y_true"])) for group in prompt_groups.values()], dtype=np.float32)
    prompt_pred = np.asarray([float(np.mean(group["y_pred"])) for group in prompt_groups.values()], dtype=np.float32)

    summary = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "setting": "prompt_plus_two_random_trajectories",
        "target_field": args.target_field,
        "model": args.model,
        "train_splits": args.train_splits,
        "test_splits": args.test_splits,
        "train_pairs_per_prompt": args.train_pairs_per_prompt,
        "test_pairs_per_prompt": args.test_pairs_per_prompt,
        "num_grouped_prompts": int(len(grouped_rows)),
        "num_train_rows": int(X_train.shape[0]),
        "num_test_rows": int(X_test.shape[0]),
        "prompt_hidden_dim": int(grouped_rows[0]["prompt_hidden"].shape[0]) if grouped_rows else 0,
        "rollout_feature_count": int(len(feature_keys)),
        "pair_feature_dim": int(X_train.shape[1]) if len(X_train) else 0,
        "test_metrics": metrics(y_test, pred_test),
        "prompt_mean_test_metrics": metrics(prompt_true, prompt_pred),
        "num_test_prompts": int(len(prompt_groups)),
        "params": {
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "random_seed": args.random_seed,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as f:
        for meta, pred in zip(test_meta, pred_test.tolist()):
            row = dict(meta)
            row["y_pred"] = float(pred)
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
