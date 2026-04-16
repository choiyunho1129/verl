from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from classifer_training.train_prompt_two_trajectory import (
    build_model,
    load_grouped_rollouts,
    metrics,
    sample_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search prompt-plus-two-trajectory models over multiple pair budgets "
            "and regressor families."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_field", type=str, default="difficulty")
    parser.add_argument("--train_splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--train_pairs_per_prompt", nargs="+", type=int, default=[4, 6, 10, 15])
    parser.add_argument("--test_pairs_per_prompt", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=12)
    parser.add_argument(
        "--include_models",
        nargs="*",
        default=["et", "cat", "xgb", "histgb"],
    )
    return parser.parse_args()


def build_candidates(include_models: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    if "et" in include_models:
        for n_estimators in (1000, 2000, 3000):
            for min_samples_leaf in (1, 3, 5):
                for max_features in (0.3, 0.5, 0.7):
                    candidates.append(
                        {
                            "name": f"et_n{n_estimators}_l{min_samples_leaf}_mf{max_features}",
                            "model": "et",
                            "n_estimators": n_estimators,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                            "max_depth": 6,
                            "learning_rate": 0.03,
                            "l2_leaf_reg": 3.0,
                        }
                    )

    if "cat" in include_models:
        for depth in (6, 8):
            for lr in (0.03, 0.05):
                for iterations in (600, 1000):
                    for l2_leaf_reg in (3.0, 10.0):
                        candidates.append(
                            {
                                "name": f"cat_d{depth}_lr{lr}_i{iterations}_l2{l2_leaf_reg:g}",
                                "model": "cat",
                                "n_estimators": iterations,
                                "min_samples_leaf": 5,
                                "max_features": 0.5,
                                "max_depth": depth,
                                "learning_rate": lr,
                                "l2_leaf_reg": l2_leaf_reg,
                            }
                        )

    if "xgb" in include_models:
        for depth in (4, 6, 8):
            for lr in (0.03, 0.05):
                for n_estimators in (300, 600):
                    for min_samples_leaf in (1, 5):
                        candidates.append(
                            {
                                "name": f"xgb_d{depth}_lr{lr}_n{n_estimators}_mcw{min_samples_leaf}",
                                "model": "xgb",
                                "n_estimators": n_estimators,
                                "min_samples_leaf": min_samples_leaf,
                                "max_features": 0.5,
                                "max_depth": depth,
                                "learning_rate": lr,
                                "l2_leaf_reg": 3.0,
                            }
                        )

    if "histgb" in include_models:
        for depth in (4, 6, 8):
            for lr in (0.03, 0.05):
                for max_iter in (300, 800):
                    candidates.append(
                        {
                            "name": f"histgb_d{depth}_lr{lr}_i{max_iter}",
                            "model": "histgb",
                            "n_estimators": max_iter,
                            "min_samples_leaf": 5,
                            "max_features": 0.5,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "l2_leaf_reg": 3.0,
                        }
                    )

    return candidates


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows, feature_keys = load_grouped_rollouts(args.manifest, args.target_field)
    candidates = build_candidates(set(args.include_models))
    progress_path = args.output_dir / "results.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    best: dict[str, Any] | None = None
    for pair_budget in args.train_pairs_per_prompt:
        X, y, splits, metadata_rows = sample_pairs(
            grouped_rows=grouped_rows,
            feature_keys=feature_keys,
            train_splits=set(args.train_splits),
            test_splits=set(args.test_splits),
            train_pairs_per_prompt=pair_budget,
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

        for candidate in candidates:
            ns = argparse.Namespace(
                model=candidate["model"],
                n_estimators=candidate["n_estimators"],
                min_samples_leaf=candidate["min_samples_leaf"],
                max_features=candidate["max_features"],
                max_depth=candidate["max_depth"],
                learning_rate=candidate["learning_rate"],
                l2_leaf_reg=candidate["l2_leaf_reg"],
                random_seed=args.random_seed,
                n_jobs=args.n_jobs,
            )
            model = build_model(ns)
            model.fit(X_train, y_train)
            pred = np.asarray(model.predict(X_test), dtype=np.float32).reshape(-1)
            pred = np.clip(pred, 0.0, 1.0)

            prompt_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
            for meta, pred_value in zip(test_meta, pred.tolist()):
                prompt_groups[str(meta["task_id"])]["y_true"].append(float(meta["y_true"]))
                prompt_groups[str(meta["task_id"])]["y_pred"].append(float(pred_value))
            prompt_true = np.asarray(
                [float(np.mean(group["y_true"])) for group in prompt_groups.values()],
                dtype=np.float32,
            )
            prompt_pred = np.asarray(
                [float(np.mean(group["y_pred"])) for group in prompt_groups.values()],
                dtype=np.float32,
            )

            row = {
                "name": f"pairs{pair_budget}__{candidate['name']}",
                "train_pairs_per_prompt": pair_budget,
                "num_train_rows": int(X_train.shape[0]),
                "num_test_rows": int(X_test.shape[0]),
                "feature_dim": int(X_train.shape[1]),
                "test_metrics": metrics(y_test, pred),
                "prompt_mean_test_metrics": metrics(prompt_true, prompt_pred),
                "num_test_prompts": int(len(prompt_groups)),
            }
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {
                        "name": row["name"],
                        "row_r2": row["test_metrics"]["r2"],
                        "prompt_mean_r2": row["prompt_mean_test_metrics"]["r2"],
                    }
                ),
                flush=True,
            )
            if best is None or row["prompt_mean_test_metrics"]["r2"] > best["prompt_mean_test_metrics"]["r2"]:
                best = row

    summary = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "target_field": args.target_field,
        "train_splits": args.train_splits,
        "test_splits": args.test_splits,
        "train_pairs_per_prompt_grid": args.train_pairs_per_prompt,
        "test_pairs_per_prompt": args.test_pairs_per_prompt,
        "best": best,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
