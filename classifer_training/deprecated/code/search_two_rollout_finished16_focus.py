from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from classifer_training.enrich_rollout_index import _single_run_features
from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.train_prompt_two_trajectory_promptsearch import (
    build_matrix,
    build_pair_rows,
    build_prompt_lookup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused finished16 search for the 2-rollout prompt probe.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--rollout_index_path", type=Path, required=True)
    parser.add_argument("--prompt_hidden_dir", type=Path, required=True)
    parser.add_argument("--prompt_index_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--train_pairs_per_prompt", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--test_pairs_per_prompt", type=int, default=10)
    parser.add_argument("--n_estimators", nargs="+", type=int, default=[1000, 1500, 2000])
    parser.add_argument("--min_samples_leaf", nargs="+", type=int, default=[5, 3, 7])
    parser.add_argument("--max_features", nargs="+", type=float, default=[0.7, 0.6, 0.8])
    parser.add_argument("--prompt_configs", nargs="+", default=[
        "last6:l10_l26",
        "last4:l10_l25",
        "last10:l10_l26",
        "last5:l10_l24",
    ])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=12)
    return parser.parse_args()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_label_buckets(run_root: Path) -> dict[str, dict[str, object]]:
    label_buckets: dict[str, dict[str, object]] = {}
    for run_dir in sorted(run_root.glob("temp0.7_seed*")):
        all_path = run_dir / "all_experiments.jsonl"
        eval_path = run_dir / "evaluation_results.jsonl"
        if not all_path.exists() or not eval_path.exists():
            continue
        experiment_rows = _load_jsonl(all_path)
        evaluation_rows = _load_jsonl(eval_path)
        if not evaluation_rows:
            continue
        correctness = evaluation_rows[-1]["correctness"]
        total = min(len(experiment_rows), len(correctness))
        for idx in range(total):
            row = experiment_rows[idx]
            task_id = str(row["task_id"])
            bucket = label_buckets.setdefault(
                task_id,
                {"split": str(row.get("split", "")), "correct": []},
            )
            bucket["correct"].append(int(correctness[idx]))
    return label_buckets


def _load_grouped_rows(index_path: Path, label_buckets: dict[str, dict[str, object]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _load_jsonl(index_path)

    sample_features: dict[str, float] | None = None
    for row in rows[:5]:
        feats = dict(row.get("rollout_features") or {})
        if any(key in row for key in ("generated_text", "reasoning_content", "answer_content")):
            feats.update(extract_rollout_numeric_features(row))
            feats.update(_single_run_features(row))
        sample_features = feats
        break
    if sample_features is None:
        raise ValueError("No rollout rows found.")
    feature_keys = sorted(sample_features.keys())

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in label_buckets:
            continue
        feats = dict(row.get("rollout_features") or {})
        if any(key in row for key in ("generated_text", "reasoning_content", "answer_content")):
            feats.update(extract_rollout_numeric_features(row))
            feats.update(_single_run_features(row))
        stats_vec = np.asarray([float(feats.get(key, 0.0)) for key in feature_keys], dtype=np.float32)
        label_bucket = label_buckets[task_id]
        group = grouped.setdefault(
            task_id,
            {
                "task_id": task_id,
                "split": str(row.get("split", label_bucket["split"])),
                "y_true": float(1.0 - (sum(label_bucket["correct"]) / len(label_bucket["correct"]))),
                "rollouts": [],
            },
        )
        group["rollouts"].append(
            {
                "rollout_row_index": int(row.get("rollout_row_index", len(group["rollouts"]))),
                "stats_vec": stats_vec,
            }
        )
    return [grouped[key] for key in sorted(grouped.keys())], feature_keys


def _parse_prompt_config(spec: str, repo_root: Path) -> tuple[str, Path, Path, str]:
    pooled_name, prompt_mode = spec.split(":", 1)
    hidden_dir = repo_root / "classifer_training" / "artifacts" / "hidden" / "dapo_math_17k" / f"qwen3_4b_instruct_2507_{pooled_name}mean"
    index_dir = repo_root / "classifer_training" / "artifacts" / "index" / "dapo_math_17k" / f"qwen3_4b_instruct_2507_{pooled_name}mean"
    return spec, hidden_dir, index_dir, prompt_mode


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    repo_root = args.run_root.expanduser().resolve().parents[4]

    seen_names: set[str] = set()
    results: list[dict[str, Any]] = []
    if results_path.exists():
        for row in _load_jsonl(results_path):
            results.append(row)
            seen_names.add(str(row["name"]))

    print(json.dumps({"stage": "load_labels", "run_root": str(args.run_root.expanduser().resolve())}), flush=True)
    label_buckets = _build_label_buckets(args.run_root.expanduser().resolve())
    print(json.dumps({"stage": "labels_ready", "num_tasks": len(label_buckets)}), flush=True)

    print(json.dumps({"stage": "load_grouped_rows", "index_path": str(args.rollout_index_path.expanduser().resolve())}), flush=True)
    grouped_rows, feature_keys = _load_grouped_rows(args.rollout_index_path.expanduser().resolve(), label_buckets)
    print(
        json.dumps(
            {
                "stage": "grouped_rows_ready",
                "num_groups": len(grouped_rows),
                "feature_dim_stats": len(feature_keys),
            }
        ),
        flush=True,
    )

    prompt_lookups: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for spec in args.prompt_configs:
        name, pooled_hidden_dir, pooled_index_dir, _ = _parse_prompt_config(spec, repo_root)
        print(
            json.dumps(
                {
                    "stage": "build_prompt_lookup",
                    "prompt_config": name,
                    "pooled_hidden_dir": str(pooled_hidden_dir),
                }
            ),
            flush=True,
        )
        prompt_lookups[name] = build_prompt_lookup(
            args.prompt_hidden_dir.expanduser().resolve(),
            args.prompt_index_dir.expanduser().resolve(),
            pooled_hidden_dir.expanduser().resolve(),
            pooled_index_dir.expanduser().resolve(),
        )
        print(
            json.dumps(
                {
                    "stage": "prompt_lookup_ready",
                    "prompt_config": name,
                    "num_prompts": len(prompt_lookups[name]),
                }
            ),
            flush=True,
        )

    train_splits = set(args.train_splits)
    test_splits = set(args.test_splits)

    best: dict[str, Any] | None = max(
        results,
        key=lambda row: row["prompt_mean_test_metrics"]["r2"],
        default=None,
    )

    for pair_budget in args.train_pairs_per_prompt:
        print(json.dumps({"stage": "build_pair_rows", "train_pairs_per_prompt": pair_budget}), flush=True)
        pair_rows = build_pair_rows(
            grouped_rows=grouped_rows,
            feature_keys=feature_keys,
            train_splits=train_splits,
            test_splits=test_splits,
            train_pairs_per_prompt=pair_budget,
            test_pairs_per_prompt=args.test_pairs_per_prompt,
            random_seed=args.random_seed,
        )
        print(
            json.dumps(
                {
                    "stage": "pair_rows_ready",
                    "train_pairs_per_prompt": pair_budget,
                    "num_pair_rows": len(pair_rows),
                }
            ),
            flush=True,
        )
        matrix_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
        for spec in args.prompt_configs:
            config_name, _, _, prompt_mode = _parse_prompt_config(spec, repo_root)
            print(
                json.dumps(
                    {
                        "stage": "build_matrix",
                        "train_pairs_per_prompt": pair_budget,
                        "prompt_config": config_name,
                    }
                ),
                flush=True,
            )
            matrix_cache[config_name] = build_matrix(
                pair_rows=pair_rows,
                prompt_lookup=prompt_lookups[config_name],
                prompt_mode=prompt_mode,
            )
            print(
                json.dumps(
                    {
                        "stage": "matrix_ready",
                        "train_pairs_per_prompt": pair_budget,
                        "prompt_config": config_name,
                        "num_rows": int(matrix_cache[config_name][0].shape[0]),
                        "feature_dim": int(matrix_cache[config_name][0].shape[1]),
                    }
                ),
                flush=True,
            )

        for spec in args.prompt_configs:
            config_name, _, _, prompt_mode = _parse_prompt_config(spec, repo_root)
            X, y, splits, metadata_rows = matrix_cache[config_name]
            train_mask = np.isin(splits, np.asarray(args.train_splits))
            test_mask = np.isin(splits, np.asarray(args.test_splits))
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            test_meta = [metadata_rows[idx] for idx, keep in enumerate(test_mask.tolist()) if keep]

            for n_estimators in args.n_estimators:
                for min_samples_leaf in args.min_samples_leaf:
                    for max_features in args.max_features:
                        name = f"pairs{pair_budget}__{config_name}__et_n{n_estimators}_l{min_samples_leaf}_mf{max_features}"
                        if name in seen_names:
                            continue
                        model = ExtraTreesRegressor(
                            n_estimators=n_estimators,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
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

                        row = {
                            "name": name,
                            "prompt_config": config_name,
                            "prompt_mode": prompt_mode,
                            "train_pairs_per_prompt": pair_budget,
                            "test_pairs_per_prompt": args.test_pairs_per_prompt,
                            "params": {
                                "n_estimators": n_estimators,
                                "min_samples_leaf": min_samples_leaf,
                                "max_features": max_features,
                                "random_seed": args.random_seed,
                            },
                            "num_train_rows": int(X_train.shape[0]),
                            "num_test_rows": int(X_test.shape[0]),
                            "pair_feature_dim": int(X_train.shape[1]),
                            "test_metrics": _metrics(y_test, pred_test),
                            "prompt_mean_test_metrics": _metrics(prompt_true, prompt_pred),
                            "num_test_prompts": int(len(prompt_groups)),
                        }
                        results.append(row)
                        seen_names.add(name)
                        with results_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + "\n")
                        print(
                            json.dumps(
                                {
                                    "name": name,
                                    "row_r2": row["test_metrics"]["r2"],
                                    "prompt_mean_r2": row["prompt_mean_test_metrics"]["r2"],
                                }
                            ),
                            flush=True,
                        )
                        if best is None or row["prompt_mean_test_metrics"]["r2"] > best["prompt_mean_test_metrics"]["r2"]:
                            best = row
                            summary_path.write_text(
                                json.dumps(
                                    {
                                        "setting": "two_rollout_prompt_search_finished16_focus",
                                        "prompt_configs": args.prompt_configs,
                                        "train_pairs_per_prompt_grid": args.train_pairs_per_prompt,
                                        "test_pairs_per_prompt": args.test_pairs_per_prompt,
                                        "best": best,
                                        "num_results": len(results),
                                    },
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )

    results.sort(key=lambda row: row["prompt_mean_test_metrics"]["r2"], reverse=True)
    summary = {
        "setting": "two_rollout_prompt_search_finished16_focus",
        "prompt_configs": args.prompt_configs,
        "train_pairs_per_prompt_grid": args.train_pairs_per_prompt,
        "test_pairs_per_prompt": args.test_pairs_per_prompt,
        "best": results[0] if results else None,
        "top10": results[:10],
        "num_results": len(results),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
