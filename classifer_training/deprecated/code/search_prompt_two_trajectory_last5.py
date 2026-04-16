from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from classifer_training.train_prompt_two_trajectory_promptsearch import (
    build_matrix,
    build_pair_rows,
    load_grouped_rollouts,
    metrics,
    _iter_hidden_rows,
)
from classifer_training.prompt_only_experiments import _hidden_relation_features, _prompt_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search last5-mean prompt representations for the 2-rollout setting."
    )
    parser.add_argument("--rollout_manifest", type=Path, required=True)
    parser.add_argument("--prompt_hidden_dir", type=Path, required=True)
    parser.add_argument("--prompt_index_dir", type=Path, required=True)
    parser.add_argument("--prompt_last5_hidden_dir", type=Path, required=True)
    parser.add_argument("--prompt_last5_index_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--train_pairs_per_prompt", nargs="+", type=int, default=[4, 6, 10])
    parser.add_argument("--test_pairs_per_prompt", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_estimators", nargs="+", type=int, default=[1000, 2000, 3000])
    parser.add_argument("--min_samples_leaf", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--max_features", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--n_jobs", type=int, default=12)
    parser.add_argument(
        "--prompt_modes",
        nargs="+",
        default=[
            "raw_last35",
            "l5_l17",
            "l5_l24",
            "l5_l35",
            "l5_l17_l24",
            "raw_last35_l5_l17",
            "raw_last35_l5_l24",
            "raw_last35_l5_l17_l24",
            "means_pair",
        ],
    )
    return parser.parse_args()


def build_prompt_lookup(
    prompt_hidden_dir: Path,
    prompt_index_dir: Path,
    prompt_last5_hidden_dir: Path,
    prompt_last5_index_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
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
            "last_l17": hidden_layers[17],
            "last_l24": hidden_layers[24],
            "last_l35": hidden_layers[35],
            "last_mean": np.stack(hidden_layers, axis=0).mean(axis=0).astype(np.float32),
        }
    for row in _iter_hidden_rows(prompt_last5_hidden_dir, prompt_last5_index_dir):
        task_id = str(row["task_id"])
        hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
        if task_id not in lookup:
            continue
        lookup[task_id].update(
            {
                "rel_l5": _hidden_relation_features(hidden_layers),
                "l5_l17": hidden_layers[17],
                "l5_l24": hidden_layers[24],
                "l5_l35": hidden_layers[35],
                "l5_mean": np.stack(hidden_layers, axis=0).mean(axis=0).astype(np.float32),
            }
        )
    return lookup


def _prompt_vector(features: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "raw_last35":
        return features["last_l35"]
    if mode == "l5_l17":
        return features["l5_l17"]
    if mode == "l5_l24":
        return features["l5_l24"]
    if mode == "l5_l35":
        return features["l5_l35"]
    if mode == "l5_l17_l24":
        return np.concatenate([features["l5_l17"], features["l5_l24"]], axis=0)
    if mode == "raw_last35_l5_l17":
        return np.concatenate([features["last_l35"], features["l5_l17"]], axis=0)
    if mode == "raw_last35_l5_l24":
        return np.concatenate([features["last_l35"], features["l5_l24"]], axis=0)
    if mode == "raw_last35_l5_l17_l24":
        return np.concatenate([features["last_l35"], features["l5_l17"], features["l5_l24"]], axis=0)
    if mode == "means_pair":
        return np.concatenate([features["last_mean"], features["l5_mean"]], axis=0)
    raise ValueError(mode)


def build_matrix_last5(
    pair_rows: list[dict],
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    prompt_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    split_rows: list[str] = []
    metadata_rows: list[dict] = []

    for row in pair_rows:
        task_id = str(row["task_id"])
        if task_id not in prompt_lookup:
            continue
        prompt_feats = prompt_lookup[task_id]["prompt_feats"]
        rel_parts = [prompt_lookup[task_id]["rel_last"]]
        if "rel_l5" in prompt_lookup[task_id]:
            rel_parts.append(prompt_lookup[task_id]["rel_l5"])
        rel_both = np.concatenate(rel_parts, axis=0)
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

    prompt_lookup = build_prompt_lookup(
        args.prompt_hidden_dir.expanduser().resolve(),
        args.prompt_index_dir.expanduser().resolve(),
        args.prompt_last5_hidden_dir.expanduser().resolve(),
        args.prompt_last5_index_dir.expanduser().resolve(),
    )
    grouped_rows, feature_keys = load_grouped_rollouts(args.rollout_manifest)
    progress_path = args.output_dir / "results.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    best: dict | None = None

    for pair_budget in args.train_pairs_per_prompt:
        pair_rows = build_pair_rows(
            grouped_rows=grouped_rows,
            feature_keys=feature_keys,
            train_splits=set(args.train_splits),
            test_splits=set(args.test_splits),
            train_pairs_per_prompt=pair_budget,
            test_pairs_per_prompt=args.test_pairs_per_prompt,
            random_seed=args.random_seed,
        )
        for prompt_mode in args.prompt_modes:
            X, y, splits, metadata_rows = build_matrix_last5(
                pair_rows=pair_rows,
                prompt_lookup=prompt_lookup,
                prompt_mode=prompt_mode,
            )
            train_mask = np.isin(splits, np.asarray(args.train_splits))
            test_mask = np.isin(splits, np.asarray(args.test_splits))
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            test_meta = [metadata_rows[idx] for idx, keep in enumerate(test_mask.tolist()) if keep]

            for n_estimators in args.n_estimators:
                for min_samples_leaf in args.min_samples_leaf:
                    for max_features in args.max_features:
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
                            "name": f"pairs{pair_budget}__{prompt_mode}__et_n{n_estimators}_l{min_samples_leaf}_mf{max_features}",
                            "prompt_mode": prompt_mode,
                            "train_pairs_per_prompt": pair_budget,
                            "num_train_rows": int(X_train.shape[0]),
                            "num_test_rows": int(X_test.shape[0]),
                            "feature_dim": int(X_train.shape[1]),
                            "test_metrics": metrics(y_test, pred_test),
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
        "setting": "two_rollout_last5_prompt_search",
        "rollout_manifest": str(args.rollout_manifest.expanduser().resolve()),
        "train_pairs_per_prompt_grid": args.train_pairs_per_prompt,
        "test_pairs_per_prompt": args.test_pairs_per_prompt,
        "prompt_modes": args.prompt_modes,
        "best": best,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
