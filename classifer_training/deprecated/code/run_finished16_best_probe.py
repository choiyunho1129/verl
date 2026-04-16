from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    workdir = Path("/home/jongwonlim/verl/yoonho/verl")
    run_root = workdir / "classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507"
    run_dirs = sorted(run_root.glob("temp0.7_seed*"))

    label_buckets: dict[str, dict[str, object]] = {}
    for run_dir in run_dirs:
        experiment_rows = _load_jsonl(run_dir / "all_experiments.jsonl")
        evaluation_rows = _load_jsonl(run_dir / "evaluation_results.jsonl")
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

    index_path = (
        workdir
        / "classifer_training/artifacts/rollout_index/dapo_math_17k/qwen3_4b_instruct_2507_promptonly_finished16/finished16_promptonly_rollout_index.jsonl"
    )
    rows = _load_jsonl(index_path)

    sample_features = None
    for row in rows[:5]:
        feats = dict(row.get("rollout_features") or {})
        feats.update(extract_rollout_numeric_features(row))
        feats.update(_single_run_features(row))
        sample_features = feats
        break
    if sample_features is None:
        raise ValueError("No rollout rows found.")
    feature_keys = sorted(sample_features.keys())

    grouped: dict[str, dict] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in label_buckets:
            continue
        feats = dict(row.get("rollout_features") or {})
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

    grouped_rows = [grouped[key] for key in sorted(grouped.keys())]

    prompt_lookup = build_prompt_lookup(
        workdir / "classifer_training/artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507",
        workdir / "classifer_training/artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507",
        workdir / "classifer_training/artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507_last6mean",
        workdir / "classifer_training/artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507_last6mean",
    )

    pair_rows = build_pair_rows(
        grouped_rows=grouped_rows,
        feature_keys=feature_keys,
        train_splits={"train", "validation"},
        test_splits={"test"},
        train_pairs_per_prompt=4,
        test_pairs_per_prompt=10,
        random_seed=42,
    )

    X, y, splits, metadata_rows = build_matrix(pair_rows, prompt_lookup, "l10_l26")
    train_mask = np.isin(splits, np.asarray(["train", "validation"]))
    test_mask = np.isin(splits, np.asarray(["test"]))
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_meta = [metadata_rows[idx] for idx, keep in enumerate(test_mask.tolist()) if keep]

    model = ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=42,
        n_jobs=12,
    )
    model.fit(X_train, y_train)
    pred_test = np.clip(np.asarray(model.predict(X_test), dtype=np.float32).reshape(-1), 0.0, 1.0)

    prompt_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for meta, pred in zip(test_meta, pred_test.tolist()):
        prompt_groups[str(meta["task_id"])]["y_true"].append(float(meta["y_true"]))
        prompt_groups[str(meta["task_id"])]["y_pred"].append(float(pred))
    prompt_true = np.asarray([float(np.mean(group["y_true"])) for group in prompt_groups.values()], dtype=np.float32)
    prompt_pred = np.asarray([float(np.mean(group["y_pred"])) for group in prompt_groups.values()], dtype=np.float32)

    summary = {
        "setting": "two_rollout_prompt_search_finished16_best_probe",
        "prompt_mode": "last6_mean+layer26",
        "train_pairs_per_prompt": 4,
        "test_pairs_per_prompt": 10,
        "params": {
            "n_estimators": 1000,
            "min_samples_leaf": 5,
            "max_features": 0.7,
            "random_seed": 42,
        },
        "num_train_rows": int(X_train.shape[0]),
        "num_test_rows": int(X_test.shape[0]),
        "pair_feature_dim": int(X_train.shape[1]),
        "test_metrics": _metrics(y_test, pred_test),
        "prompt_mean_test_metrics": _metrics(prompt_true, prompt_pred),
        "num_test_prompts": int(len(prompt_groups)),
    }

    outdir = (
        workdir
        / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_random_traj_finished16_best_last6_l26"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (outdir / "predictions_test.jsonl").open("w", encoding="utf-8") as f:
        for meta, pred in zip(test_meta, pred_test.tolist()):
            f.write(
                json.dumps(
                    {
                        "task_id": meta["task_id"],
                        "split": meta["split"],
                        "y_true": meta["y_true"],
                        "y_pred": float(pred),
                        "prompt_mode": "last6_mean+layer26",
                    }
                )
                + "\n"
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
