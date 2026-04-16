from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from classifer_training.score_two_rollout_think_cascade_best import (
    _build_target_prompt_lookup,
    _clip,
    _compose_prediction,
    _fit_detector_from_key,
    _fit_logistic,
    _fit_ridge,
    _fit_specialist_from_key,
    _group_target_rollouts,
    _load_chunk_offsets,
    _load_dataset_cache,
    _load_feature_keys,
)
from classifer_training.score_two_rollout_think_tail_balanced_best import _compose_with_easy
from classifer_training.train_two_rollout_reasoning_probe import (
    build_feature_matrix,
    build_pair_rows,
    build_rollout_hidden_lookup,
)
from classifer_training.train_two_rollout_weak_transfer_text import _reg_metrics
from classifer_training.utils import load_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tail-balanced probe transfer on base no-think sampled runs.")
    parser.add_argument("--repo_root", type=Path, default=Path("/home/jongwonlim/verl/yoonho/verl"))
    parser.add_argument("--target_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--target_last6_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--target_last6_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--target_rollout_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--target_rollout_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--target_rollout_hidden_component", type=str, default="response_hidden")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--test_pairs_per_prompt", type=int, default=10)
    return parser.parse_args()


def _load_prompt_accuracy(run_dirs: list[Path]) -> dict[str, float]:
    prompt_acc: dict[str, float] = {}
    for run_dir in run_dirs:
        eval_path = run_dir / "evaluation_results.jsonl"
        rows = load_records(eval_path)
        if not rows:
            continue
        prompt_accuracy = dict(rows[0].get("prompt_accuracy") or {})
        for task_id, acc in prompt_accuracy.items():
            prompt_acc[str(task_id)] = float(acc)
    return prompt_acc


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()
    run_dirs = [path.expanduser().resolve() for path in args.target_run_dirs]

    best_summary_path = (
        repo
        / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_tail_balanced_search/summary.json"
    )
    best_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
    best = best_summary["best"]
    base_decomp = best["base_decomp"]

    train_dataset_path = (
        repo
        / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache"
        / f"{best['dataset_key']}.npz"
    )
    train_dataset = _load_dataset_cache(train_dataset_path)
    x_trainval = np.concatenate([np.asarray(train_dataset["x_train"]), np.asarray(train_dataset["x_val"])], axis=0)
    y_trainval = np.concatenate([np.asarray(train_dataset["y_train"]), np.asarray(train_dataset["y_val"])], axis=0)

    base_alpha = 3000.0 if "a3000" in base_decomp["base_key"] else 10000.0
    base_model = _fit_ridge(base_alpha, x_trainval, y_trainval)
    hard_model = _fit_specialist_from_key(base_decomp["hard_key"], x_trainval, y_trainval)
    vhard_model = _fit_specialist_from_key(base_decomp["vhard_key"], x_trainval, y_trainval)
    p80_model = _fit_detector_from_key(base_decomp["detectors"]["p80"], x_trainval, y_trainval)
    p90_model = _fit_detector_from_key(base_decomp["detectors"]["p90"], x_trainval, y_trainval)
    p100_model = _fit_detector_from_key(base_decomp["detectors"]["p100"], x_trainval, y_trainval)
    c10 = float(str(best["easy_detector"]).split("_c")[-1])
    p10_model = _fit_logistic(c10, x_trainval, (y_trainval <= 0.1).astype(np.int32))

    feature_keys = _load_feature_keys(
        repo
        / "classifer_training/artifacts/rollout_index/dapo_math_17k/qwen3_4b_instruct_2507_promptonly_finished16/finished16_promptonly_rollout_index_compact.jsonl"
    )
    chunk_offsets = _load_chunk_offsets(repo / "classifer_training/artifacts/datasets/dapo_math_17k_full_nonzh_chunks")
    target_prompt_lookup = _build_target_prompt_lookup(
        [path.expanduser().resolve() for path in args.target_last6_hidden_paths],
        [path.expanduser().resolve() for path in args.target_last6_index_paths],
    )
    rollout_hidden_lookup = build_rollout_hidden_lookup(
        [path.expanduser().resolve() for path in args.target_rollout_hidden_paths],
        [path.expanduser().resolve() for path in args.target_rollout_index_paths],
        component_name=args.target_rollout_hidden_component,
        layer_index=0,
        pool_mode="mean",
    )
    target_grouped_rows = _group_target_rollouts(
        run_dirs,
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
    mid_pred = _compose_prediction(
        base_pred,
        hard_pred,
        vhard_pred,
        p80,
        p90,
        p100,
        **best["hard_routing"],
    )
    p10 = _clip(p10_model.predict_proba(x_target)[:, 1])
    pred = _compose_with_easy(mid_pred, p10, **best["easy_routing"])

    prompt_groups: dict[str, list[float]] = defaultdict(list)
    for task_id, pred_val in zip(task_ids, pred.tolist()):
        prompt_groups[str(task_id)].append(float(pred_val))

    prompt_acc = _load_prompt_accuracy(run_dirs)
    rows = []
    y_true: list[float] = []
    y_pred: list[float] = []
    for task_id in sorted(prompt_groups):
        if task_id not in prompt_acc:
            continue
        pred_mean = float(np.mean(prompt_groups[task_id]))
        true_difficulty = float(1.0 - prompt_acc[task_id])
        rows.append(
            {
                "task_id": task_id,
                "predicted_difficulty": pred_mean,
                "predicted_value": float(1.0 - pred_mean),
                "true_accuracy": float(prompt_acc[task_id]),
                "true_difficulty": true_difficulty,
                "probe": "two_rollout_think_tail_balanced_best",
                "rollout_hidden_component": args.target_rollout_hidden_component,
                "num_pair_predictions": int(len(prompt_groups[task_id])),
            }
        )
        y_true.append(true_difficulty)
        y_pred.append(pred_mean)

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    metrics = _reg_metrics(y_true_arr, y_pred_arr)
    bias = float(np.mean(y_pred_arr - y_true_arr)) if y_true_arr.size else float("nan")
    summary = {
        "output_rows": int(len(rows)),
        "num_target_pair_rows": int(len(task_ids)),
        "rollout_hidden_component": args.target_rollout_hidden_component,
        "train_feature_dim": int(x_trainval.shape[1]),
        "metrics": {
            **metrics,
            "pearson": _pearson(y_true_arr, y_pred_arr),
            "bias": bias,
            "true_mean_difficulty": float(np.mean(y_true_arr)) if y_true_arr.size else float("nan"),
            "pred_mean_difficulty": float(np.mean(y_pred_arr)) if y_pred_arr.size else float("nan"),
        },
        "best_summary_path": str(best_summary_path),
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
