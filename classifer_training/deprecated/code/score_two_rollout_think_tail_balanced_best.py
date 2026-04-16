from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from classifer_training.score_two_rollout_think_cascade_best import (
    _build_target_prompt_lookup,
    _clip,
    _compose_prediction,
    _extract_prompt_text,
    _fit_detector_from_key,
    _fit_logistic,
    _fit_ridge,
    _fit_specialist_from_key,
    _group_target_rollouts,
    _load_chunk_offsets,
    _load_dataset_cache,
    _load_feature_keys,
)
from classifer_training.train_two_rollout_reasoning_probe import (
    build_feature_matrix,
    build_pair_rows,
    build_rollout_hidden_lookup,
)
from classifer_training.utils import write_jsonl


def _soft_gate(prob: np.ndarray, threshold: float, gamma: float) -> np.ndarray:
    return np.clip((prob - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0) ** gamma


def _compose_with_easy(mid: np.ndarray, p10: np.ndarray, t10: float, g10: float, beta10: float) -> np.ndarray:
    w10 = _soft_gate(p10, t10, g10)
    pred = (1.0 - w10) * mid
    pred = pred - beta10 * np.clip(p10 - 0.5, 0.0, 1.0)
    return _clip(pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score full DAPO prompts with the best confirmed tail-balanced 2-rollout think probe.")
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
    tail_balanced_pred = _compose_with_easy(mid_pred, p10, **best["easy_routing"])

    prompt_groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"preds": [], "user_input": ""})
    for task_id, pred_val in zip(task_ids, tail_balanced_pred.tolist()):
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
                "probe": "two_rollout_think_tail_balanced_best",
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
