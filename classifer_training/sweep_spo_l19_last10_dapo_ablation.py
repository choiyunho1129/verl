from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import classifer_training.sweep_spo_base_rowr2_axis as axis
from classifer_training.single_rollout_hidden_utils import (
    build_rollout_hidden_lookup,
    build_rollout_index_lookup,
    load_prompt_hidden_lookup,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_INDEX_DIR = (
    ROOT / "classifer_training/artifacts/rollout_index/spo_temp1_subset0to4/Qwen_Qwen3-4B_dapo_score"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "classifer_training/artifacts/probe/"
    "spo_temp1_subset0to4_qwen3_4b_base_rowr2_L19_last10_hidden_entropy3_dapo_ablation"
)
DEFAULT_SCALARS = [
    "output_mean_token_entropy",
    "reasoning_mean_token_entropy",
    "answer_mean_token_entropy",
]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _format_fraction(value: float) -> str:
    if np.isclose(value, 1.0):
        return "full"
    if np.isclose(value, 0.5):
        return "half"
    if np.isclose(value, 0.25):
        return "quarter"
    if np.isclose(value, 0.125):
        return "eighth"
    return f"frac{value:g}".replace(".", "p")


def _label_bucket(value: float) -> float:
    if np.isclose(value, 0.0):
        return 0.0
    if np.isclose(value, 0.5):
        return 0.5
    if np.isclose(value, 1.0):
        return 1.0
    return float(round(value, 6))


def _load_rows(*, rollout_index_dir: Path, rollout_component: str, layer: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, str]]:
    axis.ROLLOUT_INDEX_DIR = rollout_index_dir
    rollout_index_paths = sorted(rollout_index_dir.glob("rollout_index.shard*.jsonl"))
    if not rollout_index_paths:
        single_path = rollout_index_dir / "rollout_index.jsonl"
        if single_path.exists():
            rollout_index_paths = [single_path]
    if not rollout_index_paths:
        raise FileNotFoundError(f"No rollout index JSONL files under {rollout_index_dir}")

    rollout_hidden_paths = sorted(axis.ROLLOUT_HIDDEN_DIR.glob("rollout_hidden_states.shard*.pt"))
    if len(rollout_hidden_paths) != len(rollout_index_paths):
        raise FileNotFoundError(
            f"Rollout hidden/index shard count mismatch: {len(rollout_hidden_paths)} hidden, "
            f"{len(rollout_index_paths)} index."
        )

    print(json.dumps({"event": "load_index_start", "rollout_index_dir": str(rollout_index_dir)}), flush=True)
    rollout_index_lookup = build_rollout_index_lookup(rollout_index_paths)
    print(json.dumps({"event": "load_index_done", "num_rows": len(rollout_index_lookup)}), flush=True)

    prompt_hidden_paths, prompt_index_paths = axis._prompt_paths()
    print(json.dumps({"event": "load_prompt_start", "component": "hidden_last10_mean", "layer": layer}), flush=True)
    prompt_lookup = load_prompt_hidden_lookup(
        prompt_hidden_paths,
        prompt_index_paths,
        layer_index=layer,
        component_name="hidden_last10_mean",
    )
    print(json.dumps({"event": "load_prompt_done", "num_rows": len(prompt_lookup)}), flush=True)

    print(json.dumps({"event": "load_rollout_hidden_start", "component": rollout_component, "layer": layer}), flush=True)
    rollout_hidden_lookup = build_rollout_hidden_lookup(
        rollout_hidden_paths,
        rollout_index_paths,
        component_name=rollout_component,
        layer_index=layer,
        pool_mode="mean",
    )
    print(json.dumps({"event": "load_rollout_hidden_done", "num_rows": len(rollout_hidden_lookup)}), flush=True)

    rows = axis._build_rows(
        rollout_component=rollout_component,
        rollout_index_lookup=rollout_index_lookup,
        rollout_hidden_lookup=rollout_hidden_lookup,
    )
    print(json.dumps({"event": "build_rows_done", "num_rows": len(rows)}), flush=True)
    return prompt_lookup, rows, axis._load_prompt_text_by_task()


def _train_prompt_groups(rows: list[dict[str, Any]]) -> dict[float, list[str]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_task[str(row["task_id"])].append(row)

    groups: dict[float, list[str]] = defaultdict(list)
    for task_id, task_rows in by_task.items():
        prompt_label = float(np.mean([float(row["rollout_correctness"]) for row in task_rows]))
        groups[_label_bucket(prompt_label)].append(task_id)
    for task_ids in groups.values():
        task_ids.sort()
    return dict(groups)


def _rows_for_tasks(rows: list[dict[str, Any]], selected_train_tasks: set[str]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["split"] == "test" or (row["split"] == "train" and str(row["task_id"]) in selected_train_tasks)
    ]


def _sample(values: list[str], count: int, rng: np.random.Generator) -> list[str]:
    if count > len(values):
        raise ValueError(f"Requested {count} examples from a bucket with only {len(values)} examples.")
    if count == len(values):
        return list(values)
    indices = rng.choice(len(values), size=count, replace=False)
    return [values[int(idx)] for idx in sorted(indices.tolist())]


def _count_ablation_tasks(groups: dict[float, list[str]], fraction: float, rng: np.random.Generator) -> tuple[set[str], dict[str, Any]]:
    selected: list[str] = []
    counts: dict[str, int] = {}
    for label_value, task_ids in sorted(groups.items()):
        count = len(task_ids) if np.isclose(fraction, 1.0) else int(round(len(task_ids) * fraction))
        count = max(count, 1) if fraction > 0 else 0
        sampled = _sample(task_ids, count, rng)
        selected.extend(sampled)
        counts[str(label_value)] = len(sampled)
    return set(selected), {
        "sampling": "stratified_by_train_prompt_label",
        "fraction": float(fraction),
        "prompt_label_counts": counts,
    }


def _natural_fixed_size_tasks(groups: dict[float, list[str]], total_prompts: int, rng: np.random.Generator) -> tuple[set[str], dict[str, Any]]:
    total_available = sum(len(task_ids) for task_ids in groups.values())
    raw_counts = {label: total_prompts * len(task_ids) / total_available for label, task_ids in groups.items()}
    counts = {label: int(np.floor(value)) for label, value in raw_counts.items()}
    remaining = total_prompts - sum(counts.values())
    for label, _ in sorted(raw_counts.items(), key=lambda item: item[1] - np.floor(item[1]), reverse=True):
        if remaining <= 0:
            break
        counts[label] += 1
        remaining -= 1

    selected: list[str] = []
    for label, count in sorted(counts.items()):
        sampled = _sample(groups[label], int(count), rng)
        selected.extend(sampled)
    return set(selected), {
        "sampling": "natural_fixed_prompt_count",
        "total_prompts": int(total_prompts),
        "prompt_label_counts": {str(label): int(count) for label, count in sorted(counts.items())},
    }


def _balance_ablation_tasks(
    groups: dict[float, list[str]],
    *,
    hard_to_mid_ratio: float,
    total_prompts: int,
    rng: np.random.Generator,
) -> tuple[set[str], dict[str, Any]]:
    mid_count = int(round(total_prompts / (1.0 + hard_to_mid_ratio)))
    hard_count = total_prompts - mid_count
    zero_count = hard_count // 2
    one_count = hard_count - zero_count
    counts = {0.0: zero_count, 0.5: mid_count, 1.0: one_count}

    selected: list[str] = []
    for label, count in sorted(counts.items()):
        sampled = _sample(groups[label], int(count), rng)
        selected.extend(sampled)

    return set(selected), {
        "sampling": "fixed_prompt_count_balanced_hard_labels",
        "total_prompts": int(total_prompts),
        "hard_to_mid_ratio": float(hard_to_mid_ratio),
        "prompt_label_counts": {str(label): int(count) for label, count in sorted(counts.items())},
    }


def _train_one(
    *,
    name: str,
    ablation: dict[str, Any],
    selected_tasks: set[str],
    all_rows: list[dict[str, Any]],
    prompt_lookup: dict[str, np.ndarray],
    prompt_text_by_task: dict[str, str],
    output_dir: Path,
    rollout_component: str,
    layer: int,
) -> dict[str, Any]:
    axis.BASE_OUTPUT = output_dir
    rows = _rows_for_tasks(all_rows, selected_tasks)
    train_label_counts = Counter(_label_bucket(float(row["value_true"])) for row in rows if row["split"] == "train")
    print(
        json.dumps(
            {
                "event": "ablation_start",
                "name": name,
                "num_selected_train_prompts": len(selected_tasks),
                "train_label_counts": {str(k): int(v) for k, v in sorted(train_label_counts.items())},
            }
        ),
        flush=True,
    )
    summary = axis._fit_one(
        name=name,
        prompt_lookup_raw=prompt_lookup,
        prompt_slug=axis.PROMPT_SLUG,
        prompt_component="hidden_last10_mean",
        prompt_layer=layer,
        rollout_component=rollout_component,
        rollout_layer=layer,
        prompt_pca_dim=32,
        rollout_pca_dim=256,
        rows_raw=rows,
        prompt_projection_cache={},
        rollout_projection_cache={},
        prompt_text_by_task=prompt_text_by_task,
    )
    summary["ablation"] = {
        **ablation,
        "num_selected_train_prompts": int(len(selected_tasks)),
        "train_value_true_counts": {str(k): int(v) for k, v in sorted(train_label_counts.items())},
    }
    condition_dir = output_dir / name
    (condition_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "ablation_done",
                "name": name,
                "row_r2": summary["test_row_metrics"]["r2"],
                "prompt_mean_r2": summary["test_prompt_mean_metrics"]["r2"],
                "num_train_rows": summary["num_train_rows"],
            }
        ),
        flush=True,
    )
    return summary


def _write_report(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    summaries = sorted(summaries, key=lambda row: row["name"])
    (output_dir / "ablation_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (output_dir / "ablation_summary.md").open("w", encoding="utf-8") as f:
        f.write("| name | type | train rows | train prompts | prompt label counts | row R2 | prompt R2 | row MAE | row RMSE |\n")
        f.write("|---|---|---:|---:|---|---:|---:|---:|---:|\n")
        for row in summaries:
            ablation = row["ablation"]
            counts = ablation.get("prompt_label_counts") or ablation.get("train_value_true_counts") or {}
            f.write(
                f"| {row['name']} | {ablation.get('type', ablation.get('sampling', ''))} | "
                f"{row['num_train_rows']} | {ablation['num_selected_train_prompts']} | "
                f"`{json.dumps(counts, sort_keys=True)}` | "
                f"{row['test_row_metrics']['r2']:.4f} | {row['test_prompt_mean_metrics']['r2']:.4f} | "
                f"{row['test_row_metrics']['mae']:.4f} | {row['test_row_metrics']['rmse']:.4f} |\n"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L19/last10 DAPO label ablations for SPO value estimator.")
    parser.add_argument("--rollout-index-dir", type=Path, default=DEFAULT_ROLLOUT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--size-fractions", default="1,0.5,0.25,0.125")
    parser.add_argument("--balance-total-prompts", type=int, default=1200)
    parser.add_argument("--balance-hard-to-mid-ratios", default="0.5,1,2,4,8")
    parser.add_argument("--skip-count-ablation", action="store_true")
    parser.add_argument("--skip-balance-ablation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    rollout_index_dir = args.rollout_index_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    axis.BASE_OUTPUT = output_dir
    axis.ROLLOUT_INDEX_DIR = rollout_index_dir
    axis.SCALARS = list(DEFAULT_SCALARS)
    axis.LABEL_SOURCE = f"math_dapo.compute_score generated from {rollout_index_dir}"
    axis.INCLUDE_PROMPT_HIDDEN = True
    axis.INCLUDE_ROLLOUT_HIDDEN = True

    rollout_component = "response_last10_mean_hidden"
    layer = 19
    prompt_lookup, rows, prompt_text_by_task = _load_rows(
        rollout_index_dir=rollout_index_dir,
        rollout_component=rollout_component,
        layer=layer,
    )
    groups = _train_prompt_groups(rows)
    group_counts = {str(label): len(task_ids) for label, task_ids in sorted(groups.items())}
    print(json.dumps({"event": "train_prompt_label_groups", "counts": group_counts}), flush=True)

    summaries: list[dict[str, Any]] = []
    if not args.skip_count_ablation:
        for fraction in _parse_csv_floats(args.size_fractions):
            rng = np.random.default_rng(args.seed + int(round(fraction * 10000)))
            selected_tasks, ablation = _count_ablation_tasks(groups, fraction, rng)
            ablation["type"] = "train_count"
            name = f"count_{_format_fraction(fraction)}"
            summaries.append(
                _train_one(
                    name=name,
                    ablation=ablation,
                    selected_tasks=selected_tasks,
                    all_rows=rows,
                    prompt_lookup=prompt_lookup,
                    prompt_text_by_task=prompt_text_by_task,
                    output_dir=output_dir,
                    rollout_component=rollout_component,
                    layer=layer,
                )
            )
            _write_report(output_dir, summaries)

    if not args.skip_balance_ablation:
        rng = np.random.default_rng(args.seed + 4242)
        selected_tasks, ablation = _natural_fixed_size_tasks(groups, args.balance_total_prompts, rng)
        ablation["type"] = "label_balance"
        summaries.append(
            _train_one(
                name=f"balance_natural_{args.balance_total_prompts}",
                ablation=ablation,
                selected_tasks=selected_tasks,
                all_rows=rows,
                prompt_lookup=prompt_lookup,
                prompt_text_by_task=prompt_text_by_task,
                output_dir=output_dir,
                rollout_component=rollout_component,
                layer=layer,
            )
        )
        _write_report(output_dir, summaries)

        for ratio in _parse_csv_floats(args.balance_hard_to_mid_ratios):
            rng = np.random.default_rng(args.seed + 5000 + int(round(ratio * 1000)))
            selected_tasks, ablation = _balance_ablation_tasks(
                groups,
                hard_to_mid_ratio=ratio,
                total_prompts=args.balance_total_prompts,
                rng=rng,
            )
            ablation["type"] = "label_balance"
            ratio_name = f"{ratio:g}".replace(".", "p")
            summaries.append(
                _train_one(
                    name=f"balance_hardmid{ratio_name}_prompts{args.balance_total_prompts}",
                    ablation=ablation,
                    selected_tasks=selected_tasks,
                    all_rows=rows,
                    prompt_lookup=prompt_lookup,
                    prompt_text_by_task=prompt_text_by_task,
                    output_dir=output_dir,
                    rollout_component=rollout_component,
                    layer=layer,
                )
            )
            _write_report(output_dir, summaries)

    _write_report(output_dir, summaries)
    best = max(summaries, key=lambda row: row["test_row_metrics"]["r2"]) if summaries else None
    print(
        json.dumps(
            {
                "event": "finished",
                "output_dir": str(output_dir),
                "num_runs": len(summaries),
                "best": None if best is None else best["name"],
                "best_row_r2": None if best is None else best["test_row_metrics"]["r2"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
