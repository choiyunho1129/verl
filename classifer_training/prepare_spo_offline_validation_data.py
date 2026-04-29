from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from classifer_training.utils import write_jsonl
from verl.utils.reward_score.math_dapo import compute_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "classifer_training/artifacts/datasets/spo_offline_subset0_1_validation_data"
DEFAULT_SUBSET_DIRS = [
    Path("/home/jongwonlim/verl/yoonho/spo/offline_value_estimation_subset_0"),
    Path("/home/jongwonlim/verl/yoonho/spo/offline_value_estimation_subset_1"),
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def _subset_id_from_path(path: Path, fallback: int) -> int:
    text = str(path)
    marker = "subset_"
    if marker in text:
        tail = text.rsplit(marker, maxsplit=1)[1]
        digits = []
        for char in tail:
            if not char.isdigit():
                break
            digits.append(char)
        if digits:
            return int("".join(digits))
    return int(fallback)


def _dapo_score(output: str, ground_truth: str) -> dict[str, Any]:
    # Match the DAPO relabeling used for the existing L19/last10 probe.
    solution = str(output or "").split("</think>", maxsplit=1)[0]
    return compute_score(solution, str(ground_truth or ""), strict_box_verify=True)


def _normalize_run(
    subset_dir: Path,
    *,
    output_run_dir: Path,
    subset_id: int,
    rescore_with_math_dapo: bool,
    max_prompts: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = subset_dir / "validation_data" / "0.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Expected validation data at {source_path}")

    source_rows = _load_jsonl(source_path)
    prompt_seen_order: list[str] = []
    prompt_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        prompt = str(row.get("input", ""))
        if prompt not in prompt_to_rows:
            prompt_seen_order.append(prompt)
        prompt_to_rows[prompt].append(row)

    if max_prompts is not None:
        keep = set(prompt_seen_order[: int(max_prompts)])
        prompt_seen_order = [prompt for prompt in prompt_seen_order if prompt in keep]
        prompt_to_rows = {prompt: rows for prompt, rows in prompt_to_rows.items() if prompt in keep}

    prompt_records: list[dict[str, Any]] = []
    experiment_rows: list[dict[str, Any]] = []
    evaluation_scores: list[float] = []
    evaluation_correctness: list[int] = []

    for prompt in prompt_seen_order:
        rows = prompt_to_rows[prompt]
        task_id = f"subset{subset_id}_{_prompt_hash(prompt)}"
        ground_truth = str(rows[0].get("gts", ""))
        prompt_records.append(
            {
                "dataset_name": "spo_offline_subset0_1_validation_data",
                "task_id": task_id,
                "split": "train",
                "user_input": prompt,
                "ground_truth": ground_truth,
                "source_subset_id": int(subset_id),
                "source_validation_data": str(source_path),
            }
        )
        for sample_index, source_row in enumerate(rows):
            output = str(source_row.get("output", ""))
            if rescore_with_math_dapo:
                scored = _dapo_score(output, ground_truth)
                score = float(scored["score"])
                correctness = 1 if bool(scored["acc"]) else 0
                pred = scored.get("pred")
                score_source = "math_dapo.compute_score(strict_box_verify=True,before_think)"
            else:
                raw_score = float(source_row.get("score", source_row.get("reward", 0.0)) or 0.0)
                score = 1.0 if raw_score >= 1.0 else -1.0
                correctness = 1 if raw_score >= 1.0 else 0
                pred = source_row.get("pred", "")
                score_source = "source_validation_data"

            experiment_row = {
                "dataset_name": "spo_offline_subset0_1_validation_data",
                "task_id": task_id,
                "split": "train",
                "user_input": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "ground_truth": ground_truth,
                "generated_text": output,
                "reasoning_content": "",
                "answer_content": "",
                "sample_index": int(sample_index),
                "sample_count": int(len(rows)),
                "score": float(score),
                "reward": float(score),
                "correctness": int(correctness),
                "pred": pred,
                "score_source": score_source,
                "legacy_score": source_row.get("score"),
                "legacy_reward": source_row.get("reward"),
                "legacy_pred": source_row.get("pred"),
                "source_subset_id": int(subset_id),
                "source_validation_data": str(source_path),
                "source_row_index": int(len(experiment_rows)),
            }
            experiment_rows.append(experiment_row)
            evaluation_scores.append(float(score))
            evaluation_correctness.append(int(correctness))

    output_run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_run_dir / "all_experiments.jsonl", experiment_rows)
    write_jsonl(
        output_run_dir / "evaluation_results.jsonl",
        [
            {
                "correctness": evaluation_correctness,
                "scores": evaluation_scores,
                "num_rows": int(len(experiment_rows)),
                "num_prompts": int(len(prompt_records)),
                "score_source": "math_dapo" if rescore_with_math_dapo else "source_validation_data",
            }
        ],
    )
    return prompt_records, {
        "subset_id": int(subset_id),
        "source_path": str(source_path),
        "run_dir": str(output_run_dir),
        "num_prompts": int(len(prompt_records)),
        "num_rollouts": int(len(experiment_rows)),
        "num_correct": int(sum(evaluation_correctness)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize SPO validation_data JSONL files into classifer_training run dirs.")
    parser.add_argument("--subset-dirs", nargs="+", type=Path, default=DEFAULT_SUBSET_DIRS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default="spo_offline_subset0_1_validation_data")
    parser.add_argument("--keep-source-labels", action="store_true")
    parser.add_argument("--max-prompts-per-subset", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    prompt_dataset_dir = output_root / "prompt_dataset"
    run_root = output_root / "runs"
    manifest_path = output_root / "manifest.json"

    if manifest_path.exists() and not args.overwrite:
        print(manifest_path)
        return

    all_prompt_records: list[dict[str, Any]] = []
    run_summaries = []
    for idx, subset_dir in enumerate(args.subset_dirs):
        subset_dir = subset_dir.expanduser().resolve()
        subset_id = _subset_id_from_path(subset_dir, idx)
        run_dir = run_root / f"offline_value_estimation_subset_{subset_id}"
        prompt_records, run_summary = _normalize_run(
            subset_dir,
            output_run_dir=run_dir,
            subset_id=subset_id,
            rescore_with_math_dapo=not bool(args.keep_source_labels),
            max_prompts=args.max_prompts_per_subset,
        )
        all_prompt_records.extend(prompt_records)
        run_summaries.append(run_summary)

    prompt_dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(prompt_dataset_dir / "train.jsonl", all_prompt_records)
    manifest = {
        "dataset_name": args.dataset_name,
        "output_root": str(output_root),
        "prompt_dataset_dir": str(prompt_dataset_dir),
        "run_dirs": [row["run_dir"] for row in run_summaries],
        "runs": run_summaries,
        "num_prompts": int(len(all_prompt_records)),
        "num_rollouts": int(sum(row["num_rollouts"] for row in run_summaries)),
        "label_source": "source_validation_data" if args.keep_source_labels else "math_dapo.compute_score",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
