from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from classifer_training.ifevalg_official import evaluate_ifevalg_response
from classifer_training.utils import load_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite a sample.py run directory using official open-instruct IFEvalG verification."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--open-instruct-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--details-filename", default="ifevalg_scores.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    experiments_path = run_dir / "all_experiments.jsonl"
    evaluations_path = run_dir / "evaluation_results.jsonl"
    if evaluations_path.exists() and not args.overwrite:
        raise FileExistsError(f"{evaluations_path} already exists. Pass --overwrite to replace it.")

    experiment_rows = load_records(experiments_path)
    correctness: list[int] = []
    scores: list[float] = []
    prompt_scores: dict[str, list[float]] = defaultdict(list)
    detail_rows: list[dict] = []

    for row_idx, row in enumerate(experiment_rows):
        result = evaluate_ifevalg_response(
            str(row.get("generated_text", "")),
            row.get("ground_truth", ""),
            open_instruct_root=args.open_instruct_root,
        )
        task_id = str(row.get("task_id", row_idx))
        score = float(result["score"])
        is_correct = int(bool(result["follow_all"]))
        scores.append(score)
        correctness.append(is_correct)
        prompt_scores[task_id].append(score)
        detail_rows.append(
            {
                "row_idx": int(row_idx),
                "task_id": task_id,
                "sample_index": row.get("sample_index"),
                "score": score,
                "follow_all": bool(result["follow_all"]),
                "num_followed": int(result["num_followed"]),
                "num_instructions": int(result["num_instructions"]),
                "per_instruction": result["per_instruction"],
            }
        )

    config = dict(experiment_rows[0].get("config") or {}) if experiment_rows else {}
    config["grader"] = "ifeval"
    evaluation_rows = [
        {
            "dataset_name": str(experiment_rows[0].get("dataset_name", "ifeval")) if experiment_rows else "ifeval",
            "num_examples": int(len(experiment_rows)),
            "num_prompts": int(len(prompt_scores)),
            "accuracy": float(sum(correctness) / max(len(correctness), 1)),
            "mean_constraint_score": float(sum(scores) / max(len(scores), 1)),
            "correctness": correctness,
            "constraint_scores": scores,
            "prompt_mean_constraint_score": {
                task_id: float(sum(values) / len(values))
                for task_id, values in sorted(prompt_scores.items())
            },
            "config": config,
        }
    ]
    write_jsonl(evaluations_path, evaluation_rows)
    write_jsonl(run_dir / args.details_filename, detail_rows)

    summary = {
        "run_dir": str(run_dir),
        "num_examples": int(len(experiment_rows)),
        "num_prompts": int(len(prompt_scores)),
        "accuracy": float(evaluation_rows[0]["accuracy"]),
        "mean_constraint_score": float(evaluation_rows[0]["mean_constraint_score"]),
        "details_path": str((run_dir / args.details_filename).resolve()),
    }
    (run_dir / "ifevalg_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
