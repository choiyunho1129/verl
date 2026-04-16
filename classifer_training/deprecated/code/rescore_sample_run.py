from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifer_training.sample import _score_generated_answer
from classifer_training.utils import load_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore an existing sample.py run directory and rewrite evaluation_results.jsonl.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--grader", choices=("math_verify", "exact"), default="math_verify")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    experiments_path = run_dir / "all_experiments.jsonl"
    evaluations_path = run_dir / "evaluation_results.jsonl"

    if not experiments_path.exists():
        raise FileNotFoundError(f"Missing {experiments_path}")
    if evaluations_path.exists() and not args.overwrite:
        raise FileExistsError(f"{evaluations_path} already exists. Pass --overwrite to replace.")

    rows = load_records(experiments_path)
    if not rows:
        raise ValueError(f"No rows found in {experiments_path}")

    correctness: list[int] = []
    prompt_correctness: dict[str, list[int]] = {}
    for row in rows:
        correct = int(
            _score_generated_answer(
                generated_text=str(row.get("generated_text", "")),
                answer_content=str(row.get("answer_content", "")),
                ground_truth=str(row.get("ground_truth", "")),
                grader=args.grader,
            )
        )
        correctness.append(correct)
        task_id = str(row.get("task_id", ""))
        prompt_correctness.setdefault(task_id, []).append(correct)

    dataset_name = str(rows[0].get("dataset_name", run_dir.name))
    config = dict(rows[0].get("config") or {})
    config["grader"] = args.grader
    evaluation_row = {
        "dataset_name": dataset_name,
        "num_examples": len(rows),
        "num_prompts": len(prompt_correctness),
        "accuracy": float(sum(correctness) / len(correctness)),
        "correctness": correctness,
        "config": config,
        "prompt_accuracy": {task_id: float(sum(vals) / len(vals)) for task_id, vals in sorted(prompt_correctness.items())},
    }
    write_jsonl(evaluations_path, [evaluation_row])
    print(json.dumps({"run_dir": str(run_dir), "accuracy": evaluation_row["accuracy"], "num_prompts": evaluation_row["num_prompts"]}, indent=2))


if __name__ == "__main__":
    main()
