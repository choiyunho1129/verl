from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


IFBENCH_ROOT = Path(__file__).resolve().parent / "external" / "IFBench"
if str(IFBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(IFBENCH_ROOT))

import evaluation_lib  # type: ignore

from classifer_training.utils import load_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite evaluation_results.jsonl for an IFBench sample run using the official verifier.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--ifbench_input_path", type=Path, required=True)
    parser.add_argument("--mode", choices=("loose", "strict"), default="loose")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    experiments_path = run_dir / "all_experiments.jsonl"
    evaluations_path = run_dir / "evaluation_results.jsonl"
    if evaluations_path.exists() and not args.overwrite:
        raise FileExistsError(f"{evaluations_path} already exists. Pass --overwrite to replace it.")

    experiment_rows = load_records(experiments_path)
    inputs = evaluation_lib.read_prompt_list(str(args.ifbench_input_path.expanduser().resolve()))
    input_by_key = {str(example.key): example for example in inputs}

    verifier: Callable[..., Any]
    if args.mode == "strict":
        verifier = evaluation_lib.test_instruction_following_strict
    else:
        verifier = evaluation_lib.test_instruction_following_loose

    correctness: list[int] = []
    prompt_correctness: dict[str, list[int]] = defaultdict(list)
    for row in experiment_rows:
        task_id = str(row["task_id"])
        input_example = input_by_key.get(task_id)
        if input_example is None:
            raise KeyError(f"task_id={task_id} from {experiments_path} is missing in {args.ifbench_input_path}.")
        sanitized_kwargs = [
            {key: value for key, value in dict(kwargs).items() if value is not None}
            for kwargs in input_example.kwargs
        ]
        input_example = dataclasses.replace(input_example, kwargs=sanitized_kwargs)
        response = str(row.get("generated_text", ""))
        result = verifier(input_example, {input_example.prompt: response})
        is_correct = int(bool(result.follow_all_instructions))
        correctness.append(is_correct)
        prompt_correctness[task_id].append(is_correct)

    config = dict(experiment_rows[0].get("config") or {}) if experiment_rows else {}
    config["grader"] = f"ifbench_{args.mode}"
    evaluation_rows = [
        {
            "dataset_name": str(experiment_rows[0].get("dataset_name", "ifbench")) if experiment_rows else "ifbench",
            "num_examples": int(len(experiment_rows)),
            "num_prompts": int(len(prompt_correctness)),
            "accuracy": float(sum(correctness) / max(len(correctness), 1)),
            "correctness": correctness,
            "prompt_accuracy": {
                task_id: float(sum(values) / len(values))
                for task_id, values in sorted(prompt_correctness.items())
            },
            "config": config,
            "evaluation_mode": str(args.mode),
            "input_data": str(args.ifbench_input_path.expanduser().resolve()),
        }
    ]
    write_jsonl(evaluations_path, evaluation_rows)

    summary = {
        "run_dir": str(run_dir),
        "ifbench_input_path": str(args.ifbench_input_path.expanduser().resolve()),
        "mode": str(args.mode),
        "num_examples": int(len(experiment_rows)),
        "num_prompts": int(len(prompt_correctness)),
        "accuracy": float(evaluation_rows[0]["accuracy"]),
    }
    (run_dir / f"ifbench_{args.mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
