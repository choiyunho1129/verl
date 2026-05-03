from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from classifer_training.sample import (
    _count_text_tokens,
    _score_generated_answer,
    _split_reasoning_and_answer,
)
from classifer_training.utils import load_records, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score existing sample.py run directories after updating answer extraction heuristics."
        )
    )
    parser.add_argument(
        "--run_dirs",
        nargs="*",
        default=[],
        help="Run directories containing all_experiments.jsonl and evaluation_results.jsonl.",
    )
    parser.add_argument(
        "--run_glob",
        type=str,
        default=None,
        help="Optional glob such as 'classifer_training/artifacts/runs/foo/temp0.7_seed*'.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args(argv)


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    if args.run_glob:
        run_dirs.extend(Path(path).expanduser().resolve() for path in sorted(glob.glob(args.run_glob)))
    return sorted({path.resolve() for path in run_dirs})


def _load_tokenizer(tokenizer_cache: dict[str, Any], model_name_or_path: str, trust_remote_code: bool):
    tokenizer = tokenizer_cache.get(model_name_or_path)
    if tokenizer is not None:
        return tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer_cache[model_name_or_path] = tokenizer
    return tokenizer


def _rescore_run_dir(run_dir: Path, tokenizer_cache: dict[str, Any], trust_remote_code: bool) -> dict[str, Any]:
    experiments_path = run_dir / "all_experiments.jsonl"
    evaluations_path = run_dir / "evaluation_results.jsonl"
    experiment_rows = load_records(experiments_path)
    evaluation_rows = load_records(evaluations_path)
    if not experiment_rows:
        raise ValueError(f"No experiment rows found in {experiments_path}.")
    if not evaluation_rows:
        raise ValueError(f"No evaluation rows found in {evaluations_path}.")

    config = dict(evaluation_rows[-1].get("config", {}))
    model_name_or_path = str(config.get("model_name_or_path", ""))
    if not model_name_or_path:
        raise ValueError(f"Missing model_name_or_path in {evaluations_path}.")
    grader = str(config.get("grader", "math_verify"))
    tokenizer = _load_tokenizer(
        tokenizer_cache=tokenizer_cache,
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    correctness: list[int] = []
    prompt_correctness: dict[str, list[int]] = {}
    for row in experiment_rows:
        generated_text = str(row.get("generated_text", ""))
        reasoning_content, answer_content = _split_reasoning_and_answer(generated_text)
        correct, _verification = _score_generated_answer(
            record=row,
            generated_text=generated_text,
            answer_content=answer_content,
            ground_truth=str(row.get("ground_truth", "")),
            grader=grader,
        )
        row["reasoning_content"] = reasoning_content
        row["answer_content"] = answer_content
        row["has_complete_answer"] = bool(answer_content.strip())
        token_stats = dict(row.get("token_stats", {}))
        token_stats["think_tokens"] = int(_count_text_tokens(tokenizer, reasoning_content))
        token_stats["answer_tokens"] = int(_count_text_tokens(tokenizer, answer_content))
        token_stats["total_tokens"] = int(row.get("output_length", token_stats.get("total_tokens", 0)))
        row["token_stats"] = token_stats
        correctness.append(int(correct))
        task_id = str(row.get("task_id", ""))
        prompt_correctness.setdefault(task_id, []).append(int(correct))

    evaluation_row = {
        "dataset_name": str(evaluation_rows[-1].get("dataset_name", experiment_rows[0].get("dataset_name", ""))),
        "num_examples": len(experiment_rows),
        "num_prompts": len(prompt_correctness),
        "accuracy": float(sum(correctness) / len(correctness)),
        "correctness": correctness,
        "prompt_accuracy": {
            task_id: float(sum(values) / len(values))
            for task_id, values in sorted(prompt_correctness.items())
        },
        "config": config,
    }
    write_jsonl(experiments_path, experiment_rows)
    write_jsonl(evaluations_path, [evaluation_row])
    return {
        "run_dir": str(run_dir),
        "num_examples": len(experiment_rows),
        "num_prompts": len(prompt_correctness),
        "accuracy": evaluation_row["accuracy"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dirs = _resolve_run_dirs(args)
    if not run_dirs:
        raise ValueError("At least one run directory is required.")

    tokenizer_cache: dict[str, Any] = {}
    summaries = [
        _rescore_run_dir(
            run_dir=run_dir,
            tokenizer_cache=tokenizer_cache,
            trust_remote_code=args.trust_remote_code,
        )
        for run_dir in run_dirs
    ]
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
