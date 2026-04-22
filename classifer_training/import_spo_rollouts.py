from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from classifer_training.rollout_utils import split_reasoning_and_answer
from classifer_training.utils import load_records, write_jsonl

_USER_ASSISTANT_WRAPPER = re.compile(r"^\s*user\s*\n(?P<user>.*)\nassistant\s*$", re.IGNORECASE | re.DOTALL)


def _stable_task_id(raw_input: str, ground_truth: str) -> str:
    payload = f"{raw_input}\n<ground_truth>\n{ground_truth}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _count_simple_tokens(text: str) -> int:
    return len([token for token in str(text).strip().split() if token])


def _extract_user_input(raw_input: str) -> str:
    text = str(raw_input or "").strip()
    match = _USER_ASSISTANT_WRAPPER.match(text)
    if match is not None:
        return match.group("user").strip()
    return text


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import SPO offline-value-estimation rollout JSONL files into classifer_training run_dir format."
    )
    parser.add_argument("--input_glob", required=True, help="Glob for SPO subset JSONL files.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--dataset_name", default="spo_offline_value_estimation")
    parser.add_argument("--split_name", default="validation")
    parser.add_argument(
        "--sample_cap_per_prompt",
        type=int,
        default=None,
        help="Optional cap on the number of rollouts imported per prompt inside each subset file.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _build_imported_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    split_name: str,
    source_jsonl: Path,
    sample_cap_per_prompt: int | None,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    imported_rows: list[dict[str, Any]] = []
    correctness: list[int] = []
    per_task_counts: Counter[str] = Counter()
    prompt_rollout_counter: Counter[str] = Counter()

    prepared_rows: list[dict[str, Any]] = []
    for source_row_index, row in enumerate(rows):
        raw_input = str(row.get("input", ""))
        generated_text = str(row.get("output", ""))
        ground_truth = str(row.get("gts", ""))
        user_input = _extract_user_input(raw_input)
        task_id = _stable_task_id(raw_input, ground_truth)
        prompt_rollout_counter[task_id] += 1
        prepared_rows.append(
            {
                "source_row_index": int(source_row_index),
                "raw_input": raw_input,
                "user_input": user_input,
                "generated_text": generated_text,
                "ground_truth": ground_truth,
                "task_id": task_id,
                "pred": row.get("pred", ""),
                "score": _coerce_float(row.get("score", row.get("reward", 0.0))),
                "reward": _coerce_float(row.get("reward", row.get("score", 0.0))),
                "step": int(row.get("step", 0)),
            }
        )

    kept_per_task: Counter[str] = Counter()
    for prepared in prepared_rows:
        task_id = prepared["task_id"]
        if sample_cap_per_prompt is not None and kept_per_task[task_id] >= sample_cap_per_prompt:
            continue

        reasoning_content, answer_content = split_reasoning_and_answer(prepared["generated_text"])
        input_length = _count_simple_tokens(prepared["user_input"])
        output_length = _count_simple_tokens(prepared["generated_text"])
        think_tokens = _count_simple_tokens(reasoning_content)
        answer_tokens = _count_simple_tokens(answer_content)

        sample_index = kept_per_task[task_id]
        kept_per_task[task_id] += 1
        reward = float(prepared["reward"])
        score = float(prepared["score"])
        correct = int(reward >= 1.0 or score >= 1.0)

        imported_rows.append(
            {
                "dataset_name": dataset_name,
                "task_id": task_id,
                "split": split_name,
                "user_input": prepared["user_input"],
                "ground_truth": prepared["ground_truth"],
                "messages": [{"role": "user", "content": prepared["user_input"]}],
                "generated_text": prepared["generated_text"],
                "reasoning_content": reasoning_content,
                "answer_content": answer_content,
                "input_length": int(input_length),
                "output_length": int(output_length),
                "generation_time": 0.0,
                "sample_index": int(sample_index),
                "sample_count": None,
                "has_complete_answer": bool(answer_content.strip()),
                "token_stats": {
                    "think_tokens": int(think_tokens),
                    "answer_tokens": int(answer_tokens),
                    "total_tokens": int(output_length),
                },
                "config": {
                    "source": "spo_offline_value_estimation_import",
                    "source_jsonl": str(source_jsonl),
                    "imported_from_field_schema": ["input", "output", "gts", "pred", "reward", "score", "step"],
                },
                "pred": prepared["pred"],
                "score": score,
                "reward": reward,
                "step": int(prepared["step"]),
                "source_jsonl": str(source_jsonl),
                "source_row_index": int(prepared["source_row_index"]),
                "raw_input": prepared["raw_input"],
            }
        )
        correctness.append(correct)
        per_task_counts[task_id] += 1

    for imported_row in imported_rows:
        imported_row["sample_count"] = int(per_task_counts[imported_row["task_id"]])

    stats = {
        "num_source_rows": int(len(rows)),
        "num_imported_rows": int(len(imported_rows)),
        "num_prompts": int(len(per_task_counts)),
        "prompt_rollout_distribution": {
            str(num_rollouts): int(num_prompts)
            for num_rollouts, num_prompts in sorted(Counter(per_task_counts.values()).items())
        },
    }
    return imported_rows, correctness, stats


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not input_paths:
        raise FileNotFoundError(f"No files matched input glob: {args.input_glob}")

    summary_rows: list[dict[str, Any]] = []
    for input_path in input_paths:
        resolved_input_path = input_path.expanduser().resolve()
        subset_name = resolved_input_path.parent.parent.name
        run_dir = output_root / subset_name
        experiments_path = run_dir / "all_experiments.jsonl"
        evaluation_path = run_dir / "evaluation_results.jsonl"
        import_summary_path = run_dir / "import_summary.json"
        if not args.overwrite and experiments_path.exists() and evaluation_path.exists() and import_summary_path.exists():
            summary_rows.append(json.loads(import_summary_path.read_text(encoding="utf-8")))
            continue

        source_rows = load_records(resolved_input_path)
        imported_rows, correctness, stats = _build_imported_rows(
            source_rows,
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            source_jsonl=resolved_input_path,
            sample_cap_per_prompt=args.sample_cap_per_prompt,
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(experiments_path, imported_rows)
        evaluation_row = {
            "dataset_name": args.dataset_name,
            "subset_name": subset_name,
            "num_examples": int(len(imported_rows)),
            "num_prompts": int(stats["num_prompts"]),
            "accuracy": float(sum(correctness) / len(correctness)) if correctness else 0.0,
            "correctness": correctness,
            "config": {
                "source": "spo_offline_value_estimation_import",
                "source_jsonl": str(resolved_input_path),
                "split_name": args.split_name,
                "sample_cap_per_prompt": args.sample_cap_per_prompt,
            },
        }
        write_jsonl(evaluation_path, [evaluation_row])

        import_summary = {
            "subset_name": subset_name,
            "source_jsonl": str(resolved_input_path),
            "run_dir": str(run_dir),
            "experiments_path": str(experiments_path),
            "evaluation_path": str(evaluation_path),
            **stats,
            "accuracy": evaluation_row["accuracy"],
        }
        import_summary_path.write_text(json.dumps(import_summary, indent=2), encoding="utf-8")
        summary_rows.append(import_summary)
        print(json.dumps(import_summary), flush=True)

    summary_path = output_root / "summary.json"
    summary = {
        "dataset_name": args.dataset_name,
        "output_root": str(output_root),
        "num_run_dirs": int(len(summary_rows)),
        "runs": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
