from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from classifer_training.acecoder_official import normalize_test_cases
from classifer_training.utils import write_jsonl


def _chunk_evenly(rows: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(max(int(num_shards), 1))]
    for idx, row in enumerate(rows):
        shards[idx % len(shards)].append(row)
    return shards


def _write_generation_shards(rows: list[dict[str, Any]], *, output_dir: Path, shard_size: int) -> list[int]:
    if shard_size < 1:
        raise ValueError("generation shard size must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_sizes: list[int] = []
    for shard_idx, start_idx in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start_idx : start_idx + shard_size]
        write_jsonl(output_dir / f"shard{shard_idx:04d}.jsonl", shard_rows)
        shard_sizes.append(len(shard_rows))
    return shard_sizes


def _messages_from_row(row: dict[str, Any], question: str) -> list[dict[str, str]]:
    messages = row.get("context_messages")
    if isinstance(messages, list) and messages:
        normalized = []
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                normalized.append(
                    {
                        "role": str(message.get("role") or "user"),
                        "content": str(message.get("content") or ""),
                    }
                )
        if normalized:
            return normalized
    return [{"role": "user", "content": question}]


def _normalize_row(
    row: dict[str, Any],
    *,
    dataset_name: str,
    output_split: str,
    dataset_id: str,
    source_split: str,
    source_row_index: int,
    selection_index: int,
) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    tests = normalize_test_cases(row.get("test_cases"))
    task_id = str(row.get("id") or row.get("task_id") or f"{output_split}_{selection_index}")
    return {
        "dataset_name": str(dataset_name),
        "dataset": str(row.get("source") or dataset_name),
        "task_id": task_id,
        "split": str(output_split),
        "user_input": question,
        "prompt": question,
        "messages": _messages_from_row(row, question),
        "ground_truth": json.dumps(tests, ensure_ascii=False),
        "test_cases": tests,
        "source": {
            "dataset_id": str(dataset_id),
            "split": str(source_split),
            "row_index": int(source_row_index),
            "selection_index": int(selection_index),
            "acecode_id": str(row.get("id") or ""),
            "acecode_source": str(row.get("source") or ""),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TIGER-Lab/AceCode-87K for rollout generation and test-case reward scoring.")
    parser.add_argument("--dataset-id", default="TIGER-Lab/AceCode-87K")
    parser.add_argument("--split", default="train", help="Source HF split to sample from.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, default=None, help="Prompt hidden extraction shard directory.")
    parser.add_argument("--dataset-name", default="acecode_87k")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--train-prompts", type=int, default=None)
    parser.add_argument("--validation-prompts", type=int, default=None)
    parser.add_argument("--train-generation-shard-size", type=int, default=512)
    parser.add_argument("--validation-generation-shard-size", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--min-tests", type=int, default=1)
    parser.add_argument("--max-tests", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_specific = args.train_prompts is not None or args.validation_prompts is not None
    if split_specific and (args.train_prompts is None or args.validation_prompts is None):
        raise ValueError("--train-prompts and --validation-prompts must be provided together.")
    if split_specific and args.max_examples is not None:
        raise ValueError("--max-examples is only supported for single-split preparation.")

    output_dir = args.output_dir.expanduser().resolve()
    shard_dir = args.shard_dir.expanduser().resolve() if args.shard_dir else None
    if args.overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
        if shard_dir is not None:
            shutil.rmtree(shard_dir, ignore_errors=True)

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_id, split=args.split)
    if split_specific:
        dataset = dataset.shuffle(seed=int(args.sample_seed))
    elif args.max_examples is not None:
        dataset = dataset.select(range(min(int(args.max_examples), len(dataset))))

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []} if split_specific else {str(args.split): []}
    skipped_rows: list[dict[str, Any]] = []

    def usable_row(row_dict: dict[str, Any]) -> tuple[bool, str]:
        question = str(row_dict.get("question") or "").strip()
        tests = normalize_test_cases(row_dict.get("test_cases"))
        if not question:
            return False, "missing_question"
        if len(tests) < int(args.min_tests):
            return False, "too_few_tests"
        if args.max_tests is not None and len(tests) > int(args.max_tests):
            row_dict["test_cases"] = tests[: int(args.max_tests)]
        return True, ""

    if split_specific:
        targets = {
            "train": int(args.train_prompts),
            "validation": int(args.validation_prompts),
        }
        total_target = sum(targets.values())
        for source_row_index, row in enumerate(dataset):
            if sum(len(values) for values in rows_by_split.values()) >= total_target:
                break
            output_split = "train" if len(rows_by_split["train"]) < targets["train"] else "validation"
            row_dict = dict(row)
            ok, reason = usable_row(row_dict)
            if not ok:
                skipped_rows.append({"row_idx": int(source_row_index), "reason": reason, "id": str(row_dict.get("id", ""))})
                continue
            rows_by_split[output_split].append(
                _normalize_row(
                    row_dict,
                    dataset_name=args.dataset_name,
                    output_split=output_split,
                    dataset_id=args.dataset_id,
                    source_split=args.split,
                    source_row_index=source_row_index,
                    selection_index=len(rows_by_split[output_split]),
                )
            )
        shortfalls = {
            split_name: targets[split_name] - len(rows)
            for split_name, rows in rows_by_split.items()
            if len(rows) < targets[split_name]
        }
        if shortfalls:
            raise ValueError(f"Not enough usable rows to satisfy requested split sizes: {shortfalls}")
    else:
        for row_idx, row in enumerate(dataset):
            row_dict = dict(row)
            ok, reason = usable_row(row_dict)
            if not ok:
                skipped_rows.append({"row_idx": int(row_idx), "reason": reason, "id": str(row_dict.get("id", ""))})
                continue
            rows_by_split[str(args.split)].append(
                _normalize_row(
                    row_dict,
                    dataset_name=args.dataset_name,
                    output_split=str(args.split),
                    dataset_id=args.dataset_id,
                    source_split=args.split,
                    source_row_index=row_idx,
                    selection_index=len(rows_by_split[str(args.split)]),
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split_name, rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)
        all_rows.extend(rows)
        split_counts[split_name] = len(rows)

    prompt_shard_sizes: list[int] = []
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)
        shards = _chunk_evenly(all_rows, args.num_shards)
        for shard_idx, shard_rows in enumerate(shards):
            write_jsonl(shard_dir / f"shard{shard_idx}.jsonl", shard_rows)
        write_jsonl(shard_dir / "all.jsonl", all_rows)
        prompt_shard_sizes = [len(shard) for shard in shards]

    generation_shards: dict[str, list[int]] = {}
    if split_specific:
        generation_shards["train"] = _write_generation_shards(
            rows_by_split["train"],
            output_dir=output_dir / "train_generation_shards",
            shard_size=int(args.train_generation_shard_size),
        )
        generation_shards["validation"] = _write_generation_shards(
            rows_by_split["validation"],
            output_dir=output_dir / "validation_generation_shards",
            shard_size=int(args.validation_generation_shard_size),
        )

    summary = {
        "dataset_id": str(args.dataset_id),
        "source_split": str(args.split),
        "dataset_name": str(args.dataset_name),
        "output_dir": str(output_dir),
        "shard_dir": str(shard_dir) if shard_dir else None,
        "sample_seed": int(args.sample_seed),
        "num_rows_loaded": int(len(dataset)),
        "num_rows_written": int(len(all_rows)),
        "split_counts": split_counts,
        "num_skipped_rows": int(len(skipped_rows)),
        "min_tests": int(args.min_tests),
        "max_tests": int(args.max_tests) if args.max_tests is not None else None,
        "num_shards": int(args.num_shards) if shard_dir else 0,
        "prompt_shard_sizes": prompt_shard_sizes,
        "generation_shard_sizes": generation_shards,
        "skipped_examples": skipped_rows[:20],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if shard_dir is not None:
        (shard_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
