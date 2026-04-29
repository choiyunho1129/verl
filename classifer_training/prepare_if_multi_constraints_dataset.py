from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from classifer_training.ifevalg_official import supported_instruction_ids
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


def _user_input_from_messages(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("IF_multi row must contain non-empty messages.")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    raise TypeError("messages must be a list of role/content dicts.")


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
    messages = row["messages"]
    user_input = _user_input_from_messages(messages)
    return {
        "dataset_name": str(dataset_name),
        "dataset": str(row.get("dataset", dataset_name)),
        "task_id": str(row.get("key", f"{output_split}_{selection_index}")),
        "split": str(output_split),
        "user_input": user_input,
        "prompt": user_input,
        "messages": messages,
        "ground_truth": row["ground_truth"],
        "constraint_type": row.get("constraint_type"),
        "constraint": row.get("constraint"),
        "source": {
            "dataset_id": str(dataset_id),
            "split": str(source_split),
            "row_index": int(source_row_index),
            "selection_index": int(selection_index),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare allenai/IF_multi_constraints_upto5 for local generation and IFEvalG verification."
    )
    parser.add_argument("--dataset-id", default="allenai/IF_multi_constraints_upto5")
    parser.add_argument("--split", default="train", help="Source HF split to sample from.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, default=None, help="Prompt hidden extraction shard directory.")
    parser.add_argument("--dataset-name", default="ifeval")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--train-prompts", type=int, default=None)
    parser.add_argument("--validation-prompts", type=int, default=None)
    parser.add_argument("--train-generation-shard-size", type=int, default=512)
    parser.add_argument("--validation-generation-shard-size", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--filter-unsupported", action="store_true")
    parser.add_argument("--open-instruct-root", type=Path, default=None)
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
    unsupported_rows: list[dict[str, Any]] = []

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
            ground_truth = row_dict["ground_truth"]
            is_supported, missing = supported_instruction_ids(
                ground_truth,
                open_instruct_root=args.open_instruct_root,
            )
            if missing:
                unsupported_rows.append(
                    {
                        "row_idx": int(source_row_index),
                        "key": str(row_dict.get("key", source_row_index)),
                        "missing_instruction_ids": missing,
                    }
                )
            if args.filter_unsupported and not is_supported:
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
            raise ValueError(f"Not enough supported rows to satisfy requested split sizes: {shortfalls}")
    else:
        for row_idx, row in enumerate(dataset):
            row_dict = dict(row)
            ground_truth = row_dict["ground_truth"]
            is_supported, missing = supported_instruction_ids(
                ground_truth,
                open_instruct_root=args.open_instruct_root,
            )
            if missing:
                unsupported_rows.append(
                    {
                        "row_idx": int(row_idx),
                        "key": str(row_dict.get("key", row_idx)),
                        "missing_instruction_ids": missing,
                    }
                )
            if args.filter_unsupported and not is_supported:
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
        "num_unsupported_rows": int(len(unsupported_rows)),
        "filter_unsupported": bool(args.filter_unsupported),
        "num_shards": int(args.num_shards) if shard_dir else 0,
        "prompt_shard_sizes": prompt_shard_sizes,
        "generation_shard_sizes": generation_shards,
        "unsupported_examples": unsupported_rows[:20],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if shard_dir is not None:
        (shard_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
