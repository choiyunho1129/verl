from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from classifer_training.utils import load_records, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "deepscaler" / "train_deepscaler.parquet"
DEFAULT_VALIDATION_PATH = REPO_ROOT / "data" / "deepscaler" / "valid_deepscaler.parquet"


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    items = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(items, list) or not items:
        raise ValueError("DeepScaleR prompt field must be a non-empty list of role/content dicts.")
    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"Prompt message must be a dict, got {type(item)!r}.")
        normalized.append(
            {
                "role": str(item.get("role", "user")),
                "content": str(item.get("content", "")),
            }
        )
    return normalized


def _build_task_id(record: dict[str, Any], split_name: str, row_idx: int) -> str:
    extra_info = record.get("extra_info")
    if isinstance(extra_info, dict):
        source_index = extra_info.get("index")
        if source_index not in (None, ""):
            return f"{split_name}_{source_index}"
    return f"{split_name}_{row_idx}"


def _normalize_record(record: dict[str, Any], *, dataset_name: str, split_name: str, row_idx: int, source_path: Path) -> dict[str, Any]:
    messages = _normalize_messages(record.get("prompt"))
    ground_truth = record.get("reward_model", {}).get("ground_truth")
    if ground_truth in (None, ""):
        raise ValueError(f"Missing reward_model.ground_truth for {split_name} row {row_idx}.")
    user_input = ""
    for item in reversed(messages):
        if item["role"] == "user":
            user_input = item["content"]
            break
    if not user_input:
        user_input = messages[-1]["content"]

    extra_info = record.get("extra_info") if isinstance(record.get("extra_info"), dict) else {}
    return {
        "dataset_name": dataset_name,
        "task_id": _build_task_id(record, split_name, row_idx),
        "split": split_name,
        "user_input": user_input,
        "ground_truth": str(ground_truth),
        "messages": messages,
        "source": {
            "path": str(source_path),
            "original_split": str(extra_info.get("split", split_name)),
            "original_index": extra_info.get("index"),
            "ability": record.get("ability"),
            "data_source": record.get("data_source"),
            "reward_style": record.get("reward_model", {}).get("style"),
        },
    }


def _select_records(
    records: list[dict[str, Any]],
    *,
    split_name: str,
    dataset_name: str,
    source_path: Path,
    sample_count: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    if sample_count < 1:
        raise ValueError(f"{split_name} sample_count must be positive.")
    if len(records) < sample_count:
        raise ValueError(f"Requested {sample_count} {split_name} prompts but only found {len(records)}.")
    shuffled = list(records)
    random.Random(sample_seed).shuffle(shuffled)
    selected = shuffled[:sample_count]
    return [
        _normalize_record(record, dataset_name=dataset_name, split_name=split_name, row_idx=row_idx, source_path=source_path)
        for row_idx, record in enumerate(selected)
    ]


def _write_generation_shards(rows: list[dict[str, Any]], *, output_dir: Path, shard_size: int) -> list[Path]:
    if shard_size < 1:
        raise ValueError("generation shard_size must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    for shard_idx, start_idx in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start_idx : start_idx + shard_size]
        shard_path = output_dir / f"shard{shard_idx:04d}.jsonl"
        write_jsonl(shard_path, shard_rows)
        shard_paths.append(shard_path)
    return shard_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a custom DeepScaleR train/validation dataset and generation shards."
    )
    parser.add_argument("--train_input_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation_input_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", default="deepscaler_train5500_validation2000")
    parser.add_argument("--train_prompts", type=int, default=5500)
    parser.add_argument("--validation_prompts", type=int, default=2000)
    parser.add_argument("--train_generation_shard_size", type=int, default=500)
    parser.add_argument("--validation_generation_shard_size", type=int, default=250)
    parser.add_argument("--sample_seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    summary_path = output_dir / "summary.json"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if summary_path.exists() and not args.overwrite:
        print(summary_path.read_text(encoding="utf-8"))
        return

    train_input_path = args.train_input_path.expanduser().resolve()
    validation_input_path = args.validation_input_path.expanduser().resolve()
    train_records = load_records(train_input_path)
    validation_records = load_records(validation_input_path)

    train_rows = _select_records(
        train_records,
        split_name="train",
        dataset_name=args.dataset_name,
        source_path=train_input_path,
        sample_count=args.train_prompts,
        sample_seed=args.sample_seed,
    )
    validation_rows = _select_records(
        validation_records,
        split_name="validation",
        dataset_name=args.dataset_name,
        source_path=validation_input_path,
        sample_count=args.validation_prompts,
        sample_seed=args.sample_seed + 1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    train_shard_paths = _write_generation_shards(
        train_rows,
        output_dir=output_dir / "train_generation_shards",
        shard_size=args.train_generation_shard_size,
    )
    validation_shard_paths = _write_generation_shards(
        validation_rows,
        output_dir=output_dir / "validation_generation_shards",
        shard_size=args.validation_generation_shard_size,
    )

    summary = {
        "dataset_name": args.dataset_name,
        "train_input_path": str(train_input_path),
        "validation_input_path": str(validation_input_path),
        "output_dir": str(output_dir),
        "sample_seed": int(args.sample_seed),
        "train_prompts": int(len(train_rows)),
        "validation_prompts": int(len(validation_rows)),
        "train_generation_shard_size": int(args.train_generation_shard_size),
        "validation_generation_shard_size": int(args.validation_generation_shard_size),
        "train_generation_num_shards": int(len(train_shard_paths)),
        "validation_generation_num_shards": int(len(validation_shard_paths)),
        "train_generation_shard_sizes": [sum(1 for _ in path.open("r", encoding="utf-8")) for path in train_shard_paths],
        "validation_generation_shard_sizes": [sum(1 for _ in path.open("r", encoding="utf-8")) for path in validation_shard_paths],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
