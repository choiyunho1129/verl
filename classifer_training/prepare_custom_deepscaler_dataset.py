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


def _has_usable_messages(value: Any) -> bool:
    items = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        if item.get("content") in (None, ""):
            return False
    return True


def _has_usable_ground_truth(record: dict[str, Any]) -> bool:
    reward_model = record.get("reward_model")
    if not isinstance(reward_model, dict):
        return False
    ground_truth = reward_model.get("ground_truth")
    return ground_truth not in (None, "")


def _filter_usable_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    usable: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        if not _has_usable_messages(record.get("prompt")):
            skipped += 1
            continue
        if not _has_usable_ground_truth(record):
            skipped += 1
            continue
        usable.append(record)
    return usable, skipped


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
) -> tuple[list[dict[str, Any]], int]:
    if sample_count < 1:
        raise ValueError(f"{split_name} sample_count must be positive.")
    usable_records, skipped_count = _filter_usable_records(records)
    if len(usable_records) < sample_count:
        raise ValueError(
            f"Requested {sample_count} {split_name} prompts but only found {len(usable_records)} usable rows "
            f"after skipping {skipped_count} invalid rows."
        )
    shuffled = list(usable_records)
    random.Random(sample_seed).shuffle(shuffled)
    selected = shuffled[:sample_count]
    normalized_rows = [
        _normalize_record(record, dataset_name=dataset_name, split_name=split_name, row_idx=row_idx, source_path=source_path)
        for row_idx, record in enumerate(selected)
    ]
    return normalized_rows, skipped_count


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


def _load_reused_train_rows(
    reuse_dir: Path,
    *,
    dataset_name: str,
    train_prompts: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_path = reuse_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing reused train dataset file: {train_path}")

    train_rows = load_records(train_path)
    if len(train_rows) != train_prompts:
        raise ValueError(
            f"Requested {train_prompts} train prompts but reused dataset has {len(train_rows)} rows: {train_path}"
        )

    normalized_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(train_rows):
        task_id = str(row.get("task_id", "")).strip()
        messages = row.get("messages")
        if not task_id:
            raise ValueError(f"Reused train row {row_idx} is missing task_id: {train_path}")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Reused train row {row_idx} is missing messages: {train_path}")
        normalized_rows.append(
            {
                **row,
                "dataset_name": dataset_name,
                "split": "train",
            }
        )

    summary: dict[str, Any] = {}
    summary_path = reuse_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return normalized_rows, summary


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
    parser.add_argument(
        "--reuse_train_dataset_dir",
        type=Path,
        default=None,
        help="Reuse normalized train rows from an existing prepared dataset dir instead of resampling train prompts.",
    )
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

    validation_input_path = args.validation_input_path.expanduser().resolve()
    validation_records = load_records(validation_input_path)
    reuse_train_dataset_dir = args.reuse_train_dataset_dir.expanduser().resolve() if args.reuse_train_dataset_dir else None
    reused_train_summary: dict[str, Any] = {}
    if reuse_train_dataset_dir is None:
        train_input_path = args.train_input_path.expanduser().resolve()
        train_records = load_records(train_input_path)
        train_rows, train_skipped = _select_records(
            train_records,
            split_name="train",
            dataset_name=args.dataset_name,
            source_path=train_input_path,
            sample_count=args.train_prompts,
            sample_seed=args.sample_seed,
        )
        train_source_rows_total = int(len(train_records))
    else:
        train_rows, reused_train_summary = _load_reused_train_rows(
            reuse_train_dataset_dir,
            dataset_name=args.dataset_name,
            train_prompts=args.train_prompts,
        )
        train_input_path = Path(str(reused_train_summary.get("train_input_path", reuse_train_dataset_dir / "train.jsonl")))
        train_skipped = int(reused_train_summary.get("train_source_rows_skipped_invalid", 0))
        train_source_rows_total = int(reused_train_summary.get("train_source_rows_total", len(train_rows)))

    validation_rows, validation_skipped = _select_records(
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
        "train_source_rows_total": int(train_source_rows_total),
        "validation_source_rows_total": int(len(validation_records)),
        "train_source_rows_skipped_invalid": int(train_skipped),
        "validation_source_rows_skipped_invalid": int(validation_skipped),
        "train_generation_shard_size": int(args.train_generation_shard_size),
        "validation_generation_shard_size": int(args.validation_generation_shard_size),
        "train_generation_num_shards": int(len(train_shard_paths)),
        "validation_generation_num_shards": int(len(validation_shard_paths)),
        "train_generation_shard_sizes": [sum(1 for _ in path.open("r", encoding="utf-8")) for path in train_shard_paths],
        "validation_generation_shard_sizes": [sum(1 for _ in path.open("r", encoding="utf-8")) for path in validation_shard_paths],
        "reused_train_dataset_dir": str(reuse_train_dataset_dir) if reuse_train_dataset_dir is not None else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
