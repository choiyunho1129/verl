from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

from classifer_training.utils import get_nested_value, load_records, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "classifer_training" / "artifacts" / "datasets"
DEFAULT_DEEPSCALER_FILES = {
    "train": REPO_ROOT / "data" / "deepscaler" / "train_deepscaler.parquet",
    "validation": REPO_ROOT / "data" / "deepscaler" / "valid_deepscaler.parquet",
}
DEFAULT_HF_DATASET_IDS = {
    "dapo_math_17k": "open-r1/DAPO-Math-17k-Processed",
}

MESSAGE_FIELD_CANDIDATES = ("messages", "source_prompt", "prompt", "conversation", "conversations", "dialog", "dialogue")
QUESTION_FIELD_CANDIDATES = ("user_input", "question", "problem", "prompt", "instruction", "query", "input")
ANSWER_FIELD_CANDIDATES = ("ground_truth", "answer", "target", "final_answer", "expected_answer", "solution")
TASK_ID_FIELD_CANDIDATES = ("task_id", "id", "index", "qid", "question_id")


def _maybe_sequence(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
    return None


def _normalize_messages(value: Any) -> list[dict[str, str]] | None:
    sequence = _maybe_sequence(value)
    if sequence is None:
        return None

    normalized: list[dict[str, str]] = []
    for item in sequence:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", "user"))
        content = item.get("content")
        if content is None:
            return None
        normalized.append({"role": role, "content": str(content)})
    return normalized if normalized else None


def _find_first_present(record: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        value = record.get(key)
        if value not in (None, ""):
            return key
    return None


def _extract_messages(record: dict[str, Any], explicit_field: str | None) -> list[dict[str, str]] | None:
    if explicit_field:
        return _normalize_messages(record.get(explicit_field))
    for field in MESSAGE_FIELD_CANDIDATES:
        messages = _normalize_messages(record.get(field))
        if messages is not None:
            return messages
    return None


def _extract_question(record: dict[str, Any], explicit_field: str | None) -> str | None:
    def coerce_question_text(value: Any) -> str | None:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dict):
            return None
        if _maybe_sequence(value) is not None:
            return None
        return str(value)

    if explicit_field:
        return coerce_question_text(record.get(explicit_field))
    for field in QUESTION_FIELD_CANDIDATES:
        value = coerce_question_text(record.get(field))
        if value is not None:
            return value
    return None


def _extract_ground_truth(record: dict[str, Any], explicit_field: str | None) -> str | None:
    if explicit_field:
        value = get_nested_value(record, explicit_field, default=None)
        return None if value in (None, "") else str(value)

    reward_model_gt = get_nested_value(record, "reward_model.ground_truth", default=None)
    if reward_model_gt not in (None, ""):
        return str(reward_model_gt)

    for field in ANSWER_FIELD_CANDIDATES:
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _extract_task_id(record: dict[str, Any], explicit_field: str | None, row_idx: int, split_name: str) -> str:
    if explicit_field:
        value = get_nested_value(record, explicit_field, default=None)
        if value not in (None, ""):
            return str(value)

    extra_index = get_nested_value(record, "extra_info.index", default=None)
    if extra_index not in (None, ""):
        return str(extra_index)

    for field in TASK_ID_FIELD_CANDIDATES:
        value = record.get(field)
        if value not in (None, ""):
            return str(value)

    return f"{split_name}_{row_idx}"


def _derive_user_input(messages: list[dict[str, str]] | None, question: str | None) -> str:
    if question:
        return question
    if messages:
        for item in reversed(messages):
            if item.get("role") == "user":
                return item.get("content", "")
        return messages[-1].get("content", "")
    raise ValueError("Could not derive user_input from either messages or question.")


def normalize_record(
    record: dict[str, Any],
    *,
    dataset_name: str,
    split_name: str,
    row_idx: int,
    question_field: str | None,
    answer_field: str | None,
    task_id_field: str | None,
    messages_field: str | None,
    source_name: str,
) -> dict[str, Any]:
    messages = _extract_messages(record, explicit_field=messages_field)
    question = _extract_question(record, explicit_field=question_field)
    if messages is None:
        if question is None:
            raise KeyError("Could not infer either messages or a question field from the dataset row.")
        messages = [{"role": "user", "content": question}]

    ground_truth = _extract_ground_truth(record, explicit_field=answer_field)
    if ground_truth is None:
        raise KeyError("Could not infer the answer / ground-truth field from the dataset row.")

    task_id = _extract_task_id(record, explicit_field=task_id_field, row_idx=row_idx, split_name=split_name)
    user_input = _derive_user_input(messages, question)

    return {
        "dataset_name": dataset_name,
        "task_id": task_id,
        "split": split_name,
        "user_input": user_input,
        "ground_truth": ground_truth,
        "messages": messages,
        "source": {
            "name": source_name,
            "split": split_name,
            "row_index": row_idx,
        },
    }


def _infer_split_name_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "train" in stem:
        return "train"
    if "valid" in stem or "val" in stem:
        return "validation"
    if "test" in stem:
        return "test"
    return "train"


def _load_local_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    return load_records(path)


def prepare_from_local_files(
    *,
    dataset_name: str,
    input_paths: list[Path],
    output_root: Path,
    question_field: str | None,
    answer_field: str | None,
    task_id_field: str | None,
    messages_field: str | None,
    limit: int | None,
    train_examples: int,
    validation_examples: int,
    test_examples: int,
    sample_seed: int,
) -> list[Path]:
    use_subsample_split_plan = any(value > 0 for value in (train_examples, validation_examples, test_examples))
    if use_subsample_split_plan:
        combined_records: list[dict[str, Any]] = []
        for input_path in input_paths:
            combined_records.extend(_load_local_records(input_path))
        if limit is not None:
            combined_records = combined_records[:limit]
        return _apply_subsample_split_plan(
            records=combined_records,
            dataset_name=dataset_name,
            output_root=output_root,
            question_field=question_field,
            answer_field=answer_field,
            task_id_field=task_id_field,
            messages_field=messages_field,
            source_name=",".join(str(path) for path in input_paths),
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            sample_seed=sample_seed,
        )

    output_paths: list[Path] = []
    for input_path in input_paths:
        split_name = _infer_split_name_from_path(input_path)
        records = _load_local_records(input_path)
        if limit is not None:
            records = records[:limit]
        normalized = [
            normalize_record(
                record,
                dataset_name=dataset_name,
                split_name=split_name,
                row_idx=row_idx,
                question_field=question_field,
                answer_field=answer_field,
                task_id_field=task_id_field,
                messages_field=messages_field,
                source_name=str(input_path),
            )
            for row_idx, record in enumerate(records)
        ]
        output_path = output_root / dataset_name / f"{split_name}.jsonl"
        write_jsonl(output_path, normalized)
        output_paths.append(output_path)
    return output_paths


def _apply_subsample_split_plan(
    records: list[dict[str, Any]],
    *,
    dataset_name: str,
    output_root: Path,
    question_field: str | None,
    answer_field: str | None,
    task_id_field: str | None,
    messages_field: str | None,
    source_name: str,
    train_examples: int,
    validation_examples: int,
    test_examples: int,
    sample_seed: int,
) -> list[Path]:
    total_requested = train_examples + validation_examples + test_examples
    if total_requested <= 0:
        raise ValueError("At least one of train_examples, validation_examples, or test_examples must be positive.")
    if len(records) < total_requested:
        raise ValueError(
            f"Requested {total_requested} total examples but only {len(records)} source records are available."
        )

    shuffled = list(records)
    rng = random.Random(sample_seed)
    rng.shuffle(shuffled)

    split_plan = [
        ("train", train_examples),
        ("validation", validation_examples),
        ("test", test_examples),
    ]
    output_paths: list[Path] = []
    cursor = 0
    for split_name, count in split_plan:
        if count <= 0:
            continue
        selected = shuffled[cursor : cursor + count]
        cursor += count
        normalized = [
            normalize_record(
                record,
                dataset_name=dataset_name,
                split_name=split_name,
                row_idx=row_idx,
                question_field=question_field,
                answer_field=answer_field,
                task_id_field=task_id_field,
                messages_field=messages_field,
                source_name=source_name,
            )
            for row_idx, record in enumerate(selected)
        ]
        output_path = output_root / dataset_name / f"{split_name}.jsonl"
        write_jsonl(output_path, normalized)
        output_paths.append(output_path)
    return output_paths


def prepare_from_huggingface(
    *,
    dataset_name: str,
    dataset_id: str,
    splits: list[str],
    output_root: Path,
    question_field: str | None,
    answer_field: str | None,
    task_id_field: str | None,
    messages_field: str | None,
    limit: int | None,
    train_examples: int,
    validation_examples: int,
    test_examples: int,
    sample_seed: int,
) -> list[Path]:
    from datasets import load_dataset

    use_subsample_split_plan = any(value > 0 for value in (train_examples, validation_examples, test_examples))
    if use_subsample_split_plan and len(splits) != 1:
        raise ValueError(
            "Subsampled train/validation/test generation currently expects exactly one source split, "
            "for example --hf_splits train."
        )

    if use_subsample_split_plan:
        dataset = load_dataset(dataset_id, split=splits[0])
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return _apply_subsample_split_plan(
            records=[dict(record) for record in dataset],
            dataset_name=dataset_name,
            output_root=output_root,
            question_field=question_field,
            answer_field=answer_field,
            task_id_field=task_id_field,
            messages_field=messages_field,
            source_name=f"{dataset_id}:{splits[0]}",
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            sample_seed=sample_seed,
        )

    output_paths: list[Path] = []
    for split_name in splits:
        dataset = load_dataset(dataset_id, split=split_name)
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        normalized = [
            normalize_record(
                dict(record),
                dataset_name=dataset_name,
                split_name=split_name,
                row_idx=row_idx,
                question_field=question_field,
                answer_field=answer_field,
                task_id_field=task_id_field,
                messages_field=messages_field,
                source_name=dataset_id,
            )
            for row_idx, record in enumerate(dataset)
        ]
        output_path = output_root / dataset_name / f"{split_name}.jsonl"
        write_jsonl(output_path, normalized)
        output_paths.append(output_path)
    return output_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare normalized prompt datasets for classifer_training. Supports "
            "local DeepScaleR parquet files and Hugging Face datasets such as DAPO-Math-17k."
        )
    )
    parser.add_argument("--dataset_name", required=True, choices=("deepscaler", "dapo_math_17k"))
    parser.add_argument(
        "--source",
        choices=("auto", "local", "huggingface"),
        default="auto",
        help="Use auto to pick local defaults for deepscaler and Hugging Face defaults for dapo_math_17k.",
    )
    parser.add_argument("--input_paths", nargs="*", default=[], help="Local input files (.parquet, .jsonl, .json, .csv, .tsv).")
    parser.add_argument("--hf_dataset_id", default=None)
    parser.add_argument("--hf_splits", nargs="*", default=None, help="Splits to download from Hugging Face. Defaults to train.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--question_field", default=None)
    parser.add_argument("--answer_field", default=None, help="Can be a dotted path such as reward_model.ground_truth.")
    parser.add_argument("--task_id_field", default=None, help="Can be a dotted path such as extra_info.index.")
    parser.add_argument("--messages_field", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train_examples", type=int, default=0, help="Subsample this many train examples from a single source split.")
    parser.add_argument("--validation_examples", type=int, default=0, help="Subsample this many validation examples from a single source split.")
    parser.add_argument("--test_examples", type=int, default=0, help="Subsample this many test examples from a single source split.")
    parser.add_argument("--sample_seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()

    if args.source == "auto":
        if args.dataset_name == "deepscaler":
            source = "local"
        else:
            source = "huggingface"
    else:
        source = args.source

    if source == "local":
        if args.input_paths:
            input_paths = [Path(path).expanduser().resolve() for path in args.input_paths]
        elif args.dataset_name == "deepscaler":
            input_paths = [path.resolve() for path in DEFAULT_DEEPSCALER_FILES.values()]
        else:
            raise ValueError("Local mode requires --input_paths unless dataset_name=deepscaler.")

        output_paths = prepare_from_local_files(
            dataset_name=args.dataset_name,
            input_paths=input_paths,
            output_root=output_root,
            question_field=args.question_field,
            answer_field=args.answer_field,
            task_id_field=args.task_id_field,
            messages_field=args.messages_field,
            limit=args.limit,
            train_examples=args.train_examples,
            validation_examples=args.validation_examples,
            test_examples=args.test_examples,
            sample_seed=args.sample_seed,
        )
    else:
        dataset_id = args.hf_dataset_id or DEFAULT_HF_DATASET_IDS.get(args.dataset_name)
        if not dataset_id:
            raise ValueError(f"No default Hugging Face dataset id is configured for {args.dataset_name}.")
        splits = args.hf_splits or ["train"]
        output_paths = prepare_from_huggingface(
            dataset_name=args.dataset_name,
            dataset_id=dataset_id,
            splits=splits,
            output_root=output_root,
            question_field=args.question_field,
            answer_field=args.answer_field,
            task_id_field=args.task_id_field,
            messages_field=args.messages_field,
            limit=args.limit,
            train_examples=args.train_examples,
            validation_examples=args.validation_examples,
            test_examples=args.test_examples,
            sample_seed=args.sample_seed,
        )

    print("Wrote normalized dataset files:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
