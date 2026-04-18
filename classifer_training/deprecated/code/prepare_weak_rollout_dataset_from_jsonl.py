from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from classifer_training.rollout_utils import extract_rollout_numeric_features


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_user_input(raw_input: str) -> str:
    text = str(raw_input)
    prefix = "user\n"
    suffix = "\nassistant\n"
    if text.startswith(prefix) and text.endswith(suffix):
        return text[len(prefix) : -len(suffix)].strip()
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text.strip()


def _split_reasoning_and_answer(generated_text: str) -> tuple[str, str]:
    text = str(generated_text)
    lower = text.lower()
    start = lower.find("<think>")
    end = lower.find("</think>")
    if start != -1 and end != -1 and end > start:
        reasoning = text[start + len("<think>") : end].strip()
        answer = text[end + len("</think>") :].strip()
        return reasoning, answer
    return "", text.strip()


def _count_ws_tokens(text: str) -> int:
    return len([tok for tok in str(text).split() if tok])


def _stable_task_id(user_input: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, user_input))


def _stable_split(task_id: str, val_ratio: float) -> str:
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if value < val_ratio else "train"


def _convert_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    source_name: str,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["input"])].append(row)

    experiment_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for raw_input, samples in grouped.items():
        user_input = _extract_user_input(raw_input)
        task_id = _stable_task_id(user_input)
        split = _stable_split(task_id, val_ratio)
        prompt_rows.append(
            {
                "dataset_name": dataset_name,
                "task_id": task_id,
                "split": split,
                "user_input": user_input,
                "prompt": user_input,
                "messages": [{"role": "user", "content": user_input}],
                "source_file": source_name,
            }
        )
        for sample_idx, row in enumerate(samples):
            generated_text = str(row.get("output", ""))
            reasoning_content, answer_content = _split_reasoning_and_answer(generated_text)
            experiment_rows.append(
                {
                    "dataset_name": dataset_name,
                    "task_id": task_id,
                    "split": split,
                    "user_input": user_input,
                    "messages": [{"role": "user", "content": user_input}],
                    "ground_truth": str(row.get("gts", "")),
                    "generated_text": generated_text,
                    "reasoning_content": reasoning_content,
                    "answer_content": answer_content,
                    "input_length": int(_count_ws_tokens(user_input)),
                    "output_length": int(_count_ws_tokens(generated_text)),
                    "generation_time": 0.0,
                    "has_complete_answer": bool(answer_content.strip()),
                    "token_stats": {
                        "think_tokens": int(_count_ws_tokens(reasoning_content)),
                        "answer_tokens": int(_count_ws_tokens(answer_content)),
                    },
                    "config": {
                        "temperature": None,
                        "backend": "offline_jsonl",
                    },
                    "reward": float(row.get("reward", 0.0)),
                    "score": float(row.get("score", row.get("reward", 0.0))),
                    "pred": str(row.get("pred", "")),
                    "gts": str(row.get("gts", "")),
                    "step": int(row.get("step", 0)),
                    "sample_index": int(sample_idx),
                    "sample_count": int(len(samples)),
                    "source_file": source_name,
                    "raw_input": raw_input,
                }
            )
    prompt_rows.sort(key=lambda row: row["task_id"])
    experiment_rows.sort(key=lambda row: (row["task_id"], row["sample_index"]))
    return experiment_rows, prompt_rows


def _build_label_records_from_experiment_rows(
    *,
    run_dir: Path,
    experiment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in experiment_rows:
        key = (str(row.get("dataset_name", "")), str(row["task_id"]))
        bucket = grouped.setdefault(
            key,
            {
                "dataset_name": str(row.get("dataset_name", "")),
                "task_id": str(row["task_id"]),
                "user_input": row.get("user_input"),
                "correctness": [],
                "temperatures": [],
                "feature_values": defaultdict(list),
            },
        )
        reward = float(row.get("reward", row.get("score", 0.0)))
        bucket["correctness"].append(int(reward >= 0.5))
        temperature = row.get("config", {}).get("temperature") if isinstance(row.get("config"), dict) else None
        if temperature is not None:
            bucket["temperatures"].append(float(temperature))
        for feature_name, feature_value in extract_rollout_numeric_features(row).items():
            bucket["feature_values"][feature_name].append(float(feature_value))

    records: list[dict[str, Any]] = []
    for bucket in grouped.values():
        correctness = np.asarray(bucket["correctness"], dtype=np.float32)
        aggregated_features: dict[str, float] = {}
        for feature_name, values in sorted(bucket["feature_values"].items()):
            values_array = np.asarray(values, dtype=np.float32)
            aggregated_features[f"{feature_name}_mean"] = float(values_array.mean())
            aggregated_features[f"{feature_name}_std"] = float(values_array.std(ddof=0))
            aggregated_features[f"{feature_name}_min"] = float(values_array.min())
            aggregated_features[f"{feature_name}_max"] = float(values_array.max())
        records.append(
            {
                "dataset_name": bucket["dataset_name"],
                "task_id": bucket["task_id"],
                "user_input": bucket["user_input"],
                "num_runs": int(len(correctness)),
                "correct_count": int(correctness.sum()),
                "wrong_count": int(len(correctness) - correctness.sum()),
                "sampling_accuracy": float(correctness.mean()) if len(correctness) else 0.0,
                "difficulty": float(1.0 - correctness.mean()) if len(correctness) else 1.0,
                "temperatures": sorted({round(temp, 8) for temp in bucket["temperatures"]}),
                "aggregated_features": aggregated_features,
                "source_run_dirs": [str(run_dir)],
            }
        )
    records.sort(key=lambda row: row["task_id"])
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert sampled JSONL shards into resume-friendly pseudo run dirs.")
    parser.add_argument("--input_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--dataset_name", default="dapo_math_17k_weak4")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs: list[Path] = []
    all_prompt_rows: dict[str, dict[str, Any]] = {}
    summary_inputs: list[dict[str, Any]] = []
    all_label_records: dict[tuple[str, str], dict[str, Any]] = {}

    for input_path in args.input_paths:
        rows = _load_jsonl(input_path.expanduser().resolve())
        run_dir = args.run_root.expanduser().resolve() / input_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        experiment_rows, prompt_rows = _convert_rows(
            rows,
            dataset_name=args.dataset_name,
            source_name=input_path.name,
            val_ratio=float(args.val_ratio),
        )
        _write_jsonl(run_dir / "all_experiments.jsonl", experiment_rows)
        correctness = [int(float(row["reward"]) >= 0.5) for row in experiment_rows]
        evaluation_rows = [
            {
                "dataset_name": args.dataset_name,
                "num_examples": int(len(experiment_rows)),
                "accuracy": float(sum(correctness) / max(len(correctness), 1)),
                "correctness": correctness,
                "config": {"backend": "offline_jsonl", "source_file": input_path.name},
            }
        ]
        _write_jsonl(run_dir / "evaluation_results.jsonl", evaluation_rows)
        run_dirs.append(run_dir)
        for row in prompt_rows:
            all_prompt_rows.setdefault(row["task_id"], row)
        for label_row in _build_label_records_from_experiment_rows(run_dir=run_dir, experiment_rows=experiment_rows):
            all_label_records[(str(label_row["dataset_name"]), str(label_row["task_id"]))] = label_row
        summary_inputs.append(
            {
                "input_path": str(input_path.expanduser().resolve()),
                "run_dir": str(run_dir),
                "num_rows": int(len(experiment_rows)),
                "num_prompts": int(len(prompt_rows)),
                "accuracy": float(sum(correctness) / max(len(correctness), 1)),
            }
        )

    label_records = sorted(all_label_records.values(), key=lambda row: (str(row["dataset_name"]), str(row["task_id"])))
    _write_jsonl(args.labels_path.expanduser().resolve(), label_records)

    prompt_dataset_dir = args.prompt_dataset_dir.expanduser().resolve()
    prompt_dataset_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows_sorted = sorted(all_prompt_rows.values(), key=lambda row: row["task_id"])
    train_rows = [row for row in prompt_rows_sorted if row["split"] == "train"]
    val_rows = [row for row in prompt_rows_sorted if row["split"] == "validation"]
    _write_jsonl(prompt_dataset_dir / "train.jsonl", train_rows)
    _write_jsonl(prompt_dataset_dir / "validation.jsonl", val_rows)

    summary = {
        "dataset_name": args.dataset_name,
        "num_input_files": int(len(args.input_paths)),
        "num_run_dirs": int(len(run_dirs)),
        "num_prompts_total": int(len(prompt_rows_sorted)),
        "num_prompts_train": int(len(train_rows)),
        "num_prompts_validation": int(len(val_rows)),
        "num_label_records": int(len(label_records)),
        "run_dirs": [str(path) for path in run_dirs],
        "inputs": summary_inputs,
        "prompt_dataset_dir": str(prompt_dataset_dir),
        "labels_path": str(args.labels_path.expanduser().resolve()),
    }
    args.summary_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.expanduser().resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
