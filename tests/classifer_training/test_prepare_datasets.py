from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from classifer_training.prepare_datasets import main as prepare_datasets_main
from classifer_training.utils import load_records


def test_prepare_datasets_from_local_generic_jsonl(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "train.jsonl"
    rows = [
        {"question": "2+2?", "answer": "4", "id": "a"},
        {"question": "3+5?", "answer": "8", "id": "b"},
    ]
    with input_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    output_root = tmp_path / "datasets"
    prepare_datasets_main(
        [
            "--dataset_name",
            "dapo_math_17k",
            "--source",
            "local",
            "--input_paths",
            str(input_path),
            "--question_field",
            "question",
            "--answer_field",
            "answer",
            "--task_id_field",
            "id",
            "--output_root",
            str(output_root),
        ]
    )

    output_rows = load_records(output_root / "dapo_math_17k" / "train.jsonl")
    assert len(output_rows) == 2
    assert output_rows[0]["task_id"] == "a"
    assert output_rows[0]["user_input"] == "2+2?"
    assert output_rows[0]["ground_truth"] == "4"
    assert output_rows[0]["messages"] == [{"role": "user", "content": "2+2?"}]


def test_prepare_datasets_from_deepscaler_style_parquet(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "train_deepscaler.parquet"
    df = pd.DataFrame(
        [
            {
                "data_source": "deepscaler",
                "prompt": [{"role": "user", "content": "What is 1+1?"}],
                "reward_model": {"ground_truth": "2", "style": "rule"},
                "extra_info": {"index": 7, "split": "train"},
            }
        ]
    )
    df.to_parquet(input_path, index=False)

    output_root = tmp_path / "datasets"
    prepare_datasets_main(
        [
            "--dataset_name",
            "deepscaler",
            "--source",
            "local",
            "--input_paths",
            str(input_path),
            "--output_root",
            str(output_root),
        ]
    )

    output_rows = load_records(output_root / "deepscaler" / "train.jsonl")
    assert len(output_rows) == 1
    assert output_rows[0]["task_id"] == "7"
    assert output_rows[0]["ground_truth"] == "2"
    assert output_rows[0]["user_input"] == "What is 1+1?"


def test_prepare_datasets_can_subsample_and_split_from_single_source(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "source.jsonl"
    rows = [{"question": f"q{i}", "answer": f"a{i}", "id": f"id{i}"} for i in range(10)]
    with input_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    output_root = tmp_path / "datasets"
    prepare_datasets_main(
        [
            "--dataset_name",
            "dapo_math_17k",
            "--source",
            "local",
            "--input_paths",
            str(input_path),
            "--question_field",
            "question",
            "--answer_field",
            "answer",
            "--task_id_field",
            "id",
            "--output_root",
            str(output_root),
            "--train_examples",
            "4",
            "--validation_examples",
            "3",
            "--test_examples",
            "2",
            "--sample_seed",
            "7",
        ]
    )

    assert len(load_records(output_root / "dapo_math_17k" / "train.jsonl")) == 4
    assert len(load_records(output_root / "dapo_math_17k" / "validation.jsonl")) == 3
    assert len(load_records(output_root / "dapo_math_17k" / "test.jsonl")) == 2


def test_prepare_datasets_handles_open_r1_processed_schema(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "open_r1.jsonl"
    rows = [
        {
            "prompt": "What is 2+2?",
            "solution": "4",
            "source_prompt": [{"role": "user", "content": "What is 2+2?"}],
            "reward_model": {"ground_truth": "4"},
            "extra_info": {"index": "uuid-1"},
        }
    ]
    with input_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    output_root = tmp_path / "datasets"
    prepare_datasets_main(
        [
            "--dataset_name",
            "dapo_math_17k",
            "--source",
            "local",
            "--input_paths",
            str(input_path),
            "--output_root",
            str(output_root),
        ]
    )

    output_rows = load_records(output_root / "dapo_math_17k" / "train.jsonl")
    assert len(output_rows) == 1
    assert output_rows[0]["task_id"] == "uuid-1"
    assert output_rows[0]["user_input"] == "What is 2+2?"
    assert output_rows[0]["ground_truth"] == "4"
    assert output_rows[0]["messages"] == [{"role": "user", "content": "What is 2+2?"}]
