from __future__ import annotations

import json
from pathlib import Path

import torch

from classifer_training.aggregate_labels import main as aggregate_main
from classifer_training.train import main as train_main
from classifer_training.utils import load_records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_run_dir(base_dir: Path, run_name: str, correctness: list[int], output_lengths: list[int]) -> Path:
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows = []
    for idx, output_length in enumerate(output_lengths):
        experiment_rows.append(
            {
                "dataset_name": "toyset",
                "task_id": str(idx),
                "user_input": f"prompt {idx}",
                "generated_text": "alpha beta beta gamma" if idx % 2 == 0 else "alpha alpha alpha",
                "reasoning_content": "step one step two",
                "answer_content": "final answer",
                "input_length": 10 + idx,
                "output_length": output_length,
                "generation_time": 0.5 + idx * 0.1,
                "has_complete_answer": True,
                "token_stats": {
                    "think_tokens": output_length - 2,
                    "answer_tokens": 2,
                },
                "config": {
                    "temperature": 0.7,
                },
            }
        )

    evaluation_row = {
        "correctness": correctness,
    }
    _write_jsonl(run_dir / "all_experiments.jsonl", experiment_rows)
    _write_jsonl(run_dir / "evaluation_results.jsonl", [evaluation_row])
    return run_dir


def _make_hidden_states(hidden_states_path: Path, num_examples: int) -> None:
    rows = []
    target_signal = [0.0, 0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0, 1.0]
    for idx in range(num_examples):
        signal = target_signal[idx]
        rows.append(
            {
                "ffn": [
                    torch.tensor([[signal, signal * 2, 1.0, 0.0]], dtype=torch.float32),
                    torch.tensor([[signal * 3, signal * 4, 0.0, 1.0]], dtype=torch.float32),
                    torch.tensor([[signal * 5, signal * 6, 1.0, 1.0]], dtype=torch.float32),
                ],
                "attn": [
                    torch.tensor([[idx, idx + 1, 0.0, 1.0]], dtype=torch.float32),
                    torch.tensor([[idx + 2, idx + 3, 1.0, 0.0]], dtype=torch.float32),
                    torch.tensor([[idx + 4, idx + 5, 1.0, 1.0]], dtype=torch.float32),
                ],
            }
        )
    torch.save(rows, hidden_states_path)


def _make_manifest(manifest_path: Path, hidden_states_path: Path, index_path: Path, labels_path: Path) -> None:
    payload = {
        "datasets": [
            {
                "name": "toyset",
                "hidden_states_path": str(hidden_states_path),
                "index_path": str(index_path),
                "labels_path": str(labels_path),
            }
        ]
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_index_with_splits(source_index_path: Path, output_path: Path) -> Path:
    rows = load_records(source_index_path)
    split_plan = ["train", "validation", "train", "validation", "train", "test", "train", "test"]
    patched_rows = []
    for row, split_name in zip(rows, split_plan):
        patched = dict(row)
        patched["split"] = split_name
        patched_rows.append(patched)
    _write_jsonl(output_path, patched_rows)
    return output_path


def test_end_to_end_aggregation_and_training(tmp_path: Path) -> None:
    run1 = _make_run_dir(tmp_path, "run1", [0, 0, 0, 0, 1, 1, 1, 1], [20, 21, 22, 23, 24, 25, 26, 27])
    run2 = _make_run_dir(tmp_path, "run2", [0, 0, 1, 0, 1, 1, 1, 1], [21, 22, 23, 24, 25, 26, 27, 28])
    run3 = _make_run_dir(tmp_path, "run3", [0, 0, 0, 1, 0, 1, 1, 1], [22, 23, 24, 25, 26, 27, 28, 29])

    labels_path = tmp_path / "sampling_labels.jsonl"
    aggregate_main(
        [
            "--run_dirs",
            str(run1),
            str(run2),
            str(run3),
            "--output_path",
            str(labels_path),
        ]
    )

    label_rows = load_records(labels_path)
    assert len(label_rows) == 8
    assert abs(label_rows[0]["sampling_accuracy"] - 0.0) < 1e-8
    assert abs(label_rows[2]["sampling_accuracy"] - (1.0 / 3.0)) < 1e-6
    assert "output_length_mean" in label_rows[0]["aggregated_features"]
    assert "output_text_entropy_mean" in label_rows[0]["aggregated_features"]

    hidden_states_path = tmp_path / "hiddenStates.pt"
    _make_hidden_states(hidden_states_path, num_examples=8)

    manifest_path = tmp_path / "manifest.json"
    _make_manifest(
        manifest_path=manifest_path,
        hidden_states_path=hidden_states_path,
        index_path=run1 / "all_experiments.jsonl",
        labels_path=labels_path,
    )

    regression_out = tmp_path / "regression_out"
    train_main(
        [
            "--manifest",
            str(manifest_path),
            "--output_dir",
            str(regression_out),
            "--task_type",
            "regression",
            "--target_field",
            "sampling_accuracy",
            "--model",
            "ridge",
            "--components",
            "ffn",
            "--layers",
            "0:2",
            "--component_pooling",
            "concat",
            "--extra_features",
            "label.aggregated_features.output_length_mean",
            "--test_size",
            "0.25",
            "--random_state",
            "0",
        ]
    )
    regression_metrics = json.loads((regression_out / "metrics.json").read_text(encoding="utf-8"))
    assert regression_metrics["task_type"] == "regression"
    assert regression_metrics["num_features"] > 0
    assert "rmse" in regression_metrics["metrics"]

    classification_out = tmp_path / "classification_out"
    train_main(
        [
            "--manifest",
            str(manifest_path),
            "--output_dir",
            str(classification_out),
            "--task_type",
            "classification",
            "--target_field",
            "sampling_accuracy",
            "--classification_threshold",
            "0.5",
            "--model",
            "logistic",
            "--components",
            "ffn",
            "--layers",
            "1:2",
            "--component_pooling",
            "mean",
            "--extra_features",
            "label.aggregated_features.output_length_mean",
            "--test_size",
            "0.25",
            "--random_state",
            "0",
        ]
    )
    classification_metrics = json.loads((classification_out / "metrics.json").read_text(encoding="utf-8"))
    assert classification_metrics["task_type"] == "classification"
    assert "accuracy" in classification_metrics["metrics"]
    assert (classification_out / "model.joblib").exists()
    assert (classification_out / "predictions.jsonl").exists()


def test_training_with_predefined_splits(tmp_path: Path) -> None:
    run1 = _make_run_dir(tmp_path, "run1", [0, 0, 0, 0, 1, 1, 1, 1], [20, 21, 22, 23, 24, 25, 26, 27])
    run2 = _make_run_dir(tmp_path, "run2", [0, 0, 1, 0, 1, 1, 1, 1], [21, 22, 23, 24, 25, 26, 27, 28])
    run3 = _make_run_dir(tmp_path, "run3", [0, 0, 0, 1, 0, 1, 1, 1], [22, 23, 24, 25, 26, 27, 28, 29])

    labels_path = tmp_path / "sampling_labels.jsonl"
    aggregate_main(
        [
            "--run_dirs",
            str(run1),
            str(run2),
            str(run3),
            "--output_path",
            str(labels_path),
        ]
    )

    hidden_states_path = tmp_path / "hiddenStates.pt"
    _make_hidden_states(hidden_states_path, num_examples=8)
    index_path = _make_index_with_splits(run1 / "all_experiments.jsonl", tmp_path / "index_with_splits.jsonl")

    manifest_path = tmp_path / "manifest.json"
    _make_manifest(
        manifest_path=manifest_path,
        hidden_states_path=hidden_states_path,
        index_path=index_path,
        labels_path=labels_path,
    )

    split_out = tmp_path / "split_out"
    train_main(
        [
            "--manifest",
            str(manifest_path),
            "--output_dir",
            str(split_out),
            "--task_type",
            "classification",
            "--target_field",
            "sampling_accuracy",
            "--classification_threshold",
            "0.5",
            "--train_splits",
            "train",
            "--eval_splits",
            "validation",
            "--test_splits",
            "test",
            "--model",
            "logistic",
            "--components",
            "ffn",
            "--layers",
            "1:2",
            "--component_pooling",
            "mean",
            "--extra_features",
            "label.aggregated_features.output_length_mean",
        ]
    )
    payload = json.loads((split_out / "metrics.json").read_text(encoding="utf-8"))
    assert payload["train_splits"] == ["train"]
    assert payload["eval_splits"] == ["validation"]
    assert payload["test_splits"] == ["test"]
    assert payload["num_train"] == 4
    assert payload["num_eval"] == 2
    assert payload["num_test"] == 2
    assert "validation" in payload["metrics_by_split"]
    assert "test" in payload["metrics_by_split"]
