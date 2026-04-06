from __future__ import annotations

from pathlib import Path

from classifer_training.aggregate_labels import load_run_examples
from classifer_training.sample import _build_experiment_row, _split_reasoning_and_answer
from classifer_training.utils import write_jsonl


class _WhitespaceTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": text.split()}


def test_split_reasoning_and_answer_parses_tagged_output() -> None:
    reasoning, answer = _split_reasoning_and_answer(
        "<think>\nstep one\nstep two\n</think>\n<answer>\\boxed{4}</answer>"
    )
    assert reasoning == "step one\nstep two"
    assert answer == "\\boxed{4}"


def test_sample_row_matches_aggregate_input_contract(tmp_path: Path) -> None:
    tokenizer = _WhitespaceTokenizer()
    config = {
        "model_name_or_path": "demo/model",
        "model_slug": "demo_model",
        "backend": "transformers",
        "grader": "exact",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_new_tokens": 32,
        "seed": 1,
    }
    record = {
        "dataset_name": "toyset",
        "task_id": "42",
        "split": "validation",
        "user_input": "What is 2 + 2?",
        "ground_truth": "\\boxed{4}",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    }

    experiment_row, correctness = _build_experiment_row(
        record=record,
        dataset_name="toyset",
        config=config,
        tokenizer=tokenizer,
        generated_text="<think>2+2=4</think><answer>\\boxed{4}</answer>",
        input_length=12,
        output_length=5,
        generation_time=0.25,
    )

    run_dir = tmp_path / "run"
    write_jsonl(run_dir / "all_experiments.jsonl", [experiment_row])
    write_jsonl(
        run_dir / "evaluation_results.jsonl",
        [
            {
                "dataset_name": "toyset",
                "num_examples": 1,
                "accuracy": float(correctness),
                "correctness": [correctness],
                "config": config,
            }
        ],
    )

    rows = load_run_examples(run_dir, extra_numeric_fields=[])
    assert len(rows) == 1
    assert rows[0]["dataset_name"] == "toyset"
    assert rows[0]["task_id"] == "42"
    assert rows[0]["correct"] == 1
    assert abs(rows[0]["temperature"] - 0.7) < 1e-8
