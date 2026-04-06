from __future__ import annotations

from classifer_training.extract_rollout_hidden_states import (
    _compute_token_level_confidence_features,
    _last_token_index_before_char,
    _last_token_index_for_span,
)
from classifer_training.rollout_utils import (
    extract_response_char_spans,
    extract_rollout_numeric_features,
    select_response_char_span,
)


def test_extract_response_char_spans_and_anchor_selection() -> None:
    text = "<think>step one step two</think><answer>\\boxed{4}</answer>"
    spans = extract_response_char_spans(text)
    assert spans["reasoning"] is not None
    assert spans["answer"] is not None
    assert text[slice(*spans["reasoning"])] == "step one step two"
    assert text[slice(*spans["answer"])] == "\\boxed{4}"

    anchor_kind, anchor_span = select_response_char_span(text, "reasoning_or_answer")
    assert anchor_kind == "answer"
    assert anchor_span == spans["answer"]


def test_extract_rollout_numeric_features_includes_single_rollout_stats() -> None:
    record = {
        "input_length": 12,
        "output_length": 5,
        "generation_time": 0.25,
        "generated_text": "alpha beta beta gamma",
        "reasoning_content": "step one step two",
        "answer_content": "final answer",
        "has_complete_answer": True,
        "token_stats": {
            "think_tokens": 3,
            "answer_tokens": 2,
        },
        "config": {
            "temperature": 0.7,
        },
    }
    features = extract_rollout_numeric_features(record, ["config.temperature"])
    assert features["input_length"] == 12.0
    assert features["output_length"] == 5.0
    assert features["think_tokens"] == 3.0
    assert features["answer_tokens"] == 2.0
    assert features["has_complete_answer"] == 1.0
    assert features["has_reasoning_content"] == 1.0
    assert features["config_temperature"] == 0.7
    assert features["output_text_entropy"] > 0.0
    assert features["reasoning_text_entropy"] > 0.0
    assert features["output_repetition_ratio"] > 0.0
    assert features["output_repeated_bigram_ratio"] >= 0.0


def test_token_index_helpers_pick_last_matching_token() -> None:
    offsets = [
        (0, 5),
        (6, 10),
        (11, 16),
        (17, 22),
    ]
    assert _last_token_index_before_char(offsets, 10) == 1
    assert _last_token_index_for_span(offsets, 6, 16) == 2


def test_confidence_feature_helper_returns_logprob_entropy_and_margin() -> None:
    import torch

    logits = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    features = _compute_token_level_confidence_features(
        logits_row=logits,
        input_ids_row=input_ids,
        token_indices=[1, 2, 3],
        prefix="output",
    )
    assert "output_mean_logprob" in features
    assert "output_last_token_entropy" in features
    assert "output_last_token_margin" in features
