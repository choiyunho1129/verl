from __future__ import annotations

import re
from typing import Any

from classifer_training.utils import (
    coerce_float,
    get_nested_value,
    sanitize_name,
)

_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_THINK_START_TAG_PATTERN = re.compile(r"<think>", re.IGNORECASE)
_THINK_END_TAG_PATTERN = re.compile(r"</think>", re.IGNORECASE)


def _tokenize_whitespace(text: str) -> list[str]:
    return [token for token in re.findall(r"\S+", text.lower()) if token]


def _unique_token_ratio(text: str) -> float | None:
    tokens = _tokenize_whitespace(text)
    if not tokens:
        return None
    return len(set(tokens)) / len(tokens)


def _repetition_ratio(text: str) -> float | None:
    unique_ratio = _unique_token_ratio(text)
    if unique_ratio is None:
        return None
    return 1.0 - unique_ratio


def _repeated_ngram_ratio(text: str, n: int) -> float | None:
    tokens = _tokenize_whitespace(text)
    if len(tokens) < n:
        return None
    ngrams = [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]
    if not ngrams:
        return None
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def _duplicate_line_ratio(text: str) -> float | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return 1.0 - (len(set(lines)) / len(lines))


def _terminal_punctuation_flag(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    return 1.0 if stripped[-1] in ".!?)]}" else 0.0


def _numeric_rollout_features(record: dict[str, Any]) -> dict[str, float]:
    numeric_features: dict[str, float] = {}
    rollout_features = record.get("rollout_features")
    if not isinstance(rollout_features, dict):
        return numeric_features
    for key, value in rollout_features.items():
        numeric = coerce_float(value)
        if numeric is not None:
            numeric_features[str(key)] = numeric
    return numeric_features


def split_reasoning_and_answer(generated_text: str) -> tuple[str, str]:
    spans = extract_response_char_spans(generated_text)
    reasoning_span = spans["reasoning"]
    answer_span = spans["answer"]
    reasoning_content = generated_text[reasoning_span[0] : reasoning_span[1]].strip() if reasoning_span else ""
    answer_content = generated_text[answer_span[0] : answer_span[1]].strip() if answer_span else ""
    return reasoning_content, answer_content


def extract_rollout_numeric_features(
    record: dict[str, Any],
    extra_numeric_fields: list[str] | None = None,
) -> dict[str, float]:
    token_stats = record.get("token_stats") or {}
    generated_text = str(record.get("generated_text", ""))
    reasoning_content = str(record.get("reasoning_content", ""))
    answer_content = str(record.get("answer_content", ""))
    if generated_text and (not reasoning_content.strip() or not answer_content.strip()):
        derived_reasoning, derived_answer = split_reasoning_and_answer(generated_text)
        if not reasoning_content.strip():
            reasoning_content = derived_reasoning
        if not answer_content.strip():
            answer_content = derived_answer

    features: dict[str, float] = {}
    builtins = {
        "input_length": coerce_float(record.get("input_length")),
        "output_length": coerce_float(record.get("output_length")),
        "generation_time": coerce_float(record.get("generation_time")),
        "think_tokens": coerce_float(token_stats.get("think_tokens")),
        "answer_tokens": coerce_float(token_stats.get("answer_tokens")),
        "has_complete_answer": 1.0 if record.get("has_complete_answer") else 0.0,
        "has_reasoning_content": 1.0 if reasoning_content.strip() else 0.0,
        "output_unique_token_ratio": _unique_token_ratio(generated_text),
        "reasoning_unique_token_ratio": _unique_token_ratio(reasoning_content),
        "answer_unique_token_ratio": _unique_token_ratio(answer_content),
        "output_repetition_ratio": _repetition_ratio(generated_text),
        "reasoning_repetition_ratio": _repetition_ratio(reasoning_content),
        "answer_repetition_ratio": _repetition_ratio(answer_content),
        "output_repeated_bigram_ratio": _repeated_ngram_ratio(generated_text, 2),
        "output_repeated_trigram_ratio": _repeated_ngram_ratio(generated_text, 3),
        "reasoning_repeated_bigram_ratio": _repeated_ngram_ratio(reasoning_content, 2),
        "reasoning_repeated_trigram_ratio": _repeated_ngram_ratio(reasoning_content, 3),
        "duplicate_line_ratio": _duplicate_line_ratio(generated_text),
        "answer_terminal_punctuation": _terminal_punctuation_flag(answer_content),
    }
    features.update({key: value for key, value in builtins.items() if value is not None})
    features.update(_numeric_rollout_features(record))

    for field_path in extra_numeric_fields or []:
        numeric = coerce_float(get_nested_value(record, field_path, default=None))
        if numeric is not None:
            features[sanitize_name(field_path)] = numeric

    return features


def _trim_char_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    start = max(start, 0)
    end = min(end, len(text))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return (start, end)


def extract_response_char_spans(generated_text: str) -> dict[str, tuple[int, int] | None]:
    reasoning_match = _THINK_PATTERN.search(generated_text)
    answer_match = _ANSWER_PATTERN.search(generated_text)
    think_start_match = _THINK_START_TAG_PATTERN.search(generated_text)
    think_end_match = _THINK_END_TAG_PATTERN.search(generated_text)

    reasoning_span = _trim_char_span(generated_text, *reasoning_match.span(1)) if reasoning_match else None
    think_start_tag_span = tuple(think_start_match.span(0)) if think_start_match else None
    think_end_tag_span = tuple(think_end_match.span(0)) if think_end_match else None
    if reasoning_span is None and think_start_match:
        reasoning_start = think_start_match.end(0)
        if answer_match:
            reasoning_end = answer_match.start(0)
        elif think_end_match:
            reasoning_end = think_end_match.start(0)
        else:
            reasoning_end = len(generated_text)
        reasoning_span = _trim_char_span(generated_text, reasoning_start, reasoning_end)

    if answer_match:
        answer_span = _trim_char_span(generated_text, *answer_match.span(1))
    elif reasoning_match and "</think>" in generated_text:
        answer_start = generated_text.index("</think>") + len("</think>")
        answer_span = _trim_char_span(generated_text, answer_start, len(generated_text))
    elif think_start_match:
        answer_span = None
    else:
        answer_span = _trim_char_span(generated_text, 0, len(generated_text))

    return {
        "reasoning": reasoning_span,
        "answer": answer_span,
        "think_start_tag": think_start_tag_span,
        "think_end_tag": think_end_tag_span,
    }


def select_response_char_span(
    generated_text: str,
    anchor_mode: str,
) -> tuple[str, tuple[int, int] | None]:
    spans = extract_response_char_spans(generated_text)

    if anchor_mode == "answer":
        return "answer", spans["answer"]
    if anchor_mode == "reasoning":
        return "reasoning", spans["reasoning"]
    if anchor_mode == "reasoning_or_answer":
        if spans["answer"] is not None:
            return "answer", spans["answer"]
        if spans["reasoning"] is not None:
            return "reasoning", spans["reasoning"]
        return "last_generated", _trim_char_span(generated_text, 0, len(generated_text))
    if anchor_mode == "last_generated":
        return "last_generated", _trim_char_span(generated_text, 0, len(generated_text))
    raise ValueError(f"Unsupported anchor mode: {anchor_mode}")
