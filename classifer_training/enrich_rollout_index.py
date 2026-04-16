from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from classifer_training.rollout_utils import extract_rollout_numeric_features

_NONSPACE_RE = re.compile(r"\S+")
_WORD_RE = re.compile(r"\b\w+\b")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")
_FRAC_RE = re.compile(r"\\frac")
_SQRT_RE = re.compile(r"\\sqrt")
_BOXED_RE = re.compile(r"\\boxed")
_BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+")
_SECTION_RE = re.compile(r"^\s*#+|^\s*Step\s+\d+", re.M)
_FINAL_RE = re.compile(r"final\s+answer|answer\s*:", re.I)
_NUMERIC_TAIL_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*$")
_SINGLE_NUM_RE = re.compile(r"^\$?[-+]?\d+(?:\.\d+)?\$?$")
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.I | re.S)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _tokens(text: str) -> list[str]:
    return [token for token in _NONSPACE_RE.findall(text.lower()) if token]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _word_count(text: str) -> int:
    return len(_NONSPACE_RE.findall(text))


def _line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _number_tokens(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _number_token_count(text: str) -> int:
    return len(_number_tokens(text))


def _uppercase_token_ratio(text: str) -> float:
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return 0.0
    uppercase = sum(token.isupper() and any(char.isalpha() for char in token) for token in tokens)
    return uppercase / len(tokens)


def _avg_line_len(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return sum(map(len, lines)) / len(lines)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _text_shape_features(prefix: str, text: str) -> dict[str, float]:
    word_count = _word_count(text)
    number_count = _number_token_count(text)
    boxed_count = len(_BOXED_RE.findall(text))
    return {
        f"{prefix}_char_len": float(len(text)),
        f"{prefix}_word_count": float(word_count),
        f"{prefix}_line_count": float(_line_count(text)),
        f"{prefix}_avg_line_len": float(_avg_line_len(text)),
        f"{prefix}_number_token_count": float(number_count),
        f"{prefix}_number_token_ratio": _ratio(number_count, word_count),
        f"{prefix}_frac_count": float(len(_FRAC_RE.findall(text))),
        f"{prefix}_sqrt_count": float(len(_SQRT_RE.findall(text))),
        f"{prefix}_boxed_count": float(boxed_count),
        f"{prefix}_boxed_flag": 1.0 if boxed_count else 0.0,
        f"{prefix}_equals_count": float(text.count("=")),
        f"{prefix}_dollar_count": float(text.count("$")),
        f"{prefix}_bullet_count": float(len(_BULLET_RE.findall(text))),
        f"{prefix}_section_count": float(len(_SECTION_RE.findall(text))),
        f"{prefix}_paren_count": float(text.count("(") + text.count(")")),
        f"{prefix}_bracket_count": float(text.count("[") + text.count("]") + text.count("{") + text.count("}")),
        f"{prefix}_comma_count": float(text.count(",")),
        f"{prefix}_period_count": float(text.count(".")),
        f"{prefix}_colon_count": float(text.count(":")),
        f"{prefix}_uppercase_token_ratio": float(_uppercase_token_ratio(text)),
        f"{prefix}_suffix_is_numeric": 1.0 if _NUMERIC_TAIL_RE.search(text.strip()[-64:]) else 0.0,
        f"{prefix}_single_number_like": 1.0 if _SINGLE_NUM_RE.fullmatch(text.strip()) else 0.0,
        f"{prefix}_contains_final_answer": 1.0 if _FINAL_RE.search(text) else 0.0,
    }


def _extract_boxed_answers(text: str) -> list[tuple[str, int]]:
    answers: list[tuple[str, int]] = []
    marker = r"\boxed{"
    search_start = 0
    while True:
        boxed_start = text.find(marker, search_start)
        if boxed_start == -1:
            break
        start = boxed_start + len(marker)
        depth = 1
        cursor = start
        while cursor < len(text) and depth > 0:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            candidate = text[start : cursor - 1].strip()
            if candidate:
                answers.append((candidate, boxed_start))
        search_start = cursor
    return answers


def _extract_final_answer(text: str) -> tuple[str, str]:
    candidates: list[tuple[str, int]] = _extract_boxed_answers(text)

    for pattern in (
        re.compile(r"(?is)final answer\s*[:\-]\s*(.+)$"),
        re.compile(r"(?is)answer\s*[:\-]\s*(.+)$"),
        re.compile(r"(?is)答案\s*[:：]\s*(.+)$"),
    ):
        match = pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                candidates.append((candidate, match.start(1)))

    if not candidates:
        return text, text.strip()

    answer, start_idx = candidates[-1]
    reasoning = text[:start_idx].strip()
    return reasoning, answer.strip()


def _single_run_features(row: dict[str, Any]) -> dict[str, float]:
    user_input = str(row.get("user_input", ""))
    generated_text = str(row.get("generated_text", ""))
    reasoning_content = str(row.get("reasoning_content", ""))
    answer_content = str(row.get("answer_content", ""))

    heuristic_reasoning, heuristic_answer = _extract_final_answer(generated_text)
    final_answer = heuristic_answer.strip() or answer_content.strip()

    prompt_tokens = _token_set(user_input)
    output_tokens = _token_set(generated_text)
    reasoning_tokens = _token_set(reasoning_content)
    answer_tokens = _token_set(answer_content)
    final_answer_tokens = _token_set(final_answer)

    prompt_numbers = set(_number_tokens(user_input))
    output_numbers = set(_number_tokens(generated_text))
    reasoning_numbers = set(_number_tokens(reasoning_content))
    answer_numbers = set(_number_tokens(answer_content))
    final_answer_numbers = set(_number_tokens(final_answer))

    output_word_count = _word_count(generated_text)
    answer_word_count = _word_count(answer_content)
    reasoning_word_count = _word_count(reasoning_content)
    final_answer_word_count = _word_count(final_answer)

    features: dict[str, float] = {}
    features.update(_text_shape_features("prompt_text", user_input))
    features.update(_text_shape_features("generated_text", generated_text))
    features.update(_text_shape_features("reasoning_text", reasoning_content))
    features.update(_text_shape_features("answer_text", answer_content))
    features.update(_text_shape_features("final_answer_text", final_answer))

    features.update(
        {
            "answer_over_output_char_ratio": _ratio(len(answer_content), len(generated_text)),
            "reasoning_over_output_char_ratio": _ratio(len(reasoning_content), len(generated_text)),
            "final_answer_over_output_char_ratio": _ratio(len(final_answer), len(generated_text)),
            "answer_over_prompt_char_ratio": _ratio(len(answer_content), len(user_input)),
            "output_over_prompt_char_ratio": _ratio(len(generated_text), len(user_input)),
            "final_answer_over_prompt_char_ratio": _ratio(len(final_answer), len(user_input)),
            "answer_word_over_output_word_ratio": _ratio(answer_word_count, output_word_count),
            "reasoning_word_over_output_word_ratio": _ratio(reasoning_word_count, output_word_count),
            "final_answer_word_over_output_word_ratio": _ratio(final_answer_word_count, output_word_count),
            "answer_num_over_output_num_ratio": _ratio(len(answer_numbers), len(output_numbers)),
            "reasoning_num_over_output_num_ratio": _ratio(len(reasoning_numbers), len(output_numbers)),
            "final_answer_num_over_output_num_ratio": _ratio(len(final_answer_numbers), len(output_numbers)),
            "tokens_per_second": _ratio(float(row.get("output_length", 0.0) or 0.0), float(row.get("generation_time", 0.0) or 0.0)),
            "words_per_second": _ratio(output_word_count, float(row.get("generation_time", 0.0) or 0.0)),
            "chars_per_second": _ratio(len(generated_text), float(row.get("generation_time", 0.0) or 0.0)),
            "output_minus_prompt_char_len": float(len(generated_text) - len(user_input)),
            "answer_minus_prompt_char_len": float(len(answer_content) - len(user_input)),
            "final_answer_minus_prompt_char_len": float(len(final_answer) - len(user_input)),
            "reasoning_empty_and_long_output": 1.0 if (not reasoning_content.strip() and len(generated_text) > 200) else 0.0,
            "prompt_output_token_jaccard": _jaccard(prompt_tokens, output_tokens),
            "prompt_reasoning_token_jaccard": _jaccard(prompt_tokens, reasoning_tokens),
            "prompt_answer_token_jaccard": _jaccard(prompt_tokens, answer_tokens),
            "prompt_final_answer_token_jaccard": _jaccard(prompt_tokens, final_answer_tokens),
            "prompt_output_token_overlap": _overlap_ratio(prompt_tokens, output_tokens),
            "prompt_reasoning_token_overlap": _overlap_ratio(prompt_tokens, reasoning_tokens),
            "prompt_answer_token_overlap": _overlap_ratio(prompt_tokens, answer_tokens),
            "prompt_final_answer_token_overlap": _overlap_ratio(prompt_tokens, final_answer_tokens),
            "prompt_output_number_overlap": _overlap_ratio(prompt_numbers, output_numbers),
            "prompt_reasoning_number_overlap": _overlap_ratio(prompt_numbers, reasoning_numbers),
            "prompt_answer_number_overlap": _overlap_ratio(prompt_numbers, answer_numbers),
            "prompt_final_answer_number_overlap": _overlap_ratio(prompt_numbers, final_answer_numbers),
            "prompt_number_jaccard_output": _jaccard(prompt_numbers, output_numbers),
            "prompt_number_jaccard_answer": _jaccard(prompt_numbers, answer_numbers),
            "prompt_number_jaccard_final_answer": _jaccard(prompt_numbers, final_answer_numbers),
            "final_answer_exists": 1.0 if final_answer else 0.0,
            "final_answer_matches_answer_text": 1.0 if final_answer and final_answer == answer_content.strip() else 0.0,
            "heuristic_reasoning_shorter_than_output": 1.0 if heuristic_reasoning and len(heuristic_reasoning) < len(generated_text) else 0.0,
        }
    )
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess a rollout index with richer single-rollout features.")
    parser.add_argument("--input_index", type=Path, required=True)
    parser.add_argument("--output_index", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, default=None)
    parser.add_argument("--hidden_states_path", type=Path, default=None)
    parser.add_argument("--labels_path", type=Path, default=None)
    parser.add_argument("--dataset_name", type=str, default="dapo_math_17k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_index = args.input_index.expanduser().resolve()
    output_index = args.output_index.expanduser().resolve()
    output_index.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with input_index.open() as src, output_index.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            rollout_features = dict(row.get("rollout_features") or {})
            rollout_features.update(extract_rollout_numeric_features(row))
            rollout_features.update(_single_run_features(row))
            row["rollout_features"] = rollout_features
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    if args.manifest_path is not None and args.hidden_states_path is not None and args.labels_path is not None:
        manifest = {
            "datasets": [
                {
                    "name": args.dataset_name,
                    "hidden_states_path": str(args.hidden_states_path.expanduser().resolve()),
                    "index_path": str(output_index),
                    "labels_path": str(args.labels_path.expanduser().resolve()),
                }
            ]
        }
        args.manifest_path.expanduser().resolve().write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "rows": rows,
                "input_index": str(input_index),
                "output_index": str(output_index),
                "manifest_path": str(args.manifest_path.expanduser().resolve()) if args.manifest_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
