import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_WARNED_MATH_VERIFY_UNAVAILABLE = False


def _normalize_math_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("\n", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\$", "")
    normalized = normalized.replace("tfrac", "frac").replace("dfrac", "frac")
    normalized = normalized.replace(" ", "")
    return normalized


def _extract_last_boxed_content(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    # Handle "\boxed x" format.
    prefix = "\\boxed "
    if text.startswith(prefix, idx):
        return text[idx + len(prefix) :].strip().split("$")[0].strip()

    # Handle "\boxed{...}" format with brace matching.
    open_idx = text.find("{", idx)
    if open_idx < 0:
        return None
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i].strip()
    return None


def _fallback_rule_score(trajectory: str, answer: str) -> float:
    boxed = _extract_last_boxed_content(trajectory)
    if boxed is None:
        return 0.0

    pred = _normalize_math_text(boxed)
    gt = _normalize_math_text(answer)
    return 1.0 if pred == gt and pred != "" else 0.0


def _compute_math_verify_score(trajectory: str, answer: str) -> Optional[float]:
    global _WARNED_MATH_VERIFY_UNAVAILABLE
    try:
        from math_verify.errors import TimeoutException
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except Exception:
        if not _WARNED_MATH_VERIFY_UNAVAILABLE:
            print(
                "[warn] math_verify is not available. "
                "Falling back to boxed-answer exact matching for correctness labels."
            )
            _WARNED_MATH_VERIFY_UNAVAILABLE = True
        return _fallback_rule_score(trajectory, answer)

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ground_truth_boxed = "\\boxed{" + answer + "}"
    try:
        score, _ = verify_func([ground_truth_boxed], [trajectory])
        return float(score)
    except TimeoutException:
        return 0.0
    except Exception:
        return 0.0


VERIFIER_SYSTEM_PROMPT = (
    "You are a strict math solution verifier. "
    "Given a math question and a solution trajectory, decide whether the solution appears logically valid "
    "and the final answer is correct. "
    "Verify the trajectory step by step then provide a final verdict in the exact format: "
    "Final Verdict: CORRECT or Final Verdict: INCORRECT."
)


def _content_to_text(content: Any) -> Optional[str]:
    if isinstance(content, str):
        text = content.strip()
        return text if text else None

    if isinstance(content, list):
        parts: List[str] = []
        for segment in content:
            if isinstance(segment, dict):
                if segment.get("type") == "text":
                    text = str(segment.get("text", "")).strip()
                elif "content" in segment:
                    text = str(segment.get("content", "")).strip()
                elif "text" in segment:
                    text = str(segment.get("text", "")).strip()
                else:
                    text = ""
            else:
                text = str(segment).strip()
            if text:
                parts.append(text)
        merged = " ".join(parts).strip()
        return merged if merged else None

    if content is None:
        return None

    text = str(content).strip()
    return text if text else None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "correct"}:
            return True
        if normalized in {"false", "0", "no", "n", "incorrect"}:
            return False
    return None


def _extract_question(item: Dict[str, Any]) -> Optional[str]:
    for key in ("question", "problem", "instruction", "input"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    extra_info = item.get("extra_info")
    if isinstance(extra_info, dict):
        for key in ("question", "problem", "instruction", "input"):
            value = extra_info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("prompt", "messages"):
        messages = item.get(key)
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            content = msg.get("content")
            text = _content_to_text(content)
            if role == "user" and text:
                return text
    return None


def _resolve_actual_correct(item: Dict[str, Any], trajectory: str, answer: Optional[str]) -> Optional[bool]:
    for key in ("actual_correct", "verified_correct", "trajectory_is_correct"):
        resolved = _to_bool(item.get(key))
        if resolved is not None:
            return resolved

    extra_info = item.get("extra_info")
    if isinstance(extra_info, dict):
        for key in ("actual_correct", "verified_correct", "trajectory_is_correct"):
            resolved = _to_bool(extra_info.get(key))
            if resolved is not None:
                return resolved

    if answer is None:
        return None

    score = _compute_math_verify_score(trajectory=trajectory, answer=answer)
    if score is None:
        return None
    return score > 0.0


def _extract_answer(item: Dict[str, Any]) -> Optional[str]:
    value = item.get("answer")
    text = _content_to_text(value)
    if text:
        return text

    extra_info = item.get("extra_info")
    if isinstance(extra_info, dict):
        text = _content_to_text(extra_info.get("answer"))
        if text:
            return text

    reward_model = item.get("reward_model")
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except Exception:
            reward_model = None
    if isinstance(reward_model, dict):
        text = _content_to_text(reward_model.get("ground_truth"))
        if text:
            return text

    text = _content_to_text(item.get("reward_model_data"))
    if text:
        return text

    return None


def _build_user_prompt(question: str, trajectory: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Solution Trajectory:\n{trajectory}\n\n"
    )


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_rows(rows: List[Dict[str, Any]], parquet_path: Path, jsonl_path: Optional[Path]) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path)
    if jsonl_path is not None:
        _write_jsonl(jsonl_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/data01/yunhochoi/verl/data/DeepMath-103K/validataion_1k_Qwen3_1.7B_trajectories_nothink.jsonl",
        help="Trajectory JSONL path generated from trajectory_generation.py.",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="/data01/yunhochoi/verl/data/DeepMath-103K/validation_1k_qwen3_1.7B_nothink_trajectory_verification.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="/data01/yunhochoi/verl/data/DeepMath-103K/validation_1k_qwen3_1.7B_nothink_trajectory_verification.jsonl",
        help="Optional JSONL dump path.",
    )
    parser.add_argument("--data-source", type=str, default="qwen3_1.7B_verification")
    parser.add_argument("--system-prompt", type=str, default=VERIFIER_SYSTEM_PROMPT)
    args = parser.parse_args()

    if "verification" not in args.data_source.strip().lower():
        print(
            f"[warn] data_source={args.data_source!r} does not include 'verification'. "
            "Use an explicit task name so unified reward routing is unambiguous."
        )

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    rows: List[Dict[str, Any]] = []
    skipped = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            question = _extract_question(item)
            trajectory = item.get("trajectory")
            answer = _extract_answer(item)
            if question is None or trajectory is None:
                skipped += 1
                continue

            question_str = str(question)
            trajectory_str = str(trajectory)
            answer_str = answer

            actual_correct = _resolve_actual_correct(item, trajectory_str, answer_str)
            if actual_correct is None:
                skipped += 1
                continue

            meta = {
                "question": question_str,
                "trajectory": trajectory_str,
                "answer": answer_str,
                "actual_correct": actual_correct,
                "source_unique_id": item.get("unique_id") or item.get("uid"),
            }
            ground_truth = json.dumps(meta, ensure_ascii=False)

            row = {
                "data_source": args.data_source,
                "prompt": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": _build_user_prompt(question_str, trajectory_str)},
                ],
                "response": "",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "reward_model_data": ground_truth,
                "extra_info": {"actual_correct": actual_correct},
                "source_unique_id": item.get("unique_id") or item.get("uid"),
            }

            rows.append(row)

    if not rows:
        raise ValueError("No valid rows found. Ensure input has question/prompt, trajectory, and answer.")

    output_parquet = Path(args.output_parquet).expanduser()
    output_jsonl = Path(args.output_jsonl).expanduser() if args.output_jsonl else None
    _save_rows(rows, output_parquet, output_jsonl)

    print(f"Loaded rows: {len(rows)} (skipped: {skipped})")
    print(f"Saved rows: {len(rows)} -> {output_parquet}")
    if output_jsonl is not None:
        print(f"Saved JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
