import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_DATA_SOURCE = "HuggingFaceH4/MATH-500"


def remove_boxed(text: str) -> str:
    if "\\boxed " in text:
        left = "\\boxed "
        assert text[: len(left)] == left
        return text[len(left) :]

    left = "\\boxed{"
    assert text[: len(left)] == left
    assert text[-1] == "}"
    return text[len(left) : -1]


def strip_boxed_answer(answer: str) -> str:
    if not isinstance(answer, str):
        return ""

    text = answer.strip()
    if "\\boxed" in text:
        start = text.find("\\boxed")
        boxed = text[start:]
        try:
            return remove_boxed(boxed)
        except Exception:
            return text
    return text


def build_rows(input_path: Path, data_source: str) -> list[dict]:
    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)

            prompt_text = record["problem"]
            answer_raw = record.get("answer", "")
            ground_truth = strip_boxed_answer(answer_raw or record.get("solution", ""))

            rows.append(
                {
                    "uid": record.get("unique_id", str(idx)),
                    "data_source": data_source,
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": ground_truth},
                    "extra_info": {
                        "index": idx,
                        "unique_id": record.get("unique_id"),
                        "subject": record.get("subject"),
                        "level": record.get("level"),
                        "answer": answer_raw,
                        "solution": record.get("solution"),
                        "question": record.get("problem"),
                        "split": "test",
                    },
                }
            )
    return rows


def convert_to_parquet(input_path: Path, output_path: Path, data_source: str) -> None:
    rows = build_rows(input_path, data_source)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Wrote {len(df)} samples to {output_path}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Convert MATH-500 jsonl to Verl parquet format.")
    parser.add_argument("--input", type=Path, default='/data1/home/yunhochoi/verl/data/MATH-500/train.jsonl')
    parser.add_argument("--output", type=Path, default='/data1/home/yunhochoi/verl/data/MATH-500/train.parquet')
    parser.add_argument("--data-source", type=str, default=DEFAULT_DATA_SOURCE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_to_parquet(args.input.expanduser(), args.output.expanduser(), args.data_source)
