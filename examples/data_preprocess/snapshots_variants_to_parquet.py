"""
Convert `data/snapshots_variants_train/data.jsonl` into a GRPO-ready parquet,
mirroring the schema used by other math-500 style datasets.

- Variants are ignored; only the top-level question/answer is used.
- No extra instruction is appended to the prompt.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATA_SOURCE = "HuggingFaceH4/MATH-500"


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def strip_boxed_answer(answer: str) -> str:
    """
    Remove a trailing \\boxed{...} if present, otherwise return the raw string.
    Matches math-500 preprocessing.
    """
    if not isinstance(answer, str):
        return ""

    text = answer.strip()
    if "\\boxed" not in text:
        return text

    # Find the boxed portion and strip it.
    start = text.find("\\boxed")
    boxed = text[start:]
    try:
        if boxed.startswith("\\boxed "):
            return boxed[len("\\boxed ") :]
        if boxed.startswith("\\boxed{") and boxed.endswith("}"):
            return boxed[len("\\boxed{") : -1]
    except Exception:
        pass
    return text


def build_rows(data: Iterable[dict], data_source: str) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(data):
        question = item.get("question", "").strip()
        answer_raw = item.get("answer", "")
        ground_truth = strip_boxed_answer(str(answer_raw))

        rows.append(
            {
                "uid": item.get("unique_id", str(idx)),
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "index": idx,
                    "unique_id": item.get("unique_id"),
                    "subject": item.get("subject"),
                    "level": item.get("level"),
                    "answer": answer_raw,
                    "question": question,
                    "snapshot": "original",
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert snapshot JSONL to GRPO parquet (math-500 style).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/snapshots_variants_train/data.jsonl"),
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/snapshots_variants_train/train_grpo.parquet"),
        help="Destination parquet path.",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=DEFAULT_DATA_SOURCE,
        help="Reward data_source tag (math reward-compatible).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(Path(args.input).expanduser())
    rows = build_rows(data=samples, data_source=args.data_source)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
