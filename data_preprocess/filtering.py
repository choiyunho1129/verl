#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Set

WS_RE = re.compile(r"\s+")


def normalize_question(q: str) -> str:
    """
    Normalize to make matching robust to whitespace/newlines and casing.
    - collapse all whitespace to single spaces
    - strip edges
    - lowercase
    """
    if q is None:
        return ""
    q = str(q).replace("\r\n", "\n").replace("\r", "\n")
    q = WS_RE.sub(" ", q).strip().lower()
    return q


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def main():
    ap = argparse.ArgumentParser(
        description="Remove records from A.jsonl whose question overlaps with any question in B.jsonl."
    )
    ap.add_argument("--a", type=str, required=True, help="First JSONL (to be filtered)")
    ap.add_argument("--b", type=str, required=True, help="Second JSONL (reference for deletion)")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument(
        "--question-key",
        type=str,
        default="question",
        help="Field name that holds the question text (default: question)",
    )
    ap.add_argument(
        "--keep-missing-question",
        action="store_true",
        help="If set, keep records that do not have the question field (default: drop them).",
    )
    args = ap.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    out_path = Path(args.out)
    qkey = args.question_key

    # 1) Build reference set from B
    b_questions: Set[str] = set()
    b_total = 0
    b_missing = 0
    for r in iter_jsonl(b_path):
        b_total += 1
        q = r.get(qkey)
        if q is None:
            b_missing += 1
            continue
        b_questions.add(normalize_question(q))

    # 2) Stream A, write only non-overlapping
    out_path.parent.mkdir(parents=True, exist_ok=True)

    a_total = 0
    removed = 0
    kept = 0
    a_missing = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for r in iter_jsonl(a_path):
            a_total += 1
            q = r.get(qkey)
            if q is None:
                a_missing += 1
                if args.keep_missing_question:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    removed += 1
                continue

            nq = normalize_question(q)
            if nq in b_questions:
                removed += 1
                continue

            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"[B] total={b_total}, missing_question={b_missing}, unique_norm_questions={len(b_questions)}\n"
        f"[A] total={a_total}, missing_question={a_missing}, kept={kept}, removed={removed}\n"
        f"Output -> {out_path}"
    )


if __name__ == "__main__":
    main()