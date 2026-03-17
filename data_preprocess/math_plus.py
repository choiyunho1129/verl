#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


WS_RE = re.compile(r"\s+")


def norm_text(s: str) -> str:
    # overlap 판단용: 공백 정규화 + lower
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = WS_RE.sub(" ", s).strip().lower()
    return s


def parse_id_mathplus(id_str: str) -> Tuple[str, Optional[int]]:
    """
    MATHplus raw ids: "<base>-0|1|2" e.g. "1018-0", "0-2"  [oai_citation:3‡Hugging Face](https://huggingface.co/datasets/flagopen/MATHplus/raw/main/mathplus.jsonl)
    Returns (base_id, variant_idx).
    """
    parts = str(id_str).split("-")
    if len(parts) >= 2 and parts[-1].isdigit():
        return ("-".join(parts[:-1]), int(parts[-1]))
    return (str(id_str), None)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build merged JSONL (question=variant0, variants snapshots=0/1/2) from "
            "flagopen/MATHplus(train), dropping any group overlapping with HuggingFaceH4/MATH-500(test)."
        )
    )
    ap.add_argument("--mathplus-repo", type=str, default="flagopen/MATHplus")
    ap.add_argument("--mathplus-split", type=str, default="train")
    ap.add_argument("--math500-repo", type=str, default="HuggingFaceH4/MATH-500")
    ap.add_argument("--math500-split", type=str, default="test")
    ap.add_argument("--out", type=str, default="data/math_plus/math_plus.jsonl")

    # optional metadata fields (없으면 비워도 된다고 해서 defaults를 둠)
    ap.add_argument("--subject", type=str, default="")
    ap.add_argument("--level", type=int, default=1)

    args = ap.parse_args()

    # 1) Load MATH-500 test and build a normalized problem set  [oai_citation:4‡Hugging Face](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
    math500 = load_dataset(args.math500_repo, split=args.math500_split)
    math500_problem_set = {norm_text(r.get("problem")) for r in math500 if r.get("problem")}

    # 2) Load MATHplus train  [oai_citation:5‡Hugging Face](https://huggingface.co/datasets/flagopen/MATHplus)
    mathplus = load_dataset(args.mathplus_repo, split=args.mathplus_split)

    # 3) Group by base_id, store variants by variant_idx; also mark contaminated groups
    grouped: Dict[str, Dict[int, Dict[str, str]]] = defaultdict(dict)
    contaminated: set[str] = set()

    total_rows = 0
    for r in mathplus:
        total_rows += 1
        rid = r.get("id")
        prob = r.get("problem")
        sol = r.get("solution")

        if rid is None or prob is None:
            continue

        base_id, var_idx = parse_id_mathplus(rid)
        if var_idx is None:
            # 기대 포맷이 아니면 스킵(또는 여기에 다른 규칙 추가)
            continue

        prob_norm = norm_text(prob)
        if prob_norm in math500_problem_set:
            contaminated.add(base_id)

        grouped[base_id][var_idx] = {
            "id": str(rid),
            "problem": str(prob),
            "solution": "" if sol is None else str(sol),
        }

    # 4) Build merged rows: keep only non-contaminated groups that have variant 0
    merged: List[Dict[str, Any]] = []
    dropped_groups = 0
    missing_v0 = 0

    for base_id in sorted(grouped.keys(), key=lambda x: (len(x), x)):
        if base_id in contaminated:
            dropped_groups += 1
            continue

        variants_map = grouped[base_id]
        if 0 not in variants_map:
            # 원문(…-0)이 없으면 요구사항(0이 기존 문제) 충족 불가
            missing_v0 += 1
            continue

        base_ex = variants_map[0]

        # variants는 존재하는 것만 0/1/2 순서로 넣음
        variants_list = []
        for k in (0, 1, 2):
            if k in variants_map:
                ex = variants_map[k]
                variants_list.append(
                    {
                        "snapshot": str(k),          # <-- 핵심: "0","1","2"
                        "question": ex["problem"],
                        "answer": ex["solution"],
                    }
                )

        merged.append(
            {
                "unique_id": str(base_id),
                "subject": args.subject,
                "level": args.level,
                "question": base_ex["problem"],
                "answer": base_ex["solution"],
                "variants": variants_list,
            }
        )

    write_jsonl(args.out, merged)

    print(
        f"Done.\n"
        f"  MATHplus rows seen: {total_rows}\n"
        f"  groups total: {len(grouped)}\n"
        f"  groups dropped (overlap with MATH-500 test): {dropped_groups}\n"
        f"  groups dropped (missing variant-0): {missing_v0}\n"
        f"  output rows (merged groups): {len(merged)} -> {args.out}"
    )


if __name__ == "__main__":
    main()