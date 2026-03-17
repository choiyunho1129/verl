#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset


# ----------------------------
# HF download helpers
# ----------------------------
def dump_hf_config_to_jsonl(
    repo_id: str,
    config_name: str,
    split: str,
    out_path: Path,
    cache_dir: Optional[str] = None,
) -> None:
    """
    Download a HF dataset split and dump it as JSONL.
    """
    ds = load_dataset(repo_id, name=config_name, split=split, cache_dir=cache_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------
# Your original merge script
# ----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_FINAL_ANS_RE = re.compile(r"####\s*([^\n\r]+)\s*$")


def extract_final_answer(answer_field: Any) -> str:
    if answer_field is None:
        return ""
    s = str(answer_field).strip()
    m = _FINAL_ANS_RE.search(s)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if nums:
        return nums[-1]
    return s


def pick_key(row: Dict[str, Any], idx: int) -> Tuple[str, Any]:
    if "original_id" in row and "instance" in row:
        return ("original_id+instance", (row.get("original_id"), row.get("instance")))
    if "unique_id" in row:
        return ("unique_id", row.get("unique_id"))
    if "id" in row and "instance" in row:
        return ("id+instance", (row.get("id"), row.get("instance")))
    return ("index", idx)


def build_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    out: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for i, r in enumerate(rows):
        k = pick_key(r, i)
        if k not in out:
            out[k] = r
    return out


def make_unique_id(base_row: Dict[str, Any], idx: int) -> str:
    if "unique_id" in base_row and base_row["unique_id"]:
        return str(base_row["unique_id"])
    return str(idx)


def get_subject(base_row: Dict[str, Any]) -> str:
    s = base_row.get("subject")
    return "" if s is None else str(s)


def get_level(base_row: Dict[str, Any]) -> Optional[int]:
    lvl = base_row.get("level")
    if lvl is None:
        return None
    try:
        return int(lvl)
    except Exception:
        return None


def get_question(row: Dict[str, Any]) -> str:
    for k in ("question", "problem", "instruction", "input"):
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def get_answer(row: Dict[str, Any]) -> str:
    for k in ("answer", "solution", "ground_truth"):
        if k in row and row[k] is not None:
            return extract_final_answer(row[k])
    return ""


def merge_to_snapshot_jsonl(
    base_rows: List[Dict[str, Any]],
    p1_rows: Optional[List[Dict[str, Any]]],
    p2_rows: Optional[List[Dict[str, Any]]],
    snapshot_main: str = "main",
    snapshot_p1: str = "p1",
    snapshot_p2: str = "p2",
) -> List[Dict[str, Any]]:
    p1_idx = build_index(p1_rows) if p1_rows else {}
    p2_idx = build_index(p2_rows) if p2_rows else {}

    merged: List[Dict[str, Any]] = []
    for i, base in enumerate(base_rows):
        key = pick_key(base, i)

        base_q = get_question(base)
        base_a = get_answer(base)

        variants: List[Dict[str, Any]] = [
            {"snapshot": snapshot_main, "question": base_q, "answer": base_a}
        ]

        if p1_rows:
            r1 = p1_idx.get(key)
            if r1 is not None:
                variants.append(
                    {"snapshot": snapshot_p1, "question": get_question(r1), "answer": get_answer(r1)}
                )

        if p2_rows:
            r2 = p2_idx.get(key)
            if r2 is not None:
                variants.append(
                    {"snapshot": snapshot_p2, "question": get_question(r2), "answer": get_answer(r2)}
                )

        out: Dict[str, Any] = {
            "unique_id": make_unique_id(base, i),
            "subject": get_subject(base),
            "question": base_q,
            "answer": base_a,
            "variants": variants,
        }

        lvl = get_level(base)
        if lvl is not None:
            out["level"] = lvl

        merged.append(out)

    return merged


def main():
    ap = argparse.ArgumentParser(
        description="Download HF apple/GSM-Symbolic (main/p1/p2 test) and merge into snapshot-variants JSONL."
    )
    ap.add_argument("--repo", type=str, default="apple/GSM-Symbolic")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--dump-dir", type=str, default="./hf_dump", help="Where to dump downloaded JSONLs")
    ap.add_argument("--cache-dir", type=str, default=None, help="HF datasets cache dir (optional)")
    ap.add_argument("--out", type=str, default="/data1/home/yunhochoi/verl/data/gsm_symbolic/gsm_symbolic_variant.jsonl")

    ap.add_argument("--snap-main", type=str, default="main")
    ap.add_argument("--snap-p1", type=str, default="p1")
    ap.add_argument("--snap-p2", type=str, default="p2")

    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    base_path = dump_dir / "main_test.jsonl"
    p1_path = dump_dir / "p1_test.jsonl"
    p2_path = dump_dir / "p2_test.jsonl"

    # 1) Download + dump JSONLs
    dump_hf_config_to_jsonl(args.repo, "main", args.split, base_path, cache_dir=args.cache_dir)
    dump_hf_config_to_jsonl(args.repo, "p1", args.split, p1_path, cache_dir=args.cache_dir)
    dump_hf_config_to_jsonl(args.repo, "p2", args.split, p2_path, cache_dir=args.cache_dir)

    # 2) Merge
    base_rows = read_jsonl(base_path)
    p1_rows = read_jsonl(p1_path)
    p2_rows = read_jsonl(p2_path)

    merged = merge_to_snapshot_jsonl(
        base_rows,
        p1_rows,
        p2_rows,
        snapshot_main=args.snap_main,
        snapshot_p1=args.snap_p1,
        snapshot_p2=args.snap_p2,
    )
    write_jsonl(Path(args.out), merged)
    print(f"Wrote {len(merged)} rows -> {args.out}")


if __name__ == "__main__":
    main()