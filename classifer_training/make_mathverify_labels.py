from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from math_verify import parse, verify


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild prompt-level sampling labels from existing run directories using "
            "official Math-Verify scoring."
        )
    )
    parser.add_argument("--base_labels", type=Path, required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--score_text",
        choices=("answer", "fulltext"),
        default="fulltext",
        help="Whether to score only answer_content or the full generated trajectory.",
    )
    parser.add_argument("--max_workers", type=int, default=8)
    return parser.parse_args(argv)


def _score_row(item: tuple[str, str, str]) -> tuple[str, int]:
    task_id, ground_truth, candidate_text = item
    correct = 0
    try:
        gold = parse(f"${ground_truth}$")
        predicted = parse(candidate_text)
        correct = int(bool(verify(gold, predicted)))
    except Exception:
        correct = 0
    return task_id, correct


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    base_records: OrderedDict[str, dict] = OrderedDict()
    with args.base_labels.open() as f:
        for line in f:
            row = json.loads(line)
            base_records[str(row["task_id"])] = row

    per_task: dict[str, list[int]] = {task_id: [] for task_id in base_records}
    run_accuracies: list[dict[str, object]] = []

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).expanduser().resolve()
        rows_to_score: list[tuple[str, str, str]] = []
        with (run_dir / "all_experiments.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                task_id = str(row["task_id"])
                if task_id not in per_task:
                    continue
                ground_truth = str(row.get("ground_truth") or "").strip()
                answer_text = str(row.get("answer_content") or "")
                full_text = str(row.get("generated_text") or "")
                candidate_text = answer_text if args.score_text == "answer" and answer_text.strip() else full_text
                rows_to_score.append((task_id, ground_truth, candidate_text))

        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            scored = list(executor.map(_score_row, rows_to_score, chunksize=64))

        run_correct = 0
        for task_id, correct in scored:
            per_task[task_id].append(int(correct))
            run_correct += int(correct)

        summary = {
            "run_dir": str(run_dir),
            "accuracy": float(run_correct / len(scored)) if scored else 0.0,
            "num_examples": len(scored),
        }
        run_accuracies.append(summary)
        print(json.dumps(summary), flush=True)

    out_rows: list[dict] = []
    for task_id, base in base_records.items():
        correctness = per_task[task_id]
        sampling_accuracy = float(sum(correctness) / len(correctness)) if correctness else 0.0
        row = dict(base)
        row["num_runs"] = len(correctness)
        row["correct_count"] = int(sum(correctness))
        row["wrong_count"] = int(len(correctness) - sum(correctness))
        row["sampling_accuracy"] = sampling_accuracy
        row["difficulty"] = 1.0 - sampling_accuracy
        row["source_run_dirs"] = sorted(set(str(Path(p).expanduser().resolve()) for p in args.run_dirs))
        out_rows.append(row)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.output_path.with_name(args.output_path.stem + "_summary.json")
    summary = {
        "num_records": len(out_rows),
        "score_text": args.score_text,
        "run_accuracies": run_accuracies,
        "mean_run_accuracy": float(sum(item["accuracy"] for item in run_accuracies) / len(run_accuracies))
        if run_accuracies
        else 0.0,
        "difficulty_mean": float(sum(row["difficulty"] for row in out_rows) / len(out_rows)) if out_rows else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(out_rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
