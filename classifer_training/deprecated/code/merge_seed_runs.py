from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    if args.run_glob:
        run_dirs.extend(sorted(Path().glob(args.run_glob)))
    return sorted({path.resolve() for path in run_dirs})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge old seed-based run directories into a single multi-sample-style run directory "
            "with sample_index and sample_count fields so they can be combined with new n-sample runs."
        )
    )
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--run_glob", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = _resolve_run_dirs(args)
    if not run_dirs:
        raise ValueError("At least one run directory is required.")

    output_dir = args.output_dir.expanduser().resolve()
    experiments_path = output_dir / "all_experiments.jsonl"
    evaluations_path = output_dir / "evaluation_results.jsonl"
    if output_dir.exists() and (experiments_path.exists() or evaluations_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains merged artifacts. Pass --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    dataset_name = ""
    merged_config: dict[str, Any] | None = None

    for run_dir in run_dirs:
        experiment_rows = _load_jsonl(run_dir / "all_experiments.jsonl")
        evaluation_rows = _load_jsonl(run_dir / "evaluation_results.jsonl")
        if not evaluation_rows:
            raise ValueError(f"No evaluation rows found in {run_dir}.")
        run_correctness = list(evaluation_rows[-1].get("correctness", []))
        total_rows = min(len(experiment_rows), len(run_correctness))
        for row_idx in range(total_rows):
            row = dict(experiment_rows[row_idx])
            row["source_run_dir"] = str(run_dir)
            row["source_seed"] = row.get("config", {}).get("seed")
            row["_correct"] = int(run_correctness[row_idx])
            grouped_rows[(str(row.get("dataset_name", "")), str(row.get("task_id", row_idx)))].append(row)
            if not dataset_name:
                dataset_name = str(row.get("dataset_name", ""))
            if merged_config is None:
                merged_config = dict(row.get("config", {}))

    merged_rows: list[dict[str, Any]] = []
    merged_correctness: list[int] = []
    for key in sorted(grouped_rows.keys()):
        rows = sorted(
            grouped_rows[key],
            key=lambda row: (
                -1 if row.get("source_seed") is None else int(row.get("source_seed")),
                int(row.get("sample_index", 0)),
            ),
        )
        sample_count = len(rows)
        for sample_index, row in enumerate(rows):
            correct = int(row.pop("_correct"))
            row["sample_index"] = int(sample_index)
            row["sample_count"] = int(sample_count)
            merged_rows.append(row)
            merged_correctness.append(correct)

    accuracy = float(sum(merged_correctness) / len(merged_correctness)) if merged_correctness else 0.0
    merged_eval = {
        "dataset_name": dataset_name,
        "num_examples": len(merged_rows),
        "num_prompts": len(grouped_rows),
        "accuracy": accuracy,
        "correctness": merged_correctness,
        "config": {
            **(merged_config or {}),
            "merged_from_run_dirs": [str(path) for path in run_dirs],
            "merge_mode": "seed_runs_to_multisample",
        },
    }

    with experiments_path.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with evaluations_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(merged_eval, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_run_dirs": len(run_dirs),
                "num_prompts": len(grouped_rows),
                "num_examples": len(merged_rows),
                "accuracy": accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
