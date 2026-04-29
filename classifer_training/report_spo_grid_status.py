from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_GRID_ROOT = Path(
    "classifer_training/artifacts/probe/spo_temp1_subset0to4_qwen3_4b_base_rowr2_pca_tied_full_grid"
)
DEFAULT_REFERENCE_NAME = "pca_tied_grid_last10_L19_p32_r256"


def _load_summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary.json")):
        with path.open() as f:
            summary = json.load(f)
        rows.append(
            {
                "name": str(summary["name"]),
                "row_r2": float(summary["test_row_metrics"]["r2"]),
                "prompt_mean_r2": float(summary["test_prompt_mean_metrics"]["r2"]),
                "path": path,
            }
        )
    return rows


def _print_metric_report(rows: list[dict[str, Any]], metric_key: str, label: str, reference: dict[str, Any] | None) -> None:
    best = max(rows, key=lambda row: row[metric_key])
    worst = min(rows, key=lambda row: row[metric_key])

    print(label)
    print(f"  best        {best[metric_key]:.6f}  {best['name']}")
    print(f"  worst       {worst[metric_key]:.6f}  {worst['name']}")
    print(f"  best-worst  {best[metric_key] - worst[metric_key]:.6f}")
    if reference is not None:
        print(f"  ref         {reference[metric_key]:.6f}  {reference['name']}")
        print(f"  best-ref    {best[metric_key] - reference[metric_key]:.6f}")
        print(f"  ref-worst   {reference[metric_key] - worst[metric_key]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report SPO tied grid best/worst and reference comparisons.")
    parser.add_argument("--root", type=Path, default=DEFAULT_GRID_ROOT, help="Grid result root containing */summary.json files.")
    parser.add_argument("--reference-name", default=DEFAULT_REFERENCE_NAME, help="Config name to compare against best/worst.")
    parser.add_argument("--expected-total", type=int, default=54, help="Expected number of grid configs.")
    parser.add_argument("--top-k", type=int, default=10, help="Print top K configs by row R2.")
    args = parser.parse_args()

    rows = _load_summaries(args.root)
    if not rows:
        raise SystemExit(f"No summary.json files found under {args.root}")

    reference = next((row for row in rows if row["name"] == args.reference_name), None)

    print(f"root: {args.root}")
    print(f"completed: {len(rows)} / {args.expected_total}")
    if reference is None:
        print(f"reference: {args.reference_name} (not complete)")
    else:
        print(
            f"reference: {reference['name']} "
            f"row_r2={reference['row_r2']:.6f} "
            f"prompt_mean_r2={reference['prompt_mean_r2']:.6f}"
        )
    print()

    _print_metric_report(rows, "row_r2", "row R2", reference)
    print()
    _print_metric_report(rows, "prompt_mean_r2", "prompt-mean R2", reference)
    print()

    print(f"top {args.top_k} by row R2")
    for rank, row in enumerate(sorted(rows, key=lambda item: item["row_r2"], reverse=True)[: args.top_k], 1):
        print(f"  {rank:2d}. row={row['row_r2']:.6f}  pm={row['prompt_mean_r2']:.6f}  {row['name']}")


if __name__ == "__main__":
    main()
