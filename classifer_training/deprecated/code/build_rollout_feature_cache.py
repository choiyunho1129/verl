from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from classifer_training.enrich_rollout_index import _single_run_features
from classifer_training.rollout_utils import extract_rollout_numeric_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact rollout feature cache with enriched scalar features but "
            "without keeping large text fields. This is much smaller and faster to reuse "
            "than a full enriched rollout index."
        )
    )
    parser.add_argument("--input_index", type=Path, required=True)
    parser.add_argument("--output_index", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 1) - 2))
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--manifest_path", type=Path, default=None)
    parser.add_argument("--hidden_states_path", type=Path, default=None)
    parser.add_argument("--labels_path", type=Path, default=None)
    return parser.parse_args()


def _compact_row(row: dict[str, Any], dataset_name_override: str | None) -> dict[str, Any]:
    rollout_features = dict(row.get("rollout_features") or {})
    rollout_features.update(extract_rollout_numeric_features(row))
    rollout_features.update(_single_run_features(row))
    return {
        "dataset_name": str(dataset_name_override or row.get("dataset_name", "")),
        "task_id": str(row["task_id"]),
        "split": row.get("split"),
        "rollout_row_index": int(row.get("rollout_row_index", 0)),
        "run_name": row.get("run_name"),
        "run_dir": row.get("run_dir"),
        "global_example_index": int(row.get("global_example_index", 0)),
        "selected_layers": row.get("selected_layers"),
        "rollout_features": rollout_features,
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    input_index = args.input_index.expanduser().resolve()
    output_index = args.output_index.expanduser().resolve()
    output_index.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_index)
    dataset_name_override = args.dataset_name
    worker_args = [(row, dataset_name_override) for row in rows]

    with mp.Pool(processes=args.workers) as pool:
        compact_rows = list(pool.starmap(_compact_row, worker_args, chunksize=args.chunk_size))

    with output_index.open("w", encoding="utf-8") as f:
        for row in compact_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.manifest_path is not None and args.hidden_states_path is not None and args.labels_path is not None:
        manifest = {
            "datasets": [
                {
                    "name": dataset_name_override or compact_rows[0]["dataset_name"],
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
                "rows": len(compact_rows),
                "input_index": str(input_index),
                "output_index": str(output_index),
                "workers": args.workers,
                "chunk_size": args.chunk_size,
                "manifest_path": str(args.manifest_path.expanduser().resolve()) if args.manifest_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
