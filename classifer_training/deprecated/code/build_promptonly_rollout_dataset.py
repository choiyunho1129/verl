from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from classifer_training.aggregate_labels import build_aggregated_label_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prompt-hidden-only rollout dataset by repeating per-prompt hidden states "
            "for each finished rollout row."
        )
    )
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--run_glob", type=str, default=None)
    parser.add_argument("--prompt_hidden_root", type=Path, required=True)
    parser.add_argument("--prompt_index_root", type=Path, required=True)
    parser.add_argument("--output_hidden_path", type=Path, required=True)
    parser.add_argument("--output_index_path", type=Path, required=True)
    parser.add_argument("--output_manifest_path", type=Path, required=True)
    parser.add_argument("--output_labels_path", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, default="dapo_math_17k")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    if args.run_glob:
        run_dirs.extend(sorted(Path().glob(args.run_glob)))
    return sorted({path.resolve() for path in run_dirs})


def load_prompt_hidden_map(prompt_hidden_root: Path, prompt_index_root: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        hidden_path = prompt_hidden_root / f"hidden_states_{split}.pt"
        index_path = prompt_index_root / f"index_{split}.jsonl"
        hidden_payload = torch.load(hidden_path, map_location="cpu")
        examples = hidden_payload["examples"]
        index_rows = load_records(index_path)
        for example, index_row in zip(examples, index_rows):
            task_id = str(example["task_id"])
            hidden_layers = example["hidden"]
            mapping[task_id] = {
                "dataset_name": str(example.get("dataset_name", index_row.get("dataset_name", ""))),
                "task_id": task_id,
                "split": str(index_row.get("split", split)),
                "prompt_hidden": [torch.as_tensor(hidden_layers[-1]).detach().cpu().to(torch.float32)],
                "index_row": index_row,
            }
    return mapping


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args)
    if not run_dirs:
        raise ValueError("At least one finished run directory is required.")

    prompt_map = load_prompt_hidden_map(
        prompt_hidden_root=args.prompt_hidden_root.expanduser().resolve(),
        prompt_index_root=args.prompt_index_root.expanduser().resolve(),
    )

    hidden_examples: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    global_example_index = 0

    for run_dir in run_dirs:
        experiments_path = run_dir / "all_experiments.jsonl"
        for rollout_row_index, row in enumerate(load_records(experiments_path)):
            task_id = str(row["task_id"])
            prompt_item = prompt_map.get(task_id)
            if prompt_item is None:
                continue
            hidden_examples.append(
                {
                    "dataset_name": prompt_item["dataset_name"],
                    "task_id": task_id,
                    "prompt_hidden": prompt_item["prompt_hidden"],
                }
            )
            merged = dict(row)
            merged["dataset_name"] = prompt_item["dataset_name"]
            merged["split"] = prompt_item["split"]
            merged["rollout_row_index"] = int(rollout_row_index)
            merged["run_dir"] = str(run_dir)
            merged["run_name"] = run_dir.name
            merged["global_example_index"] = int(global_example_index)
            merged["selected_layers"] = [35]
            index_rows.append(merged)
            global_example_index += 1

    args.output_hidden_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_index_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_labels_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "metadata": {
                "dataset_name": args.dataset_name,
                "default_component_name": "prompt_hidden",
                "num_examples": len(hidden_examples),
            },
            "examples": hidden_examples,
        },
        args.output_hidden_path,
    )

    with args.output_index_path.open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_records = build_aggregated_label_records(run_dirs=run_dirs, extra_numeric_fields=[])
    with args.output_labels_path.open("w", encoding="utf-8") as f:
        for row in label_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "datasets": [
            {
                "name": args.dataset_name,
                "hidden_states_path": str(args.output_hidden_path.expanduser().resolve()),
                "index_path": str(args.output_index_path.expanduser().resolve()),
                "labels_path": str(args.output_labels_path.expanduser().resolve()),
            }
        ]
    }
    args.output_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_count": len(run_dirs),
                "num_examples": len(hidden_examples),
                "num_labels": len(label_records),
                "output_manifest_path": str(args.output_manifest_path.expanduser().resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
