from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifer_training.utils import sanitize_name


def build_manifest(
    model_name: str,
    model_slug: str | None,
    datasets: list[str],
    hidden_root: Path,
    index_root: Path,
    labels_root: Path,
    hidden_filename: str,
    index_filename: str,
    labels_filename: str,
    default_component_name: str | None,
    hidden_glob: str | None = None,
    index_glob: str | None = None,
) -> dict:
    resolved_model_slug = model_slug or sanitize_name(model_name)
    entries = []
    for dataset_name in datasets:
        dataset_hidden_root = hidden_root / dataset_name / resolved_model_slug
        dataset_index_root = index_root / dataset_name / resolved_model_slug
        labels_path = str((labels_root / dataset_name / resolved_model_slug / labels_filename).resolve())

        if hidden_glob or index_glob:
            if not hidden_glob or not index_glob:
                raise ValueError("--hidden_glob and --index_glob must be provided together.")
            hidden_paths = sorted(dataset_hidden_root.glob(hidden_glob))
            index_paths = sorted(dataset_index_root.glob(index_glob))
            if not hidden_paths or not index_paths:
                raise FileNotFoundError(
                    f"No shard files matched hidden_glob={hidden_glob!r} or index_glob={index_glob!r} "
                    f"for dataset {dataset_name!r}."
                )
            if len(hidden_paths) != len(index_paths):
                raise ValueError(
                    f"Mismatched shard counts for dataset {dataset_name!r}: "
                    f"{len(hidden_paths)} hidden files vs {len(index_paths)} index files."
                )
            for hidden_path, index_path in zip(hidden_paths, index_paths):
                entry = {
                    "name": dataset_name,
                    "hidden_states_path": str(hidden_path.resolve()),
                    "index_path": str(index_path.resolve()),
                    "labels_path": labels_path,
                }
                if default_component_name:
                    entry["default_component_name"] = default_component_name
                entries.append(entry)
            continue

        entry = {
            "name": dataset_name,
            "hidden_states_path": str((dataset_hidden_root / hidden_filename).resolve()),
            "index_path": str((dataset_index_root / index_filename).resolve()),
            "labels_path": labels_path,
        }
        if default_component_name:
            entry["default_component_name"] = default_component_name
        entries.append(entry)
    return {"datasets": entries}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a manifest for classifer_training using a simple dataset/model "
            "directory convention."
        )
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_slug", default=None, help="Optional override for artifact directory naming.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--hidden_root", type=Path, required=True)
    parser.add_argument("--index_root", type=Path, required=True)
    parser.add_argument("--labels_root", type=Path, required=True)
    parser.add_argument("--hidden_filename", default="hidden_states.pt")
    parser.add_argument("--index_filename", default="index.jsonl")
    parser.add_argument("--labels_filename", default="sampling_labels.jsonl")
    parser.add_argument("--default_component_name", default=None)
    parser.add_argument("--hidden_glob", default=None)
    parser.add_argument("--index_glob", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_manifest(
        model_name=args.model_name,
        model_slug=args.model_slug,
        datasets=args.datasets,
        hidden_root=args.hidden_root.expanduser(),
        index_root=args.index_root.expanduser(),
        labels_root=args.labels_root.expanduser(),
        hidden_filename=args.hidden_filename,
        index_filename=args.index_filename,
        labels_filename=args.labels_filename,
        default_component_name=args.default_component_name,
        hidden_glob=args.hidden_glob,
        index_glob=args.index_glob,
    )
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {output_path}")


if __name__ == "__main__":
    main()
