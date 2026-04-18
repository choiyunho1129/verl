from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing rollout index fields by copying from fallback fields."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        action="append",
        required=True,
        help="Field mapping in rollout_features as missing_field=fallback_field.",
    )
    return parser.parse_args()


def parse_mapping(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid mapping {spec!r}; expected missing=fallback.")
    target, source = spec.split("=", 1)
    target = target.strip()
    source = source.strip()
    if not target or not source:
        raise ValueError(f"Invalid mapping {spec!r}; expected missing=fallback.")
    return target, source


def backfill_file(input_path: Path, output_path: Path, mappings: list[tuple[str, str]]) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            feats = row.setdefault("rollout_features", {})
            stats["rows"] += 1
            for target, source in mappings:
                if feats.get(target) is None and feats.get(source) is not None:
                    feats[target] = feats[source]
                    stats[f"filled:{target}"] += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mappings = [parse_mapping(spec) for spec in args.mapping]

    shard_paths = sorted(input_dir.rglob("*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No jsonl files found under {input_dir}")

    summary: dict[str, int] = defaultdict(int)
    for input_path in shard_paths:
        rel = input_path.relative_to(input_dir)
        stats = backfill_file(input_path, output_dir / rel, mappings)
        for key, value in stats.items():
            summary[key] += int(value)

    summary_path = output_dir / "backfill_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
