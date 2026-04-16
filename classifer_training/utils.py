from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("datasets", "records", "items", "examples", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON structure in {path}.")

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))

    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")

    raise ValueError(
        f"Unsupported file format for {path}. Expected .jsonl, .json, .csv, .tsv, or .parquet."
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_nested_value(record: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = record
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            converted = float(stripped)
        except ValueError:
            return None
        return converted if math.isfinite(converted) else None
    return None


def sanitize_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace(" ", "_")

def parse_layer_spec(spec: str | None, num_layers: int) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")

    if spec is None or spec.strip() in {"", "all"}:
        return list(range(num_layers))

    def normalize_index(idx: int) -> int:
        normalized = idx if idx >= 0 else num_layers + idx
        if not 0 <= normalized < num_layers:
            raise ValueError(
                f"Layer index {idx} is out of range for a component with {num_layers} layers."
            )
        return normalized

    resolved: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue

        if ":" in token:
            start_str, end_str = token.split(":", maxsplit=1)
            start = normalize_index(int(start_str)) if start_str else 0
            end = normalize_index(int(end_str)) if end_str else num_layers - 1
            if end < start:
                raise ValueError(f"Invalid layer slice {token!r}.")
            resolved.extend(range(start, end + 1))
            continue

        if "-" in token[1:]:
            start_str, end_str = token.split("-", maxsplit=1)
            start = normalize_index(int(start_str))
            end = normalize_index(int(end_str))
            step = 1 if end >= start else -1
            resolved.extend(range(start, end + step, step))
            continue

        resolved.append(normalize_index(int(token)))

    deduped: list[int] = []
    seen: set[int] = set()
    for idx in resolved:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped
