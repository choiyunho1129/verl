from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from classifer_training.utils import load_records


@dataclass
class ExampleRecord:
    dataset_name: str
    task_id: str
    split: str | None
    components: dict[str, list[np.ndarray]]
    index_row: dict[str, Any]
    label_row: dict[str, Any]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("datasets"), list):
        return payload["datasets"]
    raise ValueError("Manifest must be either a JSON list or an object with a 'datasets' list.")


def _vectorize_last_token(value: Any) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu().to(torch.float32)
    if tensor.ndim == 0:
        return tensor.reshape(1).numpy()
    if tensor.ndim == 1:
        return tensor.numpy()
    return tensor.reshape(-1, tensor.shape[-1])[-1].numpy()


def _normalize_component_layers(value: Any) -> list[np.ndarray]:
    if isinstance(value, (list, tuple)):
        return [_vectorize_last_token(item) for item in value]

    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        tensor = torch.as_tensor(value)
        if tensor.ndim <= 1:
            return [_vectorize_last_token(tensor)]
        return [_vectorize_last_token(tensor[layer_idx]) for layer_idx in range(tensor.shape[0])]

    raise TypeError(f"Unsupported hidden-state component type: {type(value)!r}")


def _is_component_payload(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value.ndim >= 1
    return False


def _build_identity(
    row_idx: int,
    index_rows: list[dict[str, Any]] | None,
    dataset_name: str | None,
    task_ids: list[Any] | None = None,
    dataset_names: list[Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    index_row: dict[str, Any] = {}
    if index_rows is not None and row_idx < len(index_rows):
        index_row = index_rows[row_idx]
        task_id = str(index_row.get("task_id", row_idx))
        resolved_dataset_name = str(index_row.get("dataset_name") or dataset_name or "")
        return resolved_dataset_name, task_id, index_row

    task_id = str(task_ids[row_idx]) if task_ids is not None else str(row_idx)
    resolved_dataset_name = (
        str(dataset_names[row_idx]) if dataset_names is not None else str(dataset_name or "")
    )
    return resolved_dataset_name, task_id, index_row


def _rows_from_list_payload(
    payload: list[Any],
    index_rows: list[dict[str, Any]] | None,
    dataset_name: str | None,
    default_component_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, item in enumerate(payload):
        resolved_dataset_name, task_id, index_row = _build_identity(
            row_idx=row_idx,
            index_rows=index_rows,
            dataset_name=dataset_name,
        )
        if isinstance(item, dict):
            components = {
                component_name: _normalize_component_layers(component_payload)
                for component_name, component_payload in item.items()
                if component_name not in {"task_id", "dataset_name", "metadata"}
                and _is_component_payload(component_payload)
            }
        else:
            components = {
                default_component_name: _normalize_component_layers(item),
            }
        rows.append(
            {
                "dataset_name": resolved_dataset_name,
                "task_id": task_id,
                "components": components,
                "index_row": index_row,
            }
        )
    return rows


def _rows_from_batched_payload(
    payload: dict[str, Any],
    index_rows: list[dict[str, Any]] | None,
    dataset_name: str | None,
) -> list[dict[str, Any]]:
    task_ids = payload.get("task_ids")
    dataset_names = payload.get("dataset_names")
    component_names = [
        key
        for key, value in payload.items()
        if key not in {"task_ids", "dataset_names", "metadata", "examples"}
        and _is_component_payload(value)
    ]
    if not component_names:
        raise ValueError("Could not find any hidden-state components in the batched payload.")

    first_component = payload[component_names[0]]
    total_rows = len(first_component)
    rows: list[dict[str, Any]] = []
    for row_idx in range(total_rows):
        resolved_dataset_name, task_id, index_row = _build_identity(
            row_idx=row_idx,
            index_rows=index_rows,
            dataset_name=dataset_name,
            task_ids=task_ids,
            dataset_names=dataset_names,
        )
        components = {
            component_name: _normalize_component_layers(payload[component_name][row_idx])
            for component_name in component_names
        }
        rows.append(
            {
                "dataset_name": resolved_dataset_name,
                "task_id": task_id,
                "components": components,
                "index_row": index_row,
            }
        )
    return rows


def load_hidden_rows(
    hidden_states_path: Path,
    index_path: Path | None = None,
    dataset_name: str | None = None,
    default_component_name: str = "hidden",
) -> list[dict[str, Any]]:
    payload = torch.load(hidden_states_path, map_location="cpu")
    index_rows = load_records(index_path) if index_path is not None else None

    if isinstance(payload, dict):
        if isinstance(payload.get("examples"), list):
            return _rows_from_list_payload(
                payload["examples"],
                index_rows=index_rows,
                dataset_name=dataset_name,
                default_component_name=default_component_name,
            )
        return _rows_from_batched_payload(
            payload,
            index_rows=index_rows,
            dataset_name=dataset_name,
        )

    if isinstance(payload, list):
        return _rows_from_list_payload(
            payload,
            index_rows=index_rows,
            dataset_name=dataset_name,
            default_component_name=default_component_name,
        )

    if torch.is_tensor(payload) or isinstance(payload, np.ndarray):
        rows = _rows_from_list_payload(
            [payload[row_idx] for row_idx in range(len(payload))],
            index_rows=index_rows,
            dataset_name=dataset_name,
            default_component_name=default_component_name,
        )
        return rows

    raise TypeError(f"Unsupported hidden-states payload type: {type(payload)!r}")


def _build_label_maps(label_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    by_task_id: dict[str, dict[str, Any]] = {}
    by_dataset_and_task: dict[tuple[str, str], dict[str, Any]] = {}
    for row in label_rows:
        task_id = str(row["task_id"])
        dataset_name = str(row.get("dataset_name", ""))
        by_task_id[task_id] = row
        by_dataset_and_task[(dataset_name, task_id)] = row
    return by_task_id, by_dataset_and_task


def load_aligned_examples(
    manifest_entries: list[dict[str, Any]],
    strict: bool = True,
) -> list[ExampleRecord]:
    examples: list[ExampleRecord] = []
    missing_labels: list[str] = []

    for entry in manifest_entries:
        dataset_name = entry.get("name")
        hidden_rows = load_hidden_rows(
            hidden_states_path=Path(entry["hidden_states_path"]).expanduser().resolve(),
            index_path=Path(entry["index_path"]).expanduser().resolve()
            if entry.get("index_path")
            else None,
            dataset_name=dataset_name,
            default_component_name=entry.get("default_component_name", "hidden"),
        )
        label_rows = load_records(Path(entry["labels_path"]).expanduser().resolve())
        labels_by_task_id, labels_by_dataset_and_task = _build_label_maps(label_rows)

        for hidden_row in hidden_rows:
            key = (hidden_row["dataset_name"], hidden_row["task_id"])
            label_row = labels_by_dataset_and_task.get(key) or labels_by_task_id.get(hidden_row["task_id"])
            if label_row is None:
                missing_labels.append(f"{hidden_row['dataset_name']}::{hidden_row['task_id']}")
                continue
            examples.append(
                ExampleRecord(
                    dataset_name=hidden_row["dataset_name"],
                    task_id=hidden_row["task_id"],
                    split=hidden_row["index_row"].get("split") if isinstance(hidden_row["index_row"], dict) else None,
                    components=hidden_row["components"],
                    index_row=hidden_row["index_row"],
                    label_row=label_row,
                )
            )

    if strict and missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise KeyError(f"Missing labels for {len(missing_labels)} hidden-state rows. Examples: {preview}")

    return examples
