from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from classifer_training.data import ExampleRecord
from classifer_training.utils import coerce_float, get_nested_value, parse_layer_spec, sanitize_name


@dataclass
class FeatureExtractionConfig:
    components: list[str] = field(default_factory=lambda: ["hidden"])
    layers: str | None = "all"
    component_pooling: str = "concat"
    extra_feature_paths: list[str] = field(default_factory=list)
    include_hidden_features: bool = True
    engineered_feature_set: str = "raw"


def _aggregate_component_vectors(
    component_name: str,
    layer_vectors: list[np.ndarray],
    layer_spec: str | None,
    pooling: str,
) -> tuple[np.ndarray, list[str]]:
    selected_layers = parse_layer_spec(layer_spec, len(layer_vectors))
    selected_vectors = [np.asarray(layer_vectors[layer_idx], dtype=np.float32) for layer_idx in selected_layers]
    if not selected_vectors:
        raise ValueError(f"No layers were selected for component {component_name!r}.")

    reference_dim = selected_vectors[0].shape[0]
    for vector in selected_vectors:
        if vector.ndim != 1 or vector.shape[0] != reference_dim:
            raise ValueError(
                f"All selected vectors for component {component_name!r} must be 1D and share the same dimension."
            )

    if pooling == "concat":
        feature_vector = np.concatenate(selected_vectors, axis=0)
        feature_names = [
            f"{component_name}_layer{layer_idx}_dim{dim_idx}"
            for layer_idx, vector in zip(selected_layers, selected_vectors)
            for dim_idx in range(vector.shape[0])
        ]
        return feature_vector, feature_names

    stacked = np.stack(selected_vectors, axis=0)
    if pooling == "mean":
        pooled = stacked.mean(axis=0)
        suffix = "mean"
    elif pooling == "max":
        pooled = stacked.max(axis=0)
        suffix = "max"
    else:
        raise ValueError(f"Unsupported component pooling mode: {pooling}")

    layer_name = sanitize_name(",".join(str(layer_idx) for layer_idx in selected_layers))
    feature_names = [f"{component_name}_{suffix}_{layer_name}_dim{dim_idx}" for dim_idx in range(pooled.shape[0])]
    return pooled.astype(np.float32), feature_names


def _resolve_numeric_feature(example: ExampleRecord, feature_path: str) -> float:
    search_order: list[tuple[dict[str, Any], str]] = []
    if feature_path.startswith("label."):
        search_order.append((example.label_row, feature_path[len("label.") :]))
    elif feature_path.startswith("index."):
        search_order.append((example.index_row, feature_path[len("index.") :]))
    else:
        search_order.extend(
            [
                (example.label_row, feature_path),
                (example.index_row, feature_path),
            ]
        )

    for source, path in search_order:
        value = get_nested_value(source, path, default=None)
        numeric = coerce_float(value)
        if numeric is not None:
            return numeric
    # Some rollout-derived scalar features are intentionally sparse, for example
    # reasoning-only statistics when a sample has no reasoning segment.
    return 0.0


def _pair_relation_features(prefix: str, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, list[str]]:
    dot = float(np.dot(left, right))
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    delta = right - left
    delta_norm = float(np.linalg.norm(delta))
    cosine = dot / max(left_norm * right_norm, 1e-6)
    values = np.asarray(
        [
            left_norm,
            right_norm,
            dot,
            cosine,
            delta_norm,
            float(np.mean(np.abs(delta))),
            float(np.max(np.abs(delta))),
            float(np.mean(delta)),
            float(np.std(delta)),
        ],
        dtype=np.float32,
    )
    names = [
        f"{prefix}_left_norm",
        f"{prefix}_right_norm",
        f"{prefix}_dot",
        f"{prefix}_cosine",
        f"{prefix}_delta_norm",
        f"{prefix}_mean_abs_delta",
        f"{prefix}_max_abs_delta",
        f"{prefix}_mean_delta",
        f"{prefix}_std_delta",
    ]
    return values, names


def _relation_feature_block(component_vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    prompt = component_vectors.get("prompt_hidden")
    response = component_vectors.get("response_hidden")
    reasoning = component_vectors.get("reasoning_hidden")

    parts: list[np.ndarray] = []
    names: list[str] = []

    if prompt is not None and response is not None:
        values, value_names = _pair_relation_features("prompt_response", prompt, response)
        parts.append(values)
        names.extend(value_names)
    if prompt is not None and reasoning is not None:
        values, value_names = _pair_relation_features("prompt_reasoning", prompt, reasoning)
        parts.append(values)
        names.extend(value_names)
    if reasoning is not None and response is not None:
        values, value_names = _pair_relation_features("reasoning_response", reasoning, response)
        parts.append(values)
        names.extend(value_names)

    if not parts:
        return np.zeros((0,), dtype=np.float32), []
    return np.concatenate(parts, axis=0).astype(np.float32), names


def _concat_parts(parts: list[np.ndarray]) -> np.ndarray:
    available = [part.astype(np.float32) for part in parts if part.size > 0]
    if not available:
        raise ValueError("At least one feature part is required.")
    return np.concatenate(available, axis=0)


def _component_names(prefix: str, vector: np.ndarray) -> list[str]:
    return [f"{prefix}_dim{dim_idx}" for dim_idx in range(vector.shape[0])]


def _build_engineered_feature_row(
    *,
    component_vectors: dict[str, np.ndarray],
    extra_values: np.ndarray,
    extra_names: list[str],
    feature_set: str,
) -> tuple[np.ndarray, list[str]]:
    prompt = component_vectors.get("prompt_hidden")
    response = component_vectors.get("response_hidden")
    reasoning = component_vectors.get("reasoning_hidden")
    relation_values, relation_names = _relation_feature_block(component_vectors)

    def require(name: str, value: np.ndarray | None) -> np.ndarray:
        if value is None:
            raise ValueError(f"Engineered feature set {feature_set!r} requires component {name!r}.")
        return value

    if feature_set == "stats_only":
        if extra_values.size == 0:
            raise ValueError("stats_only requires at least one extra numeric feature.")
        return extra_values.astype(np.float32), list(extra_names)

    if feature_set == "relation_stats":
        if relation_values.size == 0 and extra_values.size == 0:
            raise ValueError("relation_stats requires relation features or extra numeric features.")
        return _concat_parts([relation_values, extra_values]), relation_names + extra_names

    if feature_set == "prompt_only":
        prompt = require("prompt_hidden", prompt)
        return _concat_parts([prompt, extra_values, relation_values]), (
            _component_names("prompt", prompt) + extra_names + relation_names
        )

    if feature_set == "response_only":
        response = require("response_hidden", response)
        return _concat_parts([response, extra_values, relation_values]), (
            _component_names("response", response) + extra_names + relation_names
        )

    if feature_set == "prompt_response":
        prompt = require("prompt_hidden", prompt)
        response = require("response_hidden", response)
        return _concat_parts([prompt, response, extra_values, relation_values]), (
            _component_names("prompt", prompt)
            + _component_names("response", response)
            + extra_names
            + relation_names
        )

    if feature_set == "delta_only":
        prompt = require("prompt_hidden", prompt)
        response = require("response_hidden", response)
        delta = response - prompt
        return _concat_parts([delta, extra_values, relation_values]), (
            _component_names("delta", delta) + extra_names + relation_names
        )

    if feature_set == "prompt_response_delta":
        prompt = require("prompt_hidden", prompt)
        response = require("response_hidden", response)
        delta = response - prompt
        return _concat_parts([prompt, response, delta, extra_values, relation_values]), (
            _component_names("prompt", prompt)
            + _component_names("response", response)
            + _component_names("delta", delta)
            + extra_names
            + relation_names
        )

    if feature_set == "prompt_response_delta_abs":
        prompt = require("prompt_hidden", prompt)
        response = require("response_hidden", response)
        delta = response - prompt
        abs_delta = np.abs(delta)
        return _concat_parts([prompt, response, delta, abs_delta, extra_values, relation_values]), (
            _component_names("prompt", prompt)
            + _component_names("response", response)
            + _component_names("delta", delta)
            + _component_names("abs_delta", abs_delta)
            + extra_names
            + relation_names
        )

    if feature_set == "prompt_response_delta_prod":
        prompt = require("prompt_hidden", prompt)
        response = require("response_hidden", response)
        delta = response - prompt
        product = prompt * response
        return _concat_parts([prompt, response, delta, product, extra_values, relation_values]), (
            _component_names("prompt", prompt)
            + _component_names("response", response)
            + _component_names("delta", delta)
            + _component_names("product", product)
            + extra_names
            + relation_names
        )

    if feature_set == "prompt_reasoning_response":
        prompt = require("prompt_hidden", prompt)
        reasoning = require("reasoning_hidden", reasoning)
        response = require("response_hidden", response)
        return _concat_parts([prompt, reasoning, response, extra_values, relation_values]), (
            _component_names("prompt", prompt)
            + _component_names("reasoning", reasoning)
            + _component_names("response", response)
            + extra_names
            + relation_names
        )

    if feature_set == "prompt_reasoning_response_deltas":
        prompt = require("prompt_hidden", prompt)
        reasoning = require("reasoning_hidden", reasoning)
        response = require("response_hidden", response)
        reasoning_prompt_delta = reasoning - prompt
        response_reasoning_delta = response - reasoning
        response_prompt_delta = response - prompt
        return _concat_parts(
            [
                prompt,
                reasoning,
                response,
                reasoning_prompt_delta,
                response_reasoning_delta,
                response_prompt_delta,
                extra_values,
                relation_values,
            ]
        ), (
            _component_names("prompt", prompt)
            + _component_names("reasoning", reasoning)
            + _component_names("response", response)
            + _component_names("reasoning_prompt_delta", reasoning_prompt_delta)
            + _component_names("response_reasoning_delta", response_reasoning_delta)
            + _component_names("response_prompt_delta", response_prompt_delta)
            + extra_names
            + relation_names
        )

    raise ValueError(f"Unsupported engineered_feature_set: {feature_set}")


def build_feature_matrix(
    examples: list[ExampleRecord],
    config: FeatureExtractionConfig,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    feature_rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    metadata_rows: list[dict[str, Any]] = []

    for example in examples:
        parts: list[np.ndarray] = []
        names: list[str] = []
        component_vectors: dict[str, np.ndarray] = {}

        if config.include_hidden_features:
            for component_name in config.components:
                if component_name not in example.components:
                    raise KeyError(
                        f"Component {component_name!r} is missing for task {example.task_id} in dataset {example.dataset_name}."
                    )
                component_vector, component_names = _aggregate_component_vectors(
                    component_name=component_name,
                    layer_vectors=example.components[component_name],
                    layer_spec=config.layers,
                    pooling=config.component_pooling,
                )
                component_vectors[component_name] = component_vector
                if config.engineered_feature_set == "raw":
                    parts.append(component_vector)
                    names.extend(component_names)

        extra_values = np.zeros((0,), dtype=np.float32)
        extra_names: list[str] = []
        if config.extra_feature_paths:
            extra_values = np.asarray(
                [_resolve_numeric_feature(example, feature_path) for feature_path in config.extra_feature_paths],
                dtype=np.float32,
            )
            extra_names = [sanitize_name(feature_path) for feature_path in config.extra_feature_paths]
            if config.engineered_feature_set == "raw":
                parts.append(extra_values)
                names.extend(extra_names)

        if config.include_hidden_features and config.engineered_feature_set != "raw":
            row_vector, row_names = _build_engineered_feature_row(
                component_vectors=component_vectors,
                extra_values=extra_values,
                extra_names=extra_names,
                feature_set=config.engineered_feature_set,
            )
            parts = [row_vector]
            names = row_names

        if not parts:
            raise ValueError("At least one hidden-state or auxiliary feature must be enabled.")

        row_vector = np.concatenate(parts, axis=0)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature names changed across rows. Hidden-state dimensions are inconsistent.")

        feature_rows.append(row_vector)
        metadata_rows.append(
            {
                "dataset_name": example.dataset_name,
                "task_id": example.task_id,
                "split": example.split,
                "user_input": example.index_row.get("user_input") or example.label_row.get("user_input"),
            }
        )

    return np.stack(feature_rows, axis=0), feature_names or [], metadata_rows
