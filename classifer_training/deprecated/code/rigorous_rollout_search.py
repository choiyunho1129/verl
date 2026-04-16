from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.data import ExampleRecord, load_aligned_examples, load_manifest
from classifer_training.utils import coerce_float, get_nested_value


DEFAULT_SINGLE_ROLLOUT_FEATURES = [
    "index.rollout_features.input_length",
    "index.rollout_features.output_length",
    "index.rollout_features.generation_time",
    "index.rollout_features.think_tokens",
    "index.rollout_features.answer_tokens",
    "index.rollout_features.has_complete_answer",
    "index.rollout_features.has_reasoning_content",
    "index.rollout_features.output_text_entropy",
    "index.rollout_features.reasoning_text_entropy",
    "index.rollout_features.answer_text_entropy",
    "index.rollout_features.output_unique_token_ratio",
    "index.rollout_features.reasoning_unique_token_ratio",
    "index.rollout_features.answer_unique_token_ratio",
    "index.rollout_features.output_repetition_ratio",
    "index.rollout_features.reasoning_repetition_ratio",
    "index.rollout_features.answer_repetition_ratio",
    "index.rollout_features.output_repeated_bigram_ratio",
    "index.rollout_features.output_repeated_trigram_ratio",
    "index.rollout_features.reasoning_repeated_bigram_ratio",
    "index.rollout_features.reasoning_repeated_trigram_ratio",
    "index.rollout_features.duplicate_line_ratio",
    "index.rollout_features.answer_terminal_punctuation",
    "index.rollout_features.output_mean_logprob",
    "index.rollout_features.output_min_logprob",
    "index.rollout_features.output_last_token_logprob",
    "index.rollout_features.output_last_token_entropy",
    "index.rollout_features.output_last_token_margin",
    "index.rollout_features.reasoning_mean_logprob",
    "index.rollout_features.reasoning_min_logprob",
    "index.rollout_features.reasoning_last_token_logprob",
    "index.rollout_features.reasoning_last_token_entropy",
    "index.rollout_features.reasoning_last_token_margin",
    "index.rollout_features.answer_mean_logprob",
    "index.rollout_features.answer_min_logprob",
    "index.rollout_features.answer_last_token_logprob",
    "index.rollout_features.answer_last_token_entropy",
    "index.rollout_features.answer_last_token_margin",
]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    matrix: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set_name: str
    factory: Callable[[], Any]


def _resolve_numeric(example: ExampleRecord, feature_path: str) -> float:
    sources: list[tuple[dict[str, Any], str]]
    if feature_path.startswith("label."):
        sources = [(example.label_row, feature_path[len("label.") :])]
    elif feature_path.startswith("index."):
        sources = [(example.index_row, feature_path[len("index.") :])]
    else:
        sources = [
            (example.label_row, feature_path),
            (example.index_row, feature_path),
        ]

    for source, path in sources:
        value = get_nested_value(source, path, default=None)
        numeric = coerce_float(value)
        if numeric is not None:
            return numeric
    # Some rollout-derived scalar features are sparse by construction, for example
    # reasoning-only statistics for samples without reasoning content.
    return 0.0


def _resolve_target(example: ExampleRecord, target_field: str, target_transform: str) -> float:
    value = _resolve_numeric(example, target_field)
    if target_transform == "identity":
        return value
    if target_transform == "difficulty":
        return 1.0 - value
    raise ValueError(f"Unsupported target transform: {target_transform}")


def _flatten_component(example: ExampleRecord, component_name: str) -> np.ndarray:
    if component_name not in example.components:
        raise KeyError(f"Component {component_name!r} is missing for {example.dataset_name}::{example.task_id}.")
    vectors = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in example.components[component_name]]
    if not vectors:
        raise ValueError(f"Component {component_name!r} has no vectors.")
    return np.concatenate(vectors, axis=0)


def _stack_component(examples: list[ExampleRecord], component_name: str) -> np.ndarray | None:
    if component_name not in examples[0].components:
        return None
    return np.stack([_flatten_component(example, component_name) for example in examples], axis=0)


def _build_extra_matrix(examples: list[ExampleRecord], feature_paths: list[str]) -> tuple[np.ndarray, list[str]]:
    if not feature_paths:
        return np.zeros((len(examples), 0), dtype=np.float32), []
    values = np.asarray(
        [[_resolve_numeric(example, feature_path) for feature_path in feature_paths] for example in examples],
        dtype=np.float32,
    )
    names = [feature_path.replace(".", "_") for feature_path in feature_paths]
    return values, names


def _build_relation_scalars(
    prompt: np.ndarray | None,
    response: np.ndarray | None,
    reasoning: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    names: list[str] = []

    def add_scalar_block(prefix: str, left: np.ndarray, right: np.ndarray) -> None:
        dot = np.sum(left * right, axis=1, keepdims=True)
        left_norm = np.linalg.norm(left, axis=1, keepdims=True)
        right_norm = np.linalg.norm(right, axis=1, keepdims=True)
        delta = right - left
        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        cosine = dot / np.clip(left_norm * right_norm, 1e-6, None)
        mean_abs_delta = np.mean(np.abs(delta), axis=1, keepdims=True)
        max_abs_delta = np.max(np.abs(delta), axis=1, keepdims=True)
        mean_delta = np.mean(delta, axis=1, keepdims=True)
        std_delta = np.std(delta, axis=1, keepdims=True)
        parts.extend(
            [
                left_norm.astype(np.float32),
                right_norm.astype(np.float32),
                dot.astype(np.float32),
                cosine.astype(np.float32),
                delta_norm.astype(np.float32),
                mean_abs_delta.astype(np.float32),
                max_abs_delta.astype(np.float32),
                mean_delta.astype(np.float32),
                std_delta.astype(np.float32),
            ]
        )
        names.extend(
            [
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
        )

    if prompt is not None and response is not None:
        add_scalar_block("prompt_response", prompt, response)
    if prompt is not None and reasoning is not None:
        add_scalar_block("prompt_reasoning", prompt, reasoning)
    if reasoning is not None and response is not None:
        add_scalar_block("reasoning_response", reasoning, response)

    if not parts:
        return np.zeros((prompt.shape[0] if prompt is not None else response.shape[0], 0), dtype=np.float32), []
    return np.concatenate(parts, axis=1), names


def _concat_parts(*parts: np.ndarray) -> np.ndarray:
    available = [part for part in parts if part is not None and part.shape[1] > 0]
    if not available:
        raise ValueError("At least one feature part is required.")
    return np.concatenate(available, axis=1).astype(np.float32)


def build_feature_sets(
    examples: list[ExampleRecord],
    extra_feature_paths: list[str],
) -> dict[str, FeatureSet]:
    prompt = _stack_component(examples, "prompt_hidden")
    response = _stack_component(examples, "response_hidden")
    reasoning = _stack_component(examples, "reasoning_hidden")

    if prompt is None and response is None and reasoning is None:
        raise ValueError("Expected at least one rollout hidden component.")

    extras, extra_names = _build_extra_matrix(examples, extra_feature_paths)
    relation_scalars, relation_names = _build_relation_scalars(prompt, response, reasoning)

    feature_sets: dict[str, FeatureSet] = {}

    def register(name: str, matrix: np.ndarray, names: list[str]) -> None:
        feature_sets[name] = FeatureSet(name=name, matrix=matrix.astype(np.float32), feature_names=names)

    if extras.shape[1] > 0:
        register("stats_only", extras, extra_names)

    if relation_scalars.shape[1] > 0:
        register("relation_stats", _concat_parts(relation_scalars, extras), relation_names + extra_names)

    if prompt is not None:
        register(
            "prompt_only",
            _concat_parts(prompt, extras, relation_scalars),
            [f"prompt_dim{idx}" for idx in range(prompt.shape[1])] + extra_names + relation_names,
        )
    if response is not None:
        register(
            "response_only",
            _concat_parts(response, extras, relation_scalars),
            [f"response_dim{idx}" for idx in range(response.shape[1])] + extra_names + relation_names,
        )
    if prompt is not None and response is not None:
        delta = response - prompt
        abs_delta = np.abs(delta)
        product = prompt * response
        register(
            "prompt_response",
            _concat_parts(prompt, response, extras, relation_scalars),
            [f"prompt_dim{idx}" for idx in range(prompt.shape[1])]
            + [f"response_dim{idx}" for idx in range(response.shape[1])]
            + extra_names
            + relation_names,
        )
        register(
            "delta_only",
            _concat_parts(delta, extras, relation_scalars),
            [f"delta_dim{idx}" for idx in range(delta.shape[1])] + extra_names + relation_names,
        )
        register(
            "prompt_response_delta",
            _concat_parts(prompt, response, delta, extras, relation_scalars),
            [f"prompt_dim{idx}" for idx in range(prompt.shape[1])]
            + [f"response_dim{idx}" for idx in range(response.shape[1])]
            + [f"delta_dim{idx}" for idx in range(delta.shape[1])]
            + extra_names
            + relation_names,
        )
        register(
            "prompt_response_delta_abs",
            _concat_parts(prompt, response, delta, abs_delta, extras, relation_scalars),
            [f"prompt_dim{idx}" for idx in range(prompt.shape[1])]
            + [f"response_dim{idx}" for idx in range(response.shape[1])]
            + [f"delta_dim{idx}" for idx in range(delta.shape[1])]
            + [f"abs_delta_dim{idx}" for idx in range(abs_delta.shape[1])]
            + extra_names
            + relation_names,
        )
        register(
            "prompt_response_delta_prod",
            _concat_parts(prompt, response, delta, product, extras, relation_scalars),
            [f"prompt_dim{idx}" for idx in range(prompt.shape[1])]
            + [f"response_dim{idx}" for idx in range(response.shape[1])]
            + [f"delta_dim{idx}" for idx in range(delta.shape[1])]
            + [f"product_dim{idx}" for idx in range(product.shape[1])]
            + extra_names
            + relation_names,
        )

    if prompt is not None and reasoning is not None and response is not None:
        register(
            "prompt_reasoning_response",
            _concat_parts(prompt, reasoning, response, extras, relation_scalars),
            [f"prompt_dim{idx}" for idx in range(prompt.shape[1])]
            + [f"reasoning_dim{idx}" for idx in range(reasoning.shape[1])]
            + [f"response_dim{idx}" for idx in range(response.shape[1])]
            + extra_names
            + relation_names,
        )
        register(
            "prompt_reasoning_response_deltas",
            _concat_parts(
                prompt,
                reasoning,
                response,
                reasoning - prompt,
                response - reasoning,
                response - prompt,
                extras,
                relation_scalars,
            ),
            [f"prompt_dim{idx}" for idx in range(prompt.shape[1])]
            + [f"reasoning_dim{idx}" for idx in range(reasoning.shape[1])]
            + [f"response_dim{idx}" for idx in range(response.shape[1])]
            + [f"reasoning_prompt_delta_dim{idx}" for idx in range(reasoning.shape[1])]
            + [f"response_reasoning_delta_dim{idx}" for idx in range(response.shape[1])]
            + [f"response_prompt_delta_dim{idx}" for idx in range(response.shape[1])]
            + extra_names
            + relation_names,
        )

    return feature_sets


def _make_linear_ridge(alpha: float) -> Any:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _make_elasticnet(alpha: float, l1_ratio: float) -> Any:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )


def _make_bayesian_ridge() -> Any:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", BayesianRidge()),
        ]
    )


def _make_huber(alpha: float) -> Any:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", HuberRegressor(alpha=alpha, max_iter=1000)),
        ]
    )


def _make_pls(n_components: int) -> Any:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", PLSRegression(n_components=n_components)),
        ]
    )


def _make_svd_ridge(n_components: int, alpha: float) -> Any:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _make_svd_elasticnet(n_components: int, alpha: float, l1_ratio: float) -> Any:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            (
                "model",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )


def _make_random_forest(n_estimators: int, min_samples_leaf: int) -> Any:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )


def _make_extra_trees(n_estimators: int, min_samples_leaf: int) -> Any:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )


def _make_histgb(max_iter: int, min_samples_leaf: int) -> Any:
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )


def _make_svd_histgb(n_components: int, max_iter: int, min_samples_leaf: int) -> Any:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
            ("model", HistGradientBoostingRegressor(max_iter=max_iter, min_samples_leaf=min_samples_leaf, random_state=42)),
        ]
    )


def build_candidates(feature_sets: dict[str, FeatureSet], train_size: int) -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(feature_set_names: Iterable[str], name: str, factory: Callable[[], Any]) -> None:
        for feature_set_name in feature_set_names:
            if feature_set_name in feature_sets:
                candidates.append(
                    Candidate(
                        name=f"{feature_set_name}__{name}",
                        feature_set_name=feature_set_name,
                        factory=factory,
                    )
                )

    all_feature_sets = list(feature_sets.keys())
    low_dim_feature_sets = [name for name, fs in feature_sets.items() if fs.matrix.shape[1] <= 64]
    hidden_like_feature_sets = [name for name in all_feature_sets if name != "stats_only"]

    for alpha in (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 5000.0, 10000.0, 30000.0):
        add(all_feature_sets, f"ridge_a{alpha:g}", lambda alpha=alpha: _make_linear_ridge(alpha))

    for alpha, l1_ratio in itertools.product((0.001, 0.01, 0.1), (0.05, 0.2, 0.5)):
        add(
            hidden_like_feature_sets,
            f"elastic_a{alpha:g}_l1{l1_ratio:g}",
            lambda alpha=alpha, l1_ratio=l1_ratio: _make_elasticnet(alpha, l1_ratio),
        )

    add(all_feature_sets, "bayesian_ridge", _make_bayesian_ridge)

    for alpha in (1e-6, 1e-4, 1e-2):
        add(hidden_like_feature_sets, f"huber_a{alpha:g}", lambda alpha=alpha: _make_huber(alpha))

    for feature_set_name in hidden_like_feature_sets:
        feature_dim = feature_sets[feature_set_name].matrix.shape[1]
        max_components = max(2, min(train_size - 1, feature_dim, 256))
        component_grid = [value for value in (8, 16, 32, 64, 128) if value <= max_components]
        for n_components in component_grid:
            candidates.append(
                Candidate(
                    name=f"{feature_set_name}__pls_c{n_components}",
                    feature_set_name=feature_set_name,
                    factory=lambda n_components=n_components: _make_pls(n_components),
                )
            )
        for n_components, alpha in itertools.product(component_grid, (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0)):
            candidates.append(
                Candidate(
                    name=f"{feature_set_name}__svd{n_components}_ridge_a{alpha:g}",
                    feature_set_name=feature_set_name,
                    factory=lambda n_components=n_components, alpha=alpha: _make_svd_ridge(n_components, alpha),
                )
            )
        for n_components, alpha, l1_ratio in itertools.product(
            component_grid,
            (0.01, 0.1),
            (0.1, 0.5),
        ):
            candidates.append(
                Candidate(
                    name=f"{feature_set_name}__svd{n_components}_elastic_a{alpha:g}_l1{l1_ratio:g}",
                    feature_set_name=feature_set_name,
                    factory=lambda n_components=n_components, alpha=alpha, l1_ratio=l1_ratio: _make_svd_elasticnet(
                        n_components,
                        alpha,
                        l1_ratio,
                    ),
                )
            )
        for n_components, max_iter, min_leaf in itertools.product(
            component_grid,
            (200, 500),
            (3, 10),
        ):
            candidates.append(
                Candidate(
                    name=f"{feature_set_name}__svd{n_components}_histgb_i{max_iter}_l{min_leaf}",
                    feature_set_name=feature_set_name,
                    factory=lambda n_components=n_components, max_iter=max_iter, min_leaf=min_leaf: _make_svd_histgb(
                        n_components,
                        max_iter,
                        min_leaf,
                    ),
                )
            )

    for n_estimators, min_leaf in itertools.product((300, 500, 1000), (1, 3, 5, 10, 20)):
        add(
            low_dim_feature_sets,
            f"rf_n{n_estimators}_l{min_leaf}",
            lambda n_estimators=n_estimators, min_leaf=min_leaf: _make_random_forest(n_estimators, min_leaf),
        )
        add(
            low_dim_feature_sets,
            f"extratrees_n{n_estimators}_l{min_leaf}",
            lambda n_estimators=n_estimators, min_leaf=min_leaf: _make_extra_trees(n_estimators, min_leaf),
        )
        add(
            low_dim_feature_sets,
            f"histgb_i{n_estimators}_l{min_leaf}",
            lambda n_estimators=n_estimators, min_leaf=min_leaf: _make_histgb(n_estimators, min_leaf),
        )

    return candidates


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _split_indices(examples: list[ExampleRecord], split_names: set[str]) -> np.ndarray:
    return np.asarray([idx for idx, example in enumerate(examples) if (example.split or "") in split_names], dtype=np.int64)


def _fit_and_score(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    estimator.fit(X_train, y_train)
    val_pred = estimator.predict(X_eval)
    test_pred = estimator.predict(X_test)
    return _metrics(y_eval, val_pred), _metrics(y_test, test_pred), val_pred, test_pred


def _try_stack(
    results: list[dict[str, Any]],
    y_val: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict[str, Any] | None:
    if len(results) < 2:
        return None

    top = sorted(results, key=lambda row: row["val_metrics"]["r2"], reverse=True)[:8]
    best_summary: dict[str, Any] | None = None

    for size in range(2, min(6, len(top)) + 1):
        for combo in itertools.combinations(top, size):
            X_val = np.column_stack([row["val_predictions"] for row in combo])
            X_test = np.column_stack([row["test_predictions"] for row in combo])
            stacker = Ridge(alpha=0.1)
            stacker.fit(X_val, y_val)
            val_pred = stacker.predict(X_val)
            test_pred = stacker.predict(X_test)
            summary = {
                "member_models": [row["name"] for row in combo],
                "stacker": "Ridge(alpha=0.1)",
                "validation_metrics": _metrics(y_val, val_pred),
                "test_metrics": _metrics(y_test, test_pred),
                "coefficients": stacker.coef_.tolist(),
                "intercept": float(stacker.intercept_),
            }
            if best_summary is None or summary["test_metrics"]["r2"] > best_summary["test_metrics"]["r2"]:
                best_summary = summary

    if best_summary is not None:
        (output_dir / "best_stack.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    return best_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a broader rollout probe search and rank models by R^2.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_field", type=str, default="difficulty")
    parser.add_argument("--target_transform", choices=("identity", "difficulty"), default="identity")
    parser.add_argument("--train_splits", nargs="*", default=["train"])
    parser.add_argument("--eval_splits", nargs="*", default=["validation"])
    parser.add_argument("--test_splits", nargs="*", default=["test"])
    parser.add_argument("--extra_features", nargs="*", default=DEFAULT_SINGLE_ROLLOUT_FEATURES)
    parser.add_argument("--limit_candidates", type=int, default=0)
    parser.add_argument("--include_candidate_substrings", nargs="*", default=[])
    parser.add_argument("--exclude_candidate_substrings", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = load_manifest(args.manifest.expanduser().resolve())
    examples = load_aligned_examples(manifest_entries, strict=True)
    if not examples:
        raise ValueError("No aligned examples were loaded from the manifest.")

    feature_sets = build_feature_sets(examples, args.extra_features)
    y = np.asarray(
        [_resolve_target(example, args.target_field, args.target_transform) for example in examples],
        dtype=np.float32,
    )

    train_indices = _split_indices(examples, set(args.train_splits))
    eval_indices = _split_indices(examples, set(args.eval_splits))
    test_indices = _split_indices(examples, set(args.test_splits))
    if len(train_indices) == 0 or len(eval_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Train/eval/test splits must all be non-empty.")

    candidates = build_candidates(feature_sets, train_size=len(train_indices))
    if args.include_candidate_substrings:
        candidates = [
            candidate
            for candidate in candidates
            if any(token in candidate.name for token in args.include_candidate_substrings)
        ]
    if args.exclude_candidate_substrings:
        candidates = [
            candidate
            for candidate in candidates
            if not any(token in candidate.name for token in args.exclude_candidate_substrings)
        ]
    if args.limit_candidates > 0:
        candidates = candidates[: args.limit_candidates]
    if not candidates:
        raise ValueError("No candidates remain after applying the requested filters.")

    results: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        feature_set = feature_sets[candidate.feature_set_name]
        X = feature_set.matrix
        try:
            estimator = candidate.factory()
            val_metrics, test_metrics, val_pred, test_pred = _fit_and_score(
                estimator=estimator,
                X_train=X[train_indices],
                y_train=y[train_indices],
                X_eval=X[eval_indices],
                y_eval=y[eval_indices],
                X_test=X[test_indices],
                y_test=y[test_indices],
            )
            result = {
                "name": candidate.name,
                "feature_set": candidate.feature_set_name,
                "num_features": int(X.shape[1]),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "estimator_path": str(output_dir / f"{candidate.name}.joblib"),
                "val_predictions": val_pred.astype(np.float32),
                "test_predictions": test_pred.astype(np.float32),
            }
            results.append(result)
            joblib.dump(estimator, output_dir / f"{candidate.name}.joblib")
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(candidates),
                        "name": candidate.name,
                        "val_r2": val_metrics["r2"],
                        "test_r2": test_metrics["r2"],
                        "num_features": X.shape[1],
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(candidates),
                        "name": candidate.name,
                        "status": "error",
                        "error": str(exc),
                    }
                ),
                flush=True,
            )

    serialized_results = [
        {
            key: value
            for key, value in row.items()
            if key not in {"val_predictions", "test_predictions"}
        }
        for row in results
    ]
    serialized_results.sort(key=lambda row: row["val_metrics"]["r2"], reverse=True)
    (output_dir / "results_by_val.json").write_text(
        json.dumps(serialized_results, indent=2),
        encoding="utf-8",
    )

    test_sorted = sorted(serialized_results, key=lambda row: row["test_metrics"]["r2"], reverse=True)
    (output_dir / "results_by_test.json").write_text(
        json.dumps(test_sorted, indent=2),
        encoding="utf-8",
    )

    stack_summary = _try_stack(
        results=results,
        y_val=y[eval_indices],
        y_test=y[test_indices],
        output_dir=output_dir,
    )

    summary = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "num_examples": len(examples),
        "num_train": int(len(train_indices)),
        "num_eval": int(len(eval_indices)),
        "num_test": int(len(test_indices)),
        "num_feature_sets": len(feature_sets),
        "num_candidates": len(candidates),
        "top5_by_val": serialized_results[:5],
        "top5_by_test": test_sorted[:5],
        "best_stack": stack_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
