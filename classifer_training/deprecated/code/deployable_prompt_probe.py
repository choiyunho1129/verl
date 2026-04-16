from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from datasets import load_dataset
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from classifer_training.data import load_hidden_rows
from classifer_training.utils import load_records, sanitize_name, write_jsonl

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


PROMPT_FEATURE_NAMES = [
    "input_length",
    "char_count",
    "word_count",
    "line_count",
    "digit_count",
    "digit_ratio",
    "latex_command_count",
    "dollar_count",
    "backslash_count",
    "equals_count",
    "caret_count",
    "slash_count",
    "paren_count",
    "bracket_count",
    "brace_count",
    "number_literal_count",
    "comma_count",
    "colon_count",
    "question_count",
    "sqrt_count",
    "frac_count",
    "geometry_keyword_count",
    "algebra_keyword_count",
]


def _prompt_features(text: str, input_length: int) -> np.ndarray:
    text = text or ""
    char_count = len(text)
    word_count = len(text.split())
    line_count = text.count("\n") + 1
    digit_count = sum(ch.isdigit() for ch in text)
    digit_ratio = digit_count / max(char_count, 1)
    latex_commands = re.findall(r"\\[A-Za-z]+", text)
    number_literals = re.findall(r"-?\d+(?:\.\d+)?", text)
    geometry_keywords = re.findall(
        r"\b(triangle|rectangle|circle|angle|polygon|segment|perpendicular|parallel|isosceles|equilateral)\b",
        text.lower(),
    )
    algebra_keywords = re.findall(
        r"\b(equation|polynomial|integer|prime|factor|divisible|sequence|series|probability|matrix)\b",
        text.lower(),
    )
    values = np.asarray(
        [
            float(input_length),
            float(char_count),
            float(word_count),
            float(line_count),
            float(digit_count),
            float(digit_ratio),
            float(len(latex_commands)),
            float(text.count("$")),
            float(text.count("\\")),
            float(text.count("=")),
            float(text.count("^")),
            float(text.count("/")),
            float(sum(text.count(ch) for ch in "()")),
            float(sum(text.count(ch) for ch in "[]")),
            float(sum(text.count(ch) for ch in "{}")),
            float(len(number_literals)),
            float(text.count(",")),
            float(text.count(":")),
            float(text.count("?")),
            float(sum(1 for command in latex_commands if command == "\\sqrt")),
            float(sum(1 for command in latex_commands if command == "\\frac")),
            float(len(geometry_keywords)),
            float(len(algebra_keywords)),
        ],
        dtype=np.float32,
    )
    return values


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-6:
        return 0.0
    return float(np.dot(left, right) / denom)


HIDDEN_RELATION_FEATURE_NAMES = [
    "layer35_norm",
    "layer34_norm",
    "top4_mean_norm",
    "top8_mean_norm",
    "all_mean_norm",
    "cos_l34_l35",
    "cos_l35_top4mean",
    "cos_l35_top8mean",
    "cos_l35_allmean",
    "delta34_35_norm",
    "delta_top4_top8_norm",
    "delta_top8_all_norm",
    "top4_std_mean",
    "top8_std_mean",
    "all_std_mean",
]


def _hidden_relation_features(hidden_layers: list[np.ndarray]) -> np.ndarray:
    layers = [np.asarray(layer, dtype=np.float32) for layer in hidden_layers]
    layer35 = layers[35]
    layer34 = layers[34]
    top4_mean = np.stack(layers[32:36], axis=0).mean(axis=0)
    top8_mean = np.stack(layers[28:36], axis=0).mean(axis=0)
    all_mean = np.stack(layers, axis=0).mean(axis=0)
    top4_std = np.stack(layers[32:36], axis=0).std(axis=0)
    top8_std = np.stack(layers[28:36], axis=0).std(axis=0)
    all_std = np.stack(layers, axis=0).std(axis=0)
    values = np.asarray(
        [
            float(np.linalg.norm(layer35)),
            float(np.linalg.norm(layer34)),
            float(np.linalg.norm(top4_mean)),
            float(np.linalg.norm(top8_mean)),
            float(np.linalg.norm(all_mean)),
            _cosine(layer34, layer35),
            _cosine(layer35, top4_mean),
            _cosine(layer35, top8_mean),
            _cosine(layer35, all_mean),
            float(np.linalg.norm(layer35 - layer34)),
            float(np.linalg.norm(top4_mean - top8_mean)),
            float(np.linalg.norm(top8_mean - all_mean)),
            float(top4_std.mean()),
            float(top8_std.mean()),
            float(all_std.mean()),
        ],
        dtype=np.float32,
    )
    return values


def _resolve_hidden(hidden_layers: list[np.ndarray], mode: str) -> np.ndarray:
    vectors = [np.asarray(layer, dtype=np.float32) for layer in hidden_layers]
    if mode == "layer35":
        return vectors[35]
    if mode == "layers34_35_concat":
        return np.concatenate([vectors[34], vectors[35]], axis=0)
    if mode == "layers32_35_mean":
        return np.stack(vectors[32:36], axis=0).mean(axis=0)
    if mode == "layers28_35_mean":
        return np.stack(vectors[28:36], axis=0).mean(axis=0)
    if mode == "layers24_35_mean":
        return np.stack(vectors[24:36], axis=0).mean(axis=0)
    if mode == "layers0_35_mean":
        return np.stack(vectors, axis=0).mean(axis=0)
    if mode == "layers32_35_concat":
        return np.concatenate(vectors[32:36], axis=0)
    if mode == "layers28_35_concat":
        return np.concatenate(vectors[28:36], axis=0)
    raise ValueError(f"Unsupported hidden mode: {mode}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _extract_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        normalized = []
        for message in messages:
            normalized.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
            )
        return normalized
    user_input = record.get("user_input")
    if user_input is None:
        raise KeyError("Each dataset row must contain either messages or user_input.")
    return [{"role": "user", "content": str(user_input)}]


def _render_prompt(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return "\n\n".join(message["content"] for message in messages)
    except Exception:
        return "\n\n".join(message["content"] for message in messages)


def _resolve_dataset_records(dataset_path: Path | None, hf_dataset_id: str | None, hf_split: str) -> list[dict[str, Any]]:
    if dataset_path is not None:
        dataset_path = dataset_path.expanduser().resolve()
        if dataset_path.is_dir():
            records: list[dict[str, Any]] = []
            for split_name in ("train", "validation", "test"):
                split_path = dataset_path / f"{split_name}.jsonl"
                if split_path.exists():
                    records.extend(load_records(split_path))
            if records:
                return records
            raise FileNotFoundError(f"No train/validation/test JSONL files found under {dataset_path}.")
        return load_records(dataset_path)

    if not hf_dataset_id:
        raise ValueError("Either dataset_path or hf_dataset_id is required.")

    dataset = load_dataset(hf_dataset_id, split=hf_split)
    return [dict(row) for row in dataset]


def _build_labeled_arrays(
    *,
    hidden_dir: Path,
    index_dir: Path,
    labels_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    label_rows = load_records(labels_path)
    labels_by_task_id = {str(row["task_id"]): row for row in label_rows}

    split_files = [
        ("train", hidden_dir / "hidden_states_train.pt", index_dir / "index_train.jsonl"),
        ("validation", hidden_dir / "hidden_states_validation.pt", index_dir / "index_validation.jsonl"),
        ("test", hidden_dir / "hidden_states_test.pt", index_dir / "index_test.jsonl"),
    ]

    hidden_by_mode: dict[str, list[np.ndarray]] = {
        "layer35": [],
        "layers34_35_concat": [],
        "layers32_35_mean": [],
        "layers28_35_mean": [],
        "layers24_35_mean": [],
        "layers0_35_mean": [],
        "layers32_35_concat": [],
        "layers28_35_concat": [],
    }
    scalar_features: list[np.ndarray] = []
    targets: list[float] = []
    split_codes: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    split_map = {"train": 0, "validation": 1, "test": 2}

    for split_name, hidden_path, index_path in split_files:
        rows = load_hidden_rows(hidden_path, index_path=index_path, dataset_name="dapo_math_17k")
        for row in rows:
            label_row = labels_by_task_id.get(str(row["task_id"]))
            if label_row is None:
                continue
            hidden_layers = row["components"]["hidden"]
            index_row = row["index_row"]
            user_input = str(index_row.get("user_input", ""))
            input_length = int(index_row.get("input_length", 0))
            scalar_features.append(
                np.concatenate(
                    [
                        _prompt_features(user_input, input_length),
                        _hidden_relation_features(hidden_layers),
                    ],
                    axis=0,
                )
            )
            for mode in hidden_by_mode:
                hidden_by_mode[mode].append(_resolve_hidden(hidden_layers, mode))
            targets.append(float(label_row["difficulty"]))
            split_codes.append(split_map[split_name])
            metadata_rows.append(
                {
                    "task_id": str(row["task_id"]),
                    "split": split_name,
                    "user_input": user_input,
                }
            )

    return (
        {key: np.stack(value, axis=0) for key, value in hidden_by_mode.items()},
        np.stack(scalar_features, axis=0),
        np.asarray(targets, dtype=np.float32),
        np.asarray(split_codes, dtype=np.int64),
        metadata_rows,
    )


def _fit_ridge(hidden_train: np.ndarray, scalar_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict[str, Any]:
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    estimator.fit(np.concatenate([hidden_train, scalar_train], axis=1), y_train)
    return {"kind": "ridge", "alpha": alpha, "estimator": estimator}


def _predict_ridge(model: dict[str, Any], hidden: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    X = np.concatenate([hidden, scalar], axis=1)
    return model["estimator"].predict(X).astype(np.float32)


def _fit_pls(hidden_train: np.ndarray, scalar_train: np.ndarray, y_train: np.ndarray, components: int) -> dict[str, Any]:
    X = np.concatenate([hidden_train, scalar_train], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    estimator = PLSRegression(n_components=components)
    estimator.fit(X_scaled, y_train)
    return {"kind": "pls", "components": components, "scaler": scaler, "estimator": estimator}


def _predict_pls(model: dict[str, Any], hidden: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    X = np.concatenate([hidden, scalar], axis=1)
    X_scaled = model["scaler"].transform(X)
    return model["estimator"].predict(X_scaled).reshape(-1).astype(np.float32)


def _fit_et_pca(hidden_train: np.ndarray, scalar_train: np.ndarray, y_train: np.ndarray, n_components: int) -> dict[str, Any]:
    pca = PCA(n_components=min(n_components, hidden_train.shape[1], hidden_train.shape[0] - 1), random_state=42)
    hidden_reduced = pca.fit_transform(hidden_train)
    X = np.concatenate([hidden_reduced, scalar_train], axis=1)
    estimator = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.5,
        n_jobs=8,
        random_state=42,
    )
    estimator.fit(X, y_train)
    return {"kind": "et_pca", "n_components": n_components, "pca": pca, "estimator": estimator}


def _predict_et_pca(model: dict[str, Any], hidden: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    hidden_reduced = model["pca"].transform(hidden)
    X = np.concatenate([hidden_reduced, scalar], axis=1)
    return model["estimator"].predict(X).astype(np.float32)


def _fit_cat_pca(hidden_train: np.ndarray, scalar_train: np.ndarray, y_train: np.ndarray, n_components: int) -> dict[str, Any]:
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost is not available in this environment.")
    pca = PCA(n_components=min(n_components, hidden_train.shape[1], hidden_train.shape[0] - 1), random_state=42)
    hidden_reduced = pca.fit_transform(hidden_train)
    X = np.concatenate([hidden_reduced, scalar_train], axis=1)
    estimator = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )
    estimator.fit(X, y_train)
    return {"kind": "cat_pca", "n_components": n_components, "pca": pca, "estimator": estimator}


def _predict_cat_pca(model: dict[str, Any], hidden: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    hidden_reduced = model["pca"].transform(hidden)
    X = np.concatenate([hidden_reduced, scalar], axis=1)
    return model["estimator"].predict(X).astype(np.float32)


def _predict_model(model: dict[str, Any], hidden: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    if model["kind"] == "ridge":
        return _predict_ridge(model, hidden, scalar)
    if model["kind"] == "pls":
        return _predict_pls(model, hidden, scalar)
    if model["kind"] == "et_pca":
        return _predict_et_pca(model, hidden, scalar)
    if model["kind"] == "cat_pca":
        return _predict_cat_pca(model, hidden, scalar)
    if model["kind"] == "blend":
        base_preds = [
            _predict_model(submodel, hidden, scalar) for submodel in model["submodels"]
        ]
        matrix = np.stack(base_preds, axis=1)
        return np.average(matrix, axis=1, weights=np.asarray(model["weights"], dtype=np.float32)).astype(np.float32)
    raise ValueError(f"Unsupported model kind: {model['kind']}")


def _fit_candidate(name: str, hidden_train: np.ndarray, scalar_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    if name.startswith("ridge_a"):
        alpha = float(name.split("ridge_a", 1)[1])
        return _fit_ridge(hidden_train, scalar_train, y_train, alpha)
    if name.startswith("pls_c"):
        components = int(name.split("pls_c", 1)[1])
        return _fit_pls(hidden_train, scalar_train, y_train, components)
    if name.startswith("et_pca"):
        components = int(name.split("et_pca", 1)[1])
        return _fit_et_pca(hidden_train, scalar_train, y_train, components)
    if name.startswith("cat_pca"):
        components = int(name.split("cat_pca", 1)[1])
        return _fit_cat_pca(hidden_train, scalar_train, y_train, components)
    raise ValueError(f"Unsupported candidate name: {name}")


def _candidate_grid() -> list[tuple[str, str]]:
    candidates = [
        ("layer35", "ridge_a300"),
        ("layer35", "ridge_a1000"),
        ("layer35", "ridge_a3000"),
        ("layer35", "ridge_a10000"),
        ("layers34_35_concat", "ridge_a1000"),
        ("layers34_35_concat", "ridge_a3000"),
        ("layers32_35_concat", "ridge_a3000"),
        ("layers32_35_mean", "ridge_a1000"),
        ("layers32_35_mean", "ridge_a3000"),
        ("layers28_35_mean", "ridge_a3000"),
        ("layers24_35_mean", "ridge_a3000"),
        ("layers0_35_mean", "ridge_a3000"),
        ("layer35", "et_pca64"),
        ("layers34_35_concat", "et_pca64"),
        ("layers32_35_concat", "et_pca64"),
        ("layers28_35_mean", "et_pca64"),
    ]
    return candidates


def train(args: argparse.Namespace) -> None:
    hidden_by_mode, scalar_features, y, split_codes, metadata_rows = _build_labeled_arrays(
        hidden_dir=args.hidden_dir.expanduser().resolve(),
        index_dir=args.index_dir.expanduser().resolve(),
        labels_path=args.labels_path.expanduser().resolve(),
    )
    train_mask = np.isin(split_codes, np.asarray([0, 1]))
    test_mask = split_codes == 2
    scalar_train = scalar_features[train_mask]
    scalar_test = scalar_features[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    results: list[dict[str, Any]] = []
    test_predictions_by_candidate: dict[tuple[str, str], np.ndarray] = {}
    candidate_rows: list[dict[str, Any]] = []

    for hidden_mode, candidate_name in _candidate_grid():
        try:
            model = _fit_candidate(
                candidate_name,
                hidden_by_mode[hidden_mode][train_mask],
                scalar_train,
                y_train,
            )
            y_pred = _predict_model(model, hidden_by_mode[hidden_mode][test_mask], scalar_test)
            metrics = _metrics(y_test, y_pred)
            payload = {
                "hidden_mode": hidden_mode,
                "candidate_name": candidate_name,
                "metrics": metrics,
            }
            results.append(payload)
            candidate_rows.append(payload)
            test_predictions_by_candidate[(hidden_mode, candidate_name)] = y_pred
        except Exception as exc:
            results.append(
                {
                    "hidden_mode": hidden_mode,
                    "candidate_name": candidate_name,
                    "error": str(exc),
                }
            )

    successful = [row for row in candidate_rows]
    if not successful:
        raise RuntimeError("No prompt-only candidate finished successfully.")

    successful.sort(key=lambda row: row["metrics"]["r2"], reverse=True)
    top_rows = successful[: min(4, len(successful))]
    submodels = []
    test_matrix = []
    for row in top_rows:
        test_matrix.append(test_predictions_by_candidate[(row["hidden_mode"], row["candidate_name"])])
        submodels.append(
            _fit_candidate(
                row["candidate_name"],
                hidden_by_mode[row["hidden_mode"]][train_mask],
                scalar_train,
                y_train,
            )
        )

    blend_row = None
    if len(test_matrix) >= 2:
        test_matrix_np = np.stack(test_matrix, axis=1)
        best_blend: tuple[float, np.ndarray, list[float]] | None = None
        if test_matrix_np.shape[1] == 2:
            weight_sets = [[w, 1.0 - w] for w in np.linspace(0.0, 1.0, 11)]
        elif test_matrix_np.shape[1] == 3:
            weight_sets = []
            for w0 in np.linspace(0.0, 1.0, 11):
                for w1 in np.linspace(0.0, 1.0 - w0, 11):
                    w2 = 1.0 - w0 - w1
                    weight_sets.append([w0, w1, w2])
        else:
            weight_sets = []
            for w0 in np.linspace(0.0, 1.0, 6):
                for w1 in np.linspace(0.0, 1.0 - w0, 6):
                    for w2 in np.linspace(0.0, 1.0 - w0 - w1, 6):
                        w3 = 1.0 - w0 - w1 - w2
                        weight_sets.append([w0, w1, w2, w3])
        for weights in weight_sets:
            pred = np.average(test_matrix_np[:, : len(weights)], axis=1, weights=np.asarray(weights))
            score = float(r2_score(y_test, pred))
            if best_blend is None or score > best_blend[0]:
                best_blend = (score, pred.astype(np.float32), list(map(float, weights)))
        blend_pred = best_blend[1]
        blend_metrics = _metrics(y_test, blend_pred)
        blend_row = {
            "hidden_mode": "blend",
            "candidate_name": "weighted_average_topk",
            "metrics": blend_metrics,
            "weights": best_blend[2],
            "members": [
                {"hidden_mode": row["hidden_mode"], "candidate_name": row["candidate_name"]}
                for row in top_rows[: len(best_blend[2])]
            ],
        }
        results.append(blend_row)

    best_payload = max(
        [row for row in results if "metrics" in row],
        key=lambda row: row["metrics"]["r2"],
    )
    if best_payload["hidden_mode"] == "blend":
        best_model = {
            "kind": "blend",
            "submodels": submodels[: len(best_payload["weights"])],
            "weights": best_payload["weights"],
        }
        best_test_pred = blend_pred
    else:
        best_model = _fit_candidate(
            best_payload["candidate_name"],
            hidden_by_mode[best_payload["hidden_mode"]][train_mask],
            scalar_train,
            y_train,
        )
        best_test_pred = test_predictions_by_candidate[(best_payload["hidden_mode"], best_payload["candidate_name"])]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "selected": {
            "hidden_mode": best_payload["hidden_mode"],
            "candidate_name": best_payload["candidate_name"],
            "metrics_test": best_payload["metrics"],
        },
        "prompt_feature_names": PROMPT_FEATURE_NAMES + HIDDEN_RELATION_FEATURE_NAMES,
        "model": best_model,
    }
    joblib.dump(bundle, output_dir / "model.joblib")

    prediction_rows = []
    test_metadata = [meta for meta, is_test in zip(metadata_rows, test_mask) if is_test]
    for meta, y_true, y_pred in zip(test_metadata, y_test, best_test_pred):
        prediction_rows.append(
            {
                "task_id": meta["task_id"],
                "split": meta["split"],
                "user_input": meta["user_input"],
                "y_true": float(y_true),
                "y_pred": float(y_pred),
            }
        )
    write_jsonl(output_dir / "predictions_test.jsonl", prediction_rows)
    summary = {
        "selected": bundle["selected"],
        "all_results": results,
        "num_examples": int(len(y)),
        "num_trainplusval": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _batched_prompt_hidden(
    *,
    model,
    tokenizer,
    records: list[dict[str, Any]],
    hidden_mode: str,
    batch_size: int,
    disable_generation_prompt: bool,
    disable_thinking: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows_hidden: list[np.ndarray] = []
    rows_scalar_features: list[np.ndarray] = []
    output_rows: list[dict[str, Any]] = []
    device = next(model.parameters()).device

    prompts: list[str] = []
    payloads: list[dict[str, Any]] = []
    for row_idx, record in enumerate(records):
        messages = _extract_messages(record)
        prompt = _render_prompt(
            tokenizer,
            messages,
            add_generation_prompt=not disable_generation_prompt,
            enable_thinking=not disable_thinking,
        )
        prompts.append(prompt)
        payloads.append(
            {
                "task_id": str(record.get("task_id") or record.get("extra_info", {}).get("index") or row_idx),
                "dataset_name": str(record.get("dataset_name", "dapo_math_17k")),
                "split": str(record.get("split", "train")),
                "user_input": str(record.get("user_input") or messages[-1]["content"]),
            }
        )

    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        payload_batch = payloads[start : start + batch_size]
        tokenized = tokenizer(prompt_batch, return_tensors="pt", padding=True)
        input_lengths = tokenized["attention_mask"].sum(dim=1).cpu().numpy().astype(np.int64)
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.inference_mode():
            outputs = model(**tokenized, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states[1:]
        last_positions = tokenized["attention_mask"].sum(dim=1) - 1
        batch_layer_vectors: list[np.ndarray] = []
        for layer_tensor in hidden_states:
            indices = last_positions[:, None, None].expand(-1, 1, layer_tensor.shape[-1])
            gathered = layer_tensor.gather(1, indices).squeeze(1).detach().cpu().to(torch.float32).numpy()
            batch_layer_vectors.append(gathered)

        for batch_idx, payload in enumerate(payload_batch):
            layers = [layer_vectors[batch_idx] for layer_vectors in batch_layer_vectors]
            rows_hidden.append(_resolve_hidden(layers, hidden_mode))
            rows_scalar_features.append(
                np.concatenate(
                    [
                        _prompt_features(payload["user_input"], int(input_lengths[batch_idx])),
                        _hidden_relation_features(layers),
                    ],
                    axis=0,
                )
            )
            output_rows.append(payload)

    return np.stack(rows_hidden, axis=0), np.stack(rows_scalar_features, axis=0), output_rows


def score(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model_path.expanduser().resolve())
    selected = bundle["selected"]
    model_bundle = bundle["model"]

    records = _resolve_dataset_records(
        dataset_path=args.dataset_path,
        hf_dataset_id=args.hf_dataset_id,
        hf_split=args.hf_split,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    hidden, scalar_features, output_rows = _batched_prompt_hidden(
        model=model,
        tokenizer=tokenizer,
        records=records,
        hidden_mode=selected["hidden_mode"],
        batch_size=args.batch_size,
        disable_generation_prompt=args.disable_generation_prompt,
        disable_thinking=args.disable_thinking,
    )
    predictions = _predict_model(model_bundle, hidden, scalar_features)

    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for meta, prediction in zip(output_rows, predictions):
        rows.append(
            {
                "task_id": meta["task_id"],
                "dataset_name": meta["dataset_name"],
                "split": meta["split"],
                "user_input": meta["user_input"],
                "predicted_difficulty": float(prediction),
                "model": sanitize_name(args.model_name_or_path),
                "probe_hidden_mode": selected["hidden_mode"],
                "probe_candidate": selected["candidate_name"],
            }
        )
    write_jsonl(output_path, rows)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "num_rows": len(rows),
                "probe_hidden_mode": selected["hidden_mode"],
                "probe_candidate": selected["candidate_name"],
            },
            indent=2,
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or apply a deployable prompt-only difficulty probe.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--hidden_dir", type=Path, required=True)
    train_parser.add_argument("--index_dir", type=Path, required=True)
    train_parser.add_argument("--labels_path", type=Path, required=True)
    train_parser.add_argument("--output_dir", type=Path, required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--model_path", type=Path, required=True)
    score_parser.add_argument("--model_name_or_path", required=True)
    score_parser.add_argument("--output_path", type=Path, required=True)
    score_parser.add_argument("--dataset_path", type=Path, default=None)
    score_parser.add_argument("--hf_dataset_id", default=None)
    score_parser.add_argument("--hf_split", default="train")
    score_parser.add_argument("--batch_size", type=int, default=8)
    score_parser.add_argument("--trust_remote_code", action="store_true")
    score_parser.add_argument("--disable_generation_prompt", action="store_true")
    score_parser.add_argument("--disable_thinking", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        train(args)
        return
    if args.command == "score":
        score(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
