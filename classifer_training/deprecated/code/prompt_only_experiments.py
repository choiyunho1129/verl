from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.data import load_hidden_rows
from classifer_training.utils import load_records


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    out = np.empty_like(values, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


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
    return np.asarray(
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


def _hidden_relation_features(hidden_layers: list[np.ndarray]) -> np.ndarray:
    layers = [np.asarray(layer, dtype=np.float32) for layer in hidden_layers]
    last = layers[35]
    middle = layers[17]
    early = layers[8]
    top4_mean = np.stack(layers[32:36], axis=0).mean(axis=0)
    all_mean = np.stack(layers, axis=0).mean(axis=0)
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-6:
            return 0.0
        return float(np.dot(a, b) / denom)
    return np.asarray(
        [
            float(np.linalg.norm(early)),
            float(np.linalg.norm(middle)),
            float(np.linalg.norm(last)),
            float(np.linalg.norm(top4_mean)),
            float(np.linalg.norm(all_mean)),
            cosine(early, middle),
            cosine(middle, last),
            cosine(last, all_mean),
            float(np.linalg.norm(last - middle)),
            float(np.linalg.norm(last - early)),
        ],
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prompt-only prompt-feature + hidden-state probe variants.")
    parser.add_argument("--hidden_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--index_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def _build_dataset(hidden_dir: Path, index_dir: Path, labels_by_task_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if (hidden_dir / "hidden_states.pt").exists() and (index_dir / "index.jsonl").exists():
        split_files = [
            ("combined", hidden_dir / "hidden_states.pt", index_dir / "index.jsonl"),
        ]
    else:
        split_files = [
            ("train", hidden_dir / "hidden_states_train.pt", index_dir / "index_train.jsonl"),
            ("validation", hidden_dir / "hidden_states_validation.pt", index_dir / "index_validation.jsonl"),
            ("test", hidden_dir / "hidden_states_test.pt", index_dir / "index_test.jsonl"),
        ]
    hidden_modes: dict[str, list[np.ndarray]] = {
        "layer17": [],
        "layer24": [],
        "layer35": [],
        "layers0_35_mean": [],
    }
    scalar_rows: list[np.ndarray] = []
    y_reg: list[float] = []
    y_cls: list[int] = []
    split_rows: list[str] = []

    for split_name, hidden_path, index_path in split_files:
        rows = load_hidden_rows(hidden_path, index_path=index_path, dataset_name="dapo_math_17k")
        for row in rows:
            label_row = labels_by_task_id.get(str(row["task_id"]))
            if label_row is None:
                continue
            hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
            index_row = row["index_row"]
            scalar_rows.append(
                np.concatenate(
                    [
                        _prompt_features(str(index_row.get("user_input", "")), int(index_row.get("input_length", 0))),
                        _hidden_relation_features(hidden_layers),
                    ],
                    axis=0,
                )
            )
            hidden_modes["layer17"].append(hidden_layers[17])
            hidden_modes["layer24"].append(hidden_layers[24])
            hidden_modes["layer35"].append(hidden_layers[35])
            hidden_modes["layers0_35_mean"].append(np.stack(hidden_layers, axis=0).mean(axis=0))
            y_reg.append(float(label_row["difficulty"]))
            y_cls.append(int(float(label_row["sampling_accuracy"]) > 0.0))
            effective_split = str(row["index_row"].get("split", split_name))
            split_rows.append(effective_split)

    return {
        "hidden_modes": {k: np.stack(v, axis=0) for k, v in hidden_modes.items()},
        "scalar": np.stack(scalar_rows, axis=0),
        "y_reg": np.asarray(y_reg, dtype=np.float32),
        "y_cls": np.asarray(y_cls, dtype=np.int64),
        "splits": np.asarray(split_rows),
    }


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
    }


def _cls_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, prob)),
    }


def main() -> None:
    args = parse_args()
    label_rows = load_records(args.labels_path.expanduser().resolve())
    labels_by_task_id = {str(row["task_id"]): row for row in label_rows}
    payload: list[dict[str, Any]] = []

    for hidden_dir, index_dir in zip(args.hidden_dirs, args.index_dirs):
        dataset = _build_dataset(hidden_dir.expanduser().resolve(), index_dir.expanduser().resolve(), labels_by_task_id)
        train_mask = np.isin(dataset["splits"], np.asarray(["train", "validation"]))
        test_mask = dataset["splits"] == "test"
        scalar_train = dataset["scalar"][train_mask]
        scalar_test = dataset["scalar"][test_mask]
        y_reg_train = dataset["y_reg"][train_mask]
        y_reg_test = dataset["y_reg"][test_mask]
        y_cls_train = dataset["y_cls"][train_mask]
        y_cls_test = dataset["y_cls"][test_mask]

        base_name = hidden_dir.name
        for hidden_mode, hidden_array in dataset["hidden_modes"].items():
            hidden_train = hidden_array[train_mask]
            hidden_test = hidden_array[test_mask]
            X_train = np.concatenate([hidden_train, scalar_train], axis=1)
            X_test = np.concatenate([hidden_test, scalar_test], axis=1)

            ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=3000.0))])
            ridge.fit(X_train, y_reg_train)
            pred_reg = np.clip(ridge.predict(X_test).astype(np.float32), 0.0, 1.0)

            logreg = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("logreg", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, class_weight="balanced", random_state=42)),
                ]
            )
            logreg.fit(X_train, y_cls_train)
            prob_cls = _sigmoid(logreg.decision_function(X_test).astype(np.float32))
            pred_cls = (prob_cls >= 0.5).astype(np.int64)

            payload.append(
                {
                    "hidden_source": base_name,
                    "hidden_mode": hidden_mode,
                    "regression_ridge": _reg_metrics(y_reg_test, pred_reg),
                    "classification_logreg_acc_gt0": _cls_metrics(y_cls_test, prob_cls, pred_cls),
                }
            )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
