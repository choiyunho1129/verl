from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.data import load_hidden_rows
from classifer_training.single_rollout_hidden_utils import (
    PROMPT_FEATURE_NAMES,
    build_prompt_scalar_lookup,
    build_split_lookup,
    label_to_value,
    load_labels_by_task,
)
from classifer_training.utils import write_jsonl


class LogitRidgeValueEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, *, alpha: float = 3000.0, epsilon: float = 0.05, random_state: int = 42) -> None:
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.random_state = int(random_state)

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(value, dtype=np.float64), -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogitRidgeValueEstimator":
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        y_bounded = np.clip(y_array, self.epsilon, 1.0 - self.epsilon)
        target = np.log(y_bounded / (1.0 - y_bounded))
        self.model_ = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.model_.fit(x, target)
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        logit_pred = np.asarray(self.model_.predict(x), dtype=np.float64).reshape(-1)
        return self._sigmoid(logit_pred)


class TwoHeadBinaryValueEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, *, C: float = 1.0, random_state: int = 42, max_iter: int = 2000) -> None:
        self.C = float(C)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.zero_head = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        self.one_head = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TwoHeadBinaryValueEstimator":
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        self.zero_head.fit(x, np.isclose(y_array, 0.0).astype(np.int32))
        self.one_head.fit(x, np.isclose(y_array, 1.0).astype(np.int32))
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        p_zero = self.zero_head.predict_proba(x)[:, 1]
        p_one = self.one_head.predict_proba(x)[:, 1]
        denom = p_zero + p_one
        pred = np.divide(p_one, denom, out=np.full_like(p_one, 0.5, dtype=np.float64), where=denom > 1e-8)
        return np.clip(pred, 0.0, 1.0)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    layer_indices: tuple[int, ...]
    reduce: str
    include_scalars: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep prompt-only predictors for prompt mean correctness.")
    parser.add_argument("--prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--prompt_hidden_path", type=Path, required=True)
    parser.add_argument("--prompt_index_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="*", type=int, default=[0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 35])
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0])
    parser.add_argument("--logit_epsilons", nargs="*", type=float, default=[0.02, 0.05, 0.1])
    parser.add_argument("--logistic_cs", nargs="*", type=float, default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    parser.add_argument(
        "--model_families",
        nargs="+",
        choices=["ridge", "logit_ridge", "two_head_logistic"],
        default=["ridge", "logit_ridge", "two_head_logistic"],
    )
    parser.add_argument("--include_all_prompt_scalars", action="store_true")
    parser.add_argument("--disable_span_specs", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _make_specs(
    num_layers: int,
    requested_layers: list[int],
    include_scalars: bool,
    disable_span_specs: bool,
) -> list[FeatureSpec]:
    valid_layers = sorted({idx for idx in requested_layers if 0 <= idx < num_layers})
    base_specs: list[tuple[str, tuple[int, ...], str]] = [(f"layer{idx}", (idx,), "single") for idx in valid_layers]
    span_specs: list[tuple[str, tuple[int, ...], str]] = []
    if not disable_span_specs:
        for width in (2, 4, 6, 8, 12):
            if width <= num_layers:
                indices = tuple(range(num_layers - width, num_layers))
                span_specs.append((f"last{width}mean", indices, "mean"))
        if num_layers >= 28:
            span_specs.append(("mid20to27mean", tuple(range(20, 28)), "mean"))
        if num_layers >= 36:
            span_specs.append(("late24to35mean", tuple(range(24, 36)), "mean"))

    specs: list[FeatureSpec] = []
    for name, indices, reduce in base_specs + span_specs:
        specs.append(FeatureSpec(name=name, layer_indices=indices, reduce=reduce, include_scalars=False))
        if include_scalars:
            specs.append(FeatureSpec(name=f"{name}_promptscalars", layer_indices=indices, reduce=reduce, include_scalars=True))
    return specs


def _feature_matrix(
    hidden_by_task: dict[str, list[np.ndarray]],
    task_ids: list[str],
    spec: FeatureSpec,
    scalar_lookup: dict[str, np.ndarray],
) -> np.ndarray:
    rows = []
    for task_id in task_ids:
        layers = hidden_by_task[task_id]
        if spec.reduce == "single":
            hidden = np.asarray(layers[spec.layer_indices[0]], dtype=np.float32).reshape(-1)
        elif spec.reduce == "mean":
            hidden = np.mean(
                np.stack([np.asarray(layers[idx], dtype=np.float32).reshape(-1) for idx in spec.layer_indices], axis=0),
                axis=0,
            ).astype(np.float32)
        else:
            raise ValueError(f"Unsupported feature reduction: {spec.reduce}")
        pieces = [hidden]
        if spec.include_scalars:
            pieces.append(np.asarray(scalar_lookup[task_id], dtype=np.float32).reshape(-1))
        rows.append(np.concatenate(pieces, axis=0).astype(np.float32))
    return np.stack(rows, axis=0)


def _model_specs(args: argparse.Namespace) -> list[tuple[str, Any]]:
    specs: list[tuple[str, Any]] = []
    families = set(args.model_families)
    if "ridge" in families:
        for alpha in args.alphas:
            specs.append(
                (
                    f"ridge_a{alpha:g}",
                    Pipeline(
                        [("scale", StandardScaler()), ("model", Ridge(alpha=alpha, random_state=args.random_seed))]
                    ),
                )
            )
    if "logit_ridge" in families:
        for alpha in args.alphas:
            for epsilon in args.logit_epsilons:
                specs.append(
                    (
                        f"logit_ridge_a{alpha:g}_eps{epsilon:g}",
                        Pipeline(
                            [
                                ("scale", StandardScaler()),
                                (
                                    "model",
                                    LogitRidgeValueEstimator(
                                        alpha=alpha,
                                        epsilon=epsilon,
                                        random_state=args.random_seed,
                                    ),
                                ),
                            ]
                        ),
                    )
                )
    if "two_head_logistic" in families:
        for c_value in args.logistic_cs:
            specs.append(
                (
                    f"two_head_logistic_c{c_value:g}",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            ("model", TwoHeadBinaryValueEstimator(C=c_value, random_state=args.random_seed)),
                        ]
                    ),
                )
            )
    return specs

def _write_plots(output_dir: Path, results: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> None:
    top = sorted(results, key=lambda row: row["metrics"]["r2"], reverse=True)[:30]
    labels = [row["feature_spec"] + "\n" + row["model_name"] for row in top]
    r2_values = [row["metrics"]["r2"] for row in top]
    mae_values = [row["metrics"]["mae"] for row in top]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, 0.35 * len(top))))
    y_pos = np.arange(len(top))
    axes[0].barh(y_pos, r2_values, color="#386cb0")
    axes[0].set_yticks(y_pos, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Validation prompt R2")
    axes[0].set_title("Top Prompt-Only Predictors")
    axes[1].barh(y_pos, mae_values, color="#fdb462")
    axes[1].set_yticks(y_pos, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Validation prompt MAE")
    axes[1].set_title("MAE")
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_mean_sweep_top_models.png", dpi=180)
    plt.close(fig)

    y_true = np.asarray([row["value_true"] for row in predictions], dtype=np.float32)
    y_pred = np.asarray([row["value_pred"] for row in predictions], dtype=np.float32)
    order = np.argsort(y_true)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hb = axes[0].hexbin(y_true, y_pred, gridsize=32, cmap="viridis", bins="log", mincnt=1)
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1)
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("GT prompt mean")
    axes[0].set_ylabel("Pred")
    axes[0].set_title("Best Pred vs GT")
    fig.colorbar(hb, ax=axes[0], label="count")
    axes[1].plot(y_true[order], label="GT", color="black", linewidth=1.5)
    axes[1].plot(y_pred[order], label="Pred", color="#386cb0", linewidth=1.0)
    axes[1].set_title("Sorted By GT")
    axes[1].legend()
    bins = np.linspace(0, 1, 17)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_pred = []
    mean_abs_err = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (y_true >= lo) & (y_true < hi if hi < 1.0 else y_true <= hi)
        mean_pred.append(float(np.mean(y_pred[mask])) if np.any(mask) else np.nan)
        mean_abs_err.append(float(np.mean(np.abs(y_pred[mask] - y_true[mask]))) if np.any(mask) else np.nan)
    axes[2].plot(centers, mean_pred, marker="o", label="mean pred", color="#386cb0")
    axes[2].plot([0, 1], [0, 1], color="black", linewidth=1)
    ax2 = axes[2].twinx()
    ax2.bar(centers, mean_abs_err, width=0.045, alpha=0.25, color="#fdb462", label="MAE")
    axes[2].set_xlim(-0.03, 1.03)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_title("Calibration By GT Bin")
    axes[2].set_xlabel("GT prompt mean")
    axes[2].set_ylabel("mean pred")
    ax2.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_mean_best_pred_vs_gt.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_by_task = load_labels_by_task(args.labels_path)
    split_lookup = build_split_lookup(args.prompt_dataset_dir.expanduser().resolve())
    prompt_scalar_keys = list(PROMPT_FEATURE_NAMES) if args.include_all_prompt_scalars else []
    scalar_lookup = build_prompt_scalar_lookup(labels_by_task, prompt_scalar_keys)

    hidden_rows = load_hidden_rows(
        args.prompt_hidden_path.expanduser().resolve(),
        index_path=args.prompt_index_path.expanduser().resolve(),
        dataset_name="dapo_math_17k",
        default_component_name="hidden",
    )
    hidden_by_task = {str(row["task_id"]): row["components"]["hidden"] for row in hidden_rows}
    if not hidden_by_task:
        raise ValueError("No prompt hidden rows loaded.")
    num_layers = len(next(iter(hidden_by_task.values())))

    task_ids = sorted(task_id for task_id in hidden_by_task if task_id in labels_by_task and task_id in split_lookup)
    train_task_ids = [task_id for task_id in task_ids if split_lookup[task_id] == "train"]
    val_task_ids = [task_id for task_id in task_ids if split_lookup[task_id] == "validation"]
    y_train = np.asarray([label_to_value(labels_by_task[task_id]) for task_id in train_task_ids], dtype=np.float32)
    y_val = np.asarray([label_to_value(labels_by_task[task_id]) for task_id in val_task_ids], dtype=np.float32)

    specs = _make_specs(
        num_layers,
        list(args.layers),
        bool(args.include_all_prompt_scalars),
        bool(args.disable_span_specs),
    )
    model_specs = _model_specs(args)

    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for spec in specs:
        x_train = _feature_matrix(hidden_by_task, train_task_ids, spec, scalar_lookup)
        x_val = _feature_matrix(hidden_by_task, val_task_ids, spec, scalar_lookup)
        for model_name, estimator in model_specs:
            estimator.fit(x_train, y_train)
            pred = np.clip(np.asarray(estimator.predict(x_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
            metrics = _metrics(y_val, pred)
            result = {
                "feature_spec": spec.name,
                "layer_indices": list(spec.layer_indices),
                "reduce": spec.reduce,
                "include_scalars": bool(spec.include_scalars),
                "model_name": model_name,
                "metrics": metrics,
                "feature_dim": int(x_train.shape[1]),
            }
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            results.append(result)
            if best is None or metrics["r2"] > best["result"]["metrics"]["r2"]:
                best = {
                    "result": result,
                    "estimator": estimator,
                    "pred": pred,
                    "x_train_shape": list(x_train.shape),
                    "x_val_shape": list(x_val.shape),
                }

    if best is None:
        raise RuntimeError("No model was fit.")

    predictions = [
        {
            "task_id": task_id,
            "user_input": str(labels_by_task[task_id].get("user_input", "")),
            "value_true": float(true_value),
            "value_pred": float(pred_value),
            "num_rows": 1,
        }
        for task_id, true_value, pred_value in zip(val_task_ids, y_val.tolist(), best["pred"].tolist(), strict=True)
    ]
    write_jsonl(args.output_dir / "predictions_weakval.jsonl", predictions)
    joblib.dump(best["estimator"], args.output_dir / "best_estimator.joblib")
    _write_plots(args.output_dir, results, predictions)

    summary = {
        "setting": "prompt_mean_predictor_sweep",
        "num_layers": int(num_layers),
        "prompt_scalar_keys": prompt_scalar_keys,
        "num_train_prompts": int(len(train_task_ids)),
        "num_validation_prompts": int(len(val_task_ids)),
        "best": best["result"],
        "x_train_shape": best["x_train_shape"],
        "x_val_shape": best["x_val_shape"],
        "top10": sorted(results, key=lambda row: row["metrics"]["r2"], reverse=True)[:10],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
