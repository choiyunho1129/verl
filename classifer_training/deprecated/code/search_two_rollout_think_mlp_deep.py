from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def aggregate_prompt(
    task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y": [], "p": []})
    for task_id, y_val, pred_val in zip(task_ids, y_true.tolist(), y_pred.tolist()):
        groups[str(task_id)]["y"].append(float(y_val))
        groups[str(task_id)]["p"].append(float(pred_val))
    ordered = sorted(groups)
    yt = np.asarray([float(np.mean(groups[k]["y"])) for k in ordered], dtype=np.float32)
    yp = np.asarray([float(np.mean(groups[k]["p"])) for k in ordered], dtype=np.float32)
    return ordered, yt, yp


def metrics(task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], list[str], np.ndarray, np.ndarray]:
    ordered, yt, yp = aggregate_prompt(task_ids, y_true, y_pred)
    return {
        "row_r2": float(r2_score(y_true, y_pred)),
        "prompt_mean_r2": float(r2_score(yt, yp)),
        "prompt_mean_mae": float(mean_absolute_error(yt, yp)),
        "num_prompts": int(len(ordered)),
    }, ordered, yt, yp


def save_plot(path: Path, y_true_prompt: np.ndarray, y_pred_prompt: np.ndarray, title: str) -> None:
    order = np.argsort(y_true_prompt)
    ys = y_true_prompt[order]
    ps = y_pred_prompt[order]
    err = np.abs(ps - ys)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hb = axes[0].hexbin(y_true_prompt, y_pred_prompt, gridsize=32, cmap="viridis", bins="log", mincnt=1)
    axes[0].plot([0, 1], [0, 1], "--", color="tab:red", lw=1.5)
    axes[0].set_xlabel("True difficulty")
    axes[0].set_ylabel("Predicted difficulty")
    axes[0].set_title("GT vs Pred")
    fig.colorbar(hb, ax=axes[0], label="log count")
    axes[1].plot(ys, color="black", lw=2, label="true")
    axes[1].plot(ps, color="tab:purple", lw=1.5, label="pred")
    axes[1].set_title("Sorted Alignment")
    axes[1].set_xlabel("Prompts sorted by true difficulty")
    axes[1].legend(frameon=False)
    axes[2].plot(err, color="teal", lw=1.2)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Prompts sorted by true difficulty")
    axes[2].set_ylabel("|pred-true|")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_predictions(path: Path, task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    ordered, yt, yp = aggregate_prompt(task_ids, y_true, y_pred)
    with path.open("w", encoding="utf-8") as handle:
        for task_id, y_val, pred_val in zip(ordered, yt.tolist(), yp.tolist()):
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "y_true_difficulty": y_val,
                        "predicted_difficulty": pred_val,
                    }
                )
                + "\n"
            )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_dataset_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "x_train": data["x_train"],
            "y_train": data["y_train"],
            "train_task_ids": data["train_task_ids"].tolist(),
            "x_val": data["x_val"],
            "y_val": data["y_val"],
            "val_task_ids": data["val_task_ids"].tolist(),
            "x_test": data["x_test"],
            "y_test": data["y_test"],
            "test_task_ids": data["test_task_ids"].tolist(),
            "feature_dim": int(data["feature_dim"][0]),
        }


def _candidate_configs() -> list[dict[str, Any]]:
    configs = []
    for hidden_layer_sizes, alpha, learning_rate_init, max_iter in product(
        [(512, 256, 128), (768, 384, 192), (1024, 512, 256)],
        [1e-4, 1e-3],
        [3e-4, 1e-3],
        [400, 800],
    ):
        configs.append(
            {
                "hidden_layer_sizes": hidden_layer_sizes,
                "alpha": alpha,
                "learning_rate_init": learning_rate_init,
                "max_iter": max_iter,
            }
        )
    return configs


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    compare_dir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_mlp_deep_search"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_paths = sorted((compare_dir / "dataset_cache").glob("*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No dataset caches found under {compare_dir / 'dataset_cache'}")

    best: dict[str, Any] | None = None
    best_dataset: dict[str, Any] | None = None
    progress_path = outdir / "progress.json"
    for cache_path in cache_paths:
        dataset = _load_dataset_cache(cache_path)
        x_train = np.asarray(dataset["x_train"], dtype=np.float32)
        y_train = np.asarray(dataset["y_train"], dtype=np.float32)
        x_val = np.asarray(dataset["x_val"], dtype=np.float32)
        y_val = np.asarray(dataset["y_val"], dtype=np.float32)
        val_task_ids = list(dataset["val_task_ids"])
        dataset_key = cache_path.stem
        print(json.dumps({"stage": "search_dataset", "dataset_key": dataset_key}), flush=True)
        for params in _candidate_configs():
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=params["hidden_layer_sizes"],
                            alpha=params["alpha"],
                            learning_rate_init=params["learning_rate_init"],
                            activation="relu",
                            solver="adam",
                            batch_size=256,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=30,
                            max_iter=params["max_iter"],
                            random_state=42,
                        ),
                    ),
                ]
            )
            model.fit(x_train, y_train)
            pred = np.clip(np.asarray(model.predict(x_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
            metric, _, _, _ = metrics(val_task_ids, y_val, pred)
            row = {
                "dataset_key": dataset_key,
                "feature_dim": int(dataset["feature_dim"]),
                "params": {
                    "hidden_layer_sizes": list(params["hidden_layer_sizes"]),
                    "alpha": float(params["alpha"]),
                    "learning_rate_init": float(params["learning_rate_init"]),
                    "max_iter": int(params["max_iter"]),
                },
                "val": metric,
            }
            if best is None or row["val"]["prompt_mean_r2"] > best["val"]["prompt_mean_r2"]:
                best = row
                best_dataset = dataset
                _write_json(progress_path, best)

    if best is None or best_dataset is None:
        raise RuntimeError("No MLP candidates were evaluated")

    x_trainval = np.concatenate([np.asarray(best_dataset["x_train"]), np.asarray(best_dataset["x_val"])], axis=0)
    y_trainval = np.concatenate([np.asarray(best_dataset["y_train"]), np.asarray(best_dataset["y_val"])], axis=0)
    x_test = np.asarray(best_dataset["x_test"])
    y_test = np.asarray(best_dataset["y_test"])
    test_task_ids = list(best_dataset["test_task_ids"])

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=tuple(best["params"]["hidden_layer_sizes"]),
                    alpha=best["params"]["alpha"],
                    learning_rate_init=best["params"]["learning_rate_init"],
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                    max_iter=best["params"]["max_iter"],
                    random_state=42,
                ),
            ),
        ]
    )
    final_model.fit(x_trainval, y_trainval)
    pred = np.clip(np.asarray(final_model.predict(x_test), dtype=np.float32).reshape(-1), 0.0, 1.0)
    metric, _, yt, yp = metrics(test_task_ids, y_test, pred)
    summary = {
        "family": "MLP_deep",
        "best_component": best["dataset_key"],
        "feature_dim": int(best["feature_dim"]),
        "best_params_from_val": best["params"],
        "val_prompt_mean_r2": best["val"]["prompt_mean_r2"],
        "test": metric,
    }
    save_predictions(outdir / "mlp_deep_predictions_test.jsonl", test_task_ids, y_test, pred)
    save_plot(
        outdir / "mlp_deep_vs_gt.png",
        np.asarray(yt),
        np.asarray(yp),
        f"MLP_deep | {best['dataset_key']} | prompt R2={metric['prompt_mean_r2']:.3f}",
    )
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
