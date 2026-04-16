from __future__ import annotations

import json
import os

# Keep nested BLAS/OpenMP thread pools from fighting the family-level workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from classifer_training.train_two_rollout_reasoning_probe import (
    build_feature_matrix,
    build_pair_rows,
    build_prompt_lookup,
    build_rollout_hidden_lookup,
    load_grouped_rollouts,
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
    out = {
        "row_r2": float(r2_score(y_true, y_pred)),
        "prompt_mean_r2": float(r2_score(yt, yp)),
        "prompt_mean_mae": float(mean_absolute_error(yt, yp)),
        "num_prompts": int(len(ordered)),
    }
    return out, ordered, yt, yp


def fit_and_eval(model, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray, eval_task_ids: list[str]) -> dict[str, Any]:
    model.fit(x_train, y_train)
    pred = np.clip(np.asarray(model.predict(x_eval), dtype=np.float32).reshape(-1), 0.0, 1.0)
    metric, ordered, yt, yp = metrics(eval_task_ids, y_eval, pred)
    return {"metric": metric, "pred": pred, "ordered": ordered, "yt": yt, "yp": yp}


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


def bucket_edges() -> list[np.ndarray]:
    return [
        np.asarray([0.0, 0.1, 0.9, 1.01], dtype=np.float32),
        np.asarray([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01], dtype=np.float32),
        np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.01], dtype=np.float32),
    ]


def _make_mlp_pipeline(params: dict[str, object]) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
                    alpha=float(params["alpha"]),
                    learning_rate_init=float(params["learning_rate_init"]),
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=int(params["n_iter_no_change"]),
                    max_iter=int(params["max_iter"]),
                    random_state=int(params["seed"]),
                ),
            ),
        ]
    )


def build_model_specs(fast_mode: bool, random_seed: int, et_n_jobs: int) -> dict[str, list[tuple[dict[str, object], object]]]:
    model_specs: dict[str, list[tuple[dict[str, object], object]]] = {
        "ExtraTrees": [],
        "Ridge": [],
        "SVM": [],
        "MLP": [],
    }

    et_n_estimators = [1000, 2000] if fast_mode else [1000, 2000]
    et_min_leaf = [5, 7] if fast_mode else [3, 5, 7]
    et_max_features = [0.5, 0.7] if fast_mode else [0.5, 0.7, 0.9]
    for n_estimators in et_n_estimators:
        for min_samples_leaf in et_min_leaf:
            for max_features in et_max_features:
                params = {
                    "n_estimators": n_estimators,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                }
                model_specs["ExtraTrees"].append(
                    (params, ExtraTreesRegressor(random_state=random_seed, n_jobs=et_n_jobs, **params))
                )

    ridge_alphas = [300, 1000, 3000, 10000] if fast_mode else [30, 100, 300, 1000, 3000, 10000, 30000]
    for alpha in ridge_alphas:
        params = {"alpha": alpha}
        model_specs["Ridge"].append(
            (params, Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=random_seed))]))
        )

    svm_c_vals = [0.1, 0.3, 1.0] if fast_mode else [0.03, 0.1, 0.3, 1.0]
    svm_eps = [0.03, 0.05] if fast_mode else [0.01, 0.03, 0.05, 0.1]
    for c_val in svm_c_vals:
        for epsilon in svm_eps:
            params = {"C": c_val, "epsilon": epsilon}
            model_specs["SVM"].append(
                (
                    params,
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "svr",
                                LinearSVR(
                                    C=c_val,
                                    epsilon=epsilon,
                                    loss="squared_epsilon_insensitive",
                                    dual=False,
                                    random_state=random_seed,
                                    max_iter=20000,
                                ),
                            ),
                        ]
                    ),
                )
            )

    mlp_hiddens = (
        [(256, 128), (512, 256), (512, 256, 128)]
        if fast_mode
        else [(256, 128), (512, 256), (512, 256, 128), (768, 384, 192)]
    )
    mlp_alphas = [1e-4, 1e-3] if fast_mode else [1e-4, 1e-3, 1e-2]
    mlp_lrs = [3e-4, 1e-3]
    mlp_seeds = [random_seed, random_seed + 1, random_seed + 2] if fast_mode else [random_seed + idx for idx in range(5)]
    mlp_max_iter = 400 if fast_mode else 600
    mlp_n_iter_no_change = 30 if fast_mode else 40
    for hidden in mlp_hiddens:
        for alpha in mlp_alphas:
            for lr in mlp_lrs:
                for mlp_seed in mlp_seeds:
                    params = {
                        "hidden_layer_sizes": list(hidden),
                        "alpha": alpha,
                        "learning_rate_init": lr,
                        "seed": int(mlp_seed),
                        "max_iter": int(mlp_max_iter),
                        "n_iter_no_change": int(mlp_n_iter_no_change),
                    }
                    model_specs["MLP"].append((params, _make_mlp_pipeline(params)))
    return model_specs


def _candidate_score_key(row: dict[str, Any]) -> float:
    return float(row["val"]["prompt_mean_r2"])


def _family_progress_path(outdir: Path, family: str) -> Path:
    return outdir / f"{family.lower()}_progress.json"


def _family_result_path(outdir: Path, family: str) -> Path:
    return outdir / f"{family.lower()}_result.json"


def run_family_search(
    family: str,
    dataset_paths: list[str],
    model_specs: dict[str, list[tuple[dict[str, object], object]]],
    random_seed: int,
    outdir_str: str,
) -> dict[str, object]:
    outdir = Path(outdir_str)
    progress_path = _family_progress_path(outdir, family)
    result_path = _family_result_path(outdir, family)
    print(f"stage: search {family}", flush=True)
    best = None
    best_data: dict[str, Any] | None = None
    for dataset_path_str in dataset_paths:
        dataset_path = Path(dataset_path_str)
        dataset = _load_dataset_cache(dataset_path)
        x_train = np.asarray(dataset["x_train"], dtype=np.float32)
        y_train = np.asarray(dataset["y_train"], dtype=np.float32)
        x_val = np.asarray(dataset["x_val"], dtype=np.float32)
        y_val = np.asarray(dataset["y_val"], dtype=np.float32)
        val_task_ids = list(dataset["val_task_ids"])
        for params, model in model_specs[family]:
            result = fit_and_eval(model, x_train, y_train, x_val, y_val, val_task_ids)
            row = {
                "family": family,
                "dataset_key": dataset_path.stem,
                "dataset_path": str(dataset_path),
                "params": params,
                "val": result["metric"],
                "feature_dim": int(dataset["feature_dim"]),
            }
            if best is None or _candidate_score_key(row) > _candidate_score_key(best):
                best = row
                best_data = dataset
                _write_json(progress_path, best)

    if best is None or best_data is None:
        raise RuntimeError(f"No candidates evaluated for family {family}")

    x_trainval = np.concatenate([np.asarray(best_data["x_train"]), np.asarray(best_data["x_val"])], axis=0)
    y_trainval = np.concatenate([np.asarray(best_data["y_train"]), np.asarray(best_data["y_val"])], axis=0)
    x_test = np.asarray(best_data["x_test"])
    y_test = np.asarray(best_data["y_test"])
    test_task_ids = list(best_data["test_task_ids"])

    if family == "ExtraTrees":
        final_model = ExtraTreesRegressor(random_state=random_seed, **best["params"])
    elif family == "Ridge":
        final_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best["params"]["alpha"], random_state=random_seed))])
    elif family == "SVM":
        final_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svr",
                    LinearSVR(
                        C=best["params"]["C"],
                        epsilon=best["params"]["epsilon"],
                        loss="squared_epsilon_insensitive",
                        dual=False,
                        random_state=random_seed,
                        max_iter=20000,
                    ),
                ),
            ]
        )
    elif family == "MLP":
        final_model = _make_mlp_pipeline(best["params"])
    else:
        raise ValueError(f"Unsupported family {family}")

    final_result = fit_and_eval(final_model, x_trainval, y_trainval, x_test, y_test, test_task_ids)
    family_result = {
        "family": family,
        "best_component": best["dataset_key"],
        "feature_dim": int(best["feature_dim"]),
        "best_params_from_val": best["params"],
        "val_prompt_mean_r2": best["val"]["prompt_mean_r2"],
        "test": final_result["metric"],
    }
    save_predictions(outdir / f"{family.lower()}_predictions_test.jsonl", test_task_ids, y_test, final_result["pred"])
    save_plot(
        outdir / f"{family.lower()}_vs_gt.png",
        np.asarray(final_result["yt"]),
        np.asarray(final_result["yp"]),
        f"{family} | {best['dataset_key']} | prompt R2={final_result['metric']['prompt_mean_r2']:.3f}",
    )
    _write_json(result_path, family_result)
    return family_result


def run_lda_search(
    dataset_paths: list[str],
    random_seed: int,
    fast_mode: bool,
    outdir_str: str,
) -> dict[str, object]:
    outdir = Path(outdir_str)
    progress_path = _family_progress_path(outdir, "LDA")
    result_path = _family_result_path(outdir, "LDA")
    print("stage: search LDA", flush=True)
    lda_best = None
    lda_best_data: dict[str, Any] | None = None
    lda_bucket_edges = bucket_edges()[:2] if fast_mode else bucket_edges()
    lda_pca_dims = [64, 128] if fast_mode else [64, 128, 256]
    for dataset_path_str in dataset_paths:
        dataset_path = Path(dataset_path_str)
        dataset = _load_dataset_cache(dataset_path)
        x_train = np.asarray(dataset["x_train"], dtype=np.float32)
        y_train = np.asarray(dataset["y_train"], dtype=np.float32)
        x_val = np.asarray(dataset["x_val"], dtype=np.float32)
        y_val = np.asarray(dataset["y_val"], dtype=np.float32)
        val_task_ids = list(dataset["val_task_ids"])
        for edges in lda_bucket_edges:
            for pca_dim in lda_pca_dims:
                y_bin = np.clip(np.digitize(y_train, edges[1:-1], right=False), 0, len(edges) - 2)
                present = sorted(set(y_bin.tolist()))
                remap = {c: i for i, c in enumerate(present)}
                y_mapped = np.asarray([remap[c] for c in y_bin.tolist()], dtype=np.int64)
                class_means = np.asarray([float(np.mean(y_train[y_bin == c])) for c in present], dtype=np.float32)
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=min(pca_dim, x_train.shape[1]), random_state=random_seed)),
                        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                    ]
                )
                model.fit(x_train, y_mapped)
                prob = model.predict_proba(x_val)
                pred = np.clip((prob @ class_means).astype(np.float32), 0.0, 1.0)
                metric, _, _, _ = metrics(val_task_ids, y_val, pred)
                row = {
                    "family": "LDA",
                    "dataset_key": dataset_path.stem,
                    "dataset_path": str(dataset_path),
                    "params": {"edges": edges.tolist(), "pca_dim": pca_dim},
                    "val": metric,
                    "feature_dim": int(dataset["feature_dim"]),
                }
                if lda_best is None or _candidate_score_key(row) > _candidate_score_key(lda_best):
                    lda_best = row
                    lda_best_data = dataset
                    _write_json(progress_path, lda_best)

    if lda_best is None or lda_best_data is None:
        raise RuntimeError("No candidates evaluated for family LDA")

    x_trainval = np.concatenate([np.asarray(lda_best_data["x_train"]), np.asarray(lda_best_data["x_val"])], axis=0)
    y_trainval = np.concatenate([np.asarray(lda_best_data["y_train"]), np.asarray(lda_best_data["y_val"])], axis=0)
    x_test = np.asarray(lda_best_data["x_test"])
    y_test = np.asarray(lda_best_data["y_test"])
    test_task_ids = list(lda_best_data["test_task_ids"])
    edges = np.asarray(lda_best["params"]["edges"], dtype=np.float32)
    y_bin = np.clip(np.digitize(y_trainval, edges[1:-1], right=False), 0, len(edges) - 2)
    present = sorted(set(y_bin.tolist()))
    remap = {c: i for i, c in enumerate(present)}
    y_mapped = np.asarray([remap[c] for c in y_bin.tolist()], dtype=np.int64)
    class_means = np.asarray([float(np.mean(y_trainval[y_bin == c])) for c in present], dtype=np.float32)
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(int(lda_best["params"]["pca_dim"]), x_trainval.shape[1]), random_state=random_seed)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    final_model.fit(x_trainval, y_mapped)
    prob = final_model.predict_proba(x_test)
    pred = np.clip((prob @ class_means).astype(np.float32), 0.0, 1.0)
    metric, _, yt, yp = metrics(test_task_ids, y_test, pred)
    lda_result = {
        "family": "LDA",
        "best_component": lda_best["dataset_key"],
        "feature_dim": int(lda_best["feature_dim"]),
        "best_params_from_val": lda_best["params"],
        "val_prompt_mean_r2": lda_best["val"]["prompt_mean_r2"],
        "test": metric,
    }
    save_predictions(outdir / "lda_predictions_test.jsonl", test_task_ids, y_test, pred)
    save_plot(outdir / "lda_vs_gt.png", np.asarray(yt), np.asarray(yp), f"LDA | {lda_best['dataset_key']} | prompt R2={metric['prompt_mean_r2']:.3f}")
    _write_json(result_path, lda_result)
    return lda_result


def _save_dataset_cache(cache_path: Path, dataset: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        x_train=np.asarray(dataset["x_train"], dtype=np.float32),
        y_train=np.asarray(dataset["y_train"], dtype=np.float32),
        train_task_ids=np.asarray(dataset["train_task_ids"]),
        x_val=np.asarray(dataset["x_val"], dtype=np.float32),
        y_val=np.asarray(dataset["y_val"], dtype=np.float32),
        val_task_ids=np.asarray(dataset["val_task_ids"]),
        x_test=np.asarray(dataset["x_test"], dtype=np.float32),
        y_test=np.asarray(dataset["y_test"], dtype=np.float32),
        test_task_ids=np.asarray(dataset["test_task_ids"]),
        feature_dim=np.asarray([dataset["feature_dim"]], dtype=np.int32),
    )


def build_or_load_dataset_cache(
    *,
    repo: Path,
    outdir: Path,
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    component_name: str,
    pool: str,
    manifest_path: Path,
    rollout_hidden_paths: list[Path],
    rollout_index_paths: list[Path],
    prompt_mode: str,
    train_pairs_per_prompt: int,
    eval_pairs_per_prompt: int,
    random_seed: int,
    force_rebuild: bool,
) -> Path:
    dataset_key = f"{component_name}:{pool}"
    cache_path = outdir / "dataset_cache" / f"{dataset_key}.npz"
    if cache_path.exists() and not force_rebuild:
        print(json.dumps({"stage": "dataset_cache_hit", "dataset_key": dataset_key, "path": str(cache_path)}), flush=True)
        return cache_path

    print(json.dumps({"stage": "build_dataset", "component": component_name, "pool": pool}), flush=True)
    rollout_hidden_lookup = build_rollout_hidden_lookup(
        rollout_hidden_paths,
        rollout_index_paths,
        component_name=component_name,
        layer_index=0,
        pool_mode=pool,
    )
    grouped_rows, feature_keys = load_grouped_rollouts(manifest_path, rollout_hidden_lookup)
    pair_rows = build_pair_rows(
        grouped_rows,
        feature_keys,
        {"train", "validation"},
        {"test"},
        train_pairs_per_prompt,
        eval_pairs_per_prompt,
        random_seed,
    )
    x, y, splits, task_ids = build_feature_matrix(pair_rows, prompt_lookup, prompt_mode)
    train_mask = np.asarray([split == "train" for split in splits], dtype=bool)
    val_mask = np.asarray([split == "validation" for split in splits], dtype=bool)
    test_mask = np.asarray([split == "test" for split in splits], dtype=bool)
    dataset = {
        "x_train": x[train_mask],
        "y_train": y[train_mask],
        "train_task_ids": [task_ids[i] for i, keep in enumerate(train_mask.tolist()) if keep],
        "x_val": x[val_mask],
        "y_val": y[val_mask],
        "val_task_ids": [task_ids[i] for i, keep in enumerate(val_mask.tolist()) if keep],
        "x_test": x[test_mask],
        "y_test": y[test_mask],
        "test_task_ids": [task_ids[i] for i, keep in enumerate(test_mask.tolist()) if keep],
        "feature_dim": int(x.shape[1]),
    }
    _save_dataset_cache(cache_path, dataset)
    _write_json(
        outdir / "dataset_cache" / f"{dataset_key}.summary.json",
        {
            "dataset_key": dataset_key,
            "feature_dim": int(x.shape[1]),
            "num_train_rows": int(train_mask.sum()),
            "num_val_rows": int(val_mask.sum()),
            "num_test_rows": int(test_mask.sum()),
            "cache_path": str(cache_path),
        },
    )
    return cache_path


def refresh_partial_summary(outdir: Path, metadata: dict[str, Any]) -> None:
    summary = dict(metadata)
    summary["models"] = {}
    for family in ("ExtraTrees", "Ridge", "SVM", "MLP", "LDA"):
        result_path = _family_result_path(outdir, family)
        if result_path.exists():
            summary["models"][family] = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            progress_path = _family_progress_path(outdir, family)
            if progress_path.exists():
                summary["models"][family] = {"status": "running", "best_so_far": json.loads(progress_path.read_text(encoding="utf-8"))}
    _write_json(outdir / "summary.partial.json", summary)


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    manifest_path = repo / "classifer_training/artifacts/manifests/dapo_math_17k_qwen3_4b_instruct_2507_promptonly_finished16_rollout_raw.json"
    prompt_hidden_dir = repo / "classifer_training/artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507_last6mean"
    prompt_index_dir = repo / "classifer_training/artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507_last6mean"
    rollout_hidden_paths = [
        repo
        / "classifer_training/artifacts/rollout_hidden/dapo_math_17k/_data2_sangjunsong__cache_hf_hub_models--Qwen--Qwen3-4B-Instruct-2507_snapshots_cdbee75f17c01a7cc42f958dc650907174af0554/finished16_plus_extra2000v2_think_end_l26.shard00of02.pt",
        repo
        / "classifer_training/artifacts/rollout_hidden/dapo_math_17k/_data2_sangjunsong__cache_hf_hub_models--Qwen--Qwen3-4B-Instruct-2507_snapshots_cdbee75f17c01a7cc42f958dc650907174af0554/finished16_plus_extra2000v2_think_end_l26.shard01of02.pt",
    ]
    rollout_index_paths = [
        repo
        / "classifer_training/artifacts/rollout_index/dapo_math_17k/_data2_sangjunsong__cache_hf_hub_models--Qwen--Qwen3-4B-Instruct-2507_snapshots_cdbee75f17c01a7cc42f958dc650907174af0554/finished16_plus_extra2000v2_think_end_l26.shard00of02.jsonl",
        repo
        / "classifer_training/artifacts/rollout_index/dapo_math_17k/_data2_sangjunsong__cache_hf_hub_models--Qwen--Qwen3-4B-Instruct-2507_snapshots_cdbee75f17c01a7cc42f958dc650907174af0554/finished16_plus_extra2000v2_think_end_l26.shard01of02.jsonl",
    ]
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare"
    outdir.mkdir(parents=True, exist_ok=True)

    prompt_mode = "l10_l26"
    train_pairs_per_prompt = 4
    eval_pairs_per_prompt = 10
    random_seed = 42
    fast_mode = os.environ.get("FAST_COMPARE", "0") == "1"
    family_parallel = int(os.environ.get("FAMILY_PARALLEL", "4"))
    et_n_jobs = int(os.environ.get("ET_N_JOBS", str(max(1, os.cpu_count() // max(family_parallel, 1)))))
    force_rebuild = os.environ.get("FORCE_REBUILD_DATASETS", "0") == "1"

    component_specs = [
        {"name": "think_end_hidden", "pool": "mean"},
        {"name": "think_end_last10_hidden", "pool": "mean"},
    ]
    metadata = {
        "setting": "two_rollout_think_fair_compare",
        "fast_mode": fast_mode,
        "prompt_mode": "last6_mean+layer26",
        "component_candidates": component_specs,
        "family_parallel": family_parallel,
        "et_n_jobs": et_n_jobs,
        "models": {},
    }

    print("stage: load prompt lookup", flush=True)
    prompt_lookup = build_prompt_lookup(prompt_hidden_dir, prompt_index_dir)

    dataset_paths: list[str] = []
    for component_spec in component_specs:
        cache_path = build_or_load_dataset_cache(
            repo=repo,
            outdir=outdir,
            prompt_lookup=prompt_lookup,
            component_name=str(component_spec["name"]),
            pool=str(component_spec["pool"]),
            manifest_path=manifest_path,
            rollout_hidden_paths=rollout_hidden_paths,
            rollout_index_paths=rollout_index_paths,
            prompt_mode=prompt_mode,
            train_pairs_per_prompt=train_pairs_per_prompt,
            eval_pairs_per_prompt=eval_pairs_per_prompt,
            random_seed=random_seed,
            force_rebuild=force_rebuild,
        )
        dataset_paths.append(str(cache_path))
    refresh_partial_summary(outdir, metadata)

    model_specs = build_model_specs(fast_mode, random_seed, et_n_jobs)
    futures = {}
    with ProcessPoolExecutor(max_workers=min(family_parallel, 5)) as executor:
        for family in ("ExtraTrees", "Ridge", "SVM", "MLP"):
            futures[executor.submit(run_family_search, family, dataset_paths, model_specs, random_seed, str(outdir))] = family
        futures[executor.submit(run_lda_search, dataset_paths, random_seed, fast_mode, str(outdir))] = "LDA"
        for future in as_completed(futures):
            family = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                _write_json(outdir / f"{family.lower()}_error.json", {"family": family, "error": repr(exc)})
                refresh_partial_summary(outdir, metadata)
                raise
            _write_json(_family_result_path(outdir, family), result)
            refresh_partial_summary(outdir, metadata)

    final_summary = json.loads((outdir / "summary.partial.json").read_text(encoding="utf-8"))
    _write_json(outdir / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
