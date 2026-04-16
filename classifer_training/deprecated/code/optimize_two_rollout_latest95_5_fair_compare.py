from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVR

from classifer_training.search_two_rollout_finished16_focus import _build_label_buckets, _load_grouped_rows
from classifer_training.train_prompt_two_trajectory_promptsearch import (
    build_matrix,
    build_pair_rows,
    build_prompt_lookup,
)


def bucket(diff: float, low: float = 0.1, high: float = 0.9) -> int:
    if diff <= low:
        return 0
    if diff >= high:
        return 2
    return 1


def aggregate_prompt(task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y": [], "p": []})
    for t, y, p in zip(task_ids, y_true.tolist(), y_pred.tolist()):
        groups[str(t)]["y"].append(float(y))
        groups[str(t)]["p"].append(float(p))
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
    with path.open("w", encoding="utf-8") as f:
        for t, y, p in zip(ordered, yt.tolist(), yp.tolist()):
            f.write(json.dumps({"task_id": t, "y_true_difficulty": y, "predicted_difficulty": p}) + "\n")


def make_outer_inner_split(grouped_rows: list[dict], outer_seed: int = 42, inner_seed: int = 43) -> list[dict]:
    rng_outer = np.random.default_rng(outer_seed)
    rng_inner = np.random.default_rng(inner_seed)
    by_bucket: dict[int, list[str]] = defaultdict(list)
    rows_by_id = {str(row["task_id"]): row for row in grouped_rows}
    for row in grouped_rows:
        by_bucket[bucket(float(row["y_true"]))].append(str(row["task_id"]))

    test_task_ids: set[str] = set()
    for _, task_ids in by_bucket.items():
        arr = np.asarray(sorted(task_ids))
        k = max(1, int(round(len(arr) * 0.05)))
        chosen = rng_outer.choice(arr, size=min(k, len(arr)), replace=False)
        test_task_ids.update(str(x) for x in chosen.tolist())

    train_pool: dict[int, list[str]] = defaultdict(list)
    for b, task_ids in by_bucket.items():
        for task_id in task_ids:
            if task_id not in test_task_ids:
                train_pool[b].append(task_id)

    val_task_ids: set[str] = set()
    for _, task_ids in train_pool.items():
        arr = np.asarray(sorted(task_ids))
        k = max(1, int(round(len(arr) * 0.10)))
        chosen = rng_inner.choice(arr, size=min(k, len(arr)), replace=False)
        val_task_ids.update(str(x) for x in chosen.tolist())

    updated_rows = []
    for task_id, row in rows_by_id.items():
        new_row = dict(row)
        new_row["rollouts"] = row["rollouts"]
        if task_id in test_task_ids:
            new_row["split"] = "test"
        elif task_id in val_task_ids:
            new_row["split"] = "val"
        else:
            new_row["split"] = "train"
        updated_rows.append(new_row)
    updated_rows.sort(key=lambda row: str(row["task_id"]))
    return updated_rows


def fit_and_eval(model, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, eval_task_ids: list[str]) -> dict:
    model.fit(X_train, y_train)
    pred = np.clip(np.asarray(model.predict(X_eval), dtype=np.float32).reshape(-1), 0.0, 1.0)
    metric, ordered, yt, yp = metrics(eval_task_ids, y_eval, pred)
    return {"metric": metric, "pred": pred, "ordered": ordered, "yt": yt, "yp": yp}


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    run_root = repo / "classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507"
    rollout_index_path = repo / "classifer_training/artifacts/rollout_index/dapo_math_17k/qwen3_4b_instruct_2507_promptonly_finished16/finished16_promptonly_rollout_index_compact.jsonl"
    prompt_hidden_dir = repo / "classifer_training/artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507"
    prompt_index_dir = repo / "classifer_training/artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507"
    pooled_hidden_dir = repo / "classifer_training/artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507_last6mean"
    pooled_index_dir = repo / "classifer_training/artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507_last6mean"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_latest95_5_fair_optimized"
    outdir.mkdir(parents=True, exist_ok=True)

    random_seed = 42
    prompt_mode = "l10_l26"
    train_pairs_per_prompt = 4
    eval_pairs_per_prompt = 10

    print("stage: load labels", flush=True)
    label_buckets = _build_label_buckets(run_root)
    grouped_rows, feature_keys = _load_grouped_rows(rollout_index_path, label_buckets)
    print("stage: split outer/inner", flush=True)
    updated_rows = make_outer_inner_split(grouped_rows, outer_seed=random_seed, inner_seed=random_seed + 1)
    print("stage: build pair rows", flush=True)
    pair_rows = build_pair_rows(
        grouped_rows=updated_rows,
        feature_keys=feature_keys,
        train_splits={"train"},
        test_splits={"val", "test"},
        train_pairs_per_prompt=train_pairs_per_prompt,
        test_pairs_per_prompt=eval_pairs_per_prompt,
        random_seed=random_seed,
    )
    print("stage: build prompt lookup", flush=True)
    prompt_lookup = build_prompt_lookup(prompt_hidden_dir, prompt_index_dir, pooled_hidden_dir, pooled_index_dir)
    print("stage: build matrix", flush=True)
    X, y, splits, metadata = build_matrix(pair_rows, prompt_lookup, prompt_mode)
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    val_task_ids = [metadata[i]["task_id"] for i, keep in enumerate(val_mask.tolist()) if keep]
    test_task_ids = [metadata[i]["task_id"] for i, keep in enumerate(test_mask.tolist()) if keep]

    summary: dict[str, object] = {
        "setting": "two_rollout_latest95_5_fair_optimized",
        "prompt_mode": "last6_mean+layer26",
        "rows": {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]), "test": int(X_test.shape[0])},
        "models": {},
    }

    model_specs: dict[str, list[tuple[dict, object]]] = {
        "ExtraTrees": [],
        "Ridge": [],
        "SVM": [],
        "MLP": [],
        "LDA": [],
    }

    for n_estimators in [1000, 2000, 3000]:
        for min_samples_leaf in [3, 5, 7]:
            for max_features in [0.5, 0.7, 0.9]:
                params = {"n_estimators": n_estimators, "min_samples_leaf": min_samples_leaf, "max_features": max_features}
                model_specs["ExtraTrees"].append((params, ExtraTreesRegressor(random_state=random_seed, n_jobs=12, **params)))

    for alpha in [30, 100, 300, 1000, 3000, 10000, 30000]:
        params = {"alpha": alpha}
        model_specs["Ridge"].append((params, Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=random_seed))])))

    for C in [0.03, 0.1, 0.3, 1.0, 3.0]:
        for epsilon in [0.01, 0.03, 0.05, 0.1]:
            params = {"C": C, "epsilon": epsilon}
            model_specs["SVM"].append(
                (
                    params,
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "svr",
                                LinearSVR(
                                    C=C,
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

    for hidden in [(256, 128), (512, 256), (512, 256, 128)]:
        for alpha in [1e-4, 1e-3, 1e-2]:
            for lr in [3e-4, 1e-3]:
                params = {"hidden_layer_sizes": list(hidden), "alpha": alpha, "learning_rate_init": lr}
                model_specs["MLP"].append(
                    (
                        params,
                        Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                (
                                    "mlp",
                                    MLPRegressor(
                                        hidden_layer_sizes=hidden,
                                        alpha=alpha,
                                        learning_rate_init=lr,
                                        activation="relu",
                                        solver="adam",
                                        batch_size=256,
                                        early_stopping=True,
                                        validation_fraction=0.1,
                                        n_iter_no_change=15,
                                        max_iter=200,
                                        random_state=random_seed,
                                    ),
                                ),
                            ]
                        ),
                    )
                )

    edge_sets = [
        np.asarray([0.0, 0.1, 0.9, 1.01], dtype=np.float32),
        np.asarray([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01], dtype=np.float32),
        np.asarray([0.0, 0.0625, 0.25, 0.5, 0.75, 0.9375, 1.01], dtype=np.float32),
        np.asarray([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.01], dtype=np.float32),
    ]
    pca_dims = [64, 128, 256]

    for family in ["ExtraTrees", "Ridge", "SVM", "MLP"]:
        print(f"stage: search {family}", flush=True)
        best = None
        for params, model in model_specs[family]:
            result = fit_and_eval(model, X_train, y_train, X_val, y_val, val_task_ids)
            row = {"params": params, "val": result["metric"]}
            if best is None or row["val"]["prompt_mean_r2"] > best["val"]["prompt_mean_r2"]:
                best = row
        assert best is not None
        print(json.dumps({"family": family, "best_val_prompt_r2": best["val"]["prompt_mean_r2"], "params": best["params"]}), flush=True)

        # Refit on train+val and evaluate on test.
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)
        if family == "ExtraTrees":
            final_model = ExtraTreesRegressor(random_state=random_seed, n_jobs=12, **best["params"])
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
        else:
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
                            n_iter_no_change=15,
                            max_iter=200,
                            random_state=random_seed,
                        ),
                    ),
                ]
            )
        final_result = fit_and_eval(final_model, X_trainval, y_trainval, X_test, y_test, test_task_ids)
        summary["models"][family] = {
            "best_params_from_val": best["params"],
            "val_prompt_mean_r2": best["val"]["prompt_mean_r2"],
            "test": final_result["metric"],
        }
        save_predictions(outdir / f"{family.lower()}_predictions_test.jsonl", test_task_ids, y_test, final_result["pred"])
        save_plot(outdir / f"{family.lower()}_vs_gt.png", final_result["yt"], final_result["yp"], f"{family} | prompt R2={final_result['metric']['prompt_mean_r2']:.3f}")

    print("stage: search LDA", flush=True)
    lda_best = None
    for edges in edge_sets:
        for pca_dim in pca_dims:
            y_bin = np.clip(np.digitize(y_train, edges[1:-1], right=False), 0, len(edges) - 2)
            present = sorted(set(y_bin.tolist()))
            remap = {c: i for i, c in enumerate(present)}
            y_mapped = np.asarray([remap[c] for c in y_bin.tolist()], dtype=np.int64)
            class_means = np.asarray([float(np.mean(y_train[y_bin == c])) for c in present], dtype=np.float32)
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=min(pca_dim, X_train.shape[1]), random_state=random_seed)),
                    ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                ]
            )
            model.fit(X_train, y_mapped)
            prob = model.predict_proba(X_val)
            pred = np.clip((prob @ class_means).astype(np.float32), 0.0, 1.0)
            metric, _, yt, yp = metrics(val_task_ids, y_val, pred)
            row = {
                "params": {"edges": edges.tolist(), "pca_dim": pca_dim},
                "class_means": class_means.tolist(),
                "val": metric,
            }
            if lda_best is None or row["val"]["prompt_mean_r2"] > lda_best["val"]["prompt_mean_r2"]:
                lda_best = row
        print(json.dumps({"family": "LDA", "best_so_far_val_prompt_r2": None if lda_best is None else lda_best["val"]["prompt_mean_r2"]}), flush=True)

    assert lda_best is not None
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    y_bin = np.clip(np.digitize(y_trainval, np.asarray(lda_best["params"]["edges"], dtype=np.float32)[1:-1], right=False), 0, len(lda_best["params"]["edges"]) - 2)
    present = sorted(set(y_bin.tolist()))
    remap = {c: i for i, c in enumerate(present)}
    y_mapped = np.asarray([remap[c] for c in y_bin.tolist()], dtype=np.int64)
    class_means = np.asarray([float(np.mean(y_trainval[y_bin == c])) for c in present], dtype=np.float32)
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(int(lda_best["params"]["pca_dim"]), X_trainval.shape[1]), random_state=random_seed)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    final_model.fit(X_trainval, y_mapped)
    prob = final_model.predict_proba(X_test)
    pred = np.clip((prob @ class_means).astype(np.float32), 0.0, 1.0)
    metric, _, yt, yp = metrics(test_task_ids, y_test, pred)
    summary["models"]["LDA"] = {
        "best_params_from_val": lda_best["params"],
        "val_prompt_mean_r2": lda_best["val"]["prompt_mean_r2"],
        "test": metric,
    }
    save_predictions(outdir / "lda_predictions_test.jsonl", test_task_ids, y_test, pred)
    save_plot(outdir / "lda_vs_gt.png", yt, yp, f"LDA | prompt R2={metric['prompt_mean_r2']:.3f}")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
