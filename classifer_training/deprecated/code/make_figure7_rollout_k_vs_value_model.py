from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def explained_variance_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    var_y = float(np.var(y_true))
    if var_y == 0.0:
        return 0.0
    return 1.0 - float(np.var(residual)) / var_y


def load_best_predictions(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            out[str(row["task_id"])] = {
                "y_true": float(row["y_true_difficulty"]),
                "y_pred": float(row["predicted_difficulty"]),
            }
    return out


def load_seed_task_order(path: Path) -> tuple[list[str], list[str]]:
    task_ids: list[str] = []
    splits: list[str] = []
    with path.open() as handle:
        for line in handle:
            obj = json.loads(line)
            task_ids.append(str(obj["task_id"]))
            splits.append(str(obj["split"]))
    return task_ids, splits


def load_seed_correctness(path: Path) -> np.ndarray:
    with path.open() as handle:
        obj = json.loads(next(iter(handle)))
    return np.asarray(obj["correctness"], dtype=np.float32)


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    pred_path = (
        repo
        / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_tail_balanced_search/tail_balanced_predictions_test.jsonl"
    )
    seed1_experiments = (
        repo
        / "classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_seed1/all_experiments.jsonl"
    )
    seed_eval_paths = [
        repo
        / f"classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_seed{i}/evaluation_results.jsonl"
        for i in range(1, 17)
    ]
    outdir = (
        repo
        / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_tail_balanced_search/figure7_rollout_k_compare"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    best_pred = load_best_predictions(pred_path)
    ordered_pred_task_ids = sorted(best_pred)
    pred_true = np.asarray([best_pred[t]["y_true"] for t in ordered_pred_task_ids], dtype=np.float32)
    pred_probe = np.asarray([best_pred[t]["y_pred"] for t in ordered_pred_task_ids], dtype=np.float32)
    value_model_r2 = r2_score_np(pred_true, pred_probe)
    value_model_ev = explained_variance_np(pred_true, pred_probe)

    task_order, splits = load_seed_task_order(seed1_experiments)
    index_by_task = {task_id: idx for idx, task_id in enumerate(task_order)}
    test_task_ids = [task_id for task_id, split in zip(task_order, splits) if split == "test"]
    if set(test_task_ids) != set(ordered_pred_task_ids):
        raise RuntimeError("Prediction task IDs do not match test task IDs.")
    ordered_test_indices = [index_by_task[t] for t in ordered_pred_task_ids]

    correctness = np.stack([load_seed_correctness(path) for path in seed_eval_paths], axis=1)
    test_correctness = correctness[ordered_test_indices, :]
    full16 = 1.0 - np.mean(test_correctness, axis=1)
    if not np.allclose(full16, pred_true):
        raise RuntimeError("Computed rollout-16 difficulty does not match stored y_true_difficulty.")

    all_seed_indices = tuple(range(test_correctness.shape[1]))
    k_stats: list[dict[str, float | int]] = []

    for k in range(1, 17):
        r2_scores: list[float] = []
        ev_scores: list[float] = []
        subset_count = 0
        for subset in itertools.combinations(all_seed_indices, k):
            subset_count += 1
            subset_diff = 1.0 - np.mean(test_correctness[:, subset], axis=1)
            r2_scores.append(r2_score_np(full16, subset_diff))
            ev_scores.append(explained_variance_np(full16, subset_diff))
        r2_arr = np.asarray(r2_scores, dtype=np.float64)
        ev_arr = np.asarray(ev_scores, dtype=np.float64)
        k_stats.append(
            {
                "k": k,
                "num_subsets": int(subset_count),
                "r2_mean": float(np.mean(r2_arr)),
                "r2_std": float(np.std(r2_arr)),
                "r2_min": float(np.min(r2_arr)),
                "r2_p10": float(np.quantile(r2_arr, 0.1)),
                "r2_p50": float(np.quantile(r2_arr, 0.5)),
                "r2_p90": float(np.quantile(r2_arr, 0.9)),
                "r2_max": float(np.max(r2_arr)),
                "ev_mean": float(np.mean(ev_arr)),
                "ev_std": float(np.std(ev_arr)),
                "ev_min": float(np.min(ev_arr)),
                "ev_p10": float(np.quantile(ev_arr, 0.1)),
                "ev_p50": float(np.quantile(ev_arr, 0.5)),
                "ev_p90": float(np.quantile(ev_arr, 0.9)),
                "ev_max": float(np.max(ev_arr)),
            }
        )

    ks = np.asarray([row["k"] for row in k_stats], dtype=np.int32)
    r2_mean = np.asarray([row["r2_mean"] for row in k_stats], dtype=np.float64)
    r2_p10 = np.asarray([row["r2_p10"] for row in k_stats], dtype=np.float64)
    r2_p90 = np.asarray([row["r2_p90"] for row in k_stats], dtype=np.float64)
    r2_min = np.asarray([row["r2_min"] for row in k_stats], dtype=np.float64)
    r2_max = np.asarray([row["r2_max"] for row in k_stats], dtype=np.float64)
    ev_mean = np.asarray([row["ev_mean"] for row in k_stats], dtype=np.float64)
    ev_p10 = np.asarray([row["ev_p10"] for row in k_stats], dtype=np.float64)
    ev_p90 = np.asarray([row["ev_p90"] for row in k_stats], dtype=np.float64)
    ev_min = np.asarray([row["ev_min"] for row in k_stats], dtype=np.float64)
    ev_max = np.asarray([row["ev_max"] for row in k_stats], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(ks, r2_p10, r2_p90, color="tab:blue", alpha=0.18, label="Rollout-k subset 10-90%")
    ax.fill_between(ks, r2_min, r2_max, color="tab:blue", alpha=0.08, label="Rollout-k subset min-max")
    ax.plot(ks, r2_mean, color="tab:blue", marker="o", lw=2.2, label="Rollout-k mean explained variance")
    ax.axhline(value_model_r2, color="tab:red", ls="--", lw=2.0, label=f"Best value model: R²={value_model_r2:.3f}")

    ax.set_xlim(1, 16)
    ax.set_xticks(np.arange(1, 17))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Number of generations used to estimate difficulty")
    ax.set_ylabel("Explained variance vs rollout-16 difficulty (R²)")
    ax.set_title("Explained Variance of Value Model vs Rollout-k Difficulty")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "figure7_r2_rolloutk_vs_value_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(ks, ev_p10, ev_p90, color="tab:green", alpha=0.18, label="Rollout-k subset 10-90%")
    ax.fill_between(ks, ev_min, ev_max, color="tab:green", alpha=0.08, label="Rollout-k subset min-max")
    ax.plot(ks, ev_mean, color="tab:green", marker="o", lw=2.2, label="Rollout-k mean explained variance")
    ax.axhline(value_model_ev, color="tab:red", ls="--", lw=2.0, label=f"Best value model: EV={value_model_ev:.3f}")

    ax.set_xlim(1, 16)
    ax.set_xticks(np.arange(1, 17))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Number of generations used to estimate difficulty")
    ax.set_ylabel("Explained variance vs rollout-16 difficulty (EV)")
    ax.set_title("Explained Variance of Value Model vs Rollout-k Difficulty")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "figure7_ev_rolloutk_vs_value_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True, sharey=True)
    axes[0].fill_between(ks, r2_p10, r2_p90, color="tab:blue", alpha=0.18)
    axes[0].fill_between(ks, r2_min, r2_max, color="tab:blue", alpha=0.08)
    axes[0].plot(ks, r2_mean, color="tab:blue", marker="o", lw=2.2, label="Rollout-k mean")
    axes[0].axhline(value_model_r2, color="tab:red", ls="--", lw=2.0, label=f"Value model: {value_model_r2:.3f}")
    axes[0].set_title("R²")
    axes[0].set_xlabel("Number of generations")
    axes[0].set_ylabel("Metric vs rollout-16 difficulty")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].fill_between(ks, ev_p10, ev_p90, color="tab:green", alpha=0.18)
    axes[1].fill_between(ks, ev_min, ev_max, color="tab:green", alpha=0.08)
    axes[1].plot(ks, ev_mean, color="tab:green", marker="o", lw=2.2, label="Rollout-k mean")
    axes[1].axhline(value_model_ev, color="tab:red", ls="--", lw=2.0, label=f"Value model: {value_model_ev:.3f}")
    axes[1].set_title("Explained Variance")
    axes[1].set_xlabel("Number of generations")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle("Rollout-k vs Value Model: R² and Explained Variance")
    fig.tight_layout()
    fig.savefig(outdir / "figure7_r2_vs_ev_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Paper-style single figure, using explained variance on the y-axis.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(ks, ev_p10, ev_p90, color="tab:blue", alpha=0.18, label="Rollout-k 10-90%")
    ax.fill_between(ks, ev_min, ev_max, color="tab:blue", alpha=0.08, label="Rollout-k min-max")
    ax.plot(ks, ev_mean, color="tab:blue", marker="o", lw=2.4, label="Rollout-k mean")
    ax.axhline(value_model_ev, color="tab:red", ls="--", lw=2.2, label=f"Value model (EV={value_model_ev:.3f})")
    ax.axhline(value_model_r2, color="tab:orange", ls=":", lw=2.0, label=f"Value model (R²={value_model_r2:.3f})")
    ax.set_xlim(1, 16)
    ax.set_xticks(np.arange(1, 17))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Number of generations used to estimate difficulty")
    ax.set_ylabel("Explained variance")
    ax.set_title("Explained Variance of Value Model and Rollout-k Difficulty")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "figure7_paper_style_single.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "setting": "figure7_rollout_k_vs_value_model",
        "value_model_name": "two_rollout_think_tail_balanced_best",
        "value_model_r2_vs_rollout16": float(value_model_r2),
        "value_model_ev_vs_rollout16": float(value_model_ev),
        "num_test_prompts": int(full16.shape[0]),
        "num_seed_runs": 16,
        "k_stats": k_stats,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
