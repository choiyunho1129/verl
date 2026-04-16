from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr(rankdata(x), rankdata(y))


def r2(x_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((x_true - y_pred) ** 2))
    ss_tot = float(np.sum((x_true - float(np.mean(x_true))) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def mae(x_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(x_true - y_pred)))


def metrics(x_ref: np.ndarray, y_cmp: np.ndarray) -> dict[str, float]:
    return {
        "r2": r2(x_ref, y_cmp),
        "mae": mae(x_ref, y_cmp),
        "pearson": pearsonr(x_ref, y_cmp),
        "spearman": spearmanr(x_ref, y_cmp),
        "count": int(x_ref.shape[0]),
    }


def load_best_predictions(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open() as handle:
        for line in handle:
            obj = json.loads(line)
            out[str(obj["task_id"])] = {
                "y_true": float(obj["y_true"]),
                "y_pred": float(obj["y_pred"]),
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


def compare_figure(
    out_path: Path,
    x_ref: np.ndarray,
    y_cmp: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
) -> dict[str, float]:
    stat = metrics(x_ref, y_cmp)
    order = np.argsort(x_ref)
    xs = x_ref[order]
    ys = y_cmp[order]
    err = np.abs(ys - xs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hb = axes[0].hexbin(x_ref, y_cmp, gridsize=32, cmap="viridis", bins="log", mincnt=1)
    axes[0].plot([0, 1], [0, 1], "--", color="tab:red", lw=1.5)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title(f"Hexbin | R2={stat['r2']:.3f}")
    fig.colorbar(hb, ax=axes[0], label="log count")

    axes[1].plot(xs, color="black", lw=2, label="ref")
    axes[1].plot(ys, color="tab:purple", lw=1.5, label="cmp")
    axes[1].set_xlabel(f"Examples sorted by {x_label}")
    axes[1].set_title("Sorted Alignment")
    axes[1].legend(frameon=False)

    axes[2].plot(err, color="teal", lw=1.2)
    axes[2].set_xlabel(f"Examples sorted by {x_label}")
    axes[2].set_ylabel(f"|{y_label} - {x_label}|")
    axes[2].set_title(f"Absolute Error | MAE={stat['mae']:.3f}")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stat


def summary_grid(
    out_path: Path,
    full16: np.ndarray,
    probe: np.ndarray,
    rollout8: np.ndarray,
    boot_a: np.ndarray,
    boot_b: np.ndarray,
    stats: dict[str, dict[str, float]],
) -> None:
    pairs = [
        ("Probe vs Full16", full16, probe, "True difficulty (rollout-16)", "Predicted difficulty"),
        ("Probe vs Rollout8", rollout8, probe, "Difficulty from 8 rollouts", "Predicted difficulty"),
        ("Rollout8 vs Full16", full16, rollout8, "True difficulty (rollout-16)", "Difficulty from 8 rollouts"),
        ("Bootstrap8-A vs Full16", full16, boot_a, "True difficulty (rollout-16)", "Difficulty from bootstrap-8 A"),
        ("Bootstrap8-B vs Full16", full16, boot_b, "True difficulty (rollout-16)", "Difficulty from bootstrap-8 B"),
        ("Bootstrap8-A vs Bootstrap8-B", boot_a, boot_b, "Bootstrap-8 A", "Bootstrap-8 B"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    for ax, (name, x_ref, y_cmp, x_label, y_label) in zip(axes[:5], pairs):
        hb = ax.hexbin(x_ref, y_cmp, gridsize=28, cmap="viridis", bins="log", mincnt=1)
        ax.plot([0, 1], [0, 1], "--", color="tab:red", lw=1.2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(name)
        fig.colorbar(hb, ax=ax, label="log count")

    axes[5].axis("off")
    lines = [
        "Metrics vs rollout-16 / bootstrap pair",
        f"Probe: R2={stats['probe_vs_rollout16']['r2']:.3f}, MAE={stats['probe_vs_rollout16']['mae']:.3f}",
        f"Probe vs Rollout8: R2={stats['probe_vs_rollout8']['r2']:.3f}, MAE={stats['probe_vs_rollout8']['mae']:.3f}",
        f"Rollout8: R2={stats['rollout8_vs_rollout16']['r2']:.3f}, MAE={stats['rollout8_vs_rollout16']['mae']:.3f}",
        f"Bootstrap8-A: R2={stats['bootstrap8_a_vs_rollout16']['r2']:.3f}, MAE={stats['bootstrap8_a_vs_rollout16']['mae']:.3f}",
        f"Bootstrap8-B: R2={stats['bootstrap8_b_vs_rollout16']['r2']:.3f}, MAE={stats['bootstrap8_b_vs_rollout16']['mae']:.3f}",
        f"Bootstrap A vs B: R2={stats['bootstrap8_a_vs_b']['r2']:.3f}, MAE={stats['bootstrap8_a_vs_b']['mae']:.3f}",
        "",
        f"Probe Pearson/Spearman: {stats['probe_vs_rollout16']['pearson']:.3f} / {stats['probe_vs_rollout16']['spearman']:.3f}",
        f"Probe vs Rollout8 Pearson/Spearman: {stats['probe_vs_rollout8']['pearson']:.3f} / {stats['probe_vs_rollout8']['spearman']:.3f}",
        f"Rollout8 Pearson/Spearman: {stats['rollout8_vs_rollout16']['pearson']:.3f} / {stats['rollout8_vs_rollout16']['spearman']:.3f}",
        f"Bootstrap A Pearson/Spearman: {stats['bootstrap8_a_vs_rollout16']['pearson']:.3f} / {stats['bootstrap8_a_vs_rollout16']['spearman']:.3f}",
        f"Bootstrap B Pearson/Spearman: {stats['bootstrap8_b_vs_rollout16']['pearson']:.3f} / {stats['bootstrap8_b_vs_rollout16']['spearman']:.3f}",
        f"Bootstrap A/B Pearson/Spearman: {stats['bootstrap8_a_vs_b']['pearson']:.3f} / {stats['bootstrap8_a_vs_b']['spearman']:.3f}",
    ]
    axes[5].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    fig.suptitle("Best Probe vs Rollout-8 / Bootstrap-8 Noise")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    pred_path = repo / "classifer_training/artifacts/models/tmp_finished16_last6_l26_2000/predictions_test.jsonl"
    seed1_experiments = repo / "classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_seed1/all_experiments.jsonl"
    seed_eval_paths = [
        repo / f"classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_seed{i}/evaluation_results.jsonl"
        for i in range(1, 17)
    ]
    outdir = repo / "classifer_training/artifacts/models/tmp_finished16_last6_l26_2000/rollout8_bootstrap_compare"
    outdir.mkdir(parents=True, exist_ok=True)

    best_pred = load_best_predictions(pred_path)
    ordered_pred_task_ids = sorted(best_pred)
    pred_true = np.asarray([best_pred[t]["y_true"] for t in ordered_pred_task_ids], dtype=np.float32)
    pred_probe = np.asarray([best_pred[t]["y_pred"] for t in ordered_pred_task_ids], dtype=np.float32)

    task_order, splits = load_seed_task_order(seed1_experiments)
    index_by_task = {task_id: idx for idx, task_id in enumerate(task_order)}
    test_task_ids = [task_id for task_id, split in zip(task_order, splits) if split == "test"]
    assert set(test_task_ids) == set(ordered_pred_task_ids)
    ordered_test_indices = [index_by_task[t] for t in ordered_pred_task_ids]

    correctness = np.stack([load_seed_correctness(path) for path in seed_eval_paths], axis=1)
    test_correctness = correctness[ordered_test_indices, :]

    full16 = 1.0 - np.mean(test_correctness, axis=1)
    if not np.allclose(full16, pred_true):
        raise RuntimeError("Computed full16 difficulty does not match stored prediction ground truth.")

    rollout8 = 1.0 - np.mean(test_correctness[:, :8], axis=1)

    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    boot_idx_a = rng_a.integers(0, 16, size=(test_correctness.shape[0], 8))
    boot_idx_b = rng_b.integers(0, 16, size=(test_correctness.shape[0], 8))
    boot_a = 1.0 - np.take_along_axis(test_correctness, boot_idx_a, axis=1).mean(axis=1)
    boot_b = 1.0 - np.take_along_axis(test_correctness, boot_idx_b, axis=1).mean(axis=1)

    stats = {}
    stats["probe_vs_rollout16"] = compare_figure(
        outdir / "best_probe_vs_rollout16.png",
        full16,
        pred_probe,
        "Best Probe vs rollout-16",
        "True difficulty (rollout-16)",
        "Predicted difficulty",
    )
    stats["probe_vs_rollout8"] = compare_figure(
        outdir / "best_probe_vs_rollout8.png",
        rollout8,
        pred_probe,
        "Best Probe vs actual rollout-8 (seeds 1-8)",
        "Difficulty from 8 rollouts",
        "Predicted difficulty",
    )
    stats["rollout8_vs_rollout16"] = compare_figure(
        outdir / "rollout8_vs_rollout16.png",
        full16,
        rollout8,
        "Actual rollout-8 (seeds 1-8) vs rollout-16",
        "True difficulty (rollout-16)",
        "Difficulty from 8 rollouts",
    )
    stats["bootstrap8_a_vs_rollout16"] = compare_figure(
        outdir / "bootstrap8_a_vs_rollout16.png",
        full16,
        boot_a,
        "Bootstrap-8 sample A vs rollout-16",
        "True difficulty (rollout-16)",
        "Difficulty from bootstrap-8 A",
    )
    stats["bootstrap8_b_vs_rollout16"] = compare_figure(
        outdir / "bootstrap8_b_vs_rollout16.png",
        full16,
        boot_b,
        "Bootstrap-8 sample B vs rollout-16",
        "True difficulty (rollout-16)",
        "Difficulty from bootstrap-8 B",
    )
    stats["bootstrap8_a_vs_b"] = compare_figure(
        outdir / "bootstrap8_a_vs_b.png",
        boot_a,
        boot_b,
        "Bootstrap-8 sample A vs bootstrap-8 sample B",
        "Bootstrap-8 A",
        "Bootstrap-8 B",
    )

    summary_grid(outdir / "comparison_grid.png", full16, pred_probe, rollout8, boot_a, boot_b, stats)

    summary = {
        "setting": "best_probe_vs_rollout8_bootstrap8",
        "best_probe_predictions": str(pred_path),
        "fixed_rollout8_seeds": list(range(1, 9)),
        "bootstrap_seeds": {"a": 0, "b": 1},
        "num_test_prompts": int(full16.shape[0]),
        "metrics": stats,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
