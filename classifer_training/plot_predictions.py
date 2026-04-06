from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def aggregate_by_task(rows: list[dict]) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)

    aggregated: list[dict] = []
    for task_id, task_rows in by_task.items():
        aggregated.append(
            {
                "task_id": task_id,
                "y_true": float(task_rows[0]["y_true"]),
                "y_pred_mean": float(np.mean([row["y_pred"] for row in task_rows])),
                "y_pred_std": float(np.std([row["y_pred"] for row in task_rows])),
                "num_rollouts": len(task_rows),
            }
        )
    return aggregated


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _bin_curve(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(float(np.min(y_pred)), float(np.max(y_pred)), bins + 1)
    centers: list[float] = []
    means: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == edges[-1]:
            mask = (y_pred >= left) & (y_pred <= right)
        else:
            mask = (y_pred >= left) & (y_pred < right)
        if np.any(mask):
            centers.append(float(np.mean(y_pred[mask])))
            means.append(float(np.mean(y_true[mask])))
    return np.asarray(centers, dtype=np.float32), np.asarray(means, dtype=np.float32)


def make_figure(rows: list[dict], title: str) -> plt.Figure:
    rollout_true = np.asarray([row["y_true"] for row in rows], dtype=np.float32)
    rollout_pred = np.asarray([row["y_pred"] for row in rows], dtype=np.float32)
    rollout_resid = rollout_pred - rollout_true

    prompt_rows = aggregate_by_task(rows)
    prompt_true = np.asarray([row["y_true"] for row in prompt_rows], dtype=np.float32)
    prompt_pred = np.asarray([row["y_pred_mean"] for row in prompt_rows], dtype=np.float32)
    prompt_std = np.asarray([row["y_pred_std"] for row in prompt_rows], dtype=np.float32)
    prompt_resid = prompt_pred - prompt_true

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    ax = axes[0, 0]
    hb = ax.hexbin(rollout_true, rollout_pred, gridsize=28, cmap="viridis", mincnt=1)
    lo = min(float(np.min(rollout_true)), float(np.min(rollout_pred)))
    hi = max(float(np.max(rollout_true)), float(np.max(rollout_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    ax.set_title(
        f"Rollout Rows\nR^2={_r2(rollout_true, rollout_pred):.3f}, MAE={_mae(rollout_true, rollout_pred):.3f}"
    )
    ax.set_xlabel("True difficulty")
    ax.set_ylabel("Predicted difficulty")
    fig.colorbar(hb, ax=ax, label="count")

    ax = axes[0, 1]
    ax.scatter(prompt_true, prompt_pred, s=18, alpha=0.75, color="#1f77b4", edgecolors="none")
    for value in sorted(set(prompt_true.tolist())):
        ax.axvline(value, color="lightgray", linewidth=0.5, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    ax.set_title(
        f"Prompt-Aggregated Mean Prediction\nR^2={_r2(prompt_true, prompt_pred):.3f}, MAE={_mae(prompt_true, prompt_pred):.3f}"
    )
    ax.set_xlabel("True difficulty")
    ax.set_ylabel("Mean predicted difficulty over 16 rollouts")

    ax = axes[1, 0]
    ax.hist(rollout_resid, bins=40, alpha=0.45, color="#2ca02c", label="rollout residual")
    ax.hist(prompt_resid, bins=30, alpha=0.6, color="#d62728", label="prompt-mean residual")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Predicted - true")
    ax.set_ylabel("count")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    centers_rollout, means_rollout = _bin_curve(rollout_true, rollout_pred, bins=14)
    centers_prompt, means_prompt = _bin_curve(prompt_true, prompt_pred, bins=10)
    ax.plot(centers_rollout, means_rollout, marker="o", linewidth=1.5, label="rollout rows")
    ax.plot(centers_prompt, means_prompt, marker="o", linewidth=1.5, label="prompt mean")
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    ax.set_title(
        "Binned Prediction Curve\n"
        f"tasks={len(prompt_rows)}, rows={len(rows)}, mean prompt pred std={float(np.mean(prompt_std)):.3f}"
    )
    ax.set_xlabel("Predicted difficulty")
    ax.set_ylabel("Average true difficulty")
    ax.legend(frameon=False)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot regression predictions from predictions.jsonl.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Prediction Diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.predictions.expanduser().resolve())
    fig = make_figure(rows, args.title)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
