from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROMPT_DIM = 2560
PROMPT_FEATS_DIM = 23
PROMPT_REL_DIM = 10
ROLLOUT_STATS_DIM = 171
ROLLOUT_PAIR_SCALAR_DIM = 2
REASONING_DIM = 2560


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_xy(task_ids: list[str], x: np.ndarray, y: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    ys: dict[str, list[float]] = defaultdict(list)
    for task_id, xi, yi in zip(task_ids, x, y):
        key = str(task_id)
        if key not in sums:
            sums[key] = np.asarray(xi, dtype=np.float64).copy()
            counts[key] = 1
        else:
            sums[key] += np.asarray(xi, dtype=np.float64)
            counts[key] += 1
        ys[key].append(float(yi))
    ordered = sorted(sums)
    x_prompt = np.stack([sums[k] / counts[k] for k in ordered]).astype(np.float32)
    y_prompt = np.asarray([float(np.mean(ys[k])) for k in ordered], dtype=np.float32)
    return ordered, x_prompt, y_prompt


def select_last10_left_right_only(x: np.ndarray) -> np.ndarray:
    """Keep prompt/base features and only left/right reasoning hidden."""
    # Original order from train_two_rollout_reasoning_probe.build_feature_matrix:
    # prompt_vector(2560), prompt_feats(23), prompt_rel(10),
    # left_vec/right_vec/pair_* (7 * 171), cosine(1), l2(1),
    # left_reasoning/right_reasoning/reasoning_mean/reasoning_absdiff (4 * 2560),
    # reasoning_cosine(1), reasoning_l2(1)
    base_non_reasoning_dim = PROMPT_DIM + PROMPT_FEATS_DIM + PROMPT_REL_DIM + 7 * ROLLOUT_STATS_DIM + ROLLOUT_PAIR_SCALAR_DIM
    left_reasoning_start = base_non_reasoning_dim
    right_reasoning_start = left_reasoning_start + REASONING_DIM
    reasoning_mean_start = right_reasoning_start + REASONING_DIM
    # We intentionally drop reasoning_mean, reasoning_absdiff, reasoning_cosine, reasoning_l2.
    left_reasoning = x[:, left_reasoning_start:right_reasoning_start]
    right_reasoning = x[:, right_reasoning_start:reasoning_mean_start]
    base_part = x[:, :base_non_reasoning_dim]
    reduced = np.concatenate([base_part, left_reasoning, right_reasoning], axis=1).astype(np.float32, copy=False)
    expected_dim = base_non_reasoning_dim + 2 * REASONING_DIM
    if reduced.shape[1] != expected_dim:
        raise ValueError(f"Unexpected reduced dim {reduced.shape[1]} != {expected_dim}")
    return reduced


def fit_base(alpha: float, x: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=42))])
    model.fit(x, y)
    return model


def base_oof_predictions(alpha: float, x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    idx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    oof = np.zeros(len(y), dtype=np.float32)
    for fold in folds:
        train_idx = np.setdiff1d(idx, fold, assume_unique=False)
        model = fit_base(alpha, x[train_idx], y[train_idx])
        oof[fold] = np.clip(np.asarray(model.predict(x[fold]), dtype=np.float32), 0.0, 1.0)
    return oof


def classification_metrics(y_bin: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(np.int32)
    return {
        "positive_rate": float(np.mean(y_bin)),
        "pred_positive_rate": float(np.mean(pred)),
        "precision": float(precision_score(y_bin, pred, zero_division=0)),
        "recall": float(recall_score(y_bin, pred, zero_division=0)),
        "f1": float(f1_score(y_bin, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_bin, prob)),
        "ap": float(average_precision_score(y_bin, prob)),
        "brier": float(brier_score_loss(y_bin, prob)),
        "threshold": float(threshold),
    }


def best_threshold_by_f1(y_bin: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for thr in np.linspace(0.05, 0.95, 19):
        metric = classification_metrics(y_bin, prob, float(thr))
        if best is None or metric["f1"] > best["f1"]:
            best = metric
    assert best is not None
    return best


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    hard_mask = y_true >= 0.8
    vhard_mask = y_true >= 0.9
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "hard_mae": float(mean_absolute_error(y_true[hard_mask], y_pred[hard_mask])) if np.any(hard_mask) else float("nan"),
        "very_hard_mae": float(mean_absolute_error(y_true[vhard_mask], y_pred[vhard_mask])) if np.any(vhard_mask) else float("nan"),
    }


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        blocks: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.backbone = nn.Sequential(*blocks)
        final_dim = hidden_dims[-1]
        self.h85_head = nn.Linear(final_dim, 1)
        self.miss85_head = nn.Linear(final_dim, 1)
        self.delta_head = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.h85_head(h), self.miss85_head(h), self.delta_head(h)


def train_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    base_train_oof: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    base_val_pred: np.ndarray,
    x_test: np.ndarray,
    *,
    hidden_dims: tuple[int, int, int],
    dropout: float,
    seed: int,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_val_scaled = scaler.transform(x_val).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    y_h85_train = (y_train >= 0.85).astype(np.float32)
    y_miss85_train = ((y_train >= 0.85) & (base_train_oof < 0.85)).astype(np.float32)
    y_delta_train = np.clip(y_train - base_train_oof, 0.0, 1.0).astype(np.float32)
    y_hard_mask = (y_train >= 0.85).astype(np.float32)

    pos_weight_h85 = float((len(y_h85_train) - y_h85_train.sum()) / max(y_h85_train.sum(), 1.0))
    pos_weight_miss = float((len(y_miss85_train) - y_miss85_train.sum()) / max(y_miss85_train.sum(), 1.0))

    train_ds = TensorDataset(
        torch.from_numpy(x_train_scaled),
        torch.from_numpy(y_h85_train),
        torch.from_numpy(y_miss85_train),
        torch.from_numpy(y_delta_train),
        torch.from_numpy(y_hard_mask),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = RouterMLP(x_train_scaled.shape[1], hidden_dims, dropout).to(device)
    loss_h85 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_h85, dtype=torch.float32, device=device))
    loss_miss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_miss, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_val_tensor = torch.from_numpy(x_val_scaled).to(device)
    best_state = None
    best_summary: dict[str, Any] | None = None
    best_score = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, h85b, missb, deltab, hardmaskb in train_loader:
            xb = xb.to(device)
            h85b = h85b.to(device).unsqueeze(1)
            missb = missb.to(device).unsqueeze(1)
            deltab = deltab.to(device).unsqueeze(1)
            hardmaskb = hardmaskb.to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            h85_logit, miss_logit, delta_raw = model(xb)
            delta_pred = torch.sigmoid(delta_raw)

            l_h85 = loss_h85(h85_logit, h85b)
            l_miss = loss_miss(miss_logit, missb)
            delta_err = ((delta_pred - deltab) ** 2) * hardmaskb
            l_delta = delta_err.sum() / torch.clamp(hardmaskb.sum(), min=1.0)
            loss = 1.0 * l_h85 + 1.5 * l_miss + 0.5 * l_delta
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            h85_logit, miss_logit, delta_raw = model(x_val_tensor)
            p_h85 = torch.sigmoid(h85_logit).squeeze(1).cpu().numpy().astype(np.float32)
            p_miss = torch.sigmoid(miss_logit).squeeze(1).cpu().numpy().astype(np.float32)
            delta_hat = torch.sigmoid(delta_raw).squeeze(1).cpu().numpy().astype(np.float32)

        h85_val_bin = (y_val >= 0.85).astype(np.int32)
        miss_val_bin = ((y_val >= 0.85) & (base_val_pred < 0.85)).astype(np.int32)
        h85_best = best_threshold_by_f1(h85_val_bin, p_h85)
        miss_best = best_threshold_by_f1(miss_val_bin, p_miss)

        best_route: dict[str, Any] | None = None
        for w_h85 in (0.0, 0.25, 0.5, 0.75, 1.0):
            for w_miss in (0.0, 0.25, 0.5, 0.75, 1.0):
                if w_h85 == 0.0 and w_miss == 0.0:
                    continue
                gate = np.clip(w_h85 * p_h85 + w_miss * p_miss, 0.0, 1.0)
                corrected = np.clip(base_val_pred + gate * delta_hat, 0.0, 1.0)
                reg = regression_metrics(y_val, corrected)
                row = {"w_h85": w_h85, "w_miss": w_miss, "metrics": reg}
                score = (reg["r2"], -reg["hard_mae"], -reg["very_hard_mae"])
                best_route_score = None if best_route is None else (
                    best_route["metrics"]["r2"],
                    -best_route["metrics"]["hard_mae"],
                    -best_route["metrics"]["very_hard_mae"],
                )
                if best_route is None or score > best_route_score:
                    best_route = row

        assert best_route is not None
        summary = {
            "epoch": epoch,
            "h85_val_best_f1": h85_best,
            "miss85_val_best_f1": miss_best,
            "route_val": best_route,
        }
        score = (
            best_route["metrics"]["r2"],
            h85_best["f1"],
            miss_best["f1"],
            -best_route["metrics"]["hard_mae"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_summary = summary
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_state is not None and best_summary is not None
    model.load_state_dict(best_state)
    model.eval()

    def predict_scaled(x_eval_scaled: np.ndarray) -> dict[str, np.ndarray]:
        x_eval_tensor = torch.from_numpy(x_eval_scaled).to(device)
        with torch.no_grad():
            h85_logit, miss_logit, delta_raw = model(x_eval_tensor)
        return {
            "p_h85": torch.sigmoid(h85_logit).squeeze(1).cpu().numpy().astype(np.float32),
            "p_miss85": torch.sigmoid(miss_logit).squeeze(1).cpu().numpy().astype(np.float32),
            "delta_hat": torch.sigmoid(delta_raw).squeeze(1).cpu().numpy().astype(np.float32),
        }

    return predict_scaled(x_val_scaled), predict_scaled(x_test_scaled), best_summary


def apply_routing(base_pred: np.ndarray, pred: dict[str, np.ndarray], w_h85: float, w_miss: float) -> np.ndarray:
    gate = np.clip(w_h85 * pred["p_h85"] + w_miss * pred["p_miss85"], 0.0, 1.0)
    return np.clip(base_pred + gate * pred["delta_hat"], 0.0, 1.0)


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    cache_path = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache/think_end_last10_hidden:mean.npz"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_router_multitask_mlp_last10_lr_only"
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(cache_path, allow_pickle=True) as data:
        x_train = select_last10_left_right_only(data["x_train"])
        x_val = select_last10_left_right_only(data["x_val"])
        x_test = select_last10_left_right_only(data["x_test"])
        _, x_train_p, y_train_p = aggregate_xy(data["train_task_ids"].tolist(), x_train, data["y_train"])
        _, x_val_p, y_val_p = aggregate_xy(data["val_task_ids"].tolist(), x_val, data["y_val"])
        _, x_test_p, y_test_p = aggregate_xy(data["test_task_ids"].tolist(), x_test, data["y_test"])

    base_train_oof = base_oof_predictions(10000.0, x_train_p, y_train_p, seed=42)
    base_model = fit_base(10000.0, x_train_p, y_train_p)
    base_val_pred = np.clip(np.asarray(base_model.predict(x_val_p), dtype=np.float32), 0.0, 1.0)
    base_test_pred = np.clip(np.asarray(base_model.predict(x_test_p), dtype=np.float32), 0.0, 1.0)

    configs = [
        {"hidden_dims": (512, 256, 128), "dropout": 0.1, "seed": 1},
        {"hidden_dims": (512, 256, 128), "dropout": 0.1, "seed": 2},
        {"hidden_dims": (512, 256, 128), "dropout": 0.1, "seed": 3},
        {"hidden_dims": (768, 384, 192), "dropout": 0.15, "seed": 1},
        {"hidden_dims": (768, 384, 192), "dropout": 0.15, "seed": 2},
        {"hidden_dims": (768, 384, 192), "dropout": 0.15, "seed": 3},
    ]

    best: dict[str, Any] | None = None
    best_test_pred: dict[str, np.ndarray] | None = None
    for config in configs:
        print(json.dumps({"stage": "fit", "config": config}), flush=True)
        val_pred, test_pred, val_summary = train_one(
            x_train_p,
            y_train_p,
            base_train_oof,
            x_val_p,
            y_val_p,
            base_val_pred,
            x_test_p,
            hidden_dims=tuple(config["hidden_dims"]),
            dropout=float(config["dropout"]),
            seed=int(config["seed"]),
        )
        row = {
            "config": config,
            "val": {
                "h85": {
                    "threshold_0.5": classification_metrics((y_val_p >= 0.85).astype(np.int32), val_pred["p_h85"], 0.5),
                    "best_f1": best_threshold_by_f1((y_val_p >= 0.85).astype(np.int32), val_pred["p_h85"]),
                },
                "miss85": {
                    "threshold_0.5": classification_metrics(((y_val_p >= 0.85) & (base_val_pred < 0.85)).astype(np.int32), val_pred["p_miss85"], 0.5),
                    "best_f1": best_threshold_by_f1(((y_val_p >= 0.85) & (base_val_pred < 0.85)).astype(np.int32), val_pred["p_miss85"]),
                },
                "route": val_summary["route_val"],
            },
        }
        score = (
            row["val"]["route"]["metrics"]["r2"],
            row["val"]["h85"]["best_f1"]["f1"],
            row["val"]["miss85"]["best_f1"]["f1"],
            -row["val"]["route"]["metrics"]["hard_mae"],
        )
        best_score = None if best is None else (
            best["val"]["route"]["metrics"]["r2"],
            best["val"]["h85"]["best_f1"]["f1"],
            best["val"]["miss85"]["best_f1"]["f1"],
            -best["val"]["route"]["metrics"]["hard_mae"],
        )
        if best is None or score > best_score:
            best = row
            best_test_pred = test_pred
            _write_json(outdir / "progress.json", best)

    assert best is not None and best_test_pred is not None
    test_summary = {
        "h85": {
            "threshold_0.5": classification_metrics((y_test_p >= 0.85).astype(np.int32), best_test_pred["p_h85"], 0.5),
            "threshold_from_val_best_f1": classification_metrics((y_test_p >= 0.85).astype(np.int32), best_test_pred["p_h85"], best["val"]["h85"]["best_f1"]["threshold"]),
        },
        "miss85": {
            "threshold_0.5": classification_metrics(((y_test_p >= 0.85) & (base_test_pred < 0.85)).astype(np.int32), best_test_pred["p_miss85"], 0.5),
            "threshold_from_val_best_f1": classification_metrics(((y_test_p >= 0.85) & (base_test_pred < 0.85)).astype(np.int32), best_test_pred["p_miss85"], best["val"]["miss85"]["best_f1"]["threshold"]),
        },
        "route": {
            "weights": {"w_h85": best["val"]["route"]["w_h85"], "w_miss": best["val"]["route"]["w_miss"]},
            "metrics": regression_metrics(
                y_test_p,
                apply_routing(
                    base_test_pred,
                    best_test_pred,
                    best["val"]["route"]["w_h85"],
                    best["val"]["route"]["w_miss"],
                ),
            ),
        },
        "base": regression_metrics(y_test_p, base_test_pred),
    }

    summary = {
        "dataset_key": "think_end_last10_hidden:mean",
        "feature_mode": "left_right_only",
        "feature_dim": int(x_train_p.shape[1]),
        "setting": "prompt_level_router_multitask_mlp",
        "best_config": best["config"],
        "val": best["val"],
        "test": test_summary,
    }
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
