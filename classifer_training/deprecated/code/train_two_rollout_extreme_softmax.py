from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from classifer_training.search_two_rollout_finished16_focus import (
    _build_label_buckets,
    _load_grouped_rows,
)
from classifer_training.train_prompt_two_trajectory_promptsearch import (
    build_matrix,
    build_pair_rows,
    build_prompt_lookup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train balanced two-class softmax on easy vs hard 2-rollout samples.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--rollout_index_path", type=Path, required=True)
    parser.add_argument("--prompt_hidden_dir", type=Path, required=True)
    parser.add_argument("--prompt_index_dir", type=Path, required=True)
    parser.add_argument("--pooled_hidden_dir", type=Path, required=True)
    parser.add_argument("--pooled_index_dir", type=Path, required=True)
    parser.add_argument("--prompt_mode", type=str, default="l10_l26")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--low_threshold", type=float, default=0.1)
    parser.add_argument("--high_threshold", type=float, default=0.9)
    parser.add_argument("--train_splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--train_pairs_per_prompt", type=int, default=4)
    parser.add_argument("--test_pairs_per_prompt", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--global_test_fraction", type=float, default=0.0)
    parser.add_argument("--include_mid_in_test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cls_metrics(y_true: np.ndarray, prob_hard: np.ndarray, pred_hard: np.ndarray) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, pred_hard)),
        "f1": float(f1_score(y_true, pred_hard)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, prob_hard))
    else:
        out["auc"] = float("nan")
    return out


def _aggregate_prompt(task_ids: list[str], y_true: np.ndarray, prob_hard: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "prob": []})
    for task_id, target, prob in zip(task_ids, y_true.tolist(), prob_hard.tolist()):
        groups[task_id]["y_true"].append(float(target))
        groups[task_id]["prob"].append(float(prob))
    ordered = sorted(groups)
    y = np.asarray([int(round(np.mean(groups[k]["y_true"]))) for k in ordered], dtype=np.int64)
    p = np.asarray([float(np.mean(groups[k]["prob"])) for k in ordered], dtype=np.float32)
    return y, p


class SoftmaxMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.random_seed)
    train_idx, val_idx = next(splitter.split(X_train_scaled, y_train, groups=groups_train))
    X_fit, y_fit = X_train_scaled[train_idx], y_train[train_idx]
    X_val, y_val = X_train_scaled[val_idx], y_train[val_idx]

    fit_counts = np.bincount(y_fit, minlength=2).astype(np.float32)
    class_weights = fit_counts.sum() / np.maximum(fit_counts, 1.0)
    class_weights = class_weights / class_weights.mean()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoftmaxMLP(X_train_scaled.shape[1], args.hidden_dim, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit.astype(np.int64))),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(X_val).to(device)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, torch.from_numpy(y_val.astype(np.int64)).to(device)).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            best_epoch = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test_scaled).to(device))
        prob_hard = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy().astype(np.float32)

    return prob_hard, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "class_weights": class_weights.tolist(),
    }


def _write_prompt_predictions(path: Path, task_ids: list[str], y_true: np.ndarray, prob_hard: np.ndarray) -> None:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "prob": []})
    for task_id, target, prob in zip(task_ids, y_true.tolist(), prob_hard.tolist()):
        groups[task_id]["y_true"].append(float(target))
        groups[task_id]["prob"].append(float(prob))
    with path.open("w", encoding="utf-8") as f:
        for task_id in sorted(groups):
            prob = float(np.mean(groups[task_id]["prob"]))
            y = int(round(np.mean(groups[task_id]["y_true"])))
            f.write(json.dumps({"task_id": task_id, "y_true": y, "prob_hard": prob, "pred_hard": int(prob >= 0.5)}) + "\n")


def _write_prompt_predictions_continuous(
    path: Path,
    task_ids: list[str],
    y_true_diff: np.ndarray,
    y_true_cls: np.ndarray,
    prob_hard: np.ndarray,
) -> None:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"diff": [], "prob": [], "cls": []})
    for task_id, diff, cls, prob in zip(task_ids, y_true_diff.tolist(), y_true_cls.tolist(), prob_hard.tolist()):
        groups[task_id]["diff"].append(float(diff))
        groups[task_id]["prob"].append(float(prob))
        if int(cls) >= 0:
            groups[task_id]["cls"].append(float(cls))
    with path.open("w", encoding="utf-8") as f:
        for task_id in sorted(groups):
            cls_vals = groups[task_id]["cls"]
            row = {
                "task_id": task_id,
                "y_true_difficulty": float(np.mean(groups[task_id]["diff"])),
                "prob_hard": float(np.mean(groups[task_id]["prob"])),
                "pred_hard": int(float(np.mean(groups[task_id]["prob"])) >= 0.5),
                "y_true_cls": int(round(float(np.mean(cls_vals)))) if cls_vals else None,
            }
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.random_seed)

    label_buckets = _build_label_buckets(args.run_root.expanduser().resolve())
    grouped_rows, feature_keys = _load_grouped_rows(args.rollout_index_path.expanduser().resolve(), label_buckets)
    prompt_lookup = build_prompt_lookup(
        args.prompt_hidden_dir.expanduser().resolve(),
        args.prompt_index_dir.expanduser().resolve(),
        args.pooled_hidden_dir.expanduser().resolve(),
        args.pooled_index_dir.expanduser().resolve(),
    )

    if args.global_test_fraction > 0.0 and args.include_mid_in_test:
        def _bucket(diff: float) -> int:
            if diff <= args.low_threshold:
                return 0
            if diff >= args.high_threshold:
                return 2
            return 1

        rng = np.random.default_rng(args.random_seed)
        test_task_ids_set: set[str] = set()
        by_bucket: dict[int, list[str]] = defaultdict(list)
        for row in grouped_rows:
            by_bucket[_bucket(float(row["y_true"]))].append(str(row["task_id"]))
        for bucket_id, task_ids in by_bucket.items():
            arr = np.asarray(sorted(task_ids))
            if len(arr) == 0:
                continue
            k = max(1, int(round(len(arr) * args.global_test_fraction)))
            chosen = rng.choice(arr, size=min(k, len(arr)), replace=False)
            test_task_ids_set.update([str(x) for x in chosen.tolist()])

        updated_rows = []
        for row in grouped_rows:
            new_row = dict(row)
            new_row["rollouts"] = row["rollouts"]
            new_row["split"] = "test" if str(row["task_id"]) in test_task_ids_set else "train"
            updated_rows.append(new_row)
        pair_rows = build_pair_rows(
            grouped_rows=updated_rows,
            feature_keys=feature_keys,
            train_splits={"train"},
            test_splits={"test"},
            train_pairs_per_prompt=args.train_pairs_per_prompt,
            test_pairs_per_prompt=args.test_pairs_per_prompt,
            random_seed=args.random_seed,
        )
    else:
        pair_rows = build_pair_rows(
            grouped_rows=grouped_rows,
            feature_keys=feature_keys,
            train_splits=set(args.train_splits),
            test_splits=set(args.test_splits),
            train_pairs_per_prompt=args.train_pairs_per_prompt,
            test_pairs_per_prompt=args.test_pairs_per_prompt,
            random_seed=args.random_seed,
        )
    X, y_diff, splits, metadata_rows = build_matrix(pair_rows, prompt_lookup, args.prompt_mode)

    y_cls = np.full_like(y_diff, fill_value=-1, dtype=np.int64)
    y_cls[y_diff <= args.low_threshold] = 0
    y_cls[y_diff >= args.high_threshold] = 1

    if args.global_test_fraction > 0.0 and not args.include_mid_in_test:
        keep_mask = y_cls >= 0
        X = X[keep_mask]
        y_diff = y_diff[keep_mask]
        y_cls = y_cls[keep_mask]
        splits = splits[keep_mask]
        metadata_rows = [metadata_rows[idx] for idx, keep in enumerate(keep_mask.tolist()) if keep]

        all_task_ids = np.asarray([str(meta["task_id"]) for meta in metadata_rows])
        unique_task_ids = np.asarray(sorted(set(all_task_ids.tolist())))
        task_targets = []
        for task_id in unique_task_ids.tolist():
            idxs = np.where(all_task_ids == task_id)[0]
            task_targets.append(int(round(float(np.mean(y_cls[idxs])))))
        task_targets = np.asarray(task_targets, dtype=np.int64)
        rng = np.random.default_rng(args.random_seed)
        test_task_ids_list: list[str] = []
        for cls in [0, 1]:
            cls_groups = unique_task_ids[task_targets == cls]
            if len(cls_groups) == 0:
                continue
            k = max(1, int(round(len(cls_groups) * args.global_test_fraction)))
            chosen = rng.choice(cls_groups, size=min(k, len(cls_groups)), replace=False)
            test_task_ids_list.extend([str(x) for x in chosen.tolist()])
        test_task_ids_set = set(test_task_ids_list)
        test_mask = np.asarray([str(meta["task_id"]) in test_task_ids_set for meta in metadata_rows], dtype=bool)
        train_mask = ~test_mask
    else:
        train_mask = np.isin(splits, np.asarray(args.train_splits))
        test_mask = np.isin(splits, np.asarray(args.test_splits))

    X_train = X[train_mask]
    y_train = y_cls[train_mask]
    X_test = X[test_mask]
    y_test = y_cls[test_mask]
    y_diff_test = y_diff[test_mask]
    train_groups = np.asarray([metadata_rows[idx]["task_id"] for idx, keep in enumerate(train_mask.tolist()) if keep])
    test_task_ids = [metadata_rows[idx]["task_id"] for idx, keep in enumerate(test_mask.tolist()) if keep]

    if args.include_mid_in_test:
        train_keep = y_train >= 0
        X_train = X_train[train_keep]
        y_train = y_train[train_keep]
        train_groups = train_groups[train_keep]

    if args.train_fraction < 1.0:
        unique_train_groups = np.asarray(sorted(set(train_groups.tolist())))
        group_targets = []
        group_to_idx: dict[str, int] = {}
        for i, task_id in enumerate(unique_train_groups.tolist()):
            group_to_idx[task_id] = i
        for task_id in unique_train_groups.tolist():
            idxs = np.where(train_groups == task_id)[0]
            group_targets.append(int(round(float(np.mean(y_train[idxs])))))
        group_targets = np.asarray(group_targets, dtype=np.int64)

        chosen_groups: list[str] = []
        rng = np.random.default_rng(args.random_seed)
        for cls in [0, 1]:
            cls_groups = unique_train_groups[group_targets == cls]
            if len(cls_groups) == 0:
                continue
            k = max(1, int(round(len(cls_groups) * args.train_fraction)))
            chosen = rng.choice(cls_groups, size=min(k, len(cls_groups)), replace=False)
            chosen_groups.extend([str(x) for x in chosen.tolist()])
        chosen_group_set = set(chosen_groups)
        keep_idx = np.asarray([i for i, task_id in enumerate(train_groups.tolist()) if task_id in chosen_group_set], dtype=np.int64)
        move_idx = np.asarray([i for i, task_id in enumerate(train_groups.tolist()) if task_id not in chosen_group_set], dtype=np.int64)

        moved_task_ids = train_groups[move_idx].tolist()
        X_test = np.concatenate([X_test, X_train[move_idx]], axis=0)
        y_test = np.concatenate([y_test, y_train[move_idx]], axis=0)
        test_task_ids = test_task_ids + moved_task_ids
        X_train = X_train[keep_idx]
        y_train = y_train[keep_idx]
        train_groups = train_groups[keep_idx]

    prompt_train_counts = np.bincount(y_train, minlength=2).tolist()
    prompt_test_counts = np.bincount(y_test[y_test >= 0], minlength=2).tolist() if np.any(y_test >= 0) else [0, 0]

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=args.random_seed)),
        ]
    )
    logistic.fit(X_train, y_train)
    prob_test_log = logistic.predict_proba(X_test)[:, 1].astype(np.float32)
    pred_test_log = (prob_test_log >= 0.5).astype(np.int64)

    mlp_prob_test, mlp_train_info = _fit_mlp(X_train, y_train, train_groups, X_test, args)
    mlp_pred_test = (mlp_prob_test >= 0.5).astype(np.int64)

    if args.include_mid_in_test:
        extreme_test_mask = y_test >= 0
        prompt_true_log, prompt_prob_log = _aggregate_prompt(
            [task_id for task_id, keep in zip(test_task_ids, extreme_test_mask.tolist()) if keep],
            y_test[extreme_test_mask],
            prob_test_log[extreme_test_mask],
        )
        prompt_true_mlp, prompt_prob_mlp = _aggregate_prompt(
            [task_id for task_id, keep in zip(test_task_ids, extreme_test_mask.tolist()) if keep],
            y_test[extreme_test_mask],
            mlp_prob_test[extreme_test_mask],
        )
        all_prompt_diff_log, all_prompt_prob_log = _aggregate_prompt(test_task_ids, y_diff_test, prob_test_log)
        all_prompt_diff_mlp, all_prompt_prob_mlp = _aggregate_prompt(test_task_ids, y_diff_test, mlp_prob_test)
        continuous = {
            "logistic": {
                "corr": float(np.corrcoef(all_prompt_diff_log, all_prompt_prob_log)[0, 1]),
                "mae": float(np.mean(np.abs(all_prompt_diff_log - all_prompt_prob_log))),
            },
            "mlp": {
                "corr": float(np.corrcoef(all_prompt_diff_mlp, all_prompt_prob_mlp)[0, 1]),
                "mae": float(np.mean(np.abs(all_prompt_diff_mlp - all_prompt_prob_mlp))),
            },
        }
    else:
        prompt_true_log, prompt_prob_log = _aggregate_prompt(test_task_ids, y_test, prob_test_log)
        prompt_true_mlp, prompt_prob_mlp = _aggregate_prompt(test_task_ids, y_test, mlp_prob_test)
        continuous = None

    summary = {
        "setting": "two_rollout_extreme_softmax",
        "prompt_mode": args.prompt_mode,
        "difficulty_thresholds": {"easy_leq": args.low_threshold, "hard_geq": args.high_threshold},
        "train_pairs_per_prompt": args.train_pairs_per_prompt,
        "test_pairs_per_prompt": args.test_pairs_per_prompt,
        "train_fraction": args.train_fraction,
        "global_test_fraction": args.global_test_fraction,
        "pair_feature_dim": int(X_train.shape[1]) if len(X_train) else 0,
        "row_counts": {
            "train_easy": int(prompt_train_counts[0]),
            "train_hard": int(prompt_train_counts[1]),
            "test_easy": int(prompt_test_counts[0]),
            "test_hard": int(prompt_test_counts[1]),
        },
        "logistic": {
            "row_metrics": _cls_metrics(y_test[y_test >= 0], prob_test_log[y_test >= 0], pred_test_log[y_test >= 0]) if np.any(y_test >= 0) else {},
            "prompt_metrics": _cls_metrics(prompt_true_log, prompt_prob_log, (prompt_prob_log >= 0.5).astype(np.int64)),
        },
        "mlp": {
            "row_metrics": _cls_metrics(y_test[y_test >= 0], mlp_prob_test[y_test >= 0], mlp_pred_test[y_test >= 0]) if np.any(y_test >= 0) else {},
            "prompt_metrics": _cls_metrics(prompt_true_mlp, prompt_prob_mlp, (prompt_prob_mlp >= 0.5).astype(np.int64)),
            "train_info": mlp_train_info,
        },
    }
    if continuous is not None:
        summary["continuous_test_metrics"] = continuous
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.include_mid_in_test:
        _write_prompt_predictions_continuous(args.output_dir / "logistic_predictions_test.jsonl", test_task_ids, y_diff_test, y_test, prob_test_log)
        _write_prompt_predictions_continuous(args.output_dir / "mlp_predictions_test.jsonl", test_task_ids, y_diff_test, y_test, mlp_prob_test)
    else:
        _write_prompt_predictions(args.output_dir / "logistic_predictions_test.jsonl", test_task_ids, y_test, prob_test_log)
        _write_prompt_predictions(args.output_dir / "mlp_predictions_test.jsonl", test_task_ids, y_test, mlp_prob_test)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
