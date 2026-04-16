from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from classifer_training.search_two_rollout_router_multitask_mlp_last10_lr_only_pairlevel_trainplusval import (
    INNER_VAL_PROMPT_FRACTION,
    apply_routing,
    classification_metrics,
    prompt_mean_metrics,
    regression_metrics,
    select_last10_left_right_only,
    split_trainplusval_by_prompt,
    train_one,
)
from classifer_training.search_two_rollout_think_tail_balanced import (
    _clip,
    _compose_mid,
    _compose_with_easy,
    _fit_detector_from_key,
    _fit_logistic,
    _fit_ridge,
    _fit_specialist_from_key,
    _load_dataset_cache,
    _write_json,
    aggregate_prompt,
)


def _fit_tail_balanced(best: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    base_key = str(best["base_decomp"]["base_key"])
    base_alpha = 3000.0 if "a3000" in base_key else 10000.0
    p10_c = float(str(best["easy_detector"]).split("_c")[-1])
    return {
        "base": _fit_ridge(base_alpha, x_train, y_train),
        "hard": _fit_specialist_from_key(best["base_decomp"]["hard_key"], x_train, y_train),
        "vhard": _fit_specialist_from_key(best["base_decomp"]["vhard_key"], x_train, y_train),
        "p80": _fit_detector_from_key(best["base_decomp"]["detectors"]["p80"], x_train, y_train),
        "p90": _fit_detector_from_key(best["base_decomp"]["detectors"]["p90"], x_train, y_train),
        "p100": _fit_detector_from_key(best["base_decomp"]["detectors"]["p100"], x_train, y_train),
        "p10": _fit_logistic(p10_c, x_train, (y_train <= 0.1).astype(np.int32)),
    }


def _predict_tail_balanced(best: dict[str, Any], models: dict[str, Any], x_eval: np.ndarray) -> np.ndarray:
    base = _clip(models["base"].predict(x_eval))
    hard = _clip(models["hard"].predict(x_eval))
    vhard = _clip(models["vhard"].predict(x_eval))
    p80 = _clip(models["p80"].predict_proba(x_eval)[:, 1])
    p90 = _clip(models["p90"].predict_proba(x_eval)[:, 1])
    p100 = _clip(models["p100"].predict_proba(x_eval)[:, 1])
    mid = _compose_mid(base, hard, vhard, p80, p90, p100, dict(best["hard_routing"]))
    p10 = _clip(models["p10"].predict_proba(x_eval)[:, 1])
    return _compose_with_easy(mid, p10, **best["easy_routing"])


def _prompt_group_oof_tail_balanced(
    best: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    task_ids: list[str],
    seed: int = 42,
    n_folds: int = 5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_task_ids = np.asarray(sorted(set(map(str, task_ids))))
    rng.shuffle(unique_task_ids)
    folds = np.array_split(unique_task_ids, n_folds)
    oof = np.zeros(len(y_train), dtype=np.float32)
    for fold_prompts in folds:
        val_prompt_set = set(fold_prompts.tolist())
        val_mask = np.asarray([str(tid) in val_prompt_set for tid in task_ids], dtype=bool)
        train_mask = ~val_mask
        models = _fit_tail_balanced(best, x_train[train_mask], y_train[train_mask])
        oof[val_mask] = _predict_tail_balanced(best, models, x_train[val_mask])
    return oof


def _validate_alignment(main: dict[str, Any], router: dict[str, Any]) -> None:
    for split in ("train", "val", "test"):
        if list(main[f"{split}_task_ids"]) != list(router[f"{split}_task_ids"]):
            raise ValueError(f"task_id order mismatch on split={split}")
        if not np.allclose(main[f"y_{split}"], router[f"y_{split}"]):
            raise ValueError(f"label mismatch on split={split}")


def main() -> None:
    repo = Path("/home/jongwonlim/verl/yoonho/verl")
    tail_summary_path = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_tail_balanced_search/summary.json"
    main_cache_path = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache/think_end_hidden:mean.npz"
    router_cache_path = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_fair_compare/dataset_cache/think_end_last10_hidden:mean.npz"
    outdir = repo / "classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_rollout_think_tail_balanced_plus_last10_router"
    outdir.mkdir(parents=True, exist_ok=True)

    tail_summary = json.loads(tail_summary_path.read_text(encoding="utf-8"))
    best_tail = dict(tail_summary["best"])
    main_data = _load_dataset_cache(main_cache_path)
    router_data = _load_dataset_cache(router_cache_path)
    _validate_alignment(main_data, router_data)

    x_router_train = select_last10_left_right_only(np.asarray(router_data["x_train"], dtype=np.float32))
    y_train = np.asarray(router_data["y_train"], dtype=np.float32)
    train_task_ids = list(router_data["train_task_ids"])
    x_router_val = select_last10_left_right_only(np.asarray(router_data["x_val"], dtype=np.float32))
    y_val = np.asarray(router_data["y_val"], dtype=np.float32)
    val_task_ids = list(router_data["val_task_ids"])
    x_router_test = select_last10_left_right_only(np.asarray(router_data["x_test"], dtype=np.float32))
    y_test = np.asarray(router_data["y_test"], dtype=np.float32)
    test_task_ids = list(router_data["test_task_ids"])

    x_main_train = np.asarray(main_data["x_train"], dtype=np.float32)
    x_main_val = np.asarray(main_data["x_val"], dtype=np.float32)
    x_main_test = np.asarray(main_data["x_test"], dtype=np.float32)

    x_router_trainval = np.concatenate([x_router_train, x_router_val], axis=0)
    x_main_trainval = np.concatenate([x_main_train, x_main_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    task_ids_trainval = train_task_ids + val_task_ids

    train_mask, inner_val_mask = split_trainplusval_by_prompt(task_ids_trainval, INNER_VAL_PROMPT_FRACTION, seed=42)
    x_router_inner_train = x_router_trainval[train_mask]
    x_router_inner_val = x_router_trainval[inner_val_mask]
    x_main_inner_train = x_main_trainval[train_mask]
    x_main_inner_val = x_main_trainval[inner_val_mask]
    y_inner_train = y_trainval[train_mask]
    y_inner_val = y_trainval[inner_val_mask]
    task_ids_inner_train = [tid for tid, keep in zip(task_ids_trainval, train_mask.tolist()) if keep]
    task_ids_inner_val = [tid for tid, keep in zip(task_ids_trainval, inner_val_mask.tolist()) if keep]

    base_inner_train_oof = _prompt_group_oof_tail_balanced(best_tail, x_main_inner_train, y_inner_train, task_ids_inner_train)
    inner_models = _fit_tail_balanced(best_tail, x_main_inner_train, y_inner_train)
    base_inner_val_pred = _predict_tail_balanced(best_tail, inner_models, x_main_inner_val)

    configs = [
        {"hidden_dims": (256, 128, 64), "dropout": 0.3, "seed": 1, "weight_decay": 1e-3},
        {"hidden_dims": (256, 128, 64), "dropout": 0.3, "seed": 2, "weight_decay": 1e-3},
        {"hidden_dims": (256, 128, 64), "dropout": 0.3, "seed": 3, "weight_decay": 1e-3},
        {"hidden_dims": (384, 192, 96), "dropout": 0.2, "seed": 1, "weight_decay": 1e-3},
        {"hidden_dims": (384, 192, 96), "dropout": 0.2, "seed": 2, "weight_decay": 1e-3},
        {"hidden_dims": (384, 192, 96), "dropout": 0.2, "seed": 3, "weight_decay": 1e-3},
    ]

    best: dict[str, Any] | None = None
    for i, config in enumerate(configs, start=1):
        print(json.dumps({"stage": "fit", "i": i, "n": len(configs), "config": config}), flush=True)
        val_pred, _unused_test_pred, val_summary = train_one(
            x_router_inner_train,
            y_inner_train,
            base_inner_train_oof,
            x_router_inner_val,
            y_inner_val,
            base_inner_val_pred,
            x_router_test,
            hidden_dims=tuple(config["hidden_dims"]),
            dropout=float(config["dropout"]),
            seed=int(config["seed"]),
            epochs=50,
            batch_size=128,
            lr=1e-3,
            weight_decay=float(config["weight_decay"]),
            patience=5,
        )
        routed_val = apply_routing(base_inner_val_pred, val_pred, val_summary["route_val"]["w_h85"], val_summary["route_val"]["w_miss"])
        row = {
            "config": config,
            "val": {
                "h85": {"threshold_0.5": classification_metrics((y_inner_val >= 0.85).astype(np.int32), val_pred["p_h85"], 0.5)},
                "miss85": {"threshold_0.5": classification_metrics(((y_inner_val >= 0.85) & (base_inner_val_pred < 0.85)).astype(np.int32), val_pred["p_miss85"], 0.5)},
                "route_row": val_summary["route_val"],
                "route_prompt": prompt_mean_metrics(task_ids_inner_val, y_inner_val, routed_val),
                "base_prompt": prompt_mean_metrics(task_ids_inner_val, y_inner_val, base_inner_val_pred),
            },
        }
        score = (
            row["val"]["route_prompt"]["r2"],
            -row["val"]["route_prompt"]["hard_mae"],
            -row["val"]["route_prompt"]["very_hard_mae"],
        )
        best_score = None if best is None else (
            best["val"]["route_prompt"]["r2"],
            -best["val"]["route_prompt"]["hard_mae"],
            -best["val"]["route_prompt"]["very_hard_mae"],
        )
        if best is None or score > best_score:
            best = row
            _write_json(outdir / "progress.json", best)

    assert best is not None

    base_trainval_oof = _prompt_group_oof_tail_balanced(best_tail, x_main_trainval, y_trainval, task_ids_trainval)
    trainval_models = _fit_tail_balanced(best_tail, x_main_trainval, y_trainval)
    base_inner_val_pred_refit = _predict_tail_balanced(best_tail, trainval_models, x_main_inner_val)
    base_test_pred = _predict_tail_balanced(best_tail, trainval_models, x_main_test)

    _unused_val_pred, test_pred, refit_summary = train_one(
        x_router_trainval,
        y_trainval,
        base_trainval_oof,
        x_router_inner_val,
        y_inner_val,
        base_inner_val_pred_refit,
        x_router_test,
        hidden_dims=tuple(best["config"]["hidden_dims"]),
        dropout=float(best["config"]["dropout"]),
        seed=int(best["config"]["seed"]),
        epochs=50,
        batch_size=128,
        lr=1e-3,
        weight_decay=float(best["config"]["weight_decay"]),
        patience=5,
    )
    routed_test = apply_routing(base_test_pred, test_pred, best["val"]["route_row"]["w_h85"], best["val"]["route_row"]["w_miss"])

    summary = {
        "base_model": "tail_balanced",
        "router_model": "last10_left_right_only_multitask_mlp_regularized",
        "feature_dim": int(x_router_train.shape[1]),
        "pair_rows": {
            "train_plus_val": int(len(y_trainval)),
            "inner_train": int(len(y_inner_train)),
            "inner_val": int(len(y_inner_val)),
            "test": int(len(y_test)),
        },
        "prompt_counts": {
            "train_plus_val": int(len(set(task_ids_trainval))),
            "inner_train": int(len(set(task_ids_inner_train))),
            "inner_val": int(len(set(task_ids_inner_val))),
            "test": int(len(set(test_task_ids))),
        },
        "best_config": best["config"],
        "selection_val": best["val"],
        "refit_val_route_row": refit_summary["route_val"],
        "test": {
            "tail_balanced_row": regression_metrics(y_test, base_test_pred),
            "tail_balanced_prompt": prompt_mean_metrics(test_task_ids, y_test, base_test_pred),
            "routed_row": regression_metrics(y_test, routed_test),
            "routed_prompt": prompt_mean_metrics(test_task_ids, y_test, routed_test),
            "weights": {"w_h85": best["val"]["route_row"]["w_h85"], "w_miss": best["val"]["route_row"]["w_miss"]},
        },
    }
    _write_json(outdir / "summary.json", summary)

    ordered, yt, yp = aggregate_prompt(test_task_ids, y_test, routed_test)
    with (outdir / "router_predictions_test.jsonl").open("w", encoding="utf-8") as handle:
        for task_id, yv, pv in zip(ordered, yt.tolist(), yp.tolist()):
            handle.write(json.dumps({"task_id": task_id, "y_true_difficulty": yv, "predicted_difficulty": pv}) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
