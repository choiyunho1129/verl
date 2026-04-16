from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from classifer_training.prompt_only_experiments import _build_dataset
from classifer_training.utils import load_records


BASE = Path("/home/jongwonlim/verl/yoonho/verl/classifer_training")
LABELS_PATH = BASE / "artifacts/labels/dapo_math_17k/qwen3_4b_instruct_2507/sampling_labels_12seeds.jsonl"
HIDDEN_LAST = BASE / "artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507"
INDEX_LAST = BASE / "artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507"
HIDDEN_LAST10 = BASE / "artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507_last10mean"
INDEX_LAST10 = BASE / "artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507_last10mean"
OUTPUT = BASE / "artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_prompt_only_rich_search/summary.json"

PROMPT_DIM = 23


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def _load() -> dict[str, Any]:
    labels = {str(row["task_id"]): row for row in load_records(LABELS_PATH)}
    ds_last = _build_dataset(HIDDEN_LAST, INDEX_LAST, labels)
    ds_last10 = _build_dataset(HIDDEN_LAST10, INDEX_LAST10, labels)
    if not np.array_equal(ds_last["splits"], ds_last10["splits"]):
        raise ValueError("Split mismatch between hidden sources.")
    if not np.allclose(ds_last["y_reg"], ds_last10["y_reg"]):
        raise ValueError("Label mismatch between hidden sources.")
    return {"last": ds_last, "last10": ds_last10}


def _feature_blocks(datasets: dict[str, Any]) -> dict[str, np.ndarray]:
    last = datasets["last"]
    last10 = datasets["last10"]

    prompt_feats = last["scalar"][:, :PROMPT_DIM]
    rel_last = last["scalar"][:, PROMPT_DIM:]
    rel_last10 = last10["scalar"][:, PROMPT_DIM:]

    blocks = {
        "prompt_feats": prompt_feats.astype(np.float32),
        "rel_both": np.concatenate([rel_last, rel_last10], axis=1).astype(np.float32),
        "last_l17": last["hidden_modes"]["layer17"].astype(np.float32),
        "last_l24": last["hidden_modes"]["layer24"].astype(np.float32),
        "last_l35": last["hidden_modes"]["layer35"].astype(np.float32),
        "last_mean": last["hidden_modes"]["layers0_35_mean"].astype(np.float32),
        "l10_l17": last10["hidden_modes"]["layer17"].astype(np.float32),
        "l10_l24": last10["hidden_modes"]["layer24"].astype(np.float32),
        "l10_l35": last10["hidden_modes"]["layer35"].astype(np.float32),
        "l10_mean": last10["hidden_modes"]["layers0_35_mean"].astype(np.float32),
    }
    blocks["mid_pair"] = np.concatenate(
        [blocks["last_l17"], blocks["last_l24"], blocks["l10_l17"], blocks["l10_l24"]],
        axis=1,
    ).astype(np.float32)
    blocks["mid_final_pair"] = np.concatenate(
        [
            blocks["last_l17"],
            blocks["last_l24"],
            blocks["last_l35"],
            blocks["l10_l17"],
            blocks["l10_l24"],
            blocks["l10_l35"],
        ],
        axis=1,
    ).astype(np.float32)
    blocks["means_pair"] = np.concatenate([blocks["last_mean"], blocks["l10_mean"]], axis=1).astype(np.float32)
    blocks["all8"] = np.concatenate(
        [
            blocks["last_l17"],
            blocks["last_l24"],
            blocks["last_l35"],
            blocks["last_mean"],
            blocks["l10_l17"],
            blocks["l10_l24"],
            blocks["l10_l35"],
            blocks["l10_mean"],
        ],
        axis=1,
    ).astype(np.float32)
    return blocks


def _compose(blocks: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.concatenate([blocks[name] for name in names], axis=1).astype(np.float32)


def _fit_predict_train_test(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    if model_name == "ridge_a3000":
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=3000.0))])
        model.fit(X_train, y_train)
        return np.clip(model.predict(X_test).astype(np.float32), 0.0, 1.0)
    if model_name == "ridge_a10000":
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10000.0))])
        model.fit(X_train, y_train)
        return np.clip(model.predict(X_test).astype(np.float32), 0.0, 1.0)
    if model_name == "pls16":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = PLSRegression(n_components=16)
        model.fit(X_train_s, y_train)
        return np.clip(model.predict(X_test_s).reshape(-1).astype(np.float32), 0.0, 1.0)
    if model_name == "pls32":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = PLSRegression(n_components=32)
        model.fit(X_train_s, y_train)
        return np.clip(model.predict(X_test_s).reshape(-1).astype(np.float32), 0.0, 1.0)
    if model_name == "et":
        model = ExtraTreesRegressor(
            n_estimators=1200,
            min_samples_leaf=3,
            max_features=0.35,
            random_state=42,
            n_jobs=12,
        )
        model.fit(X_train, y_train)
        return np.clip(model.predict(X_test).astype(np.float32), 0.0, 1.0)
    if model_name == "cat":
        model = CatBoostRegressor(
            iterations=800,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
        )
        model.fit(X_train, y_train)
        return np.clip(model.predict(X_test).astype(np.float32), 0.0, 1.0)
    if model_name == "xgb":
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=12,
        )
        model.fit(X_train, y_train)
        return np.clip(model.predict(X_test).astype(np.float32), 0.0, 1.0)
    raise ValueError(model_name)


def _stack(train_preds: np.ndarray, y_train: np.ndarray, test_preds: np.ndarray) -> np.ndarray:
    meta = Ridge(alpha=1.0, positive=True)
    meta.fit(train_preds, y_train)
    pred = meta.predict(test_preds)
    return np.clip(pred.astype(np.float32), 0.0, 1.0), meta.coef_.astype(float).tolist(), float(meta.intercept_)


def main() -> None:
    datasets = _load()
    blocks = _feature_blocks(datasets)
    y = datasets["last"]["y_reg"].astype(np.float32)
    splits = datasets["last"]["splits"]

    train_mask = np.isin(splits, ["train", "validation"])
    test_mask = splits == "test"
    y_train = y[train_mask]
    y_test = y[test_mask]

    specs = {
        "ridge_last10_l17": (["l10_l17", "prompt_feats", "rel_both"], "ridge_a3000", None),
        "ridge_last10_l17_l24": (["l10_l17", "l10_l24", "prompt_feats", "rel_both"], "ridge_a3000", None),
        "ridge_mid_pair": (["mid_pair", "prompt_feats", "rel_both"], "ridge_a10000", None),
        "ridge_means_pair": (["means_pair", "prompt_feats", "rel_both"], "ridge_a3000", None),
        "pca64_cat_midfinal": (["mid_final_pair", "prompt_feats", "rel_both"], "cat", 64),
        "pca64_xgb_midfinal": (["mid_final_pair", "prompt_feats", "rel_both"], "xgb", 64),
        "pca64_et_all8": (["all8", "prompt_feats", "rel_both"], "et", 64),
        "pca64_pls32_all8": (["all8", "prompt_feats", "rel_both"], "pls32", 64),
    }

    results: list[dict[str, Any]] = []
    oof_preds: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    model_names: list[str] = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, (block_names, model_name, pca_dim) in specs.items():
        X_full = _compose(blocks, block_names)
        X_train = X_full[train_mask]
        X_test = X_full[test_mask]

        if pca_dim is not None:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            pca = PCA(n_components=min(pca_dim, X_train_s.shape[0], X_train_s.shape[1]), random_state=42)
            X_train_low = pca.fit_transform(X_train_s)
            X_test_low = pca.transform(X_test_s)
            # keep prompt scalars and relation features uncompressed at tail if explicitly present
            if "prompt_feats" in block_names or "rel_both" in block_names:
                tail_parts = []
                if "prompt_feats" in block_names:
                    tail_parts.append(blocks["prompt_feats"][train_mask])
                if "rel_both" in block_names:
                    tail_parts.append(blocks["rel_both"][train_mask])
                X_train_model = np.concatenate([X_train_low] + tail_parts, axis=1).astype(np.float32)

                tail_parts_test = []
                if "prompt_feats" in block_names:
                    tail_parts_test.append(blocks["prompt_feats"][test_mask])
                if "rel_both" in block_names:
                    tail_parts_test.append(blocks["rel_both"][test_mask])
                X_test_model = np.concatenate([X_test_low] + tail_parts_test, axis=1).astype(np.float32)
            else:
                X_train_model = X_train_low.astype(np.float32)
                X_test_model = X_test_low.astype(np.float32)
        else:
            X_train_model = X_train.astype(np.float32)
            X_test_model = X_test.astype(np.float32)

        pred_test = _fit_predict_train_test(model_name, X_train_model, y_train, X_test_model)
        metrics = _metrics(y_test, pred_test)
        results.append(
            {
                "name": name,
                "model": model_name,
                "blocks": block_names,
                "pca_dim": pca_dim,
                **metrics,
            }
        )

        # OOF for stacking
        oof = np.zeros_like(y_train, dtype=np.float32)
        for fold_train, fold_val in kf.split(X_train_model):
            fold_pred = _fit_predict_train_test(model_name, X_train_model[fold_train], y_train[fold_train], X_train_model[fold_val])
            oof[fold_val] = fold_pred
        oof_preds.append(oof)
        test_preds.append(pred_test.astype(np.float32))
        model_names.append(name)

    train_stack = np.stack(oof_preds, axis=1)
    test_stack = np.stack(test_preds, axis=1)
    stack_pred, coefs, intercept = _stack(train_stack, y_train, test_stack)
    stack_metrics = _metrics(y_test, stack_pred)

    payload = {
        "results": sorted(results, key=lambda row: row["r2"], reverse=True),
        "stack": {
            **stack_metrics,
            "members": model_names,
            "coefficients": coefs,
            "intercept": intercept,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
