from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from classifer_training.search_two_rollout_finished16_focus import (
    _build_label_buckets,
    _load_grouped_rows,
    _parse_prompt_config,
)
from classifer_training.train_prompt_two_trajectory_promptsearch import (
    _prompt_vector,
    build_pair_rows,
    build_prompt_lookup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-engineered search for the finished16 2-rollout probe.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--rollout_index_path", type=Path, required=True)
    parser.add_argument("--prompt_hidden_dir", type=Path, required=True)
    parser.add_argument("--prompt_index_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--test_splits", nargs="+", default=["test"])
    parser.add_argument("--prompt_configs", nargs="+", default=["last6:l10_l26", "last4:l10_l25"])
    parser.add_argument("--feature_sets", nargs="+", default=[
        "base",
        "logic+disagree",
        "logic+disagree+geom",
        "logic+disagree+geom+cross",
        "logic+disagree+geom+cross+aux",
    ])
    parser.add_argument("--train_pairs_per_prompt", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--test_pairs_per_prompt", type=int, default=10)
    parser.add_argument("--n_estimators", nargs="+", type=int, default=[1000, 2000])
    parser.add_argument("--min_samples_leaf", nargs="+", type=int, default=[5])
    parser.add_argument("--max_features", nargs="+", type=float, default=[0.7])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=12)
    parser.add_argument("--aux_n_estimators", type=int, default=400)
    parser.add_argument("--aux_cv_folds", type=int, default=5)
    return parser.parse_args()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _prompt_local_geometry(features: dict[str, np.ndarray]) -> np.ndarray:
    raw24, raw25, raw26 = features["last_l24"], features["last_l25"], features["last_l26"]
    pool24, pool25, pool26 = features["l10_l24"], features["l10_l25"], features["l10_l26"]
    vals = [
        np.linalg.norm(raw24),
        np.linalg.norm(raw25),
        np.linalg.norm(raw26),
        np.linalg.norm(pool24),
        np.linalg.norm(pool25),
        np.linalg.norm(pool26),
        _cosine(raw24, raw25),
        _cosine(raw25, raw26),
        _cosine(raw24, raw26),
        _cosine(pool24, pool25),
        _cosine(pool25, pool26),
        _cosine(pool24, pool26),
        np.linalg.norm(raw25 - raw24),
        np.linalg.norm(raw26 - raw25),
        np.linalg.norm(raw26 - raw24),
        np.linalg.norm(pool25 - pool24),
        np.linalg.norm(pool26 - pool25),
        np.linalg.norm(pool26 - pool24),
        _cosine(raw25, pool25),
        _cosine(raw26, pool26),
        np.linalg.norm(raw25 - pool25),
        np.linalg.norm(raw26 - pool26),
    ]
    return np.asarray(vals, dtype=np.float32)


def _get_idx(feature_keys: list[str], key: str) -> int | None:
    try:
        return feature_keys.index(key)
    except ValueError:
        return None


def _vec_value(vec: np.ndarray, feature_keys: list[str], key: str, default: float = 0.0) -> float:
    idx = _get_idx(feature_keys, key)
    if idx is None:
        return default
    return float(vec[idx])


def _bool_triplet(left_val: float, right_val: float, threshold: float = 0.5) -> list[float]:
    left = left_val > threshold
    right = right_val > threshold
    return [float(left and right), float(left ^ right), float((not left) and (not right))]


def _pair_logic_features(row: dict[str, Any], feature_keys: list[str]) -> np.ndarray:
    left = row["left_vec"]
    right = row["right_vec"]
    feats: list[float] = []
    for key in [
        "has_complete_answer",
        "has_reasoning_content",
        "final_answer_exists",
        "final_answer_matches_answer_text",
        "answer_text_boxed_flag",
        "final_answer_text_boxed_flag",
        "generated_text_contains_final_answer",
    ]:
        feats.extend(_bool_triplet(_vec_value(left, feature_keys, key), _vec_value(right, feature_keys, key)))

    out_l = _vec_value(left, feature_keys, "output_length")
    out_r = _vec_value(right, feature_keys, "output_length")
    trunc_l = float(out_l >= 7800.0)
    trunc_r = float(out_r >= 7800.0)
    feats.extend(_bool_triplet(trunc_l, trunc_r, threshold=0.5))
    feats.extend([float(max(out_l, out_r) >= 7800.0), float(min(out_l, out_r) >= 7800.0)])
    return np.asarray(feats, dtype=np.float32)


def _pair_disagreement_features(row: dict[str, Any], feature_keys: list[str]) -> np.ndarray:
    left = row["left_vec"]
    right = row["right_vec"]
    pair_absdiff = row["pair_absdiff"]
    pair_rel_diff = row["pair_rel_diff"]
    selected = [
        "output_length",
        "think_tokens",
        "answer_tokens",
        "output_text_entropy",
        "reasoning_text_entropy",
        "answer_text_entropy",
        "final_answer_entropy",
        "duplicate_line_ratio",
        "generated_text_number_token_ratio",
        "answer_repetition_ratio",
        "answer_unique_token_ratio",
        "chars_per_second",
        "tokens_per_second",
    ]
    vals: list[float] = []
    for key in selected:
        idx = _get_idx(feature_keys, key)
        if idx is None:
            vals.extend([0.0, 0.0, 0.0, 0.0])
            continue
        l = float(left[idx])
        r = float(right[idx])
        vals.extend([
            float(pair_absdiff[idx]),
            float(pair_rel_diff[idx]),
            float(min(l, r)),
            float(max(l, r)),
        ])
    vals.extend(
        [
            abs(_vec_value(left, feature_keys, "final_answer_exists") - _vec_value(right, feature_keys, "final_answer_exists")),
            abs(_vec_value(left, feature_keys, "answer_text_boxed_flag") - _vec_value(right, feature_keys, "answer_text_boxed_flag")),
            abs(_vec_value(left, feature_keys, "has_complete_answer") - _vec_value(right, feature_keys, "has_complete_answer")),
            abs(_vec_value(left, feature_keys, "final_answer_matches_answer_text") - _vec_value(right, feature_keys, "final_answer_matches_answer_text")),
        ]
    )
    return np.asarray(vals, dtype=np.float32)


def _selected_cross_features(
    prompt_feats: np.ndarray,
    geometry_feats: np.ndarray,
    logic_feats: np.ndarray,
    disagree_feats: np.ndarray,
) -> np.ndarray:
    # Prompt side: input length, word count proxy, digit ratio, algebra/geometry counts, top geometry deltas.
    prompt_pick = [
        float(prompt_feats[0]),
        float(prompt_feats[2]),
        float(prompt_feats[5]),
        float(prompt_feats[21]),
        float(prompt_feats[22]),
        float(geometry_feats[5]),
        float(geometry_feats[10]),
        float(geometry_feats[17]),
    ]
    disagree_pick = [
        float(disagree_feats[0]),
        float(disagree_feats[4]),
        float(disagree_feats[12]),
        float(disagree_feats[16]),
        float(disagree_feats[24]),
        float(disagree_feats[40]),
        float(disagree_feats[48]),
        float(logic_feats[1]),
        float(logic_feats[-2]),
    ]
    crosses = [p * d for p in prompt_pick for d in disagree_pick]
    return np.asarray(crosses, dtype=np.float32)


def _aggregate_prompt_predictions(task_ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for task_id, target, pred in zip(task_ids, y_true.tolist(), y_pred.tolist()):
        groups[task_id]["y_true"].append(float(target))
        groups[task_id]["y_pred"].append(float(pred))
    ordered = sorted(groups.keys())
    true = np.asarray([float(np.mean(groups[k]["y_true"])) for k in ordered], dtype=np.float32)
    pred = np.asarray([float(np.mean(groups[k]["y_pred"])) for k in ordered], dtype=np.float32)
    return true, pred


def _make_base_matrix(
    pair_rows: list[dict[str, Any]],
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    prompt_mode: str,
    feature_keys: list[str],
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    feature_tokens = set(feature_set.split("+")) if feature_set else {"base"}
    use_logic = "logic" in feature_tokens
    use_disagree = "disagree" in feature_tokens
    use_geom = "geom" in feature_tokens
    use_cross = "cross" in feature_tokens

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    split_rows: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    for row in pair_rows:
        task_id = str(row["task_id"])
        prompt = prompt_lookup.get(task_id)
        if prompt is None:
            continue

        prompt_vec = _prompt_vector(prompt, prompt_mode)
        prompt_feats = prompt["prompt_feats"]
        rel_both = np.concatenate([prompt["rel_last"], prompt["rel_l10"]], axis=0)

        pieces = [
            prompt_vec,
            prompt_feats,
            rel_both,
            row["left_vec"],
            row["right_vec"],
            row["pair_mean"],
            row["pair_absdiff"],
            row["pair_min"],
            row["pair_max"],
            row["pair_rel_diff"],
            row["cosine"],
            row["l2"],
        ]

        logic_feats = None
        disagree_feats = None
        geom_feats = None
        if use_logic:
            logic_feats = _pair_logic_features(row, feature_keys)
            pieces.append(logic_feats)
        if use_disagree:
            disagree_feats = _pair_disagreement_features(row, feature_keys)
            pieces.append(disagree_feats)
        if use_geom:
            geom_feats = _prompt_local_geometry(prompt)
            pieces.append(geom_feats)
        if use_cross:
            if logic_feats is None:
                logic_feats = _pair_logic_features(row, feature_keys)
            if disagree_feats is None:
                disagree_feats = _pair_disagreement_features(row, feature_keys)
            if geom_feats is None:
                geom_feats = _prompt_local_geometry(prompt)
            pieces.append(_selected_cross_features(prompt_feats, geom_feats, logic_feats, disagree_feats))

        X_rows.append(np.concatenate(pieces, axis=0).astype(np.float32))
        y_rows.append(float(row["y_true"]))
        split_rows.append(str(row["split"]))
        metadata_rows.append(
            {
                "task_id": task_id,
                "split": str(row["split"]),
                "y_true": float(row["y_true"]),
            }
        )

    return np.stack(X_rows), np.asarray(y_rows, dtype=np.float32), np.asarray(split_rows), metadata_rows


def _oof_aux_probs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    X_test: np.ndarray,
    random_seed: int,
    n_jobs: int,
    aux_n_estimators: int,
    aux_cv_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    targets = [
        np.isclose(y_train, 1.0).astype(np.int64),
        np.isclose(y_train, 0.0).astype(np.int64),
        (y_train >= 0.75).astype(np.int64),
    ]
    train_prob_cols: list[np.ndarray] = []
    test_prob_cols: list[np.ndarray] = []
    splitter = GroupKFold(n_splits=aux_cv_folds)

    for target in targets:
        oof = np.zeros(X_train.shape[0], dtype=np.float32)
        usable = len(np.unique(target)) > 1
        if not usable:
            train_prob_cols.append(oof[:, None])
            test_prob_cols.append(np.full((X_test.shape[0], 1), float(target[0]), dtype=np.float32))
            continue

        for fit_idx, pred_idx in splitter.split(X_train, target, groups=train_groups):
            model = ExtraTreesClassifier(
                n_estimators=aux_n_estimators,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=random_seed,
                n_jobs=n_jobs,
            )
            model.fit(X_train[fit_idx], target[fit_idx])
            oof[pred_idx] = model.predict_proba(X_train[pred_idx])[:, 1].astype(np.float32)

        full_model = ExtraTreesClassifier(
            n_estimators=aux_n_estimators,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=random_seed,
            n_jobs=n_jobs,
        )
        full_model.fit(X_train, target)
        test_probs = full_model.predict_proba(X_test)[:, 1].astype(np.float32)
        train_prob_cols.append(oof[:, None])
        test_prob_cols.append(test_probs[:, None])

    return np.concatenate(train_prob_cols, axis=1), np.concatenate(test_prob_cols, axis=1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    repo_root = args.run_root.expanduser().resolve().parents[4]

    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    results.append(row)
                    seen.add(str(row["name"]))

    label_buckets = _build_label_buckets(args.run_root.expanduser().resolve())
    grouped_rows, feature_keys = _load_grouped_rows(args.rollout_index_path.expanduser().resolve(), label_buckets)

    prompt_lookups: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for spec in args.prompt_configs:
        cfg_name, pooled_hidden_dir, pooled_index_dir, _ = _parse_prompt_config(spec, repo_root)
        prompt_lookups[cfg_name] = build_prompt_lookup(
            args.prompt_hidden_dir.expanduser().resolve(),
            args.prompt_index_dir.expanduser().resolve(),
            pooled_hidden_dir.expanduser().resolve(),
            pooled_index_dir.expanduser().resolve(),
        )

    best = max(results, key=lambda row: row["prompt_mean_test_metrics"]["r2"], default=None)

    for pair_budget in args.train_pairs_per_prompt:
        pair_rows = build_pair_rows(
            grouped_rows=grouped_rows,
            feature_keys=feature_keys,
            train_splits=set(args.train_splits),
            test_splits=set(args.test_splits),
            train_pairs_per_prompt=pair_budget,
            test_pairs_per_prompt=args.test_pairs_per_prompt,
            random_seed=args.random_seed,
        )

        for spec in args.prompt_configs:
            cfg_name, _, _, prompt_mode = _parse_prompt_config(spec, repo_root)
            prompt_lookup = prompt_lookups[cfg_name]

            for feature_set in args.feature_sets:
                X, y, splits, metadata_rows = _make_base_matrix(pair_rows, prompt_lookup, prompt_mode, feature_keys, feature_set)
                train_mask = np.isin(splits, np.asarray(args.train_splits))
                test_mask = np.isin(splits, np.asarray(args.test_splits))
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_test = X[test_mask]
                y_test = y[test_mask]
                train_groups = np.asarray([metadata_rows[idx]["task_id"] for idx, keep in enumerate(train_mask.tolist()) if keep])
                test_task_ids = [metadata_rows[idx]["task_id"] for idx, keep in enumerate(test_mask.tolist()) if keep]

                if "aux" in feature_set.split("+"):
                    aux_train, aux_test = _oof_aux_probs(
                        X_train=X_train,
                        y_train=y_train,
                        train_groups=train_groups,
                        X_test=X_test,
                        random_seed=args.random_seed,
                        n_jobs=args.n_jobs,
                        aux_n_estimators=args.aux_n_estimators,
                        aux_cv_folds=args.aux_cv_folds,
                    )
                    X_train_use = np.concatenate([X_train, aux_train], axis=1)
                    X_test_use = np.concatenate([X_test, aux_test], axis=1)
                else:
                    X_train_use = X_train
                    X_test_use = X_test

                for n_estimators in args.n_estimators:
                    for min_samples_leaf in args.min_samples_leaf:
                        for max_features in args.max_features:
                            name = f"pairs{pair_budget}__{cfg_name}__{feature_set}__et_n{n_estimators}_l{min_samples_leaf}_mf{max_features}"
                            if name in seen:
                                continue
                            model = ExtraTreesRegressor(
                                n_estimators=n_estimators,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                random_state=args.random_seed,
                                n_jobs=args.n_jobs,
                            )
                            model.fit(X_train_use, y_train)
                            pred_test = np.clip(np.asarray(model.predict(X_test_use), dtype=np.float32).reshape(-1), 0.0, 1.0)
                            prompt_true, prompt_pred = _aggregate_prompt_predictions(test_task_ids, y_test, pred_test)

                            row = {
                                "name": name,
                                "prompt_config": cfg_name,
                                "prompt_mode": prompt_mode,
                                "feature_set": feature_set,
                                "train_pairs_per_prompt": pair_budget,
                                "test_pairs_per_prompt": args.test_pairs_per_prompt,
                                "params": {
                                    "n_estimators": n_estimators,
                                    "min_samples_leaf": min_samples_leaf,
                                    "max_features": max_features,
                                    "random_seed": args.random_seed,
                                },
                                "num_train_rows": int(X_train_use.shape[0]),
                                "num_test_rows": int(X_test_use.shape[0]),
                                "pair_feature_dim": int(X_train_use.shape[1]),
                                "test_metrics": _metrics(y_test, pred_test),
                                "prompt_mean_test_metrics": _metrics(prompt_true, prompt_pred),
                                "num_test_prompts": int(prompt_true.shape[0]),
                            }
                            results.append(row)
                            seen.add(name)
                            with results_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(row) + "\n")
                            print(json.dumps({
                                "name": name,
                                "row_r2": row["test_metrics"]["r2"],
                                "prompt_mean_r2": row["prompt_mean_test_metrics"]["r2"],
                            }), flush=True)

                            if best is None or row["prompt_mean_test_metrics"]["r2"] > best["prompt_mean_test_metrics"]["r2"]:
                                best = row
                                summary_path.write_text(
                                    json.dumps(
                                        {
                                            "setting": "two_rollout_finished16_engineered_search",
                                            "best": best,
                                            "num_results": len(results),
                                        },
                                        indent=2,
                                    ),
                                    encoding="utf-8",
                                )

    results.sort(key=lambda row: row["prompt_mean_test_metrics"]["r2"], reverse=True)
    summary = {
        "setting": "two_rollout_finished16_engineered_search",
        "best": results[0] if results else None,
        "top10": results[:10],
        "num_results": len(results),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
