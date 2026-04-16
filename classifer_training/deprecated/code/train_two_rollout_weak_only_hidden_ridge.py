from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.data import load_hidden_rows, load_manifest
from classifer_training.enrich_rollout_index import _single_run_features
from classifer_training.prompt_only_experiments import _hidden_relation_features, _prompt_features
from classifer_training.rollout_utils import extract_rollout_numeric_features
from classifer_training.train_two_rollout_reasoning_probe import (
    build_feature_matrix,
    build_pair_rows,
    build_rollout_hidden_lookup,
    load_grouped_rollouts,
)
from classifer_training.train_two_rollout_weak_transfer_text import (
    BASE_ROLLOUT_FEATURE_KEYS,
    _build_split_lookup,
    _prompt_mean_metrics,
    _reg_metrics,
)
from classifer_training.utils import load_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train weak-only ridge probes using weak train only, select on weak validation, and report clean validation/test transfer."
    )
    parser.add_argument("--weak_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--weak_labels_path", type=Path, required=True)
    parser.add_argument("--weak_prompt_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_rollout_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_rollout_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--clean_rollout_manifest", type=Path, required=True)
    parser.add_argument("--clean_labels_path", type=Path, required=True)
    parser.add_argument("--clean_prompt_hidden_dir", type=Path, required=True)
    parser.add_argument("--clean_prompt_index_dir", type=Path, required=True)
    parser.add_argument("--clean_rollout_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--clean_rollout_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_mode", default="l10_l26")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--weak_pairs_per_prompt", type=int, default=6)
    parser.add_argument("--clean_validation_pairs_per_prompt", type=int, default=4)
    parser.add_argument("--clean_test_pairs_per_prompt", type=int, default=10)
    parser.add_argument("--alphas", nargs="+", type=float, default=[100.0, 300.0, 1000.0, 3000.0, 10000.0])
    parser.add_argument("--drop_mean_absdiff", action="store_true")
    return parser.parse_args()


def _build_prompt_lookup_from_hidden_sources(
    hidden_paths: list[Path],
    index_paths: list[Path],
) -> dict[str, dict[str, np.ndarray]]:
    if len(hidden_paths) != len(index_paths):
        raise ValueError("Prompt hidden/index path counts must match.")
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for hidden_path, index_path in zip(hidden_paths, index_paths):
        rows = load_hidden_rows(
            hidden_path.expanduser().resolve(),
            index_path=index_path.expanduser().resolve(),
            dataset_name="dapo_math_17k",
            default_component_name="hidden",
        )
        for row in rows:
            task_id = str(row["task_id"])
            hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
            index_row = row["index_row"]
            user_input = str(index_row.get("user_input", index_row.get("prompt", "")))
            input_length = int(index_row.get("input_length", 0))
            lookup[task_id] = {
                "prompt_feats": _prompt_features(user_input, input_length),
                "rel_l10": _hidden_relation_features(hidden_layers),
                "l10_l22": hidden_layers[22],
                "l10_l23": hidden_layers[23],
                "l10_l24": hidden_layers[24],
                "l10_l25": hidden_layers[25],
                "l10_l26": hidden_layers[26],
                "l10_l35": hidden_layers[35],
            }
    return lookup


def _iter_hidden_rows_from_dir(hidden_dir: Path, index_dir: Path):
    combined_hidden = hidden_dir / "hidden_states.pt"
    combined_index = index_dir / "index.jsonl"
    if combined_hidden.exists() and combined_index.exists():
        for row in load_hidden_rows(combined_hidden, index_path=combined_index, dataset_name="dapo_math_17k"):
            yield row
        return
    for split in ("train", "validation", "test"):
        hidden_path = hidden_dir / f"hidden_states_{split}.pt"
        index_path = index_dir / f"index_{split}.jsonl"
        if not hidden_path.exists() or not index_path.exists():
            continue
        for row in load_hidden_rows(hidden_path, index_path=index_path, dataset_name="dapo_math_17k"):
            yield row


def _build_clean_prompt_lookup(prompt_hidden_dir: Path, prompt_index_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    lookup: dict[str, dict[str, np.ndarray]] = {}
    for row in _iter_hidden_rows_from_dir(prompt_hidden_dir.expanduser().resolve(), prompt_index_dir.expanduser().resolve()):
        task_id = str(row["task_id"])
        hidden_layers = [np.asarray(layer, dtype=np.float32) for layer in row["components"]["hidden"]]
        index_row = row["index_row"]
        user_input = str(index_row.get("user_input", ""))
        input_length = int(index_row.get("input_length", 0))
        lookup[task_id] = {
            "prompt_feats": _prompt_features(user_input, input_length),
            "rel_l10": _hidden_relation_features(hidden_layers),
            "l10_l22": hidden_layers[22],
            "l10_l23": hidden_layers[23],
            "l10_l24": hidden_layers[24],
            "l10_l25": hidden_layers[25],
            "l10_l26": hidden_layers[26],
            "l10_l35": hidden_layers[35],
        }
    return lookup


def _infer_feature_keys_from_manifest(manifest_path: Path) -> list[str]:
    manifest_entries = load_manifest(manifest_path.expanduser().resolve())
    feature_keys: set[str] = set()
    for entry in manifest_entries:
        index_path = Path(entry["index_path"]).expanduser().resolve()
        for row in load_records(index_path):
            rollout_features = dict(row.get("rollout_features") or {})
            rollout_features.update(extract_rollout_numeric_features(row))
            rollout_features.update(_single_run_features(row))
            feature_keys.update(rollout_features.keys())
            if len(feature_keys) >= 50:
                break
        if feature_keys:
            break
    return [key for key in BASE_ROLLOUT_FEATURE_KEYS if key in feature_keys]


def _group_weak_rollouts_with_hidden(
    *,
    weak_run_dirs: list[Path],
    feature_keys: list[str],
    weak_labels_by_task: dict[str, dict[str, Any]],
    split_lookup: dict[str, str],
    rollout_hidden_lookup: dict[tuple[str, int], np.ndarray],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    missing_rollout_hidden = 0
    for run_dir in weak_run_dirs:
        rows = load_records(run_dir / "all_experiments.jsonl")
        run_dir_str = str(run_dir.resolve())
        for row_idx, row in enumerate(rows):
            task_id = str(row["task_id"])
            label_row = weak_labels_by_task.get(task_id)
            if label_row is None:
                continue
            rollout_hidden = rollout_hidden_lookup.get((run_dir_str, int(row_idx)))
            if rollout_hidden is None:
                missing_rollout_hidden += 1
                continue
            split = split_lookup.get(task_id, str(row.get("split", "train")))
            rollout_features = extract_rollout_numeric_features(row)
            rollout_features.update(_single_run_features(row))
            stats_vec = np.asarray([float(rollout_features.get(key, 0.0)) for key in feature_keys], dtype=np.float32)
            group = grouped.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "split": split,
                    "y_true": float(label_row["difficulty"]),
                    "rollouts": [],
                },
            )
            group["rollouts"].append(
                {
                    "rollout_row_index": int(row_idx),
                    "run_dir": run_dir_str,
                    "stats_vec": stats_vec,
                    "rollout_hidden_vec": np.asarray(rollout_hidden, dtype=np.float32),
                }
            )
    print(json.dumps({"stage": "weak_rollout_hidden_grouping", "missing_rollout_hidden": missing_rollout_hidden}), flush=True)
    return [grouped[key] for key in sorted(grouped.keys())]


def _project_grouped_rollouts_feature_space(
    grouped_rows: list[dict[str, Any]],
    source_feature_keys: list[str],
    target_feature_keys: list[str],
) -> list[dict[str, Any]]:
    if source_feature_keys == target_feature_keys:
        return grouped_rows
    source_index = {key: idx for idx, key in enumerate(source_feature_keys)}
    gather_indices = np.asarray([source_index[key] for key in target_feature_keys], dtype=np.int64)
    projected_rows: list[dict[str, Any]] = []
    for group in grouped_rows:
        new_group = dict(group)
        new_rollouts = []
        for rollout in group["rollouts"]:
            new_rollout = dict(rollout)
            new_rollout["stats_vec"] = np.asarray(rollout["stats_vec"], dtype=np.float32)[gather_indices]
            new_rollouts.append(new_rollout)
        new_group["rollouts"] = new_rollouts
        projected_rows.append(new_group)
    return projected_rows


def _build_feature_matrix_drop_mean_absdiff(
    pair_rows: list[dict[str, Any]],
    prompt_lookup: dict[str, dict[str, np.ndarray]],
    prompt_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    split_rows: list[str] = []
    task_ids: list[str] = []
    for row in pair_rows:
        task_id = str(row["task_id"])
        prompt_features = prompt_lookup.get(task_id)
        if prompt_features is None:
            continue
        prompt_vector = np.asarray(prompt_features[prompt_mode], dtype=np.float32)
        prompt_feats = np.asarray(prompt_features["prompt_feats"], dtype=np.float32)
        prompt_rel = np.asarray(prompt_features["rel_l10"], dtype=np.float32)
        x_rows.append(
            np.concatenate(
                [
                    prompt_vector,
                    prompt_feats,
                    prompt_rel,
                    row["left_vec"],
                    row["right_vec"],
                    row["pair_min"],
                    row["pair_max"],
                    row["pair_rel_diff"],
                    row["cosine"],
                    row["l2"],
                    row["left_reasoning"],
                    row["right_reasoning"],
                    row["reasoning_cosine"],
                    row["reasoning_l2"],
                ],
                axis=0,
            )
        )
        y_rows.append(float(row["y_true"]))
        split_rows.append(str(row["split"]))
        task_ids.append(task_id)
    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.float32), split_rows, task_ids


def _write_predictions(output_path: Path, prompt_rows: list[dict[str, Any]], clean_labels_by_task: dict[str, dict[str, Any]]) -> None:
    rows = []
    for row in prompt_rows:
        label_row = clean_labels_by_task[str(row["task_id"])]
        rows.append(
            {
                "task_id": str(row["task_id"]),
                "user_input": str(label_row.get("user_input", "")),
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
                "num_pairs": int(row["num_pairs"]),
            }
        )
    write_jsonl(output_path, rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    weak_labels = load_records(args.weak_labels_path.expanduser().resolve())
    clean_labels = load_records(args.clean_labels_path.expanduser().resolve())
    weak_labels_by_task = {str(row["task_id"]): row for row in weak_labels}
    clean_labels_by_task = {str(row["task_id"]): row for row in clean_labels}
    split_lookup = _build_split_lookup(args.weak_prompt_dataset_dir.expanduser().resolve())
    feature_keys = _infer_feature_keys_from_manifest(args.clean_rollout_manifest)
    print(
        json.dumps(
            {
                "stage": "loaded_inputs",
                "num_weak_labels": len(weak_labels),
                "num_clean_labels": len(clean_labels),
                "num_feature_keys": len(feature_keys),
            }
        ),
        flush=True,
    )

    prompt_lookup = {}
    prompt_lookup.update(
        _build_prompt_lookup_from_hidden_sources(
            [path.expanduser().resolve() for path in args.weak_prompt_hidden_paths],
            [path.expanduser().resolve() for path in args.weak_prompt_index_paths],
        )
    )
    prompt_lookup.update(
        _build_clean_prompt_lookup(
            args.clean_prompt_hidden_dir,
            args.clean_prompt_index_dir,
        )
    )
    print(json.dumps({"stage": "built_prompt_lookup", "num_prompts": len(prompt_lookup)}), flush=True)

    configs = [
        {"name": "think_end_hidden:mean", "component": "think_end_hidden", "pool": "mean"},
        {"name": "think_end_last10_hidden:mean", "component": "think_end_last10_hidden", "pool": "mean"},
    ]

    best_bundle: dict[str, Any] | None = None
    best_val_r2 = -1e18

    for cfg in configs:
        weak_rollout_hidden_lookup = build_rollout_hidden_lookup(
            [path.expanduser().resolve() for path in args.weak_rollout_hidden_paths],
            [path.expanduser().resolve() for path in args.weak_rollout_index_paths],
            component_name=cfg["component"],
            layer_index=0,
            pool_mode=str(cfg["pool"]),
        )
        clean_rollout_hidden_lookup = build_rollout_hidden_lookup(
            [path.expanduser().resolve() for path in args.clean_rollout_hidden_paths],
            [path.expanduser().resolve() for path in args.clean_rollout_index_paths],
            component_name=cfg["component"],
            layer_index=0,
            pool_mode=str(cfg["pool"]),
        )
        print(
            json.dumps(
                {
                    "stage": "built_rollout_hidden_lookup",
                    "dataset_key": cfg["name"],
                    "num_weak_rollout_hidden": len(weak_rollout_hidden_lookup),
                    "num_clean_rollout_hidden": len(clean_rollout_hidden_lookup),
                }
            ),
            flush=True,
        )

        weak_grouped = _group_weak_rollouts_with_hidden(
            weak_run_dirs=[path.expanduser().resolve() for path in args.weak_run_dirs],
            feature_keys=feature_keys,
            weak_labels_by_task=weak_labels_by_task,
            split_lookup=split_lookup,
            rollout_hidden_lookup=weak_rollout_hidden_lookup,
        )
        clean_grouped, clean_feature_keys = load_grouped_rollouts(
            args.clean_rollout_manifest.expanduser().resolve(),
            clean_rollout_hidden_lookup,
        )
        clean_grouped = _project_grouped_rollouts_feature_space(clean_grouped, clean_feature_keys, feature_keys)
        print(
            json.dumps(
                {
                    "stage": "grouped_rollouts",
                    "dataset_key": cfg["name"],
                    "num_weak_grouped_prompts": len(weak_grouped),
                    "num_clean_grouped_prompts": len(clean_grouped),
                    "num_selected_feature_keys": len(feature_keys),
                    "num_clean_feature_keys_before_projection": len(clean_feature_keys),
                }
            ),
            flush=True,
        )

        weak_pair_rows = build_pair_rows(
            weak_grouped,
            feature_keys,
            {"train", "validation"},
            set(),
            args.weak_pairs_per_prompt,
            0,
            args.random_seed,
        )
        clean_pair_rows = build_pair_rows(
            clean_grouped,
            feature_keys,
            {"validation"},
            {"test"},
            args.clean_validation_pairs_per_prompt,
            args.clean_test_pairs_per_prompt,
            args.random_seed,
        )

        feature_builder = _build_feature_matrix_drop_mean_absdiff if args.drop_mean_absdiff else build_feature_matrix
        weak_X, weak_y, weak_splits, weak_task_ids = feature_builder(weak_pair_rows, prompt_lookup, args.prompt_mode)
        clean_X, clean_y, clean_splits, clean_task_ids = feature_builder(clean_pair_rows, prompt_lookup, args.prompt_mode)
        weak_splits = np.asarray(weak_splits)
        clean_splits = np.asarray(clean_splits)
        print(
            json.dumps(
                {
                    "stage": "matrix_shapes",
                    "dataset_key": cfg["name"],
                    "weak_dim": int(weak_X.shape[1]),
                    "clean_dim": int(clean_X.shape[1]),
                }
            ),
            flush=True,
        )

        weak_train_mask = weak_splits == "train"
        weak_val_mask = weak_splits == "validation"
        clean_val_mask = clean_splits == "validation"
        clean_test_mask = clean_splits == "test"

        X_train, y_train = weak_X[weak_train_mask], weak_y[weak_train_mask]
        X_weak_val, y_weak_val = weak_X[weak_val_mask], weak_y[weak_val_mask]
        X_clean_val, y_clean_val = clean_X[clean_val_mask], clean_y[clean_val_mask]
        X_test, y_test = clean_X[clean_test_mask], clean_y[clean_test_mask]
        weak_val_task_ids = [weak_task_ids[idx] for idx in np.where(weak_val_mask)[0]]
        clean_val_task_ids = [clean_task_ids[idx] for idx in np.where(clean_val_mask)[0]]
        test_task_ids = [clean_task_ids[idx] for idx in np.where(clean_test_mask)[0]]
        print(
            json.dumps(
                {
                    "stage": "built_matrices",
                    "dataset_key": cfg["name"],
                    "feature_dim": int(X_train.shape[1]),
                    "num_train_rows": int(X_train.shape[0]),
                    "num_weak_val_rows": int(X_weak_val.shape[0]),
                    "num_clean_val_rows": int(X_clean_val.shape[0]),
                    "num_test_rows": int(X_test.shape[0]),
                }
            ),
            flush=True,
        )

        for alpha in args.alphas:
            name = f"ridge_a{alpha:g}"
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=alpha, random_state=args.random_seed)),
                ]
            )
            model.fit(X_train, y_train)
            weak_val_pred = np.clip(np.asarray(model.predict(X_weak_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
            weak_val_row_metrics = _reg_metrics(y_weak_val, weak_val_pred)
            weak_val_prompt_metrics, _ = _prompt_mean_metrics(
                y_weak_val,
                weak_val_pred,
                [{"task_id": task_id, "y_true": float(y)} for task_id, y in zip(weak_val_task_ids, y_weak_val.tolist())],
            )
            clean_val_pred = np.clip(np.asarray(model.predict(X_clean_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
            clean_val_row_metrics = _reg_metrics(y_clean_val, clean_val_pred)
            clean_val_prompt_metrics, _ = _prompt_mean_metrics(
                y_clean_val,
                clean_val_pred,
                [{"task_id": task_id, "y_true": float(y)} for task_id, y in zip(clean_val_task_ids, y_clean_val.tolist())],
            )
            result = {
                "dataset_key": cfg["name"],
                "name": name,
                "weak_val_row_metrics": weak_val_row_metrics,
                "weak_val_prompt_mean_metrics": weak_val_prompt_metrics,
                "clean_val_row_metrics": clean_val_row_metrics,
                "clean_val_prompt_mean_metrics": clean_val_prompt_metrics,
            }
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            print(
                json.dumps(
                    {
                        "dataset_key": cfg["name"],
                        "candidate": name,
                        "weak_val_prompt_r2": weak_val_prompt_metrics["r2"],
                        "clean_val_prompt_r2": clean_val_prompt_metrics["r2"],
                    }
                ),
                flush=True,
            )

            if weak_val_prompt_metrics["r2"] > best_val_r2:
                test_pred = np.clip(np.asarray(model.predict(X_test), dtype=np.float32).reshape(-1), 0.0, 1.0)
                test_row_metrics = _reg_metrics(y_test, test_pred)
                test_prompt_metrics, prompt_rows = _prompt_mean_metrics(
                    y_test,
                    test_pred,
                    [{"task_id": task_id, "y_true": float(y)} for task_id, y in zip(test_task_ids, y_test.tolist())],
                )
                best_val_r2 = weak_val_prompt_metrics["r2"]
                best_bundle = {
                    "dataset_key": cfg["name"],
                    "name": name,
                    "model": model,
                    "weak_val_prompt_mean_metrics": weak_val_prompt_metrics,
                    "clean_val_prompt_mean_metrics": clean_val_prompt_metrics,
                    "test_row_metrics": test_row_metrics,
                    "test_prompt_mean_metrics": test_prompt_metrics,
                    "prompt_rows": prompt_rows,
                    "num_train_rows": int(X_train.shape[0]),
                    "num_weak_val_rows": int(X_weak_val.shape[0]),
                    "num_clean_val_rows": int(X_clean_val.shape[0]),
                    "num_test_rows": int(X_test.shape[0]),
                    "feature_dim": int(X_train.shape[1]),
                }

    assert best_bundle is not None
    _write_predictions(args.output_dir / "predictions_test.jsonl", best_bundle["prompt_rows"], clean_labels_by_task)
    summary = {
        "setting": "weak_only_hidden_ridge_transfer_no_mean_absdiff" if args.drop_mean_absdiff else "weak_only_hidden_ridge_transfer",
        "prompt_mode": args.prompt_mode,
        "alphas": [float(alpha) for alpha in args.alphas],
        "drop_mean_absdiff": bool(args.drop_mean_absdiff),
        "best_dataset_key": best_bundle["dataset_key"],
        "best_model": best_bundle["name"],
        "num_train_rows": best_bundle["num_train_rows"],
        "num_weak_val_rows": best_bundle["num_weak_val_rows"],
        "num_clean_val_rows": best_bundle["num_clean_val_rows"],
        "num_test_rows": best_bundle["num_test_rows"],
        "num_test_prompts": int(len(best_bundle["prompt_rows"])),
        "feature_dim": best_bundle["feature_dim"],
        "weak_val_prompt_mean_metrics": best_bundle["weak_val_prompt_mean_metrics"],
        "clean_val_prompt_mean_metrics": best_bundle["clean_val_prompt_mean_metrics"],
        "test_row_metrics": best_bundle["test_row_metrics"],
        "test_prompt_mean_metrics": best_bundle["test_prompt_mean_metrics"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
