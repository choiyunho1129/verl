#!/usr/bin/env python3
"""Run probe training + eval for all finished extraction manifests.

This utility scans manifest.json files, filters entries that can be trained from
finished artifacts, then runs:
  - train_weak_only_single_rollout_hidden
  - eval_single_rollout_hidden_transfer

It is intentionally conservative:
- requires labels
- requires prompt + rollout hidden/index files
- requires run directories (or inferable from manifest/run naming)
- requires at least one train and one validation split from discovered tasks
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import re
from pathlib import Path
from typing import Any, Iterable

from classifer_training.data import load_hidden_rows


ROOT_DIR = Path("/data2/jongwonlim/verl/yoonho/verl/classifer_training")
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-root", type=Path, default=ARTIFACTS_DIR / "logs")
    parser.add_argument("--output-root", type=Path, default=ARTIFACTS_DIR / "probe" / "batch_completed_manifests")
    parser.add_argument("--python", type=Path, default=Path("python3"))
    parser.add_argument("--prompt-layer-index", type=int, default=19)
    parser.add_argument("--rollout-layer-index", type=int, default=19)
    parser.add_argument("--prompt-hidden-pca-dim", type=int, default=32)
    parser.add_argument("--rollout-hidden-pca-dim", type=int, default=256)
    parser.add_argument("--selection-metric", type=str, default="row_r2")
    parser.add_argument("--train-target-mode", type=str, default="other_rollout_correctness")
    parser.add_argument("--model-family", type=str, default="ridge")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--single-rollout-strategy", type=str, default="first", choices=["first", "all"])
    parser.add_argument(
        "--split-mode",
        choices=["auto", "validation_half"],
        default="auto",
        help=(
            "auto uses existing train/validation split hints. validation_half ignores the original train split "
            "and splits validation prompts 50/50 into synthetic train and validation sets."
        ),
    )
    parser.add_argument(
        "--prompt-component",
        type=str,
        default="",
        help="override prompt component (default: pick from manifest/available components)",
    )
    parser.add_argument(
        "--rollout-component",
        type=str,
        default="",
        help="override rollout component (default: pick from manifest/available components)",
    )
    parser.add_argument(
        "--prompt-feature-keys",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--rollout-scalar-keys",
        nargs="+",
        default=[
            "output_mean_token_entropy",
            "reasoning_mean_token_entropy",
            "answer_mean_token_entropy",
        ],
    )
    parser.add_argument(
        "--derived-rollout-scalar-keys",
        nargs="*",
        default=["entropy_gap_reasoning_answer", "answer_entropy_gap_vs_output"],
    )
    parser.add_argument(
        "--extra-rollout-scalar-field-paths",
        nargs="*",
        default=["rollout_features.answer_mean_token_entropy"],
    )
    parser.add_argument(
        "--allow-missing-entropy-scalars",
        action="store_true",
        default=True,
        help="Pass through to train/eval as --allow_missing_entropy_scalars.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0, help="limit number of manifest entries to process")
    return parser.parse_args()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _iter_manifest_entries(manifests_root: Path):
    seen: set[tuple[str, str | None]] = set()
    for manifest_path in sorted(manifests_root.glob("**/*manifest.json")):
        try:
            manifest = _read_json(manifest_path)
        except Exception as e:
            print(f"[skip] malformed manifest {manifest_path}: {e}")
            continue

        if isinstance(manifest.get("datasets"), dict):
            for name, ds in manifest["datasets"].items():
                if not isinstance(ds, dict):
                    continue
                key = (str(manifest_path), str(name))
                if key in seen:
                    continue
                seen.add(key)
                yield manifest_path, name, ds, manifest
        else:
            key = (str(manifest_path), None)
            if key in seen:
                continue
            seen.add(key)
            yield manifest_path, None, manifest, manifest


def _existing(p: Path) -> list[Path]:
    return [x for x in p if x.exists()]


def _infer_model_name(ds: dict[str, Any], labels_path: Path | None) -> str:
    for key in ("rollout_model_slug", "prompt_model_slug", "model_slug"):
        value = ds.get(key)
        if isinstance(value, str) and value:
            return value.split("_l", 1)[0]
    for key in ("model_name_or_path", "load_model_name_or_path"):
        value = ds.get(key)
        if isinstance(value, str) and value:
            name = Path(value).name
            if "_" in name and name.endswith("B"):
                return name
            return name
    if labels_path is not None:
        parts = list(labels_path.parts)
        # labels root has .../labels/{dataset_name}/{model_name}/...
        for idx, part in enumerate(parts):
            if part == "labels" and idx + 1 < len(parts):
                if idx + 2 < len(parts):
                    return parts[idx + 2]
    return ""


def _resolve_candidate_files(paths: list[str]) -> list[Path]:
    direct = [Path(p) for p in paths if Path(p).exists()]
    if direct:
        return direct

    # fallback: if manifest recorded a shard path that got compacted to non-sharded.
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if "shard" in p.name and p.parent.exists():
            compact_name = re.sub(r"\.shard\d+of\d+(?=\.pt$|\.jsonl$)", "", p.name)
            compact = p.parent / compact_name
            if compact.exists():
                out.append(compact)
                continue
            compact = p.parent / compact_name
            if compact.exists():
                out.append(compact)
                continue
            if p.name.endswith(".jsonl"):
                compact = p.parent / compact_name
                if compact.exists():
                    out.append(compact)
                    continue
            # last resort: pull all shards by family
            if p.name.endswith(".pt"):
                out.extend(sorted(p.parent.glob("rollout_hidden_states.shard*.pt")))
            if p.name.endswith(".jsonl"):
                out.extend(sorted(p.parent.glob("rollout_index.shard*.jsonl")))
            if out:
                break
    return sorted(dict.fromkeys(out))


def _infer_run_dirs(ds: dict[str, Any], dataset_key: str | None, rollout_dataset_name: str | None, labels_path: Path | None) -> list[Path]:
    run_dirs: list[Path] = []
    for k in ("train_run_dirs", "validation_run_dirs", "run_dirs"):
        vals = _as_list(ds.get(k))
        for p in [Path(v) for v in vals]:
            if p.exists() and (p / "all_experiments.jsonl").exists():
                run_dirs.append(p)

    if run_dirs:
        return sorted(dict.fromkeys(run_dirs))

    # Infer from explicit run_dir fields.
    explicit_run = ds.get("run_dir")
    if isinstance(explicit_run, str) and explicit_run:
        explicit_path = Path(explicit_run)
        if explicit_path.exists() and (explicit_path / "all_experiments.jsonl").exists():
            return [explicit_path]
        if explicit_path.exists():
            print(f"[warn] run_dir exists but missing all_experiments.jsonl: {explicit_path}")

    # Infer from rollout dataset naming.
    if not rollout_dataset_name:
        return []

    model_name = _infer_model_name(ds, labels_path)
    if dataset_key:
        run_name = rollout_dataset_name
        prefix = f"{dataset_key}_"
        if run_name.startswith(prefix):
            run_name = run_name[len(prefix) :]
    else:
        run_name = rollout_dataset_name

    candidates: list[Path] = []
    roots = [Path(ARTIFACTS_DIR / "runs"), ARTIFACTS_DIR / "runs"]
    if dataset_key:
        roots = [ARTIFACTS_DIR / "runs" / dataset_key]
    candidate_parent_dirs = [r for r in [*roots] if r.exists()]

    model_names = []
    if model_name:
        model_names.append(model_name)
    if "Qwen_Qwen3-4B" in rollout_dataset_name:
        model_names.append("Qwen_Qwen3-4B")
    if "deepseek" in rollout_dataset_name.lower():
        model_names.append("deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B")

    # Unique stable order
    model_names = list(dict.fromkeys(model_names))
    # Some manifests append extraction suffix to rollout_dataset_name like
    # ..._response_l18_35_last5_10_15mean ; actual run dir uses the prefix.
    if "_response_" in run_name:
        run_name = run_name.split("_response_", 1)[0]
    if "_thinkend" in run_name:
        run_name = run_name.split("_thinkend", 1)[0]

    for root in candidate_parent_dirs:
        for m in model_names:
            p = root / m / run_name
            if p.exists() and (p / "all_experiments.jsonl").exists():
                candidates.append(p)
            # if no model folder, sometimes data is directly under dataset root
            direct = root / run_name
            if direct.exists() and (direct / "all_experiments.jsonl").exists():
                candidates.append(direct)

    return sorted(dict.fromkeys(candidates))


def _choose_component(components: list[str], requested: str, default_hint: str | None = None) -> str:
    if requested and requested in components:
        return requested
    for candidate in [
        "think_end_last10_hidden",
        "response_last10_mean_hidden",
        "hidden_last10_mean",
        "response_last5_mean_hidden",
        "response_last15_mean_hidden",
        "hidden",
        "think_end_hidden",
        "response_hidden",
    ]:
        if candidate in components:
            return candidate
    if default_hint and default_hint in components:
        return default_hint
    if not components:
        return default_hint or "hidden"
    return components[0]


def _components_from_file(prompt_path: Path, index_path: Path) -> list[str]:
    try:
        rows = load_hidden_rows(prompt_path, index_path, dataset_name="dapo_math_17k", default_component_name="hidden")
    except Exception:
        return []
    for row in rows:
        return sorted(row["components"].keys())
    return []


def _infer_prompt_component(ds: dict[str, Any], prompt_paths: list[Path], prompt_index_paths: list[Path], requested: str) -> str:
    candidates = _as_list(ds.get("prompt", {}).get("components"))
    if not candidates:
        candidates = _as_list(ds.get("prompt_components"))
    if not candidates:
        candidates = []
    if requested:
        return requested
    if candidates:
        # if manifest gives prompt component names (rare but possible), prefer those
        possible = [x for x in candidates if isinstance(x, str)]
    else:
        possible = []
    if prompt_paths and prompt_index_paths:
        comps = _components_from_file(prompt_paths[0], prompt_index_paths[0])
        for key in ["hidden_last10_mean", "hidden", "prompt_hidden", "hidden_last5_mean", "hidden_last15_mean"]:
            if key in possible and key in comps:
                return key
        chosen = _choose_component(possible if possible else comps, "", "hidden_last10_mean")
        if chosen:
            return chosen
    return possible[0] if possible else "hidden_last10_mean"


def _infer_rollout_component(ds: dict[str, Any], rollout_paths: list[Path], rollout_index_paths: list[Path], requested: str) -> str:
    candidates = _as_list(ds.get("rollout_components"))
    if requested:
        return requested
    if rollout_paths and rollout_index_paths:
        comps = _components_from_file(rollout_paths[0], rollout_index_paths[0])
        for key in [
            "think_end_last10_hidden",
            "response_last10_mean_hidden",
            "response_last5_mean_hidden",
            "response_last15_mean_hidden",
            "response_hidden",
            "hidden",
        ]:
            if key in comps:
                return key
        if candidates:
            for key in candidates:
                if key in comps:
                    return key
        if comps:
            return comps[0]
    if candidates:
        return candidates[0]
    return requested or "response_last10_mean_hidden"


def _collect_split_from_runs(run_dirs: list[Path]) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for run_dir in run_dirs:
        all_experiments = run_dir / "all_experiments.jsonl"
        if not all_experiments.exists():
            continue
        for row in _iter_jsonl(all_experiments):
            task_id = str(row.get("task_id", ""))
            split = str(row.get("split", ""))
            if task_id and split:
                split_map.setdefault(task_id, split)
    return split_map


def _build_split_dataset(
    labels_path: Path,
    run_dirs: list[Path],
    out_root: Path,
    dataset_name: str,
    split_mode: str = "auto",
) -> Path:
    out_root = out_root / f"{dataset_name}_split"
    out_root.mkdir(parents=True, exist_ok=True)
    labels = _read_rows(labels_path)
    task_ids = sorted({str(row.get("task_id", "")) for row in labels if row.get("task_id") is not None})
    run_split = _collect_split_from_runs(run_dirs)

    mapped: dict[str, str] = {}
    for tid in task_ids:
        mapped[tid] = run_split.get(tid, "")

    if split_mode == "validation_half":
        candidate_ids = [tid for tid, sp in mapped.items() if sp in {"validation", "valid"}]
        if not candidate_ids:
            candidate_ids = sorted(task_ids)
        candidate_ids = sorted(set(candidate_ids))
        midpoint = max(1, len(candidate_ids) // 2)
        train_ids = candidate_ids[:midpoint]
        val_ids = candidate_ids[midpoint:]
        test_ids = []
    else:
        # prefer explicit run-split signals
        train_ids = [tid for tid, sp in mapped.items() if sp == "train"]
        val_ids = [tid for tid, sp in mapped.items() if sp in {"validation", "valid"}]
        test_ids = [tid for tid, sp in mapped.items() if sp == "test"]

        if not train_ids and not val_ids and test_ids:
            # if only test split exists (e.g., ifbench), split by task id.
            midpoint = max(1, int(len(test_ids) * 0.5))
            val_ids = sorted(test_ids)[:midpoint]
            train_ids = sorted(test_ids)[midpoint:]
        elif not train_ids and val_ids:
            # use validation as synthetic train and test as synthetic valid
            train_ids = sorted(val_ids)
            if test_ids:
                val_ids = sorted(test_ids)
        elif not val_ids and train_ids:
            # if only train exists, reserve small validation set from train.
            shuffled = sorted(train_ids)
            random.Random(42).shuffle(shuffled)
            k = max(1, max(1, int(len(shuffled) * 0.2)))
            val_ids = shuffled[:k]
        elif not val_ids and not train_ids and not test_ids:
            # no split hints anywhere: 80/20 random split
            shuffled = sorted(task_ids)
            random.Random(42).shuffle(shuffled)
            k = max(1, int(len(shuffled) * 0.8))
            train_ids = shuffled[:k]
            val_ids = shuffled[k:]

    # keep order deterministic
    train_ids = sorted(set(train_ids))
    val_ids = sorted(set(val_ids))

    # if still empty, fallback to all to train/val to avoid hard crash
    if not train_ids:
        train_ids = sorted(task_ids[: max(1, len(task_ids) // 2)])
    if not val_ids:
        val_ids = sorted(task_ids[max(1, len(task_ids) // 2) :])
        if not val_ids:
            val_ids = sorted(task_ids[:1])

    train_rows = [{"task_id": tid} for tid in train_ids]
    val_rows = [{"task_id": tid} for tid in val_ids]
    if not train_rows:
        train_rows = [{"task_id": tid} for tid in val_ids[:1]]
    if not val_rows:
        val_rows = [{"task_id": tid} for tid in train_ids[:1]]

    (out_root / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + ("\n" if train_rows else ""),
        encoding="utf-8",
    )
    (out_root / "validation.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows) + ("\n" if val_rows else ""),
        encoding="utf-8",
    )
    (out_root / "split_summary.json").write_text(
        json.dumps(
            {
                "split_mode": split_mode,
                "num_total_label_tasks": len(task_ids),
                "num_train": len(train_rows),
                "num_validation": len(val_rows),
                "num_test": len(test_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if test_ids:
        test_rows = [{"task_id": tid} for tid in sorted(test_ids)]
        (out_root / "test.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in test_rows) + ("\n" if test_rows else ""),
            encoding="utf-8",
        )

    return out_root


def _split_dataset_is_ready(dataset_dir: Path) -> bool:
    return (dataset_dir / "train.jsonl").exists() and (dataset_dir / "validation.jsonl").exists()


def _resolve_dataset_dir(
    manifest_dir: Path,
    dataset_key: str | None,
    ds: dict[str, Any],
    run_dirs: list[Path],
    labels_path: Path,
    split_mode: str = "auto",
) -> Path:
    if split_mode != "auto":
        return _build_split_dataset(
            labels_path,
            run_dirs,
            manifest_dir / f"_tmp_split_dataset_{split_mode}",
            f"{manifest_dir.name}_{(dataset_key or 'dataset')}",
            split_mode,
        )

    # 1) explicit dataset_dir in manifest
    explicit = ds.get("dataset_dir")
    if isinstance(explicit, str) and explicit:
        p = Path(explicit)
        if p.exists() and _split_dataset_is_ready(p):
            return p
        if p.exists():
            # try to build synthetic split from available run data + labels
            synthetic = manifest_dir / "_tmp_split_dataset"
            return _build_split_dataset(labels_path, run_dirs, synthetic, f"{manifest_dir.name}_{(dataset_key or 'dataset')}")

    # 2) infer from prompt hidden path names e.g. .../<dataset>_shard0/...
    dataset_candidates = _as_list(ds.get("dataset_slug"))
    for c in dataset_candidates:
        if c:
            candidates = [
                ARTIFACTS_DIR / "datasets" / c,
                ARTIFACTS_DIR / "datasets" / f"{c}_shards2",
                ARTIFACTS_DIR / "datasets" / f"{c}_shards4",
            ]
            for p in candidates:
                if p.exists() and _split_dataset_is_ready(p):
                    return p

    prompt_paths = [Path(p) for p in _as_list(ds.get("prompt", {}).get("hidden"))]
    for p in prompt_paths:
        # .../hidden/<dataset>_shard0/<component>/...
        stem = p.parent.parent.name
        if "_shard" in stem:
            base = stem.split("_shard", 1)[0]
            for suffix in ("", "_shards1", "_shards2", "_shards4"):
                cand = ARTIFACTS_DIR / "datasets" / f"{base}{suffix}"
                if cand.exists() and _split_dataset_is_ready(cand):
                    return cand
                if cand.exists() and (cand / "all.jsonl").exists():
                    # no official split file; build synthetic split
                    return _build_split_dataset(labels_path, run_dirs, cand.parent / f"{cand.name}_split", base)
    # 3) fallback: if available run dirs reveal split hints, build synthetic dataset
    fallback = _build_split_dataset(labels_path, run_dirs, manifest_dir / "_tmp_split_dataset", f"{manifest_dir.name}_{(dataset_key or 'dataset')}")
    return fallback


def _resolve_manifest_parts(ds: dict[str, Any], manifest_path: Path, dataset_key: str | None) -> dict[str, Any]:
    prompt_paths = [Path(p) for p in _as_list(ds.get("prompt", {}).get("hidden"))]
    prompt_index_paths = [Path(p) for p in _as_list(ds.get("prompt", {}).get("index"))]
    rollout_paths = [Path(p) for p in _as_list(ds.get("rollout", {}).get("hidden"))]
    rollout_index_paths = [Path(p) for p in _as_list(ds.get("rollout", {}).get("index"))]

    prompt_paths = _resolve_candidate_files([str(p) for p in prompt_paths])
    prompt_index_paths = _resolve_candidate_files([str(p) for p in prompt_index_paths])
    rollout_paths = _resolve_candidate_files([str(p) for p in rollout_paths])
    rollout_index_paths = _resolve_candidate_files([str(p) for p in rollout_index_paths])

    # align counts by truncating to min length to avoid corruption
    if len(prompt_paths) != len(prompt_index_paths):
        n = min(len(prompt_paths), len(prompt_index_paths))
        prompt_paths = prompt_paths[:n]
        prompt_index_paths = prompt_index_paths[:n]
    if len(rollout_paths) != len(rollout_index_paths):
        n = min(len(rollout_paths), len(rollout_index_paths))
        rollout_paths = rollout_paths[:n]
        rollout_index_paths = rollout_index_paths[:n]

    return {
        "prompt_hidden_paths": prompt_paths,
        "prompt_index_paths": prompt_index_paths,
        "rollout_hidden_paths": rollout_paths,
        "rollout_index_paths": rollout_index_paths,
    }


def _run_cmd(cmd: list[str], dry_run: bool) -> int:
    printable = " ".join(cmd)
    if dry_run:
        print("[dry-run]", printable)
        return 0
    subprocess.run(cmd, check=True)
    return 0


def _clear_if_needed(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        path.unlink()
        return
    for child in sorted(path.glob("*"), key=lambda x: len(str(x))):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    os.environ.setdefault("PYTHONPATH", str(ROOT_DIR))

    args.output_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for manifest_path, dataset_key, ds, _top in _iter_manifest_entries(args.manifests_root):
        if args.max_jobs and count >= args.max_jobs:
            break

        try:
            labels_path = Path(ds.get("labels_path", ""))
            if not labels_path.exists():
                print(f"[skip] {manifest_path.name} missing labels {labels_path}")
                continue

            manifest_tag = f"{manifest_path.parent.name}"
            ds_tag = dataset_key or ds.get("dataset_slug") or ds.get("dataset_name") or "dataset"
            base_run_name = f"{manifest_tag}__{ds_tag}"
            out_dir = args.output_root / base_run_name

            resolved_paths = _resolve_manifest_parts(ds, manifest_path, dataset_key)
            prompt_hidden_paths = resolved_paths["prompt_hidden_paths"]
            prompt_index_paths = resolved_paths["prompt_index_paths"]
            rollout_hidden_paths = resolved_paths["rollout_hidden_paths"]
            rollout_index_paths = resolved_paths["rollout_index_paths"]

            if not prompt_hidden_paths or not prompt_index_paths:
                print(f"[skip] {base_run_name}: missing prompt files")
                continue
            if not rollout_hidden_paths or not rollout_index_paths:
                print(f"[skip] {base_run_name}: missing rollout files")
                continue

            run_dirs = _infer_run_dirs(
                ds=ds,
                dataset_key=dataset_key,
                rollout_dataset_name=ds.get("rollout_dataset_name"),
                labels_path=labels_path,
            )
            if not run_dirs:
                # as last resort, no run data means no weak rows to train
                print(f"[skip] {base_run_name}: no run dirs found")
                continue

            # ensure path list order stable
            run_dirs = sorted(dict.fromkeys(run_dirs))
            # check for all_experiments rows
            active_run_dirs = [p for p in run_dirs if (p / "all_experiments.jsonl").exists()]
            if not active_run_dirs:
                print(f"[skip] {base_run_name}: no all_experiments.jsonl under inferred run dirs")
                continue

            dataset_dir = _resolve_dataset_dir(
                manifest_path.parent,
                dataset_key,
                ds,
                active_run_dirs,
                labels_path,
                args.split_mode,
            )
            if not _split_dataset_is_ready(dataset_dir):
                print(f"[skip] {base_run_name}: cannot resolve train/validation split dataset")
                continue

            prompt_component = _infer_prompt_component(ds, prompt_hidden_paths, prompt_index_paths, args.prompt_component)
            rollout_component = _infer_rollout_component(ds, rollout_hidden_paths, rollout_index_paths, args.rollout_component)

            if out_dir.exists() and not args.rerun:
                print(f"[skip] {base_run_name}: output exists {out_dir}")
                continue
            if args.rerun and out_dir.exists():
                _clear_if_needed(out_dir)

            if out_dir.exists() and not list(out_dir.iterdir()):
                pass

            train_cmd: list[str] = [
                str(args.python),
                "-u",
                "-m",
                "classifer_training.train_weak_only_single_rollout_hidden",
                "--weak_run_dirs",
                *[str(p) for p in active_run_dirs],
                "--weak_prompt_dataset_dir",
                str(dataset_dir),
                "--weak_labels_path",
                str(labels_path),
                "--weak_prompt_hidden_paths",
                *[str(p) for p in prompt_hidden_paths],
                "--weak_prompt_index_paths",
                *[str(p) for p in prompt_index_paths],
                "--weak_rollout_hidden_paths",
                *[str(p) for p in rollout_hidden_paths],
                "--weak_rollout_index_paths",
                *[str(p) for p in rollout_index_paths],
                "--output_dir",
                str(out_dir),
                "--prompt_hidden_component",
                prompt_component,
                "--prompt_layer_index",
                str(args.prompt_layer_index),
                "--rollout_component",
                rollout_component,
                "--rollout_layer_index",
                str(args.rollout_layer_index),
                "--rollout_pool_mode",
                "mean",
                "--feature_mode",
                "prompt_plus_rollout",
                "--prompt_hidden_pca_dim",
                str(args.prompt_hidden_pca_dim),
                "--rollout_hidden_pca_dim",
                str(args.rollout_hidden_pca_dim),
                "--model_family",
                args.model_family,
                "--train_target_mode",
                args.train_target_mode,
                "--selection_metric",
                args.selection_metric,
                "--single_rollout_strategy",
                args.single_rollout_strategy,
                "--alphas",
                *[str(v) for v in args.alphas],
            ]
            if args.prompt_feature_keys:
                train_cmd += ["--prompt_feature_keys", *args.prompt_feature_keys]
            if args.rollout_scalar_keys:
                train_cmd += ["--rollout_scalar_keys", *args.rollout_scalar_keys]
            if args.derived_rollout_scalar_keys:
                train_cmd += [
                    "--derived_rollout_scalar_keys",
                    *args.derived_rollout_scalar_keys,
                ]
            if args.extra_rollout_scalar_field_paths:
                train_cmd += [
                    "--extra_rollout_scalar_field_paths",
                    *args.extra_rollout_scalar_field_paths,
                ]
            if args.allow_missing_entropy_scalars:
                train_cmd.append("--allow_missing_entropy_scalars")

            _run_cmd(train_cmd, args.dry_run)

            model_path = out_dir / "model.joblib"
            if not model_path.exists():
                # if dry-run, we don't require it
                if not args.dry_run:
                    print(f"[warn] training did not produce model for {base_run_name}")
                continue

            eval_out_dir = out_dir / "eval"
            eval_out_dir.mkdir(parents=True, exist_ok=True)
            eval_cmd: list[str] = [
                str(args.python),
                "-u",
                "-m",
                "classifer_training.eval_single_rollout_hidden_transfer",
                "--model_path",
                str(model_path),
                "--labels_path",
                str(labels_path),
                "--prompt_hidden_paths",
                *[str(p) for p in prompt_hidden_paths],
                "--prompt_index_paths",
                *[str(p) for p in prompt_index_paths],
                "--eval_rollout_hidden_paths",
                *[str(p) for p in rollout_hidden_paths],
                "--eval_rollout_index_paths",
                *[str(p) for p in rollout_index_paths],
                "--output_dir",
                str(eval_out_dir),
                "--allowed_splits",
                "validation",
                "test",
                "--prompt_hidden_component_override",
                prompt_component,
                "--rollout_component_override",
                rollout_component,
            ]
            if args.allow_missing_entropy_scalars:
                eval_cmd.append("--allow_missing_entropy_scalars")

            _run_cmd(eval_cmd, args.dry_run)

            count += 1
            print(f"[done] {base_run_name}")
        except Exception as e:
            print(f"[error] {manifest_path.name}/{dataset_key}: {e}")


if __name__ == "__main__":
    main()
