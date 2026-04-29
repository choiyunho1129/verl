from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from classifer_training.utils import coerce_float
from verl.utils.reward_score.math_dapo import compute_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    ROOT / "classifer_training/artifacts/rollout_index/spo_temp1_subset0to4/Qwen_Qwen3-4B"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "classifer_training/artifacts/rollout_index/spo_temp1_subset0to4/Qwen_Qwen3-4B_dapo_score"
)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _is_correct_from_score_fields(row: dict[str, Any]) -> bool:
    reward = coerce_float(row.get("reward"))
    score = coerce_float(row.get("score"))
    return bool((reward is not None and reward >= 1.0) or (score is not None and score >= 1.0))


def _choose_solution_text(row: dict[str, Any], preferred_field: str) -> str:
    if preferred_field in {"generated_text_before_think", "pre_think", "before_think"}:
        generated_text = row.get("generated_text")
        if generated_text is not None and str(generated_text).strip():
            return str(generated_text).split("</think>", maxsplit=1)[0]

    fields = [
        preferred_field,
        "generated_text",
        "response",
        "solution",
        "answer_content",
        "reasoning_content",
    ]
    seen: set[str] = set()
    for field in fields:
        if not field or field in seen:
            continue
        seen.add(field)
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _rescore_row(
    row: dict[str, Any],
    *,
    solution_field: str,
    ground_truth_field: str,
    strict_box_verify: bool,
    update_score_fields: bool,
) -> tuple[dict[str, Any], bool, bool, str | None]:
    previous_correct = _is_correct_from_score_fields(row)
    solution = _choose_solution_text(row, solution_field)
    ground_truth = row.get(ground_truth_field)
    error = None

    try:
        result = compute_score(
            solution_str=solution,
            ground_truth="" if ground_truth is None else str(ground_truth),
            strict_box_verify=strict_box_verify,
        )
        dapo_score = float(result["score"])
        dapo_acc = bool(result["acc"])
        dapo_pred = _jsonable(result.get("pred"))
    except Exception as exc:  # Keep long rescoring jobs from dying on one malformed row.
        dapo_score = -1.0
        dapo_acc = False
        dapo_pred = None
        error = f"{type(exc).__name__}: {exc}"

    updated = dict(row)
    updated.setdefault("legacy_score", row.get("score"))
    updated.setdefault("legacy_reward", row.get("reward"))
    updated.setdefault("legacy_pred", row.get("pred"))
    updated["dapo_score"] = dapo_score
    updated["dapo_correctness"] = 1.0 if dapo_acc else 0.0
    updated["dapo_acc"] = dapo_acc
    updated["dapo_pred"] = dapo_pred
    updated["dapo_label_source"] = "verl.utils.reward_score.math_dapo.compute_score"
    updated["dapo_solution_field"] = solution_field
    updated["dapo_ground_truth_field"] = ground_truth_field
    updated["dapo_strict_box_verify"] = bool(strict_box_verify)
    if error is not None:
        updated["dapo_error"] = error

    if update_score_fields:
        updated["score"] = dapo_score
        updated["reward"] = dapo_score
        updated["pred"] = dapo_pred
        updated["score_source"] = "math_dapo.compute_score"

    return updated, previous_correct, dapo_acc, error


def _rescore_file(
    input_path: Path,
    output_path: Path,
    *,
    solution_field: str,
    ground_truth_field: str,
    strict_box_verify: bool,
    update_score_fields: bool,
    max_rows: int | None,
    progress_interval: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_rows": 0,
        "legacy_correct": 0,
        "dapo_correct": 0,
        "changed_correctness": 0,
        "errors": 0,
    }
    started_at = time.time()
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_idx, line in enumerate(src):
            if max_rows is not None and line_idx >= max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            updated, previous_correct, dapo_correct, error = _rescore_row(
                row,
                solution_field=solution_field,
                ground_truth_field=ground_truth_field,
                strict_box_verify=strict_box_verify,
                update_score_fields=update_score_fields,
            )
            dst.write(json.dumps(updated, ensure_ascii=False) + "\n")
            stats["num_rows"] += 1
            stats["legacy_correct"] += int(previous_correct)
            stats["dapo_correct"] += int(dapo_correct)
            stats["changed_correctness"] += int(previous_correct != dapo_correct)
            stats["errors"] += int(error is not None)
            if progress_interval > 0 and stats["num_rows"] % progress_interval == 0:
                elapsed = max(time.time() - started_at, 1e-6)
                print(
                    json.dumps(
                        {
                            "event": "rescore_progress",
                            "file": input_path.name,
                            "num_rows": stats["num_rows"],
                            "rows_per_sec": stats["num_rows"] / elapsed,
                            "legacy_correct": stats["legacy_correct"],
                            "dapo_correct": stats["dapo_correct"],
                            "changed_correctness": stats["changed_correctness"],
                            "errors": stats["errors"],
                        }
                    ),
                    flush=True,
                )
    stats["elapsed_sec"] = time.time() - started_at
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score SPO rollout index JSONL files with verl.utils.reward_score.math_dapo.compute_score."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="rollout_index.shard*.jsonl")
    parser.add_argument(
        "--solution-field",
        default="generated_text",
        help="Response text field. Pseudo-field generated_text_before_think scores generated_text before </think>.",
    )
    parser.add_argument("--ground-truth-field", default="ground_truth")
    parser.add_argument("--strict-box-verify", action="store_true")
    parser.add_argument("--no-update-score-fields", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("Refusing to write DAPO rescored rollout index files in-place.")

    input_paths = sorted(input_dir.glob(args.pattern))
    if not input_paths:
        single_path = input_dir / "rollout_index.jsonl"
        if single_path.exists():
            input_paths = [single_path]
    if not input_paths:
        raise FileNotFoundError(f"No rollout index files found under {input_dir} with pattern {args.pattern!r}.")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Use --overwrite or choose a new path.")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_stats = []
    for input_path in input_paths:
        output_path = output_dir / input_path.name
        file_stats.append(
            _rescore_file(
                input_path,
                output_path,
                solution_field=args.solution_field,
                ground_truth_field=args.ground_truth_field,
                strict_box_verify=args.strict_box_verify,
                update_score_fields=not args.no_update_score_fields,
                max_rows=args.max_rows_per_file,
                progress_interval=args.progress_interval,
            )
        )

    totals = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "label_source": "verl.utils.reward_score.math_dapo.compute_score",
        "solution_field": args.solution_field,
        "ground_truth_field": args.ground_truth_field,
        "strict_box_verify": bool(args.strict_box_verify),
        "updated_score_fields": not args.no_update_score_fields,
        "files": file_stats,
        "num_rows": int(sum(row["num_rows"] for row in file_stats)),
        "legacy_correct": int(sum(row["legacy_correct"] for row in file_stats)),
        "dapo_correct": int(sum(row["dapo_correct"] for row in file_stats)),
        "changed_correctness": int(sum(row["changed_correctness"] for row in file_stats)),
        "errors": int(sum(row["errors"] for row in file_stats)),
    }
    (output_dir / "rescore_manifest.json").write_text(json.dumps(totals, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "rescore_finished", **totals}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
