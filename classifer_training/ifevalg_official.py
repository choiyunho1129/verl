from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_OPEN_INSTRUCT_ROOT = Path(__file__).resolve().parent / "external" / "open-instruct"
_INSTRUCTION_DICT_CACHE: dict[str, dict[str, type]] = {}


def ensure_open_instruct_on_path(open_instruct_root: Path | str | None = None) -> Path:
    root = Path(
        open_instruct_root
        or os.environ.get("OPEN_INSTRUCT_ROOT", "")
        or DEFAULT_OPEN_INSTRUCT_ROOT
    ).expanduser().resolve()
    if not (root / "open_instruct" / "IFEvalG" / "instructions_registry.py").exists():
        raise FileNotFoundError(
            "Official open-instruct IFEvalG implementation was not found. "
            f"Expected {root / 'open_instruct' / 'IFEvalG' / 'instructions_registry.py'}. "
            "Clone https://github.com/allenai/open-instruct.git or set OPEN_INSTRUCT_ROOT."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_instruction_dict(open_instruct_root: Path | str | None = None) -> dict[str, type]:
    root = ensure_open_instruct_on_path(open_instruct_root)
    cache_key = str(root)
    if cache_key in _INSTRUCTION_DICT_CACHE:
        return _INSTRUCTION_DICT_CACHE[cache_key]
    from open_instruct.IFEvalG import instructions_registry  # type: ignore

    _INSTRUCTION_DICT_CACHE[cache_key] = dict(instructions_registry.INSTRUCTION_DICT)
    return _INSTRUCTION_DICT_CACHE[cache_key]


def remove_thinking_section(prediction: str) -> str:
    # Mirrors open_instruct.ground_truth_utils.remove_thinking_section without
    # importing the whole RL stack and its heavy optional dependencies.
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def parse_ground_truth_label(label: Any) -> list[dict[str, Any]]:
    if isinstance(label, list):
        parsed = label
    elif isinstance(label, dict):
        parsed = [label]
    elif isinstance(label, str):
        stripped = label.strip()
        if not stripped:
            raise ValueError("Empty IFEvalG ground_truth label.")
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = json.loads(stripped)
    else:
        raise TypeError(f"Unsupported IFEvalG ground_truth type: {type(label)!r}")

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise TypeError(f"IFEvalG ground_truth must parse to a list or dict, got {type(parsed)!r}.")

    normalized: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, str):
            item = json.loads(item)
        if not isinstance(item, dict):
            raise TypeError(f"Each IFEvalG constraint entry must be a dict, got {type(item)!r}.")
        normalized.append(item)
    return normalized


def _sanitize_kwargs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"IFEvalG kwargs entries must be dicts or None, got {type(value)!r}.")
    return {key: val for key, val in value.items() if val is not None}


def evaluate_ifevalg_response(
    response: str,
    ground_truth: Any,
    *,
    open_instruct_root: Path | str | None = None,
) -> dict[str, Any]:
    instruction_dict = load_instruction_dict(open_instruct_root)
    answer = remove_thinking_section(str(response))
    specs = parse_ground_truth_label(ground_truth)
    per_instruction: list[dict[str, Any]] = []

    if not answer:
        return {
            "score": 0.0,
            "follow_all": False,
            "num_instructions": 0,
            "num_followed": 0,
            "answer": answer,
            "per_instruction": per_instruction,
        }

    for spec in specs:
        instruction_ids = spec.get("instruction_id")
        kwargs_list = spec.get("kwargs")
        if not isinstance(instruction_ids, list) or not isinstance(kwargs_list, list):
            raise ValueError(f"Invalid IFEvalG spec; expected instruction_id and kwargs lists: {spec!r}")
        if len(instruction_ids) != len(kwargs_list):
            raise ValueError(
                f"IFEvalG instruction_id/kwargs length mismatch: {len(instruction_ids)} vs {len(kwargs_list)}"
            )

        for instruction_id, raw_kwargs in zip(instruction_ids, kwargs_list):
            instruction_id = str(instruction_id)
            if instruction_id not in instruction_dict:
                raise KeyError(f"Unsupported IFEvalG instruction_id: {instruction_id}")
            kwargs = _sanitize_kwargs(raw_kwargs)
            instruction_cls = instruction_dict[instruction_id]
            instruction = instruction_cls(instruction_id)
            instruction.build_description(**kwargs)
            followed = bool(instruction.check_following(answer))
            per_instruction.append(
                {
                    "instruction_id": instruction_id,
                    "kwargs": kwargs,
                    "followed": followed,
                }
            )

    num_instructions = len(per_instruction)
    num_followed = sum(1 for item in per_instruction if item["followed"])
    score = float(num_followed / num_instructions) if num_instructions else 0.0
    return {
        "score": score,
        "follow_all": bool(num_instructions and num_followed == num_instructions),
        "num_instructions": int(num_instructions),
        "num_followed": int(num_followed),
        "answer": answer,
        "per_instruction": per_instruction,
    }


def supported_instruction_ids(ground_truth: Any, *, open_instruct_root: Path | str | None = None) -> tuple[bool, list[str]]:
    instruction_dict = load_instruction_dict(open_instruct_root)
    missing: list[str] = []
    for spec in parse_ground_truth_label(ground_truth):
        instruction_ids = spec.get("instruction_id", [])
        if not isinstance(instruction_ids, list):
            missing.append("<invalid_instruction_id_field>")
            continue
        missing.extend(str(item) for item in instruction_ids if str(item) not in instruction_dict)
    return not missing, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify one response with the official open-instruct IFEvalG checks.")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--open-instruct-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_ifevalg_response(
        args.response,
        args.ground_truth,
        open_instruct_root=args.open_instruct_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
