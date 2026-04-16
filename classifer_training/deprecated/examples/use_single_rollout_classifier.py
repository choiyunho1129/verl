from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from verl.utils.single_rollout_classifier import load_single_rollout_difficulty_classifier


def _load_prompt_hidden(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if path.suffix.lower() == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32).reshape(-1)
    raise ValueError(f"Unsupported prompt hidden format for {path}. Expected .npy or .json")


def _load_single_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
        raise ValueError(f"No records found in {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if not payload:
                raise ValueError(f"No records found in {path}")
            return payload[0]
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Unsupported rollout file format for {path}. Expected .json or .jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal runner for the compatibility single-rollout difficulty classifier."
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--prompt_hidden_path", type=Path, required=True)
    parser.add_argument("--rollout_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = load_single_rollout_difficulty_classifier(args.model_path)
    prompt_hidden = _load_prompt_hidden(args.prompt_hidden_path)
    rollout = _load_single_record(args.rollout_path)

    difficulty = classifier.predict_difficulty(prompt_hidden=prompt_hidden, rollout_record=rollout)
    payload: dict[str, Any] = {
        "predicted_difficulty": float(difficulty),
    }

    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
