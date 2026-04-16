from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from verl.utils.single_trajectory_estimator import load_single_trajectory_estimator


def _load_prompt_hidden(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if path.suffix.lower() == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32).reshape(-1)
    raise ValueError(f"Unsupported prompt hidden format for {path}. Expected .npy or .json")


def _load_single_feature_row(path: Path) -> dict[str, Any]:
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
    raise ValueError(f"Unsupported feature file format for {path}. Expected .json or .jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal standalone runner for the prompt+single-trajectory estimator.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--prompt_hidden_path", type=Path, required=True)
    parser.add_argument("--response_hidden_path", "--trajectory_hidden_path", type=Path, required=True)
    parser.add_argument("--response_features_path", "--trajectory_features_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimator = load_single_trajectory_estimator(args.model_path)
    prompt_hidden = _load_prompt_hidden(args.prompt_hidden_path)
    response_hidden = _load_prompt_hidden(args.response_hidden_path)
    response_features = _load_single_feature_row(args.response_features_path)

    value = estimator.predict_value(
        prompt_hidden=prompt_hidden,
        response_hidden=response_hidden,
        response_features=response_features,
    )
    payload = {
        "predicted_value": float(value),
        "config": estimator.config.to_dict(),
    }
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
