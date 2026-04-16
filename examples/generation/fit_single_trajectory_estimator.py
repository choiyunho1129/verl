from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from verl.utils.single_trajectory_estimator import (
    FeatureBuilderConfig,
    SingleTrajectoryEstimatorFitConfig,
    fit_single_trajectory_estimator,
    save_single_trajectory_estimator_bundle,
)


SUPPORT_DIR = (
    Path(__file__).resolve().parents[2]
    / "verl"
    / "utils"
    / "single_trajectory_estimator_support"
)


def _load_json_or_jsonl(path: Path) -> Any:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file format for {path}. Expected .json or .jsonl")


def _load_numeric_rows(path: Path) -> list[np.ndarray]:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".npy":
        array = np.asarray(np.load(path), dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return [np.asarray(row, dtype=np.float32).reshape(-1) for row in array]
    payload = _load_json_or_jsonl(path)
    if isinstance(payload, dict):
        if "rows" in payload:
            payload = payload["rows"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list-like payload in {path}")
    return [np.asarray(row, dtype=np.float32).reshape(-1) for row in payload]


def _load_targets(path: Path) -> list[float]:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1).tolist()
    payload = _load_json_or_jsonl(path)
    if isinstance(payload, dict):
        if "targets" in payload:
            payload = payload["targets"]
        else:
            payload = list(payload.values())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list-like payload in {path}")
    return [float(value) for value in payload]


def _load_response_feature_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_or_jsonl(path)
    if isinstance(payload, dict):
        if "records" in payload:
            payload = payload["records"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list-like payload in {path}")
    records: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"Response feature rows must be objects. Got {type(row)!r}")
        records.append(row)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic trainer for the single-trajectory estimator bundle."
    )
    parser.add_argument("--prompt_hidden_path", type=Path, required=True)
    parser.add_argument("--response_hidden_path", "--trajectory_hidden_path", type=Path, required=True)
    parser.add_argument("--response_features_path", "--trajectory_features_path", type=Path, required=True)
    parser.add_argument("--targets_path", type=Path, required=True)
    parser.add_argument("--output_model_path", type=Path, required=True)
    parser.add_argument("--output_config_path", type=Path)
    parser.add_argument(
        "--feature_builder_config_path",
        type=Path,
        default=SUPPORT_DIR / "default_feature_builder_config.json",
    )
    parser.add_argument(
        "--estimator_fit_config_path",
        type=Path,
        default=SUPPORT_DIR / "default_estimator_fit_config.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_hidden_rows = _load_numeric_rows(args.prompt_hidden_path)
    response_hidden_rows = _load_numeric_rows(args.response_hidden_path)
    response_feature_rows = _load_response_feature_rows(args.response_features_path)
    targets = _load_targets(args.targets_path)

    feature_builder_payload = _load_json_or_jsonl(args.feature_builder_config_path)
    if not isinstance(feature_builder_payload, dict):
        raise ValueError("feature_builder_config_path must point to a single JSON object.")
    feature_builder_config = FeatureBuilderConfig.from_dict(feature_builder_payload)

    fit_payload = _load_json_or_jsonl(args.estimator_fit_config_path)
    if not isinstance(fit_payload, dict):
        raise ValueError("estimator_fit_config_path must point to a single JSON object.")
    fit_config = SingleTrajectoryEstimatorFitConfig(
        prompt_hidden_pca_dim=int(fit_payload.get("prompt_hidden_pca_dim", 0)),
        response_hidden_pca_dim=int(
            fit_payload.get("response_hidden_pca_dim", fit_payload.get("trajectory_hidden_pca_dim", 0))
        ),
        alpha=float(fit_payload.get("alpha", 300.0)),
        random_seed=int(fit_payload.get("random_seed", 42)),
        clip_min=float(fit_payload.get("clip_min", 0.0)),
        clip_max=float(fit_payload.get("clip_max", 1.0)),
    )
    bundle = fit_single_trajectory_estimator(
        prompt_hidden_rows=prompt_hidden_rows,
        response_hidden_rows=response_hidden_rows,
        response_feature_rows=response_feature_rows,
        targets=targets,
        feature_builder_config=feature_builder_config,
        fit_config=fit_config,
    )
    save_single_trajectory_estimator_bundle(bundle, args.output_model_path)

    output_config_path = args.output_config_path
    if output_config_path is None:
        output_config_path = args.output_model_path.expanduser().resolve().with_suffix(".config.json")
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(json.dumps(bundle["config"], indent=2), encoding="utf-8")

    summary = {
        "output_model_path": str(args.output_model_path.expanduser().resolve()),
        "output_config_path": str(output_config_path.resolve()),
        "feature_builder_config_path": None
        if args.feature_builder_config_path is None
        else str(args.feature_builder_config_path.expanduser().resolve()),
        "estimator_fit_config_path": None
        if args.estimator_fit_config_path is None
        else str(args.estimator_fit_config_path.expanduser().resolve()),
        "num_rows": int(len(targets)),
        "feature_dim": int(bundle["config"]["model"]["feature_dim"]),
        "prompt_hidden_projection": bundle["config"]["prompt_hidden_projection"],
        "response_hidden_projection": bundle["config"]["response_hidden_projection"],
        "response_feature_keys": list(bundle["config"]["response_feature_keys"]),
        "derived_response_feature_keys": list(bundle["config"]["derived_response_feature_keys"]),
        "alpha": float(bundle["config"]["model"]["alpha"]),
        "feature_builder_scalar_keys": list(feature_builder_config.rollout_scalars.scalar_keys),
        "feature_builder_derived_scalar_keys": list(feature_builder_config.rollout_scalars.derived_scalar_keys),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
