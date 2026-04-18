from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib


def _build_support_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "alpha": model_config.get("alpha"),
        "clip_min": model_config.get("clip_min", 0.0),
        "clip_max": model_config.get("clip_max", 1.0),
        "feature_dim": model_config.get("feature_dim"),
    }


def _patch_config(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    prompt_cfg = config.get("prompt")
    response_cfg = config.get("response")
    if isinstance(prompt_cfg, dict) and "hidden_projection" in prompt_cfg:
        if "prompt_hidden_projection" not in config:
            config["prompt_hidden_projection"] = dict(prompt_cfg["hidden_projection"])
            changed = True
    if isinstance(response_cfg, dict) and "hidden_projection" in response_cfg:
        if "response_hidden_projection" not in config:
            config["response_hidden_projection"] = dict(response_cfg["hidden_projection"])
            changed = True
        if "response_feature_keys" not in config:
            config["response_feature_keys"] = list(response_cfg.get("scalar_keys", []))
            changed = True
        if "derived_response_feature_keys" not in config:
            config["derived_response_feature_keys"] = list(response_cfg.get("derived_scalar_keys", []))
            changed = True

    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        required_model_keys = {"alpha", "clip_min", "clip_max", "feature_dim"}
        if set(model_cfg.keys()) != required_model_keys:
            if "model_full" not in config:
                config["model_full"] = dict(model_cfg)
            config["model"] = _build_support_model_config(model_cfg)
            changed = True
    return config, changed


def _patch_bundle(bundle: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    config = bundle.get("config")
    if isinstance(config, dict):
        bundle["config"], config_changed = _patch_config(config)
        changed = changed or config_changed

    rollout_hidden_pca = bundle.get("rollout_hidden_pca")
    for alias_key in ("response_hidden_pca", "trajectory_hidden_pca", "think_end_hidden_pca"):
        if alias_key not in bundle and "rollout_hidden_pca" in bundle:
            bundle[alias_key] = rollout_hidden_pca
            changed = True

    if bundle.get("bundle_version", 1) < 2:
        bundle["bundle_version"] = 2
        changed = True
    return bundle, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch trained estimator bundles in-place so they are loadable by single_trajectory_estimator_support.")
    parser.add_argument("--model_paths", nargs="+", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patched: list[dict[str, Any]] = []
    for model_path in [path.expanduser().resolve() for path in args.model_paths]:
        bundle = joblib.load(model_path)
        bundle, changed = _patch_bundle(bundle)
        if changed:
            joblib.dump(bundle, model_path)
            estimator_config_path = model_path.with_name("estimator_config.json")
            if estimator_config_path.exists() and isinstance(bundle.get("config"), dict):
                estimator_config_path.write_text(json.dumps(bundle["config"], indent=2), encoding="utf-8")
        patched.append({"model_path": str(model_path), "changed": bool(changed)})
    print(json.dumps({"patched": patched}, indent=2), flush=True)


if __name__ == "__main__":
    main()
