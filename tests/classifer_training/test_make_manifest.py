from __future__ import annotations

import json
from pathlib import Path

from classifer_training.make_manifest import main as make_manifest_main


def test_make_manifest_builds_expected_paths(tmp_path: Path) -> None:
    output_path = tmp_path / "manifest.json"
    make_manifest_main(
        [
            "--model_name",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "--datasets",
            "deepscaler",
            "dapo_math_17k",
            "--hidden_root",
            str(tmp_path / "hidden"),
            "--index_root",
            str(tmp_path / "index"),
            "--labels_root",
            str(tmp_path / "labels"),
            "--output_path",
            str(output_path),
        ]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in payload["datasets"]] == ["deepscaler", "dapo_math_17k"]
    assert payload["datasets"][0]["hidden_states_path"].endswith(
        "/hidden/deepscaler/deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B/hidden_states.pt"
    )
    assert payload["datasets"][0]["index_path"].endswith(
        "/index/deepscaler/deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B/index.jsonl"
    )
    assert payload["datasets"][1]["labels_path"].endswith(
        "/labels/dapo_math_17k/deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B/sampling_labels.jsonl"
    )
