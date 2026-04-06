from __future__ import annotations

import numpy as np

from classifer_training.data import ExampleRecord
from classifer_training.features import FeatureExtractionConfig, build_feature_matrix


def _make_example(task_id: str, prompt: list[float], response: list[float]) -> ExampleRecord:
    return ExampleRecord(
        dataset_name="toyset",
        task_id=task_id,
        split="train",
        components={
            "prompt_hidden": [np.asarray(prompt, dtype=np.float32)],
            "response_hidden": [np.asarray(response, dtype=np.float32)],
        },
        index_row={
            "dataset_name": "toyset",
            "task_id": task_id,
            "split": "train",
            "rollout_features": {
                "output_length": 10.0,
            },
        },
        label_row={
            "dataset_name": "toyset",
            "task_id": task_id,
            "difficulty": 0.5,
        },
    )


def test_prompt_response_delta_prod_feature_set_builds_expected_shape() -> None:
    examples = [
        _make_example("a", [1.0, 2.0], [3.0, 5.0]),
        _make_example("b", [2.0, 4.0], [4.0, 8.0]),
    ]

    X, feature_names, metadata = build_feature_matrix(
        examples,
        FeatureExtractionConfig(
            components=["prompt_hidden", "response_hidden"],
            layers="all",
            component_pooling="concat",
            extra_feature_paths=["index.rollout_features.output_length"],
            engineered_feature_set="prompt_response_delta_prod",
        ),
    )

    assert X.shape == (2, 18)
    assert len(feature_names) == 18
    assert feature_names[:2] == ["prompt_dim0", "prompt_dim1"]
    assert "product_dim0" in feature_names
    assert "index_rollout_features_output_length" in feature_names
    assert "prompt_response_cosine" in feature_names
    assert metadata[0]["task_id"] == "a"
