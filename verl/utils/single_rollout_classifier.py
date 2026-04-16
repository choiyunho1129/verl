from __future__ import annotations

"""Deprecated compatibility shim.

New code should import from `verl.utils.single_trajectory_estimator`.
"""

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .single_trajectory_estimator import SingleTrajectoryEstimator, load_single_trajectory_estimator


class SingleRolloutValueEstimator:
    """Thin wrapper around `SingleTrajectoryEstimator` for older imports."""

    def __init__(self, estimator: SingleTrajectoryEstimator) -> None:
        self._estimator = estimator
        self.config = estimator.config

    @classmethod
    def load(cls, model_path: str | Path) -> "SingleRolloutValueEstimator":
        return cls(load_single_trajectory_estimator(model_path))

    def predict_value(
        self,
        *,
        prompt_hidden: np.ndarray | Sequence[float],
        response_hidden: np.ndarray | Sequence[float],
        response_features: dict[str, Any],
    ) -> float:
        return self._estimator.predict_value(
            prompt_hidden=prompt_hidden,
            response_hidden=response_hidden,
            response_features=response_features,
        )


class SingleRolloutDifficultyClassifier(SingleRolloutValueEstimator):
    """Backward-compatible alias.

    This class exists only so older imports do not break.
    """

    def predict_difficulty(
        self,
        *,
        prompt_hidden: np.ndarray | Sequence[float],
        response_hidden: np.ndarray | Sequence[float],
        response_features: dict[str, Any],
    ) -> float:
        value = self.predict_value(
            prompt_hidden=prompt_hidden,
            response_hidden=response_hidden,
            response_features=response_features,
        )
        return float(1.0 - value)


def load_single_rollout_value_estimator(model_path: str | Path) -> SingleRolloutValueEstimator:
    return SingleRolloutValueEstimator.load(model_path)


def load_single_rollout_difficulty_classifier(model_path: str | Path) -> SingleRolloutDifficultyClassifier:
    return SingleRolloutDifficultyClassifier.load(model_path)


__all__ = [
    "SingleRolloutDifficultyClassifier",
    "SingleRolloutValueEstimator",
    "load_single_rollout_difficulty_classifier",
    "load_single_rollout_value_estimator",
]
