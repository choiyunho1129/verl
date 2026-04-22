from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_training.single_rollout_hidden_utils import (
    apply_prompt_hidden_pca,
    apply_rollout_hidden_pca,
    build_matrix,
    build_prompt_scalar_lookup,
    build_rollout_hidden_lookup,
    build_rollout_index_lookup,
    build_split_lookup,
    fit_prompt_hidden_pca,
    fit_rollout_hidden_pca,
    group_weak_rollouts,
    load_labels_by_task,
    load_prompt_hidden_lookup,
    prompt_mean_metrics,
    reg_metrics,
    save_diagnostics_plot,
    select_single_rollout,
    write_predictions,
)


class TwoHeadBinaryValueEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, *, C: float = 1.0, random_state: int = 42, max_iter: int = 2000) -> None:
        self.C = float(C)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.zero_head = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        self.one_head = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TwoHeadBinaryValueEstimator":
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        zero_targets = np.isclose(y_array, 0.0).astype(np.int32)
        one_targets = np.isclose(y_array, 1.0).astype(np.int32)
        self.zero_head.fit(x, zero_targets)
        self.one_head.fit(x, one_targets)
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        p_zero = self.zero_head.predict_proba(x)[:, 1]
        p_one = self.one_head.predict_proba(x)[:, 1]
        denom = p_zero + p_one
        pred = np.divide(p_one, denom, out=np.full_like(p_one, 0.5, dtype=np.float64), where=denom > 1e-8)
        return np.clip(pred, 0.0, 1.0)


class PromptResidualRidgeValueEstimator(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        prompt_feature_dim: int,
        prompt_alpha: float = 3000.0,
        residual_alpha: float = 30000.0,
        residual_scale: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.prompt_feature_dim = int(prompt_feature_dim)
        self.prompt_alpha = float(prompt_alpha)
        self.residual_alpha = float(residual_alpha)
        self.residual_scale = float(residual_scale)
        self.random_state = int(random_state)

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(x, dtype=np.float32)
        prompt_x = x_array[:, : self.prompt_feature_dim]
        residual_x = x_array[:, self.prompt_feature_dim :]
        if residual_x.shape[1] == 0:
            residual_x = prompt_x
        return prompt_x, residual_x

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PromptResidualRidgeValueEstimator":
        prompt_x, residual_x = self._split(x)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)

        self.prompt_scaler_ = StandardScaler()
        prompt_x_scaled = self.prompt_scaler_.fit_transform(prompt_x)
        self.prompt_model_ = Ridge(alpha=self.prompt_alpha, random_state=self.random_state)
        self.prompt_model_.fit(prompt_x_scaled, y_array)

        prompt_pred = np.clip(np.asarray(self.prompt_model_.predict(prompt_x_scaled), dtype=np.float32), 0.0, 1.0)
        residual_y = y_array - prompt_pred
        self.residual_scaler_ = StandardScaler()
        residual_x_scaled = self.residual_scaler_.fit_transform(residual_x)
        self.residual_model_ = Ridge(alpha=self.residual_alpha, random_state=self.random_state)
        self.residual_model_.fit(residual_x_scaled, residual_y)
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        prompt_x, residual_x = self._split(x)
        prompt_x_scaled = self.prompt_scaler_.transform(prompt_x)
        residual_x_scaled = self.residual_scaler_.transform(residual_x)
        prompt_pred = np.asarray(self.prompt_model_.predict(prompt_x_scaled), dtype=np.float32)
        residual_pred = np.asarray(self.residual_model_.predict(residual_x_scaled), dtype=np.float32)
        return np.clip(prompt_pred + self.residual_scale * residual_pred, 0.0, 1.0)


class PromptValuePriorResidualRidgeValueEstimator(RegressorMixin, BaseEstimator):
    uses_prompt_value_target = True

    def __init__(
        self,
        *,
        prompt_feature_dim: int,
        prompt_alpha: float = 3000.0,
        residual_alpha: float = 30000.0,
        residual_scale: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.prompt_feature_dim = int(prompt_feature_dim)
        self.prompt_alpha = float(prompt_alpha)
        self.residual_alpha = float(residual_alpha)
        self.residual_scale = float(residual_scale)
        self.random_state = int(random_state)

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(x, dtype=np.float32)
        prompt_x = x_array[:, : self.prompt_feature_dim]
        residual_x = x_array[:, self.prompt_feature_dim :]
        if residual_x.shape[1] == 0:
            residual_x = prompt_x
        return prompt_x, residual_x

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PromptValuePriorResidualRidgeValueEstimator":
        prompt_x, residual_x = self._split(x)
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim == 2 and y_array.shape[1] >= 2:
            row_target = y_array[:, 0].reshape(-1)
            prompt_target = y_array[:, 1].reshape(-1)
        else:
            row_target = y_array.reshape(-1)
            prompt_target = row_target

        self.prompt_scaler_ = StandardScaler()
        prompt_x_scaled = self.prompt_scaler_.fit_transform(prompt_x)
        self.prompt_model_ = Ridge(alpha=self.prompt_alpha, random_state=self.random_state)
        self.prompt_model_.fit(prompt_x_scaled, prompt_target)

        prompt_pred = np.clip(np.asarray(self.prompt_model_.predict(prompt_x_scaled), dtype=np.float32), 0.0, 1.0)
        residual_y = row_target - prompt_pred
        self.residual_scaler_ = StandardScaler()
        residual_x_scaled = self.residual_scaler_.fit_transform(residual_x)
        self.residual_model_ = Ridge(alpha=self.residual_alpha, random_state=self.random_state)
        self.residual_model_.fit(residual_x_scaled, residual_y)
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        prompt_x, residual_x = self._split(x)
        prompt_x_scaled = self.prompt_scaler_.transform(prompt_x)
        residual_x_scaled = self.residual_scaler_.transform(residual_x)
        prompt_pred = np.asarray(self.prompt_model_.predict(prompt_x_scaled), dtype=np.float32)
        residual_pred = np.asarray(self.residual_model_.predict(residual_x_scaled), dtype=np.float32)
        return np.clip(prompt_pred + self.residual_scale * residual_pred, 0.0, 1.0)


class PromptPriorBlendRidgeValueEstimator(RegressorMixin, BaseEstimator):
    uses_prompt_value_target = True

    def __init__(
        self,
        *,
        prompt_feature_dim: int,
        prompt_alpha: float = 3000.0,
        row_alpha: float = 10000.0,
        prompt_weight: float = 0.65,
        random_state: int = 42,
    ) -> None:
        self.prompt_feature_dim = int(prompt_feature_dim)
        self.prompt_alpha = float(prompt_alpha)
        self.row_alpha = float(row_alpha)
        self.prompt_weight = float(prompt_weight)
        self.random_state = int(random_state)

    def _split_prompt(self, x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=np.float32)
        return x_array[:, : self.prompt_feature_dim]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PromptPriorBlendRidgeValueEstimator":
        x_array = np.asarray(x, dtype=np.float32)
        prompt_x = self._split_prompt(x_array)
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim == 2 and y_array.shape[1] >= 2:
            row_target = y_array[:, 0].reshape(-1)
            prompt_target = y_array[:, 1].reshape(-1)
        else:
            row_target = y_array.reshape(-1)
            prompt_target = row_target

        self.prompt_scaler_ = StandardScaler()
        prompt_x_scaled = self.prompt_scaler_.fit_transform(prompt_x)
        self.prompt_model_ = Ridge(alpha=self.prompt_alpha, random_state=self.random_state)
        self.prompt_model_.fit(prompt_x_scaled, prompt_target)

        self.row_scaler_ = StandardScaler()
        x_scaled = self.row_scaler_.fit_transform(x_array)
        self.row_model_ = Ridge(alpha=self.row_alpha, random_state=self.random_state)
        self.row_model_.fit(x_scaled, row_target)
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=np.float32)
        prompt_x = self._split_prompt(x_array)
        prompt_pred = np.clip(
            np.asarray(self.prompt_model_.predict(self.prompt_scaler_.transform(prompt_x)), dtype=np.float32),
            0.0,
            1.0,
        )
        row_pred = np.clip(
            np.asarray(self.row_model_.predict(self.row_scaler_.transform(x_array)), dtype=np.float32),
            0.0,
            1.0,
        )
        return np.clip(self.prompt_weight * prompt_pred + (1.0 - self.prompt_weight) * row_pred, 0.0, 1.0)


class PromptTrajectoryMeanRidgeValueEstimator(RegressorMixin, BaseEstimator):
    uses_prompt_value_target = True

    def __init__(
        self,
        *,
        prompt_feature_dim: int,
        prompt_alpha: float = 30000.0,
        trajectory_alpha: float = 30000.0,
        random_state: int = 42,
    ) -> None:
        self.prompt_feature_dim = int(prompt_feature_dim)
        self.prompt_alpha = float(prompt_alpha)
        self.trajectory_alpha = float(trajectory_alpha)
        self.random_state = int(random_state)

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(x, dtype=np.float32)
        prompt_x = x_array[:, : self.prompt_feature_dim]
        trajectory_x = x_array[:, self.prompt_feature_dim :]
        if trajectory_x.shape[1] == 0:
            trajectory_x = prompt_x
        return prompt_x, trajectory_x

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PromptTrajectoryMeanRidgeValueEstimator":
        prompt_x, trajectory_x = self._split(x)
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim == 2 and y_array.shape[1] >= 2:
            trajectory_target = y_array[:, 0].reshape(-1)
            prompt_target = y_array[:, 1].reshape(-1)
        else:
            trajectory_target = y_array.reshape(-1)
            prompt_target = trajectory_target

        self.prompt_scaler_ = StandardScaler()
        prompt_x_scaled = self.prompt_scaler_.fit_transform(prompt_x)
        self.prompt_model_ = Ridge(alpha=self.prompt_alpha, random_state=self.random_state)
        self.prompt_model_.fit(prompt_x_scaled, prompt_target)

        self.trajectory_scaler_ = StandardScaler()
        trajectory_x_scaled = self.trajectory_scaler_.fit_transform(trajectory_x)
        self.trajectory_model_ = Ridge(alpha=self.trajectory_alpha, random_state=self.random_state)
        self.trajectory_model_.fit(trajectory_x_scaled, trajectory_target)
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        prompt_x, trajectory_x = self._split(x)
        prompt_pred = np.clip(
            np.asarray(self.prompt_model_.predict(self.prompt_scaler_.transform(prompt_x)), dtype=np.float32),
            0.0,
            1.0,
        )
        trajectory_pred = np.clip(
            np.asarray(
                self.trajectory_model_.predict(self.trajectory_scaler_.transform(trajectory_x)),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        return np.clip(0.5 * (prompt_pred + trajectory_pred), 0.0, 1.0)


class PromptTrajectoryScoreStackRidgeValueEstimator(RegressorMixin, BaseEstimator):
    uses_prompt_value_target = True

    def __init__(
        self,
        *,
        prompt_feature_dim: int,
        prompt_alpha: float = 30000.0,
        trajectory_alpha: float = 30000.0,
        combiner_alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.prompt_feature_dim = int(prompt_feature_dim)
        self.prompt_alpha = float(prompt_alpha)
        self.trajectory_alpha = float(trajectory_alpha)
        self.combiner_alpha = float(combiner_alpha)
        self.random_state = int(random_state)

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(x, dtype=np.float32)
        prompt_x = x_array[:, : self.prompt_feature_dim]
        trajectory_x = x_array[:, self.prompt_feature_dim :]
        if trajectory_x.shape[1] == 0:
            trajectory_x = prompt_x
        return prompt_x, trajectory_x

    @staticmethod
    def _score_features(prompt_pred: np.ndarray, trajectory_pred: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(prompt_pred, dtype=np.float32).reshape(-1),
                np.asarray(trajectory_pred, dtype=np.float32).reshape(-1),
            ],
            axis=1,
        ).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PromptTrajectoryScoreStackRidgeValueEstimator":
        prompt_x, trajectory_x = self._split(x)
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim == 2 and y_array.shape[1] >= 2:
            trajectory_target = y_array[:, 0].reshape(-1)
            prompt_target = y_array[:, 1].reshape(-1)
        else:
            trajectory_target = y_array.reshape(-1)
            prompt_target = trajectory_target

        self.prompt_scaler_ = StandardScaler()
        prompt_x_scaled = self.prompt_scaler_.fit_transform(prompt_x)
        self.prompt_model_ = Ridge(alpha=self.prompt_alpha, random_state=self.random_state)
        self.prompt_model_.fit(prompt_x_scaled, prompt_target)

        self.trajectory_scaler_ = StandardScaler()
        trajectory_x_scaled = self.trajectory_scaler_.fit_transform(trajectory_x)
        self.trajectory_model_ = Ridge(alpha=self.trajectory_alpha, random_state=self.random_state)
        self.trajectory_model_.fit(trajectory_x_scaled, trajectory_target)

        prompt_pred = np.clip(np.asarray(self.prompt_model_.predict(prompt_x_scaled), dtype=np.float32), 0.0, 1.0)
        trajectory_pred = np.clip(
            np.asarray(self.trajectory_model_.predict(trajectory_x_scaled), dtype=np.float32),
            0.0,
            1.0,
        )
        combiner_x = self._score_features(prompt_pred, trajectory_pred)
        self.combiner_scaler_ = StandardScaler()
        combiner_x_scaled = self.combiner_scaler_.fit_transform(combiner_x)
        self.combiner_model_ = Ridge(alpha=self.combiner_alpha, random_state=self.random_state)
        self.combiner_model_.fit(combiner_x_scaled, prompt_target)
        self.is_fitted_ = True
        return self

    def predict_components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        prompt_x, trajectory_x = self._split(x)
        prompt_pred = np.clip(
            np.asarray(self.prompt_model_.predict(self.prompt_scaler_.transform(prompt_x)), dtype=np.float32),
            0.0,
            1.0,
        )
        trajectory_pred = np.clip(
            np.asarray(
                self.trajectory_model_.predict(self.trajectory_scaler_.transform(trajectory_x)),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        combiner_x = self._score_features(prompt_pred, trajectory_pred)
        pred = np.clip(
            np.asarray(self.combiner_model_.predict(self.combiner_scaler_.transform(combiner_x)), dtype=np.float32),
            0.0,
            1.0,
        )
        return {
            "prompt_pred": prompt_pred,
            "trajectory_pred": trajectory_pred,
            "value_pred": pred,
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_components(x)["value_pred"]


class _ScoreCombinerMLP(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for _ in range(int(num_layers)):
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).reshape(-1)


class PromptTrajectoryScoreMLPValueEstimator(RegressorMixin, BaseEstimator):
    uses_prompt_value_target = True

    def __init__(
        self,
        *,
        prompt_feature_dim: int,
        prompt_alpha: float = 30000.0,
        trajectory_alpha: float = 30000.0,
        score_feature_mode: str = "scores_only",
        hidden_dim: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
        mid_sample_weight: float = 1.0,
        half_sample_weight: float = 1.0,
        max_epochs: int = 800,
        patience: int = 80,
        validation_fraction: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.prompt_feature_dim = int(prompt_feature_dim)
        self.prompt_alpha = float(prompt_alpha)
        self.trajectory_alpha = float(trajectory_alpha)
        self.score_feature_mode = str(score_feature_mode)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.mid_sample_weight = float(mid_sample_weight)
        self.half_sample_weight = float(half_sample_weight)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.random_state = int(random_state)

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(x, dtype=np.float32)
        prompt_x = x_array[:, : self.prompt_feature_dim]
        trajectory_x = x_array[:, self.prompt_feature_dim :]
        if trajectory_x.shape[1] == 0:
            trajectory_x = prompt_x
        return prompt_x, trajectory_x

    def _score_features(self, prompt_pred: np.ndarray, trajectory_pred: np.ndarray) -> np.ndarray:
        prompt = np.asarray(prompt_pred, dtype=np.float32).reshape(-1)
        trajectory = np.asarray(trajectory_pred, dtype=np.float32).reshape(-1)
        if self.score_feature_mode == "scores_only":
            pieces = [prompt, trajectory]
        elif self.score_feature_mode == "expanded":
            diff = trajectory - prompt
            pieces = [
                prompt,
                trajectory,
                diff,
                np.abs(diff),
                np.abs(prompt - 0.5),
                np.abs(trajectory - 0.5),
                prompt * trajectory,
                np.minimum(prompt, trajectory),
                np.maximum(prompt, trajectory),
            ]
        else:
            raise ValueError(f"Unsupported score_feature_mode: {self.score_feature_mode}")
        return np.stack(pieces, axis=1).astype(np.float32)

    def _fit_score_scaler(self, x: np.ndarray) -> np.ndarray:
        self.score_mean_ = np.mean(x, axis=0, keepdims=True).astype(np.float32)
        self.score_scale_ = np.std(x, axis=0, keepdims=True).astype(np.float32)
        self.score_scale_ = np.where(self.score_scale_ < 1e-6, 1.0, self.score_scale_).astype(np.float32)
        return ((x - self.score_mean_) / self.score_scale_).astype(np.float32)

    def _transform_score_features(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.score_mean_) / self.score_scale_).astype(np.float32)

    def _sample_weights(self, target: np.ndarray) -> np.ndarray:
        target_array = np.asarray(target, dtype=np.float32).reshape(-1)
        weights = np.ones_like(target_array, dtype=np.float32)
        mid_mask = (target_array >= 0.375) & (target_array <= 0.625)
        half_mask = np.isclose(target_array, 0.5)
        weights[mid_mask] = np.maximum(weights[mid_mask], self.mid_sample_weight)
        weights[half_mask] = np.maximum(weights[half_mask], self.half_sample_weight)
        return weights.astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PromptTrajectoryScoreMLPValueEstimator":
        prompt_x, trajectory_x = self._split(x)
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim == 2 and y_array.shape[1] >= 2:
            trajectory_target = y_array[:, 0].reshape(-1)
            prompt_target = y_array[:, 1].reshape(-1)
        else:
            trajectory_target = y_array.reshape(-1)
            prompt_target = trajectory_target

        self.prompt_scaler_ = StandardScaler()
        prompt_x_scaled = self.prompt_scaler_.fit_transform(prompt_x)
        self.prompt_model_ = Ridge(alpha=self.prompt_alpha, random_state=self.random_state)
        self.prompt_model_.fit(prompt_x_scaled, prompt_target)

        self.trajectory_scaler_ = StandardScaler()
        trajectory_x_scaled = self.trajectory_scaler_.fit_transform(trajectory_x)
        self.trajectory_model_ = Ridge(alpha=self.trajectory_alpha, random_state=self.random_state)
        self.trajectory_model_.fit(trajectory_x_scaled, trajectory_target)

        prompt_pred = np.clip(np.asarray(self.prompt_model_.predict(prompt_x_scaled), dtype=np.float32), 0.0, 1.0)
        trajectory_pred = np.clip(
            np.asarray(self.trajectory_model_.predict(trajectory_x_scaled), dtype=np.float32),
            0.0,
            1.0,
        )
        score_x = self._fit_score_scaler(self._score_features(prompt_pred, trajectory_pred))
        target = np.asarray(prompt_target, dtype=np.float32).reshape(-1)
        sample_weights = self._sample_weights(target)

        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(len(target))
        val_count = int(round(len(indices) * self.validation_fraction))
        if val_count <= 0 or len(indices) - val_count < 2:
            train_indices = indices
            val_indices = indices
        else:
            val_indices = indices[:val_count]
            train_indices = indices[val_count:]

        torch.manual_seed(self.random_state)
        self.combiner_model_ = _ScoreCombinerMLP(
            input_dim=int(score_x.shape[1]),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        torch.set_num_threads(1)
        optimizer = torch.optim.AdamW(
            self.combiner_model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_x = torch.as_tensor(score_x[train_indices], dtype=torch.float32)
        train_y = torch.as_tensor(target[train_indices], dtype=torch.float32)
        train_w = torch.as_tensor(sample_weights[train_indices], dtype=torch.float32)
        val_x = torch.as_tensor(score_x[val_indices], dtype=torch.float32)
        val_y = torch.as_tensor(target[val_indices], dtype=torch.float32)
        val_w = torch.as_tensor(sample_weights[val_indices], dtype=torch.float32)

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0
        epochs_run = 0
        min_delta = 1e-7
        for epoch in range(max(1, self.max_epochs)):
            self.combiner_model_.train()
            optimizer.zero_grad(set_to_none=True)
            train_pred = self.combiner_model_(train_x)
            train_loss = (torch.square(train_pred - train_y) * train_w).sum() / torch.clamp(train_w.sum(), min=1.0)
            train_loss.backward()
            optimizer.step()

            self.combiner_model_.eval()
            with torch.no_grad():
                val_pred = self.combiner_model_(val_x)
                val_loss = float(
                    ((torch.square(val_pred - val_y) * val_w).sum() / torch.clamp(val_w.sum(), min=1.0)).item()
                )
            epochs_run = epoch + 1
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in self.combiner_model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.combiner_model_.load_state_dict(best_state)
        self.num_epochs_trained_ = int(epochs_run)
        self.best_validation_loss_ = float(best_loss)
        self.is_fitted_ = True
        return self

    def predict_components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        prompt_x, trajectory_x = self._split(x)
        prompt_pred = np.clip(
            np.asarray(self.prompt_model_.predict(self.prompt_scaler_.transform(prompt_x)), dtype=np.float32),
            0.0,
            1.0,
        )
        trajectory_pred = np.clip(
            np.asarray(
                self.trajectory_model_.predict(self.trajectory_scaler_.transform(trajectory_x)),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        score_x = self._transform_score_features(self._score_features(prompt_pred, trajectory_pred))
        self.combiner_model_.eval()
        with torch.no_grad():
            pred = self.combiner_model_(torch.as_tensor(score_x, dtype=torch.float32)).cpu().numpy()
        return {
            "prompt_pred": prompt_pred,
            "trajectory_pred": trajectory_pred,
            "value_pred": np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_components(x)["value_pred"]


class PromptMiddleGatedRidgeValueEstimator(RegressorMixin, BaseEstimator):
    uses_prompt_value_target = True

    def __init__(
        self,
        *,
        prompt_feature_dim: int,
        prompt_alpha: float = 30000.0,
        trajectory_alpha: float = 30000.0,
        gate_min_prompt_weight: float = 0.1,
        gate_max_prompt_weight: float = 0.9,
        gate_width: float = 0.35,
        gate_power: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.prompt_feature_dim = int(prompt_feature_dim)
        self.prompt_alpha = float(prompt_alpha)
        self.trajectory_alpha = float(trajectory_alpha)
        self.gate_min_prompt_weight = float(gate_min_prompt_weight)
        self.gate_max_prompt_weight = float(gate_max_prompt_weight)
        self.gate_width = float(gate_width)
        self.gate_power = float(gate_power)
        self.random_state = int(random_state)

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(x, dtype=np.float32)
        prompt_x = x_array[:, : self.prompt_feature_dim]
        trajectory_x = x_array[:, self.prompt_feature_dim :]
        if trajectory_x.shape[1] == 0:
            trajectory_x = prompt_x
        return prompt_x, trajectory_x

    def _prompt_weight(self, prompt_pred: np.ndarray) -> np.ndarray:
        if self.gate_width <= 0.0:
            raise ValueError(f"gate_width must be positive, got {self.gate_width}.")
        if self.gate_power <= 0.0:
            raise ValueError(f"gate_power must be positive, got {self.gate_power}.")
        if self.gate_max_prompt_weight < self.gate_min_prompt_weight:
            raise ValueError(
                "gate_max_prompt_weight must be >= gate_min_prompt_weight, "
                f"got {self.gate_max_prompt_weight} < {self.gate_min_prompt_weight}."
            )

        prompt = np.clip(np.asarray(prompt_pred, dtype=np.float32).reshape(-1), 0.0, 1.0)
        middle_score = np.clip(1.0 - (np.abs(prompt - 0.5) / self.gate_width), 0.0, 1.0)
        middle_score = np.power(middle_score, self.gate_power)
        return (
            self.gate_min_prompt_weight
            + (self.gate_max_prompt_weight - self.gate_min_prompt_weight) * middle_score
        ).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PromptMiddleGatedRidgeValueEstimator":
        prompt_x, trajectory_x = self._split(x)
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim == 2 and y_array.shape[1] >= 2:
            trajectory_target = y_array[:, 0].reshape(-1)
            prompt_target = y_array[:, 1].reshape(-1)
        else:
            trajectory_target = y_array.reshape(-1)
            prompt_target = trajectory_target

        self.prompt_scaler_ = StandardScaler()
        prompt_x_scaled = self.prompt_scaler_.fit_transform(prompt_x)
        self.prompt_model_ = Ridge(alpha=self.prompt_alpha, random_state=self.random_state)
        self.prompt_model_.fit(prompt_x_scaled, prompt_target)

        self.trajectory_scaler_ = StandardScaler()
        trajectory_x_scaled = self.trajectory_scaler_.fit_transform(trajectory_x)
        self.trajectory_model_ = Ridge(alpha=self.trajectory_alpha, random_state=self.random_state)
        self.trajectory_model_.fit(trajectory_x_scaled, trajectory_target)
        self.is_fitted_ = True
        return self

    def predict_components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        prompt_x, trajectory_x = self._split(x)
        prompt_pred = np.clip(
            np.asarray(self.prompt_model_.predict(self.prompt_scaler_.transform(prompt_x)), dtype=np.float32),
            0.0,
            1.0,
        )
        trajectory_pred = np.clip(
            np.asarray(
                self.trajectory_model_.predict(self.trajectory_scaler_.transform(trajectory_x)),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        prompt_weight = self._prompt_weight(prompt_pred)
        return {
            "prompt_pred": prompt_pred,
            "trajectory_pred": trajectory_pred,
            "prompt_weight": prompt_weight,
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        components = self.predict_components(x)
        prompt_weight = components["prompt_weight"]
        pred = (
            prompt_weight * components["prompt_pred"]
            + (1.0 - prompt_weight) * components["trajectory_pred"]
        )
        return np.clip(pred, 0.0, 1.0)


class PromptTrajectoryStackedRidgeValueEstimator(RegressorMixin, BaseEstimator):
    uses_prompt_value_target = True

    def __init__(
        self,
        *,
        prompt_feature_dim: int,
        prompt_alpha: float = 30000.0,
        trajectory_alpha: float = 30000.0,
        combiner_alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.prompt_feature_dim = int(prompt_feature_dim)
        self.prompt_alpha = float(prompt_alpha)
        self.trajectory_alpha = float(trajectory_alpha)
        self.combiner_alpha = float(combiner_alpha)
        self.random_state = int(random_state)

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(x, dtype=np.float32)
        prompt_x = x_array[:, : self.prompt_feature_dim]
        trajectory_x = x_array[:, self.prompt_feature_dim :]
        if trajectory_x.shape[1] == 0:
            trajectory_x = prompt_x
        return prompt_x, trajectory_x

    @staticmethod
    def _score_features(prompt_pred: np.ndarray, trajectory_pred: np.ndarray) -> np.ndarray:
        prompt = np.asarray(prompt_pred, dtype=np.float32).reshape(-1)
        trajectory = np.asarray(trajectory_pred, dtype=np.float32).reshape(-1)
        diff = trajectory - prompt
        prompt_conf = np.abs(prompt - 0.5)
        trajectory_conf = np.abs(trajectory - 0.5)
        pieces = [
            prompt,
            trajectory,
            diff,
            np.abs(diff),
            prompt_conf,
            trajectory_conf,
            prompt * trajectory,
            prompt * prompt_conf,
            trajectory * trajectory_conf,
            np.minimum(prompt, trajectory),
            np.maximum(prompt, trajectory),
        ]
        return np.stack(pieces, axis=1).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PromptTrajectoryStackedRidgeValueEstimator":
        prompt_x, trajectory_x = self._split(x)
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim == 2 and y_array.shape[1] >= 2:
            trajectory_target = y_array[:, 0].reshape(-1)
            prompt_target = y_array[:, 1].reshape(-1)
        else:
            trajectory_target = y_array.reshape(-1)
            prompt_target = trajectory_target

        self.prompt_scaler_ = StandardScaler()
        prompt_x_scaled = self.prompt_scaler_.fit_transform(prompt_x)
        self.prompt_model_ = Ridge(alpha=self.prompt_alpha, random_state=self.random_state)
        self.prompt_model_.fit(prompt_x_scaled, prompt_target)

        self.trajectory_scaler_ = StandardScaler()
        trajectory_x_scaled = self.trajectory_scaler_.fit_transform(trajectory_x)
        self.trajectory_model_ = Ridge(alpha=self.trajectory_alpha, random_state=self.random_state)
        self.trajectory_model_.fit(trajectory_x_scaled, trajectory_target)

        prompt_pred = np.clip(np.asarray(self.prompt_model_.predict(prompt_x_scaled), dtype=np.float32), 0.0, 1.0)
        trajectory_pred = np.clip(
            np.asarray(self.trajectory_model_.predict(trajectory_x_scaled), dtype=np.float32),
            0.0,
            1.0,
        )
        combiner_x = self._score_features(prompt_pred, trajectory_pred)
        self.combiner_scaler_ = StandardScaler()
        combiner_x_scaled = self.combiner_scaler_.fit_transform(combiner_x)
        self.combiner_model_ = Ridge(alpha=self.combiner_alpha, random_state=self.random_state)
        self.combiner_model_.fit(combiner_x_scaled, prompt_target)
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        prompt_x, trajectory_x = self._split(x)
        prompt_pred = np.clip(
            np.asarray(self.prompt_model_.predict(self.prompt_scaler_.transform(prompt_x)), dtype=np.float32),
            0.0,
            1.0,
        )
        trajectory_pred = np.clip(
            np.asarray(
                self.trajectory_model_.predict(self.trajectory_scaler_.transform(trajectory_x)),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        combiner_x = self._score_features(prompt_pred, trajectory_pred)
        pred = self.combiner_model_.predict(self.combiner_scaler_.transform(combiner_x))
        return np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)


class LogitRidgeValueEstimator(RegressorMixin, BaseEstimator):
    def __init__(self, *, alpha: float = 3000.0, epsilon: float = 0.05, random_state: int = 42) -> None:
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.random_state = int(random_state)

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(value, dtype=np.float64), -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogitRidgeValueEstimator":
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        y_bounded = np.clip(y_array, self.epsilon, 1.0 - self.epsilon)
        target = np.log(y_bounded / (1.0 - y_bounded))
        self.model_ = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.model_.fit(x, target)
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        logit_pred = np.asarray(self.model_.predict(x), dtype=np.float64).reshape(-1)
        return self._sigmoid(logit_pred)


def _build_support_model_config(*, alpha: float | None, feature_dim: int) -> dict[str, float | int | None]:
    return {
        "alpha": float(alpha) if alpha is not None else None,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "feature_dim": int(feature_dim),
    }


def _add_support_compatibility(
    *,
    estimator_config: dict,
    rollout_hidden_pca,
    bundle: dict,
) -> tuple[dict, dict]:
    prompt_projection = dict(estimator_config["prompt"]["hidden_projection"])
    response_projection = dict(estimator_config["response"]["hidden_projection"])
    response_feature_keys = list(estimator_config["response"].get("scalar_keys", []))
    derived_response_feature_keys = list(estimator_config["response"].get("derived_scalar_keys", []))

    original_model_config = dict(estimator_config["model"])
    support_model_config = _build_support_model_config(
        alpha=original_model_config.get("alpha"),
        feature_dim=int(original_model_config["feature_dim"]),
    )

    estimator_config["model_full"] = original_model_config
    estimator_config["model"] = support_model_config
    estimator_config["prompt_hidden_projection"] = prompt_projection
    estimator_config["response_hidden_projection"] = response_projection
    estimator_config["response_feature_keys"] = response_feature_keys
    estimator_config["derived_response_feature_keys"] = derived_response_feature_keys

    bundle["bundle_version"] = 2
    bundle["response_hidden_pca"] = rollout_hidden_pca
    bundle["trajectory_hidden_pca"] = rollout_hidden_pca
    bundle["think_end_hidden_pca"] = rollout_hidden_pca
    return estimator_config, bundle


def _row_prediction_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for true_value, pred_value, meta in zip(y_true.tolist(), y_pred.tolist(), metadata_rows, strict=True):
        rows.append(
            {
                "task_id": str(meta["task_id"]),
                "split": str(meta["split"]),
                "value_true": float(true_value),
                "value_pred": float(pred_value),
                "rollout_row_index": int(meta.get("rollout_row_index", -1)),
                "sample_index": int(meta.get("sample_index", -1)),
                "num_rows": 1,
            }
        )
    return rows


def _write_row_predictions(
    output_path: Path,
    row_predictions: list[dict[str, Any]],
    labels_by_task: dict[str, dict[str, Any]],
) -> None:
    rows = []
    for row in row_predictions:
        label_row = labels_by_task[str(row["task_id"])]
        rows.append(
            {
                "task_id": str(row["task_id"]),
                "user_input": str(label_row.get("user_input", "")),
                "value_true": float(row["value_true"]),
                "value_pred": float(row["value_pred"]),
                "rollout_row_index": int(row.get("rollout_row_index", -1)),
                "sample_index": int(row.get("sample_index", -1)),
                "num_rows": 1,
            }
        )
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _apply_train_target_mode(rows: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode == "prompt_value":
        return rows, {"mode": mode}
    if mode != "other_rollout_correctness":
        raise ValueError(f"Unsupported train target mode: {mode}")

    train_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("split")) == "train":
            train_groups.setdefault(str(row["task_id"]), []).append(row)

    updated_rows: list[dict[str, Any]] = []
    changed_train_rows = 0
    for row in rows:
        if str(row.get("split")) != "train":
            updated_rows.append(row)
            continue

        group = train_groups[str(row["task_id"])]
        if len(group) < 2:
            raise ValueError(
                "other_rollout_correctness requires at least two selected train rollouts per prompt. "
                f"task_id={row['task_id']} has {len(group)}."
            )
        if row.get("rollout_correctness") is None:
            raise ValueError(f"Missing rollout_correctness for train task_id={row['task_id']}.")

        sibling_correctness = [
            float(other["rollout_correctness"])
            for other in group
            if not (
                str(other.get("run_dir", "")) == str(row.get("run_dir", ""))
                and int(other.get("rollout_row_index", -1)) == int(row.get("rollout_row_index", -1))
            )
        ]
        if not sibling_correctness:
            raise ValueError(f"No sibling rollout correctness available for task_id={row['task_id']}.")

        updated = dict(row)
        updated["prompt_value_true"] = float(row["value_true"])
        updated["value_true"] = float(np.mean(np.asarray(sibling_correctness, dtype=np.float32)))
        updated_rows.append(updated)
        changed_train_rows += 1

    group_sizes = [len(group) for group in train_groups.values()]
    return updated_rows, {
        "mode": mode,
        "num_train_prompts": int(len(train_groups)),
        "num_train_rows_retargeted": int(changed_train_rows),
        "min_train_rollouts_per_prompt": int(min(group_sizes)) if group_sizes else 0,
        "max_train_rollouts_per_prompt": int(max(group_sizes)) if group_sizes else 0,
    }


def _balanced_train_indices(
    y_train: np.ndarray,
    *,
    mode: str,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "none":
        return np.arange(len(y_train), dtype=np.int64), {"mode": mode, "num_train_rows_before": int(len(y_train))}
    if mode != "downsample_label_values":
        raise ValueError(f"Unsupported train balance mode: {mode}")

    rng = np.random.default_rng(int(random_seed))
    rounded = np.round(np.asarray(y_train, dtype=np.float32).reshape(-1), 6)
    class_values = sorted(float(value) for value in np.unique(rounded))
    indices_by_class = {value: np.where(np.isclose(rounded, value))[0] for value in class_values}
    class_counts_before = {f"{value:g}": int(len(indices)) for value, indices in indices_by_class.items()}
    if not indices_by_class:
        raise ValueError("Cannot balance an empty train set.")
    target_count = min(len(indices) for indices in indices_by_class.values())
    if target_count <= 0:
        raise ValueError(f"Cannot balance train labels with empty class counts: {class_counts_before}")

    selected_parts = []
    for value in class_values:
        indices = indices_by_class[value]
        selected_parts.append(rng.choice(indices, size=target_count, replace=False))
    selected = np.concatenate(selected_parts).astype(np.int64)
    rng.shuffle(selected)
    return selected, {
        "mode": mode,
        "num_train_rows_before": int(len(y_train)),
        "num_train_rows_after": int(len(selected)),
        "target_count_per_label": int(target_count),
        "class_counts_before": class_counts_before,
        "class_counts_after": {f"{value:g}": int(target_count) for value in class_values},
    }


def _selection_score(
    metric_name: str,
    *,
    row_metrics: dict[str, float],
    prompt_metrics: dict[str, float],
    row_subset_metrics: dict[str, dict[str, float]],
    prompt_subset_metrics: dict[str, dict[str, float]],
) -> float:
    metrics = {
        "row_r2": float(row_metrics["r2"]),
        "row_mae": -float(row_metrics["mae"]),
        "row_rmse": -float(row_metrics["rmse"]),
        "prompt_mean_r2": float(prompt_metrics["r2"]),
        "prompt_mean_mae": -float(prompt_metrics["mae"]),
        "prompt_mean_rmse": -float(prompt_metrics["rmse"]),
        "row_half_mae": -float(row_subset_metrics["half"]["mae"]),
        "row_mid_mae": -float(row_subset_metrics["mid"]["mae"]),
        "row_non_extreme_mae": -float(row_subset_metrics["non_extreme"]["mae"]),
        "prompt_half_mae": -float(prompt_subset_metrics["half"]["mae"]),
        "prompt_mid_mae": -float(prompt_subset_metrics["mid"]["mae"]),
        "prompt_non_extreme_mae": -float(prompt_subset_metrics["non_extreme"]["mae"]),
    }
    if metric_name not in metrics:
        raise ValueError(f"Unsupported selection metric: {metric_name}")
    return metrics[metric_name]


def _safe_subset_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    y_array = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred_array = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    masks = {
        "half": np.isclose(y_array, 0.5),
        "mid": (y_array >= 0.375) & (y_array <= 0.625),
        "non_extreme": (y_array > 0.0) & (y_array < 1.0),
    }
    output: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if count == 0:
            output[name] = {
                "n": 0,
                "r2": 0.0,
                "mae": float("inf"),
                "rmse": float("inf"),
                "bias": 0.0,
                "pred_mean": 0.0,
                "pred_median": 0.0,
                "frac_pred_0.4_0.6": 0.0,
            }
            continue

        subset_true = y_array[mask]
        subset_pred = pred_array[mask]
        subset_metrics = reg_metrics(subset_true, subset_pred) if len(np.unique(subset_true)) > 1 else {
            "r2": 0.0,
            "mae": float(np.mean(np.abs(subset_pred - subset_true))),
            "rmse": float(np.sqrt(np.mean(np.square(subset_pred - subset_true)))),
        }
        output[name] = {
            "n": count,
            **subset_metrics,
            "bias": float(np.mean(subset_pred - subset_true)),
            "pred_mean": float(np.mean(subset_pred)),
            "pred_median": float(np.median(subset_pred)),
            "frac_pred_0.4_0.6": float(np.mean((subset_pred >= 0.4) & (subset_pred <= 0.6))),
        }
    return output


def _subsample_validation_rollouts(
    rows: list[dict[str, Any]],
    *,
    per_prompt: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if per_prompt <= 0:
        return rows, {"mode": "all", "per_prompt": None}

    rng = np.random.default_rng(int(random_seed))
    validation_groups: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        if str(row.get("split")) == "validation":
            validation_groups.setdefault(str(row["task_id"]), []).append(idx)

    keep_indices = {idx for idx, row in enumerate(rows) if str(row.get("split")) != "validation"}
    selected_counts = []
    original_counts = []
    for task_id in sorted(validation_groups):
        indices = validation_groups[task_id]
        original_counts.append(len(indices))
        if len(indices) <= per_prompt:
            selected = np.asarray(indices, dtype=np.int64)
        else:
            selected = rng.choice(np.asarray(indices, dtype=np.int64), size=per_prompt, replace=False)
        selected_counts.append(int(len(selected)))
        keep_indices.update(int(idx) for idx in selected.tolist())

    filtered_rows = [row for idx, row in enumerate(rows) if idx in keep_indices]
    return filtered_rows, {
        "mode": "random_per_prompt",
        "per_prompt": int(per_prompt),
        "random_seed": int(random_seed),
        "num_validation_prompts": int(len(validation_groups)),
        "num_validation_rows_before": int(sum(original_counts)),
        "num_validation_rows_after": int(sum(selected_counts)),
        "min_validation_rollouts_before": int(min(original_counts)) if original_counts else 0,
        "max_validation_rollouts_before": int(max(original_counts)) if original_counts else 0,
        "min_validation_rollouts_after": int(min(selected_counts)) if selected_counts else 0,
        "max_validation_rollouts_after": int(max(selected_counts)) if selected_counts else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a weak-only single-rollout Ridge value estimator."
    )
    parser.add_argument("--weak_run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_dataset_dir", type=Path, required=True)
    parser.add_argument("--weak_labels_path", type=Path, required=True)
    parser.add_argument("--weak_prompt_hidden_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_prompt_index_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--weak_rollout_hidden_paths", nargs="+", type=Path)
    parser.add_argument("--weak_rollout_index_paths", nargs="+", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_layer_index", type=int, default=26)
    parser.add_argument("--rollout_component", type=str, default="think_end_last10_hidden")
    parser.add_argument("--rollout_pool_mode", type=str, default="mean")
    parser.add_argument("--feature_mode", choices=["prompt_only", "prompt_plus_rollout"], required=True)
    parser.add_argument("--prompt_feature_keys", nargs="*", default=[])
    parser.add_argument("--rollout_scalar_keys", nargs="*", default=[])
    parser.add_argument("--derived_rollout_scalar_keys", nargs="*", default=[])
    parser.add_argument("--extra_rollout_scalar_field_paths", nargs="*", default=[])
    parser.add_argument(
        "--allow_missing_entropy_scalars",
        action="store_true",
        help="Fill missing rollout entropy/logprob scalar fields with 0.0 instead of failing.",
    )
    parser.add_argument("--prompt_hidden_pca_dim", type=int, default=0)
    parser.add_argument("--rollout_hidden_pca_dim", type=int, default=0)
    parser.add_argument("--single_rollout_strategy", choices=["first", "all"], default="first")
    parser.add_argument(
        "--model_family",
        choices=[
            "ridge",
            "logit_ridge",
            "two_head_logistic",
            "prompt_residual_ridge",
            "prompt_value_prior_residual_ridge",
            "prompt_prior_blend_ridge",
            "prompt_trajectory_mean_ridge",
            "prompt_trajectory_score_stack_ridge",
            "prompt_trajectory_score_mlp",
            "prompt_middle_gated_ridge",
            "prompt_trajectory_stacked_ridge",
        ],
        default="ridge",
    )
    parser.add_argument(
        "--train_target_mode",
        choices=["prompt_value", "other_rollout_correctness"],
        default="prompt_value",
        help=(
            "Training target for train rows. prompt_value uses the prompt-level mean correctness label. "
            "other_rollout_correctness replaces each train row target with the mean correctness of sibling "
            "rollouts for the same prompt; with two train rollouts this is the other trajectory's label."
        ),
    )
    parser.add_argument("--train_balance_mode", choices=["none", "downsample_label_values"], default="none")
    parser.add_argument(
        "--validation_rollouts_per_prompt",
        type=int,
        default=0,
        help="If positive, randomly keep this many validation rollouts per prompt after rollout selection.",
    )
    parser.add_argument("--validation_rollout_seed", type=int, default=42)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--selection_metric",
        choices=[
            "row_r2",
            "row_mae",
            "row_rmse",
            "row_half_mae",
            "row_mid_mae",
            "row_non_extreme_mae",
            "prompt_mean_r2",
            "prompt_mean_mae",
            "prompt_mean_rmse",
            "prompt_half_mae",
            "prompt_mid_mae",
            "prompt_non_extreme_mae",
        ],
        default="row_r2",
        help="Validation metric used to choose the saved best model.",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[100.0, 300.0, 1000.0, 3000.0, 10000.0])
    parser.add_argument("--logit_epsilons", nargs="+", type=float, default=[0.02, 0.05, 0.1])
    parser.add_argument("--logistic_cs", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--residual_alphas", nargs="+", type=float, default=[10000.0, 30000.0, 100000.0])
    parser.add_argument("--residual_scales", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--blend_prompt_weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 1.0])
    parser.add_argument("--trajectory_alphas", nargs="+", type=float, default=[10000.0, 30000.0, 100000.0])
    parser.add_argument("--combiner_alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--gate_min_prompt_weights", nargs="+", type=float, default=[0.1, 0.25])
    parser.add_argument("--gate_max_prompt_weights", nargs="+", type=float, default=[0.75, 0.9])
    parser.add_argument("--gate_widths", nargs="+", type=float, default=[0.25, 0.35, 0.5])
    parser.add_argument("--gate_powers", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument(
        "--score_mlp_feature_modes",
        nargs="+",
        choices=["scores_only", "expanded"],
        default=["scores_only"],
    )
    parser.add_argument("--score_mlp_hidden_dims", nargs="+", type=int, default=[8])
    parser.add_argument("--score_mlp_num_layers", nargs="+", type=int, default=[1])
    parser.add_argument("--score_mlp_dropouts", nargs="+", type=float, default=[0.0])
    parser.add_argument("--score_mlp_learning_rates", nargs="+", type=float, default=[0.01])
    parser.add_argument("--score_mlp_weight_decays", nargs="+", type=float, default=[0.001])
    parser.add_argument("--score_mlp_mid_weights", nargs="+", type=float, default=[1.0])
    parser.add_argument("--score_mlp_half_weights", nargs="+", type=float, default=[1.0])
    parser.add_argument("--score_mlp_max_epochs", type=int, default=800)
    parser.add_argument("--score_mlp_patience", type=int, default=80)
    parser.add_argument("--score_mlp_validation_fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    weak_labels_by_task = load_labels_by_task(args.weak_labels_path)
    split_lookup = build_split_lookup(args.weak_prompt_dataset_dir.expanduser().resolve())
    prompt_scalar_lookup = build_prompt_scalar_lookup(weak_labels_by_task, list(args.prompt_feature_keys))

    prompt_lookup = load_prompt_hidden_lookup(
        [path.expanduser().resolve() for path in args.weak_prompt_hidden_paths],
        [path.expanduser().resolve() for path in args.weak_prompt_index_paths],
        layer_index=args.prompt_layer_index,
    )
    prompt_hidden_pca = fit_prompt_hidden_pca(prompt_lookup, split_lookup, int(args.prompt_hidden_pca_dim))
    prompt_lookup = apply_prompt_hidden_pca(prompt_lookup, prompt_hidden_pca)

    rollout_hidden_lookup = None
    if args.feature_mode == "prompt_plus_rollout":
        if not args.weak_rollout_hidden_paths or not args.weak_rollout_index_paths:
            raise ValueError("Prompt+rollout mode requires weak rollout hidden/index paths.")
        rollout_hidden_lookup = build_rollout_hidden_lookup(
            [path.expanduser().resolve() for path in args.weak_rollout_hidden_paths],
            [path.expanduser().resolve() for path in args.weak_rollout_index_paths],
            component_name=args.rollout_component,
            layer_index=0,
            pool_mode=args.rollout_pool_mode,
        )

    rollout_index_lookup = None
    if args.weak_rollout_index_paths:
        rollout_index_lookup = build_rollout_index_lookup(
            [path.expanduser().resolve() for path in args.weak_rollout_index_paths]
        )

    weak_grouped = group_weak_rollouts(
        weak_run_dirs=[path.expanduser().resolve() for path in args.weak_run_dirs],
        split_lookup=split_lookup,
        labels_by_task=weak_labels_by_task,
        rollout_hidden_lookup=rollout_hidden_lookup,
        rollout_index_lookup=rollout_index_lookup,
        rollout_scalar_keys=list(args.rollout_scalar_keys),
        derived_rollout_scalar_keys=list(args.derived_rollout_scalar_keys),
        extra_rollout_scalar_field_paths=list(args.extra_rollout_scalar_field_paths),
        strict_missing_entropy=not bool(args.allow_missing_entropy_scalars),
    )
    weak_rows = select_single_rollout(weak_grouped, args.single_rollout_strategy)
    weak_rows, train_target_summary = _apply_train_target_mode(weak_rows, args.train_target_mode)
    weak_rows, validation_rollout_summary = _subsample_validation_rollouts(
        weak_rows,
        per_prompt=int(args.validation_rollouts_per_prompt),
        random_seed=int(args.validation_rollout_seed),
    )
    rollout_hidden_pca = None
    if args.feature_mode == "prompt_plus_rollout":
        rollout_hidden_pca = fit_rollout_hidden_pca(weak_rows, int(args.rollout_hidden_pca_dim))
        weak_rows = apply_rollout_hidden_pca(weak_rows, rollout_hidden_pca)
    weak_x, weak_y, weak_splits, weak_meta = build_matrix(
        weak_rows,
        prompt_lookup,
        prompt_scalar_lookup,
        feature_mode=args.feature_mode,
    )
    if not prompt_lookup:
        raise ValueError("No prompt hidden vectors were loaded.")
    prompt_hidden_feature_dim = int(np.asarray(next(iter(prompt_lookup.values())), dtype=np.float32).reshape(-1).shape[0])
    prompt_scalar_feature_dim = int(len(args.prompt_feature_keys))
    prompt_feature_dim = prompt_hidden_feature_dim + prompt_scalar_feature_dim

    weak_train_mask = weak_splits == "train"
    weak_val_mask = weak_splits == "validation"
    if not np.any(weak_train_mask):
        raise ValueError("No weak train rows were built.")
    if not np.any(weak_val_mask):
        raise ValueError("No weak validation rows were built.")

    weak_prompt_y = np.asarray(
        [float(meta.get("prompt_value_true", meta["value_true"])) for meta in weak_meta],
        dtype=np.float32,
    )
    x_train, y_train = weak_x[weak_train_mask], weak_y[weak_train_mask]
    y_prompt_train = weak_prompt_y[weak_train_mask]
    balanced_train_indices, train_balance_summary = _balanced_train_indices(
        y_train,
        mode=args.train_balance_mode,
        random_seed=args.random_seed,
    )
    x_train, y_train = x_train[balanced_train_indices], y_train[balanced_train_indices]
    y_prompt_train = y_prompt_train[balanced_train_indices]
    x_weak_val, y_weak_val = weak_x[weak_val_mask], weak_y[weak_val_mask]
    weak_val_meta = [weak_meta[idx] for idx in np.where(weak_val_mask)[0]]

    best_bundle: dict[str, Any] | None = None
    best_selection_score = -1e18
    model_specs: list[tuple[str, Any]] = []
    if args.model_family == "ridge":
        for alpha in args.alphas:
            model_specs.append(
                (
                    f"ridge_a{alpha:g}",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            ("model", Ridge(alpha=alpha, random_state=args.random_seed)),
                        ]
                    ),
                )
            )
    elif args.model_family == "logit_ridge":
        for alpha in args.alphas:
            for epsilon in args.logit_epsilons:
                model_specs.append(
                    (
                        f"logit_ridge_a{alpha:g}_eps{epsilon:g}",
                        Pipeline(
                            [
                                ("scale", StandardScaler()),
                                (
                                    "model",
                                    LogitRidgeValueEstimator(
                                        alpha=alpha,
                                        epsilon=epsilon,
                                        random_state=args.random_seed,
                                    ),
                                ),
                            ]
                        ),
                    )
                )
    elif args.model_family == "two_head_logistic":
        for c_value in args.logistic_cs:
            model_specs.append(
                (
                    f"two_head_logistic_c{c_value:g}",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                TwoHeadBinaryValueEstimator(C=c_value, random_state=args.random_seed),
                            ),
                        ]
                    ),
                )
            )
    elif args.model_family == "prompt_residual_ridge":
        for prompt_alpha in args.alphas:
            for residual_alpha in args.residual_alphas:
                for residual_scale in args.residual_scales:
                    model_specs.append(
                        (
                            f"prompt_residual_ridge_pa{prompt_alpha:g}_ra{residual_alpha:g}_s{residual_scale:g}",
                            PromptResidualRidgeValueEstimator(
                                prompt_feature_dim=prompt_feature_dim,
                                prompt_alpha=prompt_alpha,
                                residual_alpha=residual_alpha,
                                residual_scale=residual_scale,
                                random_state=args.random_seed,
                            ),
                        )
                    )
    elif args.model_family == "prompt_value_prior_residual_ridge":
        for prompt_alpha in args.alphas:
            for residual_alpha in args.residual_alphas:
                for residual_scale in args.residual_scales:
                    model_specs.append(
                        (
                            (
                                "prompt_value_prior_residual_ridge"
                                f"_pa{prompt_alpha:g}_ra{residual_alpha:g}_s{residual_scale:g}"
                            ),
                            PromptValuePriorResidualRidgeValueEstimator(
                                prompt_feature_dim=prompt_feature_dim,
                                prompt_alpha=prompt_alpha,
                                residual_alpha=residual_alpha,
                                residual_scale=residual_scale,
                                random_state=args.random_seed,
                            ),
                        )
                    )
    elif args.model_family == "prompt_prior_blend_ridge":
        for prompt_alpha in args.alphas:
            for row_alpha in args.alphas:
                for prompt_weight in args.blend_prompt_weights:
                    model_specs.append(
                        (
                            f"prompt_prior_blend_ridge_pa{prompt_alpha:g}_ra{row_alpha:g}_pw{prompt_weight:g}",
                            PromptPriorBlendRidgeValueEstimator(
                                prompt_feature_dim=prompt_feature_dim,
                                prompt_alpha=prompt_alpha,
                                row_alpha=row_alpha,
                                prompt_weight=prompt_weight,
                                random_state=args.random_seed,
                            ),
                        )
                    )
    elif args.model_family == "prompt_trajectory_mean_ridge":
        for prompt_alpha in args.alphas:
            for trajectory_alpha in args.trajectory_alphas:
                model_specs.append(
                    (
                        f"prompt_trajectory_mean_ridge_pa{prompt_alpha:g}_ta{trajectory_alpha:g}",
                        PromptTrajectoryMeanRidgeValueEstimator(
                            prompt_feature_dim=prompt_feature_dim,
                            prompt_alpha=prompt_alpha,
                            trajectory_alpha=trajectory_alpha,
                            random_state=args.random_seed,
                        ),
                    )
                )
    elif args.model_family == "prompt_trajectory_score_stack_ridge":
        for prompt_alpha in args.alphas:
            for trajectory_alpha in args.trajectory_alphas:
                for combiner_alpha in args.combiner_alphas:
                    model_specs.append(
                        (
                            (
                                "prompt_trajectory_score_stack_ridge"
                                f"_pa{prompt_alpha:g}_ta{trajectory_alpha:g}_ca{combiner_alpha:g}"
                            ),
                            PromptTrajectoryScoreStackRidgeValueEstimator(
                                prompt_feature_dim=prompt_feature_dim,
                                prompt_alpha=prompt_alpha,
                                trajectory_alpha=trajectory_alpha,
                                combiner_alpha=combiner_alpha,
                                random_state=args.random_seed,
                            ),
                        )
                    )
    elif args.model_family == "prompt_trajectory_score_mlp":
        for prompt_alpha in args.alphas:
            for trajectory_alpha in args.trajectory_alphas:
                for score_feature_mode in args.score_mlp_feature_modes:
                    for hidden_dim in args.score_mlp_hidden_dims:
                        for num_layers in args.score_mlp_num_layers:
                            for dropout in args.score_mlp_dropouts:
                                for learning_rate in args.score_mlp_learning_rates:
                                    for weight_decay in args.score_mlp_weight_decays:
                                        for mid_sample_weight in args.score_mlp_mid_weights:
                                            for half_sample_weight in args.score_mlp_half_weights:
                                                model_specs.append(
                                                    (
                                                        (
                                                            "prompt_trajectory_score_mlp"
                                                            f"_pa{prompt_alpha:g}_ta{trajectory_alpha:g}"
                                                            f"_fm{score_feature_mode}_h{hidden_dim:g}_l{num_layers:g}"
                                                            f"_d{dropout:g}_lr{learning_rate:g}_wd{weight_decay:g}"
                                                            f"_mw{mid_sample_weight:g}_hw{half_sample_weight:g}"
                                                        ),
                                                        PromptTrajectoryScoreMLPValueEstimator(
                                                            prompt_feature_dim=prompt_feature_dim,
                                                            prompt_alpha=prompt_alpha,
                                                            trajectory_alpha=trajectory_alpha,
                                                            score_feature_mode=score_feature_mode,
                                                            hidden_dim=hidden_dim,
                                                            num_layers=num_layers,
                                                            dropout=dropout,
                                                            learning_rate=learning_rate,
                                                            weight_decay=weight_decay,
                                                            mid_sample_weight=mid_sample_weight,
                                                            half_sample_weight=half_sample_weight,
                                                            max_epochs=args.score_mlp_max_epochs,
                                                            patience=args.score_mlp_patience,
                                                            validation_fraction=args.score_mlp_validation_fraction,
                                                            random_state=args.random_seed,
                                                        ),
                                                    )
                                                )
    elif args.model_family == "prompt_middle_gated_ridge":
        for prompt_alpha in args.alphas:
            for trajectory_alpha in args.trajectory_alphas:
                for gate_min_prompt_weight in args.gate_min_prompt_weights:
                    for gate_max_prompt_weight in args.gate_max_prompt_weights:
                        if gate_max_prompt_weight < gate_min_prompt_weight:
                            continue
                        for gate_width in args.gate_widths:
                            for gate_power in args.gate_powers:
                                model_specs.append(
                                    (
                                        (
                                            "prompt_middle_gated_ridge"
                                            f"_pa{prompt_alpha:g}_ta{trajectory_alpha:g}"
                                            f"_gmin{gate_min_prompt_weight:g}_gmax{gate_max_prompt_weight:g}"
                                            f"_gw{gate_width:g}_gp{gate_power:g}"
                                        ),
                                        PromptMiddleGatedRidgeValueEstimator(
                                            prompt_feature_dim=prompt_feature_dim,
                                            prompt_alpha=prompt_alpha,
                                            trajectory_alpha=trajectory_alpha,
                                            gate_min_prompt_weight=gate_min_prompt_weight,
                                            gate_max_prompt_weight=gate_max_prompt_weight,
                                            gate_width=gate_width,
                                            gate_power=gate_power,
                                            random_state=args.random_seed,
                                        ),
                                    )
                                )
    elif args.model_family == "prompt_trajectory_stacked_ridge":
        for prompt_alpha in args.alphas:
            for trajectory_alpha in args.trajectory_alphas:
                for combiner_alpha in args.combiner_alphas:
                    model_specs.append(
                        (
                            (
                                "prompt_trajectory_stacked_ridge"
                                f"_pa{prompt_alpha:g}_ta{trajectory_alpha:g}_ca{combiner_alpha:g}"
                            ),
                            PromptTrajectoryStackedRidgeValueEstimator(
                                prompt_feature_dim=prompt_feature_dim,
                                prompt_alpha=prompt_alpha,
                                trajectory_alpha=trajectory_alpha,
                                combiner_alpha=combiner_alpha,
                                random_state=args.random_seed,
                            ),
                        )
                    )
    else:
        raise ValueError(f"Unsupported model family: {args.model_family}")

    for model_name, estimator in model_specs:
        estimator_step_for_fit = (
            estimator.named_steps.get("model", estimator) if hasattr(estimator, "named_steps") else estimator
        )
        fit_target = (
            np.stack([y_train, y_prompt_train], axis=1)
            if bool(getattr(estimator_step_for_fit, "uses_prompt_value_target", False))
            else y_train
        )
        estimator.fit(x_train, fit_target)
        weak_val_pred = np.clip(np.asarray(estimator.predict(x_weak_val), dtype=np.float32).reshape(-1), 0.0, 1.0)
        weak_val_row_metrics = reg_metrics(y_weak_val, weak_val_pred)
        weak_val_prompt_metrics, weak_val_prompt_rows = prompt_mean_metrics(y_weak_val, weak_val_pred, weak_val_meta)
        weak_val_row_subset_metrics = _safe_subset_metrics(y_weak_val, weak_val_pred)
        weak_val_prompt_true = np.asarray([float(row["value_true"]) for row in weak_val_prompt_rows], dtype=np.float32)
        weak_val_prompt_pred = np.asarray([float(row["value_pred"]) for row in weak_val_prompt_rows], dtype=np.float32)
        weak_val_prompt_subset_metrics = _safe_subset_metrics(weak_val_prompt_true, weak_val_prompt_pred)
        result = {
            "name": model_name,
            "weak_val_row_metrics": weak_val_row_metrics,
            "weak_val_row_subset_metrics": weak_val_row_subset_metrics,
            "weak_val_prompt_mean_metrics": weak_val_prompt_metrics,
            "weak_val_prompt_subset_metrics": weak_val_prompt_subset_metrics,
            "selection_metric": args.selection_metric,
            "selection_score": _selection_score(
                args.selection_metric,
                row_metrics=weak_val_row_metrics,
                prompt_metrics=weak_val_prompt_metrics,
                row_subset_metrics=weak_val_row_subset_metrics,
                prompt_subset_metrics=weak_val_prompt_subset_metrics,
            ),
        }
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        if result["selection_score"] > best_selection_score:
            best_selection_score = float(result["selection_score"])
            best_bundle = {
                "name": model_name,
                "estimator": estimator,
                "weak_val_row_metrics": weak_val_row_metrics,
                "weak_val_row_subset_metrics": weak_val_row_subset_metrics,
                "weak_val_prompt_mean_metrics": weak_val_prompt_metrics,
                "weak_val_prompt_subset_metrics": weak_val_prompt_subset_metrics,
                "selection_metric": args.selection_metric,
                "selection_score": float(result["selection_score"]),
                "weak_val_row_rows": _row_prediction_rows(y_weak_val, weak_val_pred, weak_val_meta),
                "weak_val_prompt_rows": weak_val_prompt_rows,
                "feature_dim": int(x_train.shape[1]),
                "num_train_rows": int(x_train.shape[0]),
                "num_weak_val_rows": int(x_weak_val.shape[0]),
                "num_weak_val_prompts": int(len(weak_val_prompt_rows)),
            }

    if best_bundle is None:
        raise RuntimeError("Failed to fit any model.")

    estimator_pipeline = best_bundle["estimator"]
    estimator_step = estimator_pipeline.named_steps.get("model", estimator_pipeline) if hasattr(estimator_pipeline, "named_steps") else estimator_pipeline
    pipeline_steps = ["standard_scaler", type(estimator_step).__name__.lower()] if hasattr(estimator_pipeline, "named_steps") else [type(estimator_step).__name__.lower()]
    estimator_config = {
        "prediction_target": "value",
        "model_family": args.model_family,
        "train_target_mode": args.train_target_mode,
        "train_target_summary": train_target_summary,
        "train_balance_mode": args.train_balance_mode,
        "train_balance_summary": train_balance_summary,
        "validation_rollout_summary": validation_rollout_summary,
        "selection_metric": args.selection_metric,
        "selection_score": best_bundle["selection_score"],
        "prompt": {
            "hidden_layer_index": int(args.prompt_layer_index),
            "hidden_projection": {
                "type": None if prompt_hidden_pca is None else "pca",
                "input_dim": None if prompt_hidden_pca is None else int(prompt_hidden_pca.n_features_in_),
                "output_dim": None if prompt_hidden_pca is None else int(prompt_hidden_pca.n_components_),
            },
            "prompt_scalar_keys": list(args.prompt_feature_keys),
        },
        "response": {
            "hidden_component": args.rollout_component if args.feature_mode == "prompt_plus_rollout" else None,
            "hidden_pool_mode": args.rollout_pool_mode if args.feature_mode == "prompt_plus_rollout" else None,
            "hidden_projection": {
                "type": None if rollout_hidden_pca is None else "pca",
                "input_dim": None if rollout_hidden_pca is None else int(rollout_hidden_pca.n_features_in_),
                "output_dim": None if rollout_hidden_pca is None else int(rollout_hidden_pca.n_components_),
            },
            "scalar_keys": list(args.rollout_scalar_keys),
            "derived_scalar_keys": list(args.derived_rollout_scalar_keys),
            "extra_scalar_field_paths": list(args.extra_rollout_scalar_field_paths),
            "allow_missing_entropy_scalars": bool(args.allow_missing_entropy_scalars),
        },
        "model": {
            "pipeline": pipeline_steps,
            "estimator_class": type(estimator_step).__name__,
            "alpha": float(getattr(estimator_step, "alpha", 0.0)) if hasattr(estimator_step, "alpha") else None,
            "logit_epsilon": float(getattr(estimator_step, "epsilon", 0.0)) if hasattr(estimator_step, "epsilon") else None,
            "prompt_alpha": float(getattr(estimator_step, "prompt_alpha", 0.0)) if hasattr(estimator_step, "prompt_alpha") else None,
            "row_alpha": float(getattr(estimator_step, "row_alpha", 0.0)) if hasattr(estimator_step, "row_alpha") else None,
            "trajectory_alpha": float(getattr(estimator_step, "trajectory_alpha", 0.0)) if hasattr(estimator_step, "trajectory_alpha") else None,
            "combiner_alpha": float(getattr(estimator_step, "combiner_alpha", 0.0)) if hasattr(estimator_step, "combiner_alpha") else None,
            "residual_alpha": float(getattr(estimator_step, "residual_alpha", 0.0)) if hasattr(estimator_step, "residual_alpha") else None,
            "residual_scale": float(getattr(estimator_step, "residual_scale", 0.0)) if hasattr(estimator_step, "residual_scale") else None,
            "prompt_weight": float(getattr(estimator_step, "prompt_weight", 0.0)) if hasattr(estimator_step, "prompt_weight") else None,
            "gate_min_prompt_weight": float(getattr(estimator_step, "gate_min_prompt_weight", 0.0)) if hasattr(estimator_step, "gate_min_prompt_weight") else None,
            "gate_max_prompt_weight": float(getattr(estimator_step, "gate_max_prompt_weight", 0.0)) if hasattr(estimator_step, "gate_max_prompt_weight") else None,
            "gate_width": float(getattr(estimator_step, "gate_width", 0.0)) if hasattr(estimator_step, "gate_width") else None,
            "gate_power": float(getattr(estimator_step, "gate_power", 0.0)) if hasattr(estimator_step, "gate_power") else None,
            "score_feature_mode": str(getattr(estimator_step, "score_feature_mode", "")) if hasattr(estimator_step, "score_feature_mode") else None,
            "score_mlp_hidden_dim": int(getattr(estimator_step, "hidden_dim", 0)) if hasattr(estimator_step, "hidden_dim") else None,
            "score_mlp_num_layers": int(getattr(estimator_step, "num_layers", 0)) if hasattr(estimator_step, "num_layers") else None,
            "score_mlp_dropout": float(getattr(estimator_step, "dropout", 0.0)) if hasattr(estimator_step, "dropout") else None,
            "score_mlp_learning_rate": float(getattr(estimator_step, "learning_rate", 0.0)) if hasattr(estimator_step, "learning_rate") else None,
            "score_mlp_weight_decay": float(getattr(estimator_step, "weight_decay", 0.0)) if hasattr(estimator_step, "weight_decay") else None,
            "score_mlp_mid_sample_weight": float(getattr(estimator_step, "mid_sample_weight", 0.0)) if hasattr(estimator_step, "mid_sample_weight") else None,
            "score_mlp_half_sample_weight": float(getattr(estimator_step, "half_sample_weight", 0.0)) if hasattr(estimator_step, "half_sample_weight") else None,
            "score_mlp_max_epochs": int(getattr(estimator_step, "max_epochs", 0)) if hasattr(estimator_step, "max_epochs") else None,
            "score_mlp_patience": int(getattr(estimator_step, "patience", 0)) if hasattr(estimator_step, "patience") else None,
            "score_mlp_validation_fraction": float(getattr(estimator_step, "validation_fraction", 0.0)) if hasattr(estimator_step, "validation_fraction") else None,
            "score_mlp_num_epochs_trained": int(getattr(estimator_step, "num_epochs_trained_", 0)) if hasattr(estimator_step, "num_epochs_trained_") else None,
            "score_mlp_best_validation_loss": float(getattr(estimator_step, "best_validation_loss_", 0.0)) if hasattr(estimator_step, "best_validation_loss_") else None,
            "prompt_feature_dim": int(prompt_feature_dim),
            "clip_min": 0.0,
            "clip_max": 1.0,
            "best_model_name": best_bundle["name"],
            "feature_dim": int(best_bundle["feature_dim"]),
        },
    }
    bundle = {
        "bundle_type": "single_rollout_value_estimator",
        "bundle_version": 1,
        "config": estimator_config,
        "feature_mode": args.feature_mode,
        "model_family": args.model_family,
        "train_target_mode": args.train_target_mode,
        "train_target_summary": train_target_summary,
        "train_balance_mode": args.train_balance_mode,
        "train_balance_summary": train_balance_summary,
        "validation_rollouts_per_prompt": int(args.validation_rollouts_per_prompt),
        "validation_rollout_seed": int(args.validation_rollout_seed),
        "validation_rollout_summary": validation_rollout_summary,
        "selection_metric": args.selection_metric,
        "selection_score": best_bundle["selection_score"],
        "single_rollout_strategy": args.single_rollout_strategy,
        "rollout_component": args.rollout_component if args.feature_mode == "prompt_plus_rollout" else None,
        "rollout_pool_mode": args.rollout_pool_mode if args.feature_mode == "prompt_plus_rollout" else None,
        "estimator": estimator_pipeline,
        "prompt_hidden_pca": prompt_hidden_pca,
        "rollout_hidden_pca": rollout_hidden_pca,
    }
    estimator_config, bundle = _add_support_compatibility(
        estimator_config=estimator_config,
        rollout_hidden_pca=rollout_hidden_pca,
        bundle=bundle,
    )
    joblib.dump(bundle, args.output_dir / "model.joblib")
    (args.output_dir / "estimator_config.json").write_text(json.dumps(estimator_config, indent=2), encoding="utf-8")

    _write_row_predictions(args.output_dir / "predictions_weakval_rows.jsonl", best_bundle["weak_val_row_rows"], weak_labels_by_task)
    save_diagnostics_plot(
        args.output_dir / "prediction_diagnostics_weakval_rows.png",
        best_bundle["weak_val_row_rows"],
        f"Weak Validation Rows: {best_bundle['name']}",
    )
    write_predictions(args.output_dir / "predictions_weakval.jsonl", best_bundle["weak_val_prompt_rows"], weak_labels_by_task)
    save_diagnostics_plot(
        args.output_dir / "prediction_diagnostics_weakval.png",
        best_bundle["weak_val_prompt_rows"],
        f"Weak Validation: {best_bundle['name']}",
    )

    summary = {
        "setting": "weak_only_single_rollout_hidden",
        "prediction_target": "value",
        "feature_mode": args.feature_mode,
        "model_family": args.model_family,
        "prompt_layer_index": int(args.prompt_layer_index),
        "prompt_hidden_pca_dim": int(args.prompt_hidden_pca_dim),
        "rollout_hidden_pca_dim": int(args.rollout_hidden_pca_dim),
        "prompt_feature_keys": list(args.prompt_feature_keys),
        "rollout_scalar_keys": list(args.rollout_scalar_keys),
        "derived_rollout_scalar_keys": list(args.derived_rollout_scalar_keys),
        "extra_rollout_scalar_field_paths": list(args.extra_rollout_scalar_field_paths),
        "allow_missing_entropy_scalars": bool(args.allow_missing_entropy_scalars),
        "rollout_component": args.rollout_component if args.feature_mode == "prompt_plus_rollout" else None,
        "rollout_pool_mode": args.rollout_pool_mode if args.feature_mode == "prompt_plus_rollout" else None,
        "train_target_mode": args.train_target_mode,
        "train_target_summary": train_target_summary,
        "train_balance_mode": args.train_balance_mode,
        "train_balance_summary": train_balance_summary,
        "validation_rollouts_per_prompt": int(args.validation_rollouts_per_prompt),
        "validation_rollout_seed": int(args.validation_rollout_seed),
        "validation_rollout_summary": validation_rollout_summary,
        "selection_metric": args.selection_metric,
        "selection_score": best_bundle["selection_score"],
        "single_rollout_strategy": args.single_rollout_strategy,
        "alphas": [float(alpha) for alpha in args.alphas],
        "logit_epsilons": [float(epsilon) for epsilon in args.logit_epsilons],
        "logistic_cs": [float(c_value) for c_value in args.logistic_cs],
        "residual_alphas": [float(alpha) for alpha in args.residual_alphas],
        "residual_scales": [float(scale) for scale in args.residual_scales],
        "blend_prompt_weights": [float(weight) for weight in args.blend_prompt_weights],
        "trajectory_alphas": [float(alpha) for alpha in args.trajectory_alphas],
        "combiner_alphas": [float(alpha) for alpha in args.combiner_alphas],
        "gate_min_prompt_weights": [float(weight) for weight in args.gate_min_prompt_weights],
        "gate_max_prompt_weights": [float(weight) for weight in args.gate_max_prompt_weights],
        "gate_widths": [float(width) for width in args.gate_widths],
        "gate_powers": [float(power) for power in args.gate_powers],
        "score_mlp_feature_modes": list(args.score_mlp_feature_modes),
        "score_mlp_hidden_dims": [int(value) for value in args.score_mlp_hidden_dims],
        "score_mlp_num_layers": [int(value) for value in args.score_mlp_num_layers],
        "score_mlp_dropouts": [float(value) for value in args.score_mlp_dropouts],
        "score_mlp_learning_rates": [float(value) for value in args.score_mlp_learning_rates],
        "score_mlp_weight_decays": [float(value) for value in args.score_mlp_weight_decays],
        "score_mlp_mid_weights": [float(value) for value in args.score_mlp_mid_weights],
        "score_mlp_half_weights": [float(value) for value in args.score_mlp_half_weights],
        "score_mlp_max_epochs": int(args.score_mlp_max_epochs),
        "score_mlp_patience": int(args.score_mlp_patience),
        "score_mlp_validation_fraction": float(args.score_mlp_validation_fraction),
        "best_model": best_bundle["name"],
        "feature_dim": best_bundle["feature_dim"],
        "prompt_feature_dim": int(prompt_feature_dim),
        "num_train_rows": best_bundle["num_train_rows"],
        "num_weak_val_rows": best_bundle["num_weak_val_rows"],
        "num_weak_val_prompts": best_bundle["num_weak_val_prompts"],
        "weak_val_row_metrics": best_bundle["weak_val_row_metrics"],
        "weak_val_row_subset_metrics": best_bundle["weak_val_row_subset_metrics"],
        "weak_val_prompt_mean_metrics": best_bundle["weak_val_prompt_mean_metrics"],
        "weak_val_prompt_subset_metrics": best_bundle["weak_val_prompt_subset_metrics"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
