from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_hidden_layer_sizes(spec: str) -> tuple[int, ...]:
    if not spec.strip():
        return (256, 64)
    return tuple(int(token.strip()) for token in spec.split(",") if token.strip())


def build_estimator(
    task_type: str,
    model_name: str,
    *,
    hidden_layer_sizes: str,
    alpha: float,
    max_iter: int,
    n_estimators: int,
    min_samples_leaf: int,
    random_state: int,
    class_weight_balanced: bool,
):
    hidden_sizes = parse_hidden_layer_sizes(hidden_layer_sizes)
    class_weight = "balanced" if class_weight_balanced else None

    if task_type == "regression":
        if model_name == "linear":
            return make_pipeline(StandardScaler(), LinearRegression())
        if model_name == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=random_state))
        if model_name == "mlp":
            return make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=hidden_sizes,
                    alpha=alpha,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            )
        if model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
            )
        if model_name == "hist_gb":
            return HistGradientBoostingRegressor(
                max_iter=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
        raise ValueError(f"Unsupported regression model: {model_name}")

    if task_type == "classification":
        if model_name == "logistic":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            )
        if model_name == "linear_svm":
            return make_pipeline(
                StandardScaler(),
                LinearSVC(
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            )
        if model_name == "mlp":
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=hidden_sizes,
                    alpha=alpha,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            )
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
                class_weight=class_weight,
            )
        if model_name == "hist_gb":
            return HistGradientBoostingClassifier(
                max_iter=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
        raise ValueError(f"Unsupported classification model: {model_name}")

    raise ValueError(f"Unsupported task type: {task_type}")


def _extract_classification_scores(estimator, features: np.ndarray) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(features)
        if isinstance(scores, np.ndarray):
            return scores
    return None


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(
    estimator,
    X_eval: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        "confusion_matrix_labels": [0, 1],
    }
    if len(np.unique(y_true)) > 1:
        scores = _extract_classification_scores(estimator, X_eval)
        if scores is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    return metrics
