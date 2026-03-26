"""Isolation Forest anomaly detector for unsupervised fraud detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from sklearn.ensemble import IsolationForest

from src.config import settings

logger = structlog.get_logger(__name__)


class AnomalyDetector:
    """Isolation Forest-based anomaly detector for transaction data.

    Provides unsupervised anomaly scores that complement the supervised
    ensemble model for detecting novel fraud patterns.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or settings.ensemble_config.get("isolation_forest", {})
        self.n_estimators: int = cfg.get("n_estimators", 200)
        self.contamination: float = cfg.get("contamination", 0.02)
        self.random_state: int = cfg.get("random_state", 42)
        self.model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> AnomalyDetector:
        """Fit the Isolation Forest on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Self for method chaining.
        """
        logger.info(
            "anomaly_detector_fit",
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            samples=X.shape[0],
        )
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        logger.info("anomaly_detector_fitted")
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1] range (higher = more anomalous).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of anomaly scores normalized to [0, 1].
        """
        if self.model is None:
            raise RuntimeError("AnomalyDetector has not been fitted. Call fit() first.")

        # decision_function returns negative scores for outliers
        raw_scores = self.model.decision_function(X)

        # Normalize to [0, 1] — lower decision_function means more anomalous
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score == min_score:
            return np.full(len(raw_scores), 0.5)

        normalized = 1.0 - (raw_scores - min_score) / (max_score - min_score)
        return normalized.astype(np.float64)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels (1 = anomaly, 0 = normal).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Binary array of anomaly labels.
        """
        if self.model is None:
            raise RuntimeError("AnomalyDetector has not been fitted. Call fit() first.")

        # sklearn returns -1 for outliers, 1 for inliers
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)
