"""Ensemble fraud detector combining XGBoost, LightGBM, and Isolation Forest."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import settings
from src.models.anomaly import AnomalyDetector

logger = structlog.get_logger(__name__)


@dataclass
class FraudPrediction:
    """Prediction result from the ensemble model."""

    transaction_id: str
    fraud_probability: float
    label: str  # "fraud", "review", "legitimate"
    model_scores: dict[str, float] = field(default_factory=dict)


class EnsembleFraudDetector:
    """Weighted ensemble combining XGBoost + LightGBM + Isolation Forest.

    The ensemble computes a weighted average of fraud probabilities from
    each sub-model to produce a final fraud score.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or settings.ensemble_config
        weights_cfg = cfg.get("weights", {})
        self.weights = {
            "xgboost": weights_cfg.get("xgboost", 0.40),
            "lightgbm": weights_cfg.get("lightgbm", 0.40),
            "isolation_forest": weights_cfg.get("isolation_forest", 0.20),
        }

        threshold_cfg = settings.threshold_config if config is None else cfg.get("threshold", {})
        self.fraud_threshold: float = threshold_cfg.get("fraud", 0.70)
        self.review_threshold: float = threshold_cfg.get("review", 0.40)

        # Initialize sub-models
        xgb_cfg = cfg.get("xgboost", {})
        self.xgb_model = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 500),
            max_depth=xgb_cfg.get("max_depth", 8),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=xgb_cfg.get("scale_pos_weight", 10),
            eval_metric=xgb_cfg.get("eval_metric", "aucpr"),
            random_state=xgb_cfg.get("random_state", 42),
            use_label_encoder=False,
        )

        lgb_cfg = cfg.get("lightgbm", {})
        self.lgb_model = LGBMClassifier(
            n_estimators=lgb_cfg.get("n_estimators", 500),
            max_depth=lgb_cfg.get("max_depth", 8),
            learning_rate=lgb_cfg.get("learning_rate", 0.05),
            subsample=lgb_cfg.get("subsample", 0.8),
            colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=lgb_cfg.get("scale_pos_weight", 10),
            random_state=lgb_cfg.get("random_state", 42),
            verbose=-1,
        )

        self.anomaly_detector = AnomalyDetector(config=cfg.get("isolation_forest", {}))

        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnsembleFraudDetector:
        """Train all sub-models on the provided data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target labels (1 = fraud, 0 = legitimate).

        Returns:
            Self for method chaining.
        """
        logger.info("ensemble_training_start", samples=X.shape[0], features=X.shape[1])

        logger.info("training_xgboost")
        self.xgb_model.fit(X, y)

        logger.info("training_lightgbm")
        self.lgb_model.fit(X, y)

        logger.info("training_isolation_forest")
        self.anomaly_detector.fit(X)

        self._is_fitted = True
        logger.info("ensemble_training_complete")
        return self

    def predict(
        self, X: np.ndarray, transaction_ids: list[str] | None = None
    ) -> list[FraudPrediction]:
        """Generate fraud predictions for a batch of transactions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            transaction_ids: Optional list of transaction identifiers.

        Returns:
            List of FraudPrediction objects with scores and labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble model has not been fitted. Call fit() first.")

        if transaction_ids is None:
            transaction_ids = [f"TXN-{i}" for i in range(X.shape[0])]

        # Get probability scores from each model
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        iso_scores = self.anomaly_detector.predict_scores(X)

        # Weighted ensemble
        ensemble_proba = (
            self.weights["xgboost"] * xgb_proba
            + self.weights["lightgbm"] * lgb_proba
            + self.weights["isolation_forest"] * iso_scores
        )

        predictions: list[FraudPrediction] = []
        for i, txn_id in enumerate(transaction_ids):
            prob = float(ensemble_proba[i])
            label = self._classify(prob)
            predictions.append(
                FraudPrediction(
                    transaction_id=txn_id,
                    fraud_probability=round(prob, 4),
                    label=label,
                    model_scores={
                        "xgboost": round(float(xgb_proba[i]), 4),
                        "lightgbm": round(float(lgb_proba[i]), 4),
                        "isolation_forest": round(float(iso_scores[i]), 4),
                    },
                )
            )

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble fraud probability for each sample.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of ensemble fraud probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble model has not been fitted. Call fit() first.")

        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        iso_scores = self.anomaly_detector.predict_scores(X)

        return (
            self.weights["xgboost"] * xgb_proba
            + self.weights["lightgbm"] * lgb_proba
            + self.weights["isolation_forest"] * iso_scores
        )

    def _classify(self, probability: float) -> str:
        """Map fraud probability to a decision label."""
        if probability >= self.fraud_threshold:
            return "fraud"
        if probability >= self.review_threshold:
            return "review"
        return "legitimate"
