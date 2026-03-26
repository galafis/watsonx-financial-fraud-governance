"""Training pipeline with cross-validation and hyperparameter optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.models.ensemble import EnsembleFraudDetector

logger = structlog.get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Aggregate training metrics across CV folds."""

    auc_roc: float
    auc_pr: float
    precision: float
    recall: float
    f1: float
    fold_scores: list[dict[str, float]]


class FraudModelTrainer:
    """Training pipeline for the ensemble fraud detector.

    Supports stratified k-fold cross-validation to evaluate model
    performance on imbalanced fraud datasets.
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        ensemble_config: dict[str, Any] | None = None,
    ) -> None:
        self.n_folds = n_folds
        self.random_state = random_state
        self.ensemble_config = ensemble_config
        self.model: EnsembleFraudDetector | None = None
        self.metrics: TrainingMetrics | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> EnsembleFraudDetector:
        """Train the ensemble model on full data after CV evaluation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target labels.

        Returns:
            Fitted EnsembleFraudDetector.
        """
        logger.info("training_pipeline_start", samples=X.shape[0], features=X.shape[1])

        # Run cross-validation to estimate performance
        self.metrics = self._cross_validate(X, y)
        logger.info(
            "cross_validation_complete",
            auc_roc=round(self.metrics.auc_roc, 4),
            auc_pr=round(self.metrics.auc_pr, 4),
            f1=round(self.metrics.f1, 4),
        )

        # Train final model on full data
        self.model = EnsembleFraudDetector(config=self.ensemble_config)
        self.model.fit(X, y)

        logger.info("final_model_trained")
        return self.model

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> TrainingMetrics:
        """Perform stratified k-fold cross-validation.

        Args:
            X: Feature matrix.
            y: Binary target labels.

        Returns:
            Aggregate metrics across all folds.
        """
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_scores: list[dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info("cv_fold_start", fold=fold_idx + 1, total=self.n_folds)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = EnsembleFraudDetector(config=self.ensemble_config)
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_val)
            y_pred = (proba >= model.fraud_threshold).astype(int)

            scores = {
                "auc_roc": float(roc_auc_score(y_val, proba)),
                "auc_pr": float(average_precision_score(y_val, proba)),
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
                "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            }
            fold_scores.append(scores)

            logger.info(
                "cv_fold_complete",
                fold=fold_idx + 1,
                auc_roc=round(scores["auc_roc"], 4),
                auc_pr=round(scores["auc_pr"], 4),
            )

        # Aggregate across folds
        return TrainingMetrics(
            auc_roc=float(np.mean([s["auc_roc"] for s in fold_scores])),
            auc_pr=float(np.mean([s["auc_pr"] for s in fold_scores])),
            precision=float(np.mean([s["precision"] for s in fold_scores])),
            recall=float(np.mean([s["recall"] for s in fold_scores])),
            f1=float(np.mean([s["f1"] for s in fold_scores])),
            fold_scores=fold_scores,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Return training metrics as a dictionary.

        Returns:
            Dictionary of aggregate and per-fold metrics.
        """
        if self.metrics is None:
            return {"status": "not_trained"}

        return {
            "auc_roc": self.metrics.auc_roc,
            "auc_pr": self.metrics.auc_pr,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "f1": self.metrics.f1,
            "n_folds": self.n_folds,
            "fold_scores": self.metrics.fold_scores,
        }
