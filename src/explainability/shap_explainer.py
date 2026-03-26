"""SHAP-based model explainability for fraud predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ShapExplanation:
    """SHAP explanation for a single prediction."""

    transaction_id: str
    base_value: float
    feature_contributions: dict[str, float]
    top_positive_features: list[tuple[str, float]] = field(default_factory=list)
    top_negative_features: list[tuple[str, float]] = field(default_factory=list)


class ShapExplainer:
    """Generate SHAP explanations for ensemble fraud detector predictions.

    Uses TreeExplainer for the XGBoost and LightGBM components to provide
    fast, exact SHAP values for individual fraud predictions.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._xgb_explainer: Any = None
        self._lgb_explainer: Any = None

    def fit(self, ensemble_model: Any) -> ShapExplainer:
        """Initialize SHAP explainers for the ensemble sub-models.

        Args:
            ensemble_model: A fitted EnsembleFraudDetector instance.

        Returns:
            Self for method chaining.
        """
        import shap

        logger.info("initializing_shap_explainers")
        self._xgb_explainer = shap.TreeExplainer(ensemble_model.xgb_model)
        self._lgb_explainer = shap.TreeExplainer(ensemble_model.lgb_model)
        self._weights = ensemble_model.weights
        logger.info("shap_explainers_ready")
        return self

    def explain(
        self,
        X: np.ndarray,
        feature_names: list[str],
        transaction_ids: list[str] | None = None,
    ) -> list[ShapExplanation]:
        """Generate SHAP explanations for a batch of predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            feature_names: Names for each feature column.
            transaction_ids: Optional transaction identifiers.

        Returns:
            List of ShapExplanation objects with feature contributions.
        """
        if self._xgb_explainer is None or self._lgb_explainer is None:
            raise RuntimeError("ShapExplainer not fitted. Call fit() first.")

        if transaction_ids is None:
            transaction_ids = [f"TXN-{i}" for i in range(X.shape[0])]

        logger.info("computing_shap_values", samples=X.shape[0])

        # Get SHAP values from both tree models
        xgb_shap = self._xgb_explainer.shap_values(X)
        lgb_shap = self._lgb_explainer.shap_values(X)

        # Handle binary classification output shape
        if isinstance(xgb_shap, list):
            xgb_shap = xgb_shap[1]  # class 1 (fraud)
        if isinstance(lgb_shap, list):
            lgb_shap = lgb_shap[1]

        # Weighted average of SHAP values
        w_xgb = self._weights.get("xgboost", 0.4)
        w_lgb = self._weights.get("lightgbm", 0.4)
        total_w = w_xgb + w_lgb
        combined_shap = (w_xgb * xgb_shap + w_lgb * lgb_shap) / total_w

        # Base values
        xgb_base = float(self._xgb_explainer.expected_value)
        lgb_base = float(self._lgb_explainer.expected_value)
        if isinstance(self._xgb_explainer.expected_value, np.ndarray):
            xgb_base = float(self._xgb_explainer.expected_value[1])
        if isinstance(self._lgb_explainer.expected_value, np.ndarray):
            lgb_base = float(self._lgb_explainer.expected_value[1])
        base_value = (w_xgb * xgb_base + w_lgb * lgb_base) / total_w

        explanations: list[ShapExplanation] = []
        for i, txn_id in enumerate(transaction_ids):
            contributions = {
                feat: round(float(combined_shap[i, j]), 6)
                for j, feat in enumerate(feature_names)
            }

            # Sort by absolute contribution
            sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_pos = [
                (name, val) for name, val in sorted_contribs if val > 0
            ][: self.top_k]
            top_neg = [
                (name, val) for name, val in sorted_contribs if val < 0
            ][: self.top_k]

            explanations.append(
                ShapExplanation(
                    transaction_id=txn_id,
                    base_value=round(base_value, 6),
                    feature_contributions=contributions,
                    top_positive_features=top_pos,
                    top_negative_features=top_neg,
                )
            )

        logger.info("shap_explanations_generated", count=len(explanations))
        return explanations

    def explain_single(
        self,
        X: np.ndarray,
        feature_names: list[str],
        transaction_id: str = "TXN-0",
    ) -> ShapExplanation:
        """Generate a SHAP explanation for a single transaction.

        Args:
            X: Feature vector of shape (1, n_features).
            feature_names: Names for each feature column.
            transaction_id: Transaction identifier.

        Returns:
            ShapExplanation for the single prediction.
        """
        explanations = self.explain(X, feature_names, [transaction_id])
        return explanations[0]
