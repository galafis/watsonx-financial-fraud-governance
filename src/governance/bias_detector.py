"""Bias detection across protected attributes using fairness metrics.

Implements demographic parity and equalized odds checks for fraud
detection models across age, gender, and geographic groups.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class BiasDetector:
    """Detect bias across protected attributes in fraud predictions.

    Evaluates demographic parity (equal positive prediction rates) and
    equalized odds (equal TPR and FPR) across protected groups.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or settings.governance_config.get("bias", {})
        self.dp_threshold: float = cfg.get("demographic_parity_threshold", 0.10)
        self.eo_threshold: float = cfg.get("equalized_odds_threshold", 0.10)
        self.protected_attributes: list[str] = cfg.get(
            "protected_attributes", ["age_group", "gender", "geography"]
        )

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Evaluate bias across all protected attributes.

        Args:
            y_true: Ground truth labels (1 = fraud, 0 = legitimate).
            y_pred: Predicted labels (1 = fraud, 0 = legitimate).
            protected_df: DataFrame with protected attribute columns.

        Returns:
            Comprehensive bias report with per-attribute metrics and alerts.
        """
        logger.info("bias_evaluation_start", attributes=self.protected_attributes)

        report: dict[str, Any] = {
            "overall_positive_rate": float(y_pred.mean()),
            "attributes": {},
            "alerts": [],
            "is_fair": True,
        }

        for attr in self.protected_attributes:
            if attr not in protected_df.columns:
                logger.warning("attribute_missing", attribute=attr)
                continue

            attr_report = self._evaluate_attribute(
                y_true, y_pred, protected_df[attr].values, attr
            )
            report["attributes"][attr] = attr_report

            if attr_report.get("demographic_parity_violation", False):
                report["alerts"].append(
                    f"Demographic parity violation on '{attr}': "
                    f"max gap = {attr_report['dp_max_gap']:.4f} "
                    f"(threshold: {self.dp_threshold})"
                )
                report["is_fair"] = False

            if attr_report.get("equalized_odds_violation", False):
                report["alerts"].append(
                    f"Equalized odds violation on '{attr}': "
                    f"max TPR gap = {attr_report['eo_tpr_max_gap']:.4f}, "
                    f"max FPR gap = {attr_report['eo_fpr_max_gap']:.4f} "
                    f"(threshold: {self.eo_threshold})"
                )
                report["is_fair"] = False

        logger.info("bias_evaluation_complete", is_fair=report["is_fair"])
        return report

    def _evaluate_attribute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        attribute_name: str,
    ) -> dict[str, Any]:
        """Evaluate fairness metrics for a single protected attribute."""
        unique_groups = np.unique(groups)

        group_metrics: dict[str, dict[str, float]] = {}
        positive_rates: list[float] = []
        tpr_values: list[float] = []
        fpr_values: list[float] = []

        for group in unique_groups:
            mask = groups == group
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]

            n_total = int(mask.sum())
            n_positive = int(group_y_pred.sum())
            positive_rate = float(group_y_pred.mean()) if n_total > 0 else 0.0
            positive_rates.append(positive_rate)

            # True Positive Rate (recall)
            true_positives = int(((group_y_true == 1) & (group_y_pred == 1)).sum())
            actual_positives = int((group_y_true == 1).sum())
            tpr = true_positives / actual_positives if actual_positives > 0 else 0.0
            tpr_values.append(tpr)

            # False Positive Rate
            false_positives = int(((group_y_true == 0) & (group_y_pred == 1)).sum())
            actual_negatives = int((group_y_true == 0).sum())
            fpr = false_positives / actual_negatives if actual_negatives > 0 else 0.0
            fpr_values.append(fpr)

            group_metrics[str(group)] = {
                "count": n_total,
                "positive_predictions": n_positive,
                "positive_rate": round(positive_rate, 4),
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
            }

        dp_max_gap = max(positive_rates) - min(positive_rates) if positive_rates else 0.0
        eo_tpr_max_gap = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
        eo_fpr_max_gap = max(fpr_values) - min(fpr_values) if fpr_values else 0.0

        return {
            "attribute": attribute_name,
            "groups": group_metrics,
            "dp_max_gap": round(dp_max_gap, 4),
            "demographic_parity_violation": dp_max_gap > self.dp_threshold,
            "eo_tpr_max_gap": round(eo_tpr_max_gap, 4),
            "eo_fpr_max_gap": round(eo_fpr_max_gap, 4),
            "equalized_odds_violation": (
                eo_tpr_max_gap > self.eo_threshold or eo_fpr_max_gap > self.eo_threshold
            ),
        }
