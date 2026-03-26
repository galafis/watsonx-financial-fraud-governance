"""Data drift and model performance drift monitoring.

Implements Population Stability Index (PSI) and Kolmogorov-Smirnov test
for detecting distribution shifts between reference and production data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from scipy import stats

from src.config import settings

logger = structlog.get_logger(__name__)


class DriftMonitor:
    """Monitor data drift and model performance drift.

    Uses PSI for overall distribution comparison and KS-test for
    per-feature drift detection.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or settings.governance_config.get("drift", {})
        self.psi_threshold: float = cfg.get("psi_threshold", 0.20)
        self.ks_threshold: float = cfg.get("ks_threshold", 0.05)
        self.monitoring_window_days: int = cfg.get("monitoring_window_days", 30)

    def compute_psi(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index between two distributions.

        PSI measures how much a distribution has shifted. Values:
        - < 0.10: No significant change
        - 0.10 - 0.20: Moderate change (monitor)
        - > 0.20: Significant change (alert)

        Args:
            reference: Reference distribution (training data).
            production: Production distribution (recent data).
            n_bins: Number of bins for discretization.

        Returns:
            PSI value.
        """
        eps = 1e-6

        # Create bins from reference distribution
        breakpoints = np.linspace(
            min(reference.min(), production.min()),
            max(reference.max(), production.max()),
            n_bins + 1,
        )

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        prod_counts = np.histogram(production, bins=breakpoints)[0]

        # Normalize to proportions
        ref_pct = ref_counts / len(reference) + eps
        prod_pct = prod_counts / len(production) + eps

        psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        return round(psi, 6)

    def compute_ks_test(
        self,
        reference: np.ndarray,
        production: np.ndarray,
    ) -> dict[str, float]:
        """Perform Kolmogorov-Smirnov test between two distributions.

        Args:
            reference: Reference distribution.
            production: Production distribution.

        Returns:
            Dictionary with KS statistic and p-value.
        """
        ks_stat, p_value = stats.ks_2samp(reference, production)
        return {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "significant_drift": p_value < self.ks_threshold,
        }

    def evaluate_drift(
        self,
        reference_data: np.ndarray,
        production_data: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Evaluate data drift across all features.

        Args:
            reference_data: Reference feature matrix (n_ref, n_features).
            production_data: Production feature matrix (n_prod, n_features).
            feature_names: Names for each feature.

        Returns:
            Comprehensive drift report with per-feature and aggregate metrics.
        """
        logger.info(
            "drift_evaluation_start",
            reference_samples=reference_data.shape[0],
            production_samples=production_data.shape[0],
            features=len(feature_names),
        )

        feature_reports: dict[str, dict[str, Any]] = {}
        drifted_features: list[str] = []

        for i, feat_name in enumerate(feature_names):
            ref_col = reference_data[:, i]
            prod_col = production_data[:, i]

            psi = self.compute_psi(ref_col, prod_col)
            ks = self.compute_ks_test(ref_col, prod_col)

            has_drift = psi > self.psi_threshold or ks["significant_drift"]

            feature_reports[feat_name] = {
                "psi": psi,
                "psi_alert": psi > self.psi_threshold,
                "ks_statistic": ks["ks_statistic"],
                "ks_p_value": ks["p_value"],
                "ks_alert": ks["significant_drift"],
                "has_drift": has_drift,
                "reference_mean": round(float(ref_col.mean()), 4),
                "production_mean": round(float(prod_col.mean()), 4),
                "reference_std": round(float(ref_col.std()), 4),
                "production_std": round(float(prod_col.std()), 4),
            }

            if has_drift:
                drifted_features.append(feat_name)

        report: dict[str, Any] = {
            "total_features": len(feature_names),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "drift_detected": len(drifted_features) > 0,
            "psi_threshold": self.psi_threshold,
            "ks_threshold": self.ks_threshold,
            "feature_reports": feature_reports,
        }

        logger.info(
            "drift_evaluation_complete",
            drifted=len(drifted_features),
            total=len(feature_names),
        )
        return report

    def evaluate_performance_drift(
        self,
        reference_scores: np.ndarray,
        production_scores: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate model output score drift.

        Args:
            reference_scores: Model scores on reference data.
            production_scores: Model scores on production data.

        Returns:
            Score drift report.
        """
        psi = self.compute_psi(reference_scores, production_scores)
        ks = self.compute_ks_test(reference_scores, production_scores)

        return {
            "score_psi": psi,
            "score_psi_alert": psi > self.psi_threshold,
            "score_ks_statistic": ks["ks_statistic"],
            "score_ks_p_value": ks["p_value"],
            "score_ks_alert": ks["significant_drift"],
            "reference_mean_score": round(float(reference_scores.mean()), 4),
            "production_mean_score": round(float(production_scores.mean()), 4),
            "performance_drift_detected": psi > self.psi_threshold or ks["significant_drift"],
        }
