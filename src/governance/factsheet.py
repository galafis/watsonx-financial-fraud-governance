"""EU AI Act Article 11 model factsheet generation.

Generates structured documentation for high-risk AI systems as required
by the EU AI Act, including model metadata, performance metrics,
training data characteristics, and governance controls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class ModelFactsheet:
    """Generate EU AI Act Article 11 compliant model factsheets.

    Produces structured documentation covering model purpose, training data,
    performance metrics, fairness assessment, and risk controls.
    """

    def __init__(self) -> None:
        gov_cfg = settings.governance_config.get("factsheet", {})
        self.eu_article: str = gov_cfg.get("eu_ai_act_article", "Article 11")
        self.risk_classification: str = gov_cfg.get("risk_classification", "High-Risk")
        self.version: str = gov_cfg.get("version", "1.0.0")

    def generate(
        self,
        model_metrics: dict[str, Any] | None = None,
        bias_report: dict[str, Any] | None = None,
        drift_report: dict[str, Any] | None = None,
        training_data_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a complete model factsheet.

        Args:
            model_metrics: Performance metrics from training/evaluation.
            bias_report: Bias detection results across protected attributes.
            drift_report: Data and model drift monitoring results.
            training_data_summary: Summary of training data characteristics.

        Returns:
            Structured factsheet dictionary compliant with EU AI Act Article 11.
        """
        logger.info("generating_factsheet", version=self.version)

        factsheet: dict[str, Any] = {
            "metadata": {
                "factsheet_version": self.version,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "eu_ai_act_article": self.eu_article,
                "risk_classification": self.risk_classification,
                "system_name": "Watsonx Financial Fraud Detector",
                "system_version": "1.0.0",
                "provider": "Gabriel Demetrios Lafis",
            },
            "purpose_and_scope": {
                "intended_use": (
                    "Automated detection of potentially fraudulent financial transactions "
                    "using ensemble machine learning models with human-in-the-loop review."
                ),
                "intended_users": [
                    "Fraud analysts",
                    "Risk management teams",
                    "Compliance officers",
                ],
                "deployment_context": "Financial services transaction monitoring",
                "prohibited_uses": [
                    "Autonomous transaction blocking without human review",
                    "Credit scoring or lending decisions",
                    "Customer profiling beyond fraud detection",
                ],
            },
            "technical_specification": {
                "model_architecture": "Weighted ensemble: XGBoost + LightGBM + Isolation Forest",
                "input_features": "Transaction amount, velocity, temporal patterns, "
                "merchant statistics, geographic anomalies",
                "output": "Fraud probability score [0, 1] with decision label",
                "decision_thresholds": {
                    "fraud": settings.threshold_config.get("fraud", 0.70),
                    "review": settings.threshold_config.get("review", 0.40),
                    "legitimate": "< review threshold",
                },
                "explainability_method": "SHAP (TreeExplainer) with Granite narrative generation",
            },
            "training_data": training_data_summary
            or {
                "description": "Synthetic financial transaction data",
                "size": "Not available",
                "fraud_rate": "Not available",
                "date_range": "Not available",
            },
            "performance_metrics": model_metrics
            or {
                "status": "Not evaluated",
            },
            "fairness_assessment": bias_report
            or {
                "status": "Not evaluated",
            },
            "robustness_monitoring": drift_report
            or {
                "status": "Not evaluated",
            },
            "risk_management": {
                "identified_risks": [
                    "Model performance degradation due to data drift",
                    "Bias against protected groups (age, gender, geography)",
                    "Adversarial manipulation of transaction patterns",
                    "Over-reliance on automated decisions",
                ],
                "mitigation_measures": [
                    "Continuous drift monitoring with PSI and KS-test alerts",
                    "Regular bias audits across protected attributes",
                    "Human-in-the-loop review for flagged transactions",
                    "SHAP explanations for every flagged transaction",
                    "Granite narrative generation for analyst review",
                ],
                "human_oversight": {
                    "review_requirement": "All fraud-labeled transactions require analyst review",
                    "escalation_process": "High-confidence alerts escalated to senior analysts",
                    "override_capability": "Analysts can override model decisions",
                },
            },
            "logging_and_audit": {
                "prediction_logging": True,
                "explanation_logging": True,
                "drift_monitoring": True,
                "audit_trail": "Complete prediction + explanation + decision audit trail",
                "data_retention": "As per organizational data retention policy",
            },
        }

        logger.info("factsheet_generated", sections=list(factsheet.keys()))
        return factsheet
