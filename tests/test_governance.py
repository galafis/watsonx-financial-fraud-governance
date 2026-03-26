"""Tests for governance modules: BiasDetector, DriftMonitor, ModelFactsheet."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.governance.bias_detector import BiasDetector
from src.governance.drift_monitor import DriftMonitor
from src.governance.factsheet import ModelFactsheet

# ── BiasDetector Tests ──────────────────────────────────────────────────────


class TestBiasDetector:
    """Tests for the BiasDetector class."""

    def setup_method(self) -> None:
        """Initialize with test configuration."""
        self.config = {
            "demographic_parity_threshold": 0.10,
            "equalized_odds_threshold": 0.10,
            "protected_attributes": ["age_group", "gender"],
        }
        self.detector = BiasDetector(config=self.config)

    def test_evaluate_fair_predictions(self) -> None:
        """Uniform predictions across groups are reported as fair."""
        n = 100
        y_true = np.zeros(n)
        y_pred = np.zeros(n)
        protected_df = pd.DataFrame(
            {
                "age_group": np.repeat(["18-25", "26-35", "36-50", "51-65"], 25),
                "gender": np.tile(["M", "F"], 50),
            }
        )

        report = self.detector.evaluate(y_true, y_pred, protected_df)
        assert report["is_fair"] is True
        assert len(report["alerts"]) == 0

    def test_evaluate_biased_predictions(self) -> None:
        """Biased predictions trigger demographic parity violations."""
        n = 200
        y_true = np.zeros(n)

        # Group A: 50% flagged, Group B: 0% flagged
        y_pred = np.concatenate([np.ones(50), np.zeros(50), np.zeros(100)])

        protected_df = pd.DataFrame(
            {
                "age_group": np.concatenate([np.repeat("18-25", 100), np.repeat("36-50", 100)]),
                "gender": np.tile(["M", "F"], 100),
            }
        )

        report = self.detector.evaluate(y_true, y_pred, protected_df)
        assert report["is_fair"] is False
        assert len(report["alerts"]) > 0

    def test_evaluate_returns_overall_positive_rate(self) -> None:
        """Report includes the overall positive prediction rate."""
        n = 100
        y_true = np.zeros(n)
        y_pred = np.concatenate([np.ones(20), np.zeros(80)])
        protected_df = pd.DataFrame(
            {"age_group": np.repeat(["A", "B"], 50), "gender": np.tile(["M", "F"], 50)}
        )

        report = self.detector.evaluate(y_true, y_pred, protected_df)
        assert report["overall_positive_rate"] == pytest.approx(0.20)

    def test_evaluate_missing_attribute_skipped(self) -> None:
        """Missing protected attributes are skipped without error."""
        n = 50
        y_true = np.zeros(n)
        y_pred = np.zeros(n)
        protected_df = pd.DataFrame({"age_group": np.repeat(["A", "B"], 25)})

        report = self.detector.evaluate(y_true, y_pred, protected_df)
        assert "age_group" in report["attributes"]
        assert "gender" not in report["attributes"]

    def test_evaluate_group_metrics_structure(self) -> None:
        """Each group has count, positive_rate, tpr, and fpr."""
        y_true = np.concatenate([np.ones(10), np.zeros(90)])
        y_pred = np.concatenate([np.ones(5), np.zeros(5), np.ones(5), np.zeros(85)])
        protected_df = pd.DataFrame(
            {"age_group": np.repeat(["A", "B"], 50), "gender": np.tile(["M", "F"], 50)}
        )

        report = self.detector.evaluate(y_true, y_pred, protected_df)
        for _attr_name, attr_report in report["attributes"].items():
            for _group_name, metrics in attr_report["groups"].items():
                assert "count" in metrics
                assert "positive_rate" in metrics
                assert "tpr" in metrics
                assert "fpr" in metrics

    def test_equalized_odds_violation(self) -> None:
        """Large TPR gap triggers equalized odds violation."""
        # Group A: TPR = 1.0 (5 TP out of 5), Group B: TPR = 0.0 (0 TP out of 5)
        y_true = np.concatenate([np.ones(5), np.zeros(95), np.ones(5), np.zeros(95)])
        y_pred = np.concatenate([np.ones(5), np.zeros(95), np.zeros(5), np.zeros(95)])
        protected_df = pd.DataFrame(
            {"age_group": np.concatenate([np.repeat("A", 100), np.repeat("B", 100)])}
        )

        detector = BiasDetector(
            config={
                "demographic_parity_threshold": 0.10,
                "equalized_odds_threshold": 0.10,
                "protected_attributes": ["age_group"],
            }
        )
        report = detector.evaluate(y_true, y_pred, protected_df)
        age_report = report["attributes"]["age_group"]
        assert age_report["equalized_odds_violation"] is True


# ── DriftMonitor Tests ──────────────────────────────────────────────────────


class TestDriftMonitor:
    """Tests for the DriftMonitor class."""

    def setup_method(self) -> None:
        """Initialize with test configuration."""
        self.config = {
            "psi_threshold": 0.20,
            "ks_threshold": 0.05,
            "monitoring_window_days": 30,
        }
        self.monitor = DriftMonitor(config=self.config)

    def test_psi_identical_distributions(self) -> None:
        """PSI of identical distributions is approximately zero."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000)
        psi = self.monitor.compute_psi(data, data)
        assert psi < 0.01

    def test_psi_shifted_distribution(self) -> None:
        """PSI detects significant distribution shift."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal(1000)
        production = rng.standard_normal(1000) + 3.0  # Large shift
        psi = self.monitor.compute_psi(reference, production)
        assert psi > 0.20

    def test_psi_is_non_negative(self) -> None:
        """PSI is always non-negative."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal(500)
        prod = rng.standard_normal(500) + 0.5
        psi = self.monitor.compute_psi(ref, prod)
        assert psi >= 0.0

    def test_ks_test_identical(self) -> None:
        """KS-test reports no drift for identical distributions."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        result = self.monitor.compute_ks_test(data, data)
        assert result["ks_statistic"] == pytest.approx(0.0)
        assert result["significant_drift"] is False

    def test_ks_test_shifted(self) -> None:
        """KS-test detects drift for shifted distributions."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal(500)
        production = rng.standard_normal(500) + 2.0
        result = self.monitor.compute_ks_test(reference, production)
        assert result["significant_drift"] is True
        assert result["p_value"] < 0.05

    def test_evaluate_drift_no_drift(self) -> None:
        """Full drift evaluation shows no drift for similar distributions."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((500, 3))
        prod = rng.standard_normal((500, 3))
        feature_names = ["f1", "f2", "f3"]

        report = self.monitor.evaluate_drift(ref, prod, feature_names)
        assert report["total_features"] == 3
        assert isinstance(report["drifted_features"], list)
        assert isinstance(report["drift_detected"], bool)

    def test_evaluate_drift_with_drift(self) -> None:
        """Full evaluation detects drift when production shifts."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((500, 2))
        prod = np.column_stack([rng.standard_normal(500) + 5.0, rng.standard_normal(500)])
        feature_names = ["drifted_feature", "stable_feature"]

        report = self.monitor.evaluate_drift(ref, prod, feature_names)
        assert report["drift_detected"] is True
        assert "drifted_feature" in report["drifted_features"]

    def test_evaluate_drift_report_structure(self) -> None:
        """Drift report has all required keys."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 2))
        prod = rng.standard_normal((100, 2))

        report = self.monitor.evaluate_drift(ref, prod, ["f1", "f2"])
        assert "total_features" in report
        assert "drifted_features_count" in report
        assert "drifted_features" in report
        assert "drift_detected" in report
        assert "psi_threshold" in report
        assert "ks_threshold" in report
        assert "feature_reports" in report

    def test_evaluate_performance_drift(self) -> None:
        """Performance drift detects score distribution shift."""
        rng = np.random.default_rng(42)
        ref_scores = rng.uniform(0, 0.3, 500)
        prod_scores = rng.uniform(0.5, 1.0, 500)

        report = self.monitor.evaluate_performance_drift(ref_scores, prod_scores)
        assert report["performance_drift_detected"] is True
        assert "score_psi" in report
        assert "score_ks_statistic" in report

    def test_evaluate_performance_drift_stable(self) -> None:
        """No performance drift for similar score distributions."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 500)

        report = self.monitor.evaluate_performance_drift(scores, scores)
        assert report["performance_drift_detected"] is False


# ── ModelFactsheet Tests ────────────────────────────────────────────────────


class TestModelFactsheet:
    """Tests for the EU AI Act Article 11 factsheet generator."""

    def test_generate_default_factsheet(self) -> None:
        """Default factsheet has all required sections."""
        factsheet_gen = ModelFactsheet()
        result = factsheet_gen.generate()

        required_sections = [
            "metadata",
            "purpose_and_scope",
            "technical_specification",
            "training_data",
            "performance_metrics",
            "fairness_assessment",
            "robustness_monitoring",
            "risk_management",
            "logging_and_audit",
        ]
        for section in required_sections:
            assert section in result, f"Missing section: {section}"

    def test_factsheet_metadata(self) -> None:
        """Metadata includes version, timestamp, and EU AI Act reference."""
        factsheet_gen = ModelFactsheet()
        result = factsheet_gen.generate()
        metadata = result["metadata"]

        assert "factsheet_version" in metadata
        assert "generated_at" in metadata
        assert "eu_ai_act_article" in metadata
        assert "risk_classification" in metadata
        assert metadata["risk_classification"] == "High-Risk"

    def test_factsheet_with_metrics(self) -> None:
        """Custom metrics are included in the factsheet."""
        factsheet_gen = ModelFactsheet()
        metrics = {"auc_roc": 0.95, "precision": 0.88, "recall": 0.82}
        result = factsheet_gen.generate(model_metrics=metrics)

        assert result["performance_metrics"]["auc_roc"] == 0.95

    def test_factsheet_with_bias_report(self) -> None:
        """Custom bias report is included in the factsheet."""
        factsheet_gen = ModelFactsheet()
        bias = {"is_fair": True, "alerts": []}
        result = factsheet_gen.generate(bias_report=bias)

        assert result["fairness_assessment"]["is_fair"] is True

    def test_factsheet_with_drift_report(self) -> None:
        """Custom drift report is included in the factsheet."""
        factsheet_gen = ModelFactsheet()
        drift = {"drift_detected": False, "drifted_features": []}
        result = factsheet_gen.generate(drift_report=drift)

        assert result["robustness_monitoring"]["drift_detected"] is False

    def test_factsheet_risk_management_structure(self) -> None:
        """Risk management section has identified risks and mitigations."""
        factsheet_gen = ModelFactsheet()
        result = factsheet_gen.generate()
        risk = result["risk_management"]

        assert "identified_risks" in risk
        assert "mitigation_measures" in risk
        assert "human_oversight" in risk
        assert len(risk["identified_risks"]) > 0
        assert len(risk["mitigation_measures"]) > 0

    def test_factsheet_purpose_scope(self) -> None:
        """Purpose and scope section includes intended use and prohibited uses."""
        factsheet_gen = ModelFactsheet()
        result = factsheet_gen.generate()
        purpose = result["purpose_and_scope"]

        assert "intended_use" in purpose
        assert "intended_users" in purpose
        assert "prohibited_uses" in purpose
        assert len(purpose["prohibited_uses"]) > 0

    def test_factsheet_logging_audit(self) -> None:
        """Logging and audit section is configured correctly."""
        factsheet_gen = ModelFactsheet()
        result = factsheet_gen.generate()
        logging = result["logging_and_audit"]

        assert logging["prediction_logging"] is True
        assert logging["explanation_logging"] is True
        assert logging["drift_monitoring"] is True
