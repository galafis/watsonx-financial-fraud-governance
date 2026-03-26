"""Tests for SHAP explainability and narrative generation modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.explainability.shap_explainer import ShapExplainer, ShapExplanation

# ── ShapExplanation Tests ───────────────────────────────────────────────────


class TestShapExplanation:
    """Tests for the ShapExplanation dataclass."""

    def test_basic_creation(self) -> None:
        """ShapExplanation stores all fields correctly."""
        explanation = ShapExplanation(
            transaction_id="TXN-001",
            base_value=0.02,
            feature_contributions={"amount": 0.35, "hour": -0.10},
            top_positive_features=[("amount", 0.35)],
            top_negative_features=[("hour", -0.10)],
        )
        assert explanation.transaction_id == "TXN-001"
        assert explanation.base_value == 0.02
        assert explanation.feature_contributions["amount"] == 0.35
        assert len(explanation.top_positive_features) == 1
        assert len(explanation.top_negative_features) == 1

    def test_default_empty_lists(self) -> None:
        """Default top features are empty lists."""
        explanation = ShapExplanation(
            transaction_id="TXN-002",
            base_value=0.01,
            feature_contributions={},
        )
        assert explanation.top_positive_features == []
        assert explanation.top_negative_features == []


# ── ShapExplainer Tests ─────────────────────────────────────────────────────


class TestShapExplainer:
    """Tests for the ShapExplainer class."""

    def test_init_default_top_k(self) -> None:
        """Default top_k is 5."""
        explainer = ShapExplainer()
        assert explainer.top_k == 5

    def test_init_custom_top_k(self) -> None:
        """Custom top_k is respected."""
        explainer = ShapExplainer(top_k=3)
        assert explainer.top_k == 3

    def test_explain_not_fitted_raises(self) -> None:
        """explain() raises RuntimeError when not fitted."""
        explainer = ShapExplainer()
        X = np.random.default_rng(42).standard_normal((5, 3))
        with pytest.raises(RuntimeError, match="not fitted"):
            explainer.explain(X, ["f1", "f2", "f3"])

    def test_explain_single_not_fitted_raises(self) -> None:
        """explain_single() raises RuntimeError when not fitted."""
        explainer = ShapExplainer()
        X = np.random.default_rng(42).standard_normal((1, 3))
        with pytest.raises(RuntimeError, match="not fitted"):
            explainer.explain_single(X, ["f1", "f2", "f3"])

    @patch("src.explainability.shap_explainer.ShapExplainer.explain")
    def test_explain_single_delegates(self, mock_explain: MagicMock) -> None:
        """explain_single delegates to explain with single sample."""
        expected = ShapExplanation(
            transaction_id="TXN-0",
            base_value=0.01,
            feature_contributions={"f1": 0.1},
        )
        mock_explain.return_value = [expected]

        explainer = ShapExplainer()
        explainer._xgb_explainer = MagicMock()  # bypass fitted check in explain_single
        explainer._lgb_explainer = MagicMock()

        X = np.array([[1.0, 2.0, 3.0]])
        result = explainer.explain_single(X, ["f1", "f2", "f3"], "TXN-TEST")

        assert result.transaction_id == "TXN-0"

    def test_fit_with_mock_ensemble(self) -> None:
        """fit() initializes SHAP explainers for ensemble sub-models."""
        mock_ensemble = MagicMock()
        mock_ensemble.xgb_model = MagicMock()
        mock_ensemble.lgb_model = MagicMock()
        mock_ensemble.weights = {"xgboost": 0.4, "lightgbm": 0.4, "isolation_forest": 0.2}

        with patch("shap.TreeExplainer") as mock_tree_explainer:
            mock_tree_explainer.return_value = MagicMock()
            explainer = ShapExplainer()
            result = explainer.fit(mock_ensemble)

            assert result is explainer
            assert explainer._xgb_explainer is not None
            assert explainer._lgb_explainer is not None
            assert mock_tree_explainer.call_count == 2

    def test_explain_with_mock_shap(self) -> None:
        """explain() returns ShapExplanation objects with correct structure."""
        explainer = ShapExplainer(top_k=2)

        # Mock SHAP explainers
        n_features = 3
        n_samples = 2

        mock_xgb = MagicMock()
        mock_xgb.shap_values.return_value = np.array([[0.3, -0.1, 0.2], [0.1, 0.4, -0.2]])
        mock_xgb.expected_value = 0.02

        mock_lgb = MagicMock()
        mock_lgb.shap_values.return_value = np.array([[0.2, -0.05, 0.15], [0.15, 0.3, -0.1]])
        mock_lgb.expected_value = 0.03

        explainer._xgb_explainer = mock_xgb
        explainer._lgb_explainer = mock_lgb
        explainer._weights = {"xgboost": 0.5, "lightgbm": 0.5}

        X = np.random.default_rng(42).standard_normal((n_samples, n_features))
        feature_names = ["amount", "hour", "velocity"]
        txn_ids = ["TXN-A", "TXN-B"]

        explanations = explainer.explain(X, feature_names, txn_ids)

        assert len(explanations) == 2
        assert explanations[0].transaction_id == "TXN-A"
        assert explanations[1].transaction_id == "TXN-B"
        assert "amount" in explanations[0].feature_contributions
        assert isinstance(explanations[0].top_positive_features, list)
        assert isinstance(explanations[0].top_negative_features, list)

    def test_explain_auto_transaction_ids(self) -> None:
        """Auto-created transaction IDs are used when none provided."""
        explainer = ShapExplainer(top_k=1)

        mock_xgb = MagicMock()
        mock_xgb.shap_values.return_value = np.array([[0.1, 0.2]])
        mock_xgb.expected_value = 0.01

        mock_lgb = MagicMock()
        mock_lgb.shap_values.return_value = np.array([[0.1, 0.2]])
        mock_lgb.expected_value = 0.01

        explainer._xgb_explainer = mock_xgb
        explainer._lgb_explainer = mock_lgb
        explainer._weights = {"xgboost": 0.5, "lightgbm": 0.5}

        X = np.array([[1.0, 2.0]])
        explanations = explainer.explain(X, ["f1", "f2"])

        assert explanations[0].transaction_id == "TXN-0"

    def test_explain_handles_list_shap_values(self) -> None:
        """explain() handles SHAP values returned as list (binary classification)."""
        explainer = ShapExplainer(top_k=1)

        mock_xgb = MagicMock()
        mock_xgb.shap_values.return_value = [
            np.array([[-0.1, -0.2]]),  # class 0
            np.array([[0.1, 0.2]]),  # class 1 (fraud)
        ]
        mock_xgb.expected_value = np.array([0.98, 0.02])

        mock_lgb = MagicMock()
        mock_lgb.shap_values.return_value = [
            np.array([[-0.15, -0.25]]),
            np.array([[0.15, 0.25]]),
        ]
        mock_lgb.expected_value = np.array([0.97, 0.03])

        explainer._xgb_explainer = mock_xgb
        explainer._lgb_explainer = mock_lgb
        explainer._weights = {"xgboost": 0.5, "lightgbm": 0.5}

        X = np.array([[1.0, 2.0]])
        explanations = explainer.explain(X, ["f1", "f2"])

        assert len(explanations) == 1
        # Should use class 1 SHAP values
        assert explanations[0].feature_contributions["f1"] > 0


# ── NarrativeGenerator Tests ───────────────────────────────────────────────


class TestNarrativeGenerator:
    """Tests for the NarrativeGenerator class (mock Watsonx)."""

    def test_mock_narrative_generation(self) -> None:
        """Mock narrative is generated when Watsonx API key is not set."""
        from src.explainability.narrative_generator import NarrativeGenerator

        generator = NarrativeGenerator()

        explanation = ShapExplanation(
            transaction_id="TXN-TEST",
            base_value=0.02,
            feature_contributions={"amount": 0.35, "velocity": 0.20, "hour": -0.05},
            top_positive_features=[("amount", 0.35), ("velocity", 0.20)],
            top_negative_features=[("hour", -0.05)],
        )

        narrative = generator.generate(
            explanation=explanation,
            fraud_probability=0.85,
            label="fraud",
        )

        assert isinstance(narrative, str)
        assert "TXN-TEST" in narrative
        assert "fraud" in narrative
        assert len(narrative) > 20

    def test_mock_narrative_includes_factors(self) -> None:
        """Mock narrative references top risk factors."""
        from src.explainability.narrative_generator import NarrativeGenerator

        generator = NarrativeGenerator()

        explanation = ShapExplanation(
            transaction_id="TXN-002",
            base_value=0.01,
            feature_contributions={"amount": 0.50},
            top_positive_features=[("amount", 0.50), ("is_night", 0.15)],
            top_negative_features=[],
        )

        narrative = generator.generate(
            explanation=explanation,
            fraud_probability=0.75,
            label="fraud",
        )

        assert "amount" in narrative

    def test_mock_narrative_no_factors(self) -> None:
        """Mock narrative handles case with no positive features."""
        from src.explainability.narrative_generator import NarrativeGenerator

        generator = NarrativeGenerator()

        explanation = ShapExplanation(
            transaction_id="TXN-003",
            base_value=0.01,
            feature_contributions={},
            top_positive_features=[],
            top_negative_features=[],
        )

        narrative = generator.generate(
            explanation=explanation,
            fraud_probability=0.20,
            label="legitimate",
        )

        assert isinstance(narrative, str)
        assert "TXN-003" in narrative
