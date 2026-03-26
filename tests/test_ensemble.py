"""Tests for ensemble fraud detector and prediction models."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.anomaly import AnomalyDetector
from src.models.ensemble import EnsembleFraudDetector, FraudPrediction


class TestFraudPrediction:
    """Tests for the FraudPrediction dataclass."""

    def test_basic_creation(self) -> None:
        """FraudPrediction stores all fields correctly."""
        pred = FraudPrediction(
            transaction_id="TXN-001",
            fraud_probability=0.85,
            label="fraud",
            model_scores={"xgboost": 0.90, "lightgbm": 0.80, "isolation_forest": 0.75},
        )
        assert pred.transaction_id == "TXN-001"
        assert pred.fraud_probability == 0.85
        assert pred.label == "fraud"
        assert pred.model_scores["xgboost"] == 0.90

    def test_default_model_scores(self) -> None:
        """Default model_scores is an empty dict."""
        pred = FraudPrediction(
            transaction_id="TXN-002",
            fraud_probability=0.10,
            label="legitimate",
        )
        assert pred.model_scores == {}


class TestAnomalyDetector:
    """Tests for the Isolation Forest anomaly detector."""

    def setup_method(self) -> None:
        """Initialize with test configuration."""
        self.config = {
            "n_estimators": 50,
            "contamination": 0.05,
            "random_state": 42,
        }
        self.detector = AnomalyDetector(config=self.config)

    def test_fit_returns_self(self) -> None:
        """fit() returns self for method chaining."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        result = self.detector.fit(X)
        assert result is self.detector

    def test_predict_scores_range(self) -> None:
        """Anomaly scores are normalized to [0, 1]."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        self.detector.fit(X)

        scores = self.detector.predict_scores(X)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_predict_scores_shape(self) -> None:
        """Output shape matches number of input samples."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        self.detector.fit(X)

        scores = self.detector.predict_scores(X)
        assert scores.shape == (100,)

    def test_predict_labels_binary(self) -> None:
        """Labels are binary (0 or 1)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        self.detector.fit(X)

        labels = self.detector.predict_labels(X)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_predict_scores_not_fitted_raises(self) -> None:
        """predict_scores raises RuntimeError when not fitted."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            self.detector.predict_scores(X)

    def test_predict_labels_not_fitted_raises(self) -> None:
        """predict_labels raises RuntimeError when not fitted."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            self.detector.predict_labels(X)

    def test_constant_data_returns_half(self) -> None:
        """Constant data produces scores of 0.5 (uniform anomaly)."""
        X = np.ones((50, 3))
        self.detector.fit(X)
        scores = self.detector.predict_scores(X)
        assert np.allclose(scores, 0.5)


class TestEnsembleFraudDetector:
    """Tests for the weighted ensemble fraud detector."""

    def setup_method(self) -> None:
        """Initialize ensemble with small test configuration."""
        self.config = {
            "weights": {
                "xgboost": 0.40,
                "lightgbm": 0.40,
                "isolation_forest": 0.20,
            },
            "threshold": {
                "fraud": 0.70,
                "review": 0.40,
            },
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": 42,
                "scale_pos_weight": 5,
            },
            "lightgbm": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": 42,
                "scale_pos_weight": 5,
            },
            "isolation_forest": {
                "n_estimators": 20,
                "contamination": 0.05,
                "random_state": 42,
            },
        }

    def _make_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create a simple classification dataset."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 5))
        y = np.concatenate([np.zeros(180), np.ones(20)])
        idx = rng.permutation(n)
        return X[idx], y[idx]

    def test_fit_returns_self(self) -> None:
        """fit() returns self for method chaining."""
        X, y = self._make_data()
        model = EnsembleFraudDetector(config=self.config)
        result = model.fit(X, y)
        assert result is model

    def test_predict_returns_fraud_predictions(self) -> None:
        """predict() returns a list of FraudPrediction objects."""
        X, y = self._make_data()
        model = EnsembleFraudDetector(config=self.config)
        model.fit(X, y)

        predictions = model.predict(X[:5])
        assert len(predictions) == 5
        assert all(isinstance(p, FraudPrediction) for p in predictions)

    def test_predict_labels_valid(self) -> None:
        """All prediction labels are one of fraud, review, legitimate."""
        X, y = self._make_data()
        model = EnsembleFraudDetector(config=self.config)
        model.fit(X, y)

        predictions = model.predict(X)
        valid_labels = {"fraud", "review", "legitimate"}
        for p in predictions:
            assert p.label in valid_labels

    def test_predict_probability_range(self) -> None:
        """Fraud probabilities are in [0, 1]."""
        X, y = self._make_data()
        model = EnsembleFraudDetector(config=self.config)
        model.fit(X, y)

        predictions = model.predict(X)
        for p in predictions:
            assert 0.0 <= p.fraud_probability <= 1.0

    def test_predict_with_transaction_ids(self) -> None:
        """Custom transaction IDs are propagated."""
        X, y = self._make_data()
        model = EnsembleFraudDetector(config=self.config)
        model.fit(X, y)

        ids = ["A", "B", "C"]
        predictions = model.predict(X[:3], transaction_ids=ids)
        assert [p.transaction_id for p in predictions] == ids

    def test_predict_not_fitted_raises(self) -> None:
        """predict() raises RuntimeError when model is not fitted."""
        model = EnsembleFraudDetector(config=self.config)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_predict_proba_shape(self) -> None:
        """predict_proba() returns correct shape."""
        X, y = self._make_data()
        model = EnsembleFraudDetector(config=self.config)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X),)

    def test_predict_proba_not_fitted_raises(self) -> None:
        """predict_proba() raises RuntimeError when not fitted."""
        model = EnsembleFraudDetector(config=self.config)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_model_scores_contain_all_submodels(self) -> None:
        """Each prediction includes scores from all three sub-models."""
        X, y = self._make_data()
        model = EnsembleFraudDetector(config=self.config)
        model.fit(X, y)

        predictions = model.predict(X[:1])
        scores = predictions[0].model_scores
        assert "xgboost" in scores
        assert "lightgbm" in scores
        assert "isolation_forest" in scores

    def test_classify_thresholds(self) -> None:
        """Classification thresholds produce correct labels."""
        model = EnsembleFraudDetector(config=self.config)
        assert model._classify(0.85) == "fraud"
        assert model._classify(0.70) == "fraud"
        assert model._classify(0.55) == "review"
        assert model._classify(0.40) == "review"
        assert model._classify(0.30) == "legitimate"
        assert model._classify(0.0) == "legitimate"
