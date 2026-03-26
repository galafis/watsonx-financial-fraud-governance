"""Tests for FastAPI routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.routes import app


@pytest.fixture()
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Health check returns 200 with correct structure."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_loaded" in data
        assert "ensemble_models" in data

    def test_health_check_lists_models(self, client: TestClient) -> None:
        """Health check includes all ensemble model names."""
        response = client.get("/health")
        data = response.json()
        assert "xgboost" in data["ensemble_models"]
        assert "lightgbm" in data["ensemble_models"]
        assert "isolation_forest" in data["ensemble_models"]


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def _make_request(self, amount: float = 250.0) -> dict:
        """Create a valid prediction request payload."""
        return {
            "transaction_id": "TXN-TEST-001",
            "amount": amount,
            "merchant_id": "MERCH-001",
            "customer_id": "CUST-001",
            "timestamp": "2024-03-15T14:30:00Z",
            "merchant_category": "electronics",
            "latitude": 40.7128,
            "longitude": -74.0060,
        }

    def test_predict_single(self, client: TestClient) -> None:
        """Single prediction returns correct structure."""
        response = client.post("/predict", json=self._make_request())
        assert response.status_code == 200

        data = response.json()
        assert "transaction_id" in data
        assert "fraud_probability" in data
        assert "label" in data
        assert data["transaction_id"] == "TXN-TEST-001"

    def test_predict_label_is_valid(self, client: TestClient) -> None:
        """Prediction label is one of the expected values."""
        response = client.post("/predict", json=self._make_request())
        data = response.json()
        assert data["label"] in {"fraud", "review", "legitimate"}

    def test_predict_probability_range(self, client: TestClient) -> None:
        """Fraud probability is in [0, 1]."""
        response = client.post("/predict", json=self._make_request())
        data = response.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_high_amount_higher_score(self, client: TestClient) -> None:
        """Higher amounts produce higher fraud scores (mock logic)."""
        low_resp = client.post("/predict", json=self._make_request(amount=10.0))
        high_resp = client.post("/predict", json=self._make_request(amount=10000.0))

        low_score = low_resp.json()["fraud_probability"]
        high_score = high_resp.json()["fraud_probability"]
        assert high_score > low_score

    def test_predict_missing_required_field(self, client: TestClient) -> None:
        """Missing required fields return 422."""
        response = client.post("/predict", json={"amount": 100.0})
        assert response.status_code == 422

    def test_predict_invalid_amount(self, client: TestClient) -> None:
        """Amount <= 0 returns 422 validation error."""
        request = self._make_request()
        request["amount"] = -100.0
        response = client.post("/predict", json=request)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict(self, client: TestClient) -> None:
        """Batch prediction returns results for all transactions."""
        transactions = [
            {
                "transaction_id": f"TXN-BATCH-{i}",
                "amount": 100.0 * (i + 1),
                "merchant_id": "MERCH-001",
                "customer_id": "CUST-001",
                "timestamp": "2024-03-15T14:30:00Z",
            }
            for i in range(3)
        ]
        response = client.post("/predict/batch", json={"transactions": transactions})
        assert response.status_code == 200
        assert len(response.json()) == 3

    def test_batch_predict_empty(self, client: TestClient) -> None:
        """Empty batch returns 422."""
        response = client.post("/predict/batch", json={"transactions": []})
        assert response.status_code == 422


class TestExplainEndpoint:
    """Tests for the /explain endpoint."""

    def test_explain_returns_explanation(self, client: TestClient) -> None:
        """Explanation includes risk factors and narrative."""
        request = {
            "transaction_id": "TXN-EXPLAIN-001",
            "amount": 5000.0,
            "merchant_id": "MERCH-001",
            "customer_id": "CUST-001",
            "timestamp": "2024-03-15T14:30:00Z",
        }
        response = client.post("/explain", json=request)
        assert response.status_code == 200

        data = response.json()
        assert "transaction_id" in data
        assert "fraud_probability" in data
        assert "label" in data
        assert "top_risk_factors" in data
        assert "top_mitigating_factors" in data
        assert "narrative" in data

    def test_explain_narrative_not_empty(self, client: TestClient) -> None:
        """Narrative explanation is a non-empty string."""
        request = {
            "transaction_id": "TXN-EXPLAIN-002",
            "amount": 2500.0,
            "merchant_id": "MERCH-001",
            "customer_id": "CUST-001",
            "timestamp": "2024-03-15T14:30:00Z",
        }
        response = client.post("/explain", json=request)
        data = response.json()
        assert len(data["narrative"]) > 0


class TestGovernanceEndpoints:
    """Tests for governance-related endpoints."""

    def test_governance_metrics(self, client: TestClient) -> None:
        """Governance metrics returns correct structure."""
        response = client.get("/governance/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "total_predictions" in data
        assert "fraud_rate" in data
        assert "review_rate" in data
        assert "avg_fraud_probability" in data

    def test_factsheet_returns_dict(self, client: TestClient) -> None:
        """Factsheet returns structured JSON."""
        response = client.get("/governance/factsheet")
        assert response.status_code == 200

        data = response.json()
        assert "metadata" in data
        assert "purpose_and_scope" in data
        assert "technical_specification" in data
        assert "risk_management" in data

    def test_factsheet_eu_ai_act_reference(self, client: TestClient) -> None:
        """Factsheet references EU AI Act Article 11."""
        response = client.get("/governance/factsheet")
        data = response.json()
        assert data["metadata"]["eu_ai_act_article"] == "Article 11"

    def test_bias_report(self, client: TestClient) -> None:
        """Bias report returns correct structure."""
        response = client.get("/governance/bias-report")
        assert response.status_code == 200

        data = response.json()
        assert "is_fair" in data
        assert "alerts" in data
        assert "protected_attributes" in data

    def test_drift_report(self, client: TestClient) -> None:
        """Drift report returns correct structure."""
        response = client.get("/governance/drift-report")
        assert response.status_code == 200

        data = response.json()
        assert "drift_detected" in data
        assert "drifted_features_count" in data


class TestOpenAPIDocs:
    """Tests for API documentation."""

    def test_openapi_schema_available(self, client: TestClient) -> None:
        """OpenAPI schema is served at /openapi.json."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_docs_page_available(self, client: TestClient) -> None:
        """Swagger UI docs page is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
