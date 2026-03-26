"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request Models ───────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Request model for fraud prediction."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    customer_id: str = Field(..., description="Customer identifier")
    timestamp: str = Field(..., description="Transaction timestamp (ISO 8601)")
    merchant_category: str = Field(default="", description="Merchant category")
    latitude: float = Field(default=0.0, description="Transaction latitude")
    longitude: float = Field(default=0.0, description="Transaction longitude")


class BatchPredictRequest(BaseModel):
    """Request model for batch fraud prediction."""

    transactions: list[PredictRequest] = Field(
        ..., min_length=1, max_length=1000, description="List of transactions"
    )


# ── Response Models ──────────────────────────────────────────────────────────


class ModelScores(BaseModel):
    """Individual model scores within the ensemble."""

    xgboost: float = 0.0
    lightgbm: float = 0.0
    isolation_forest: float = 0.0


class PredictResponse(BaseModel):
    """Response model for fraud prediction."""

    transaction_id: str
    fraud_probability: float
    label: str  # fraud, review, legitimate
    model_scores: ModelScores = ModelScores()
    narrative: str = ""


class ExplainResponse(BaseModel):
    """Response model for prediction explanation."""

    transaction_id: str
    fraud_probability: float
    label: str
    top_risk_factors: list[dict[str, float]] = []
    top_mitigating_factors: list[dict[str, float]] = []
    narrative: str = ""


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    model_loaded: bool
    ensemble_models: list[str] = []


class GovernanceMetricsResponse(BaseModel):
    """Response model for governance metrics."""

    total_predictions: int = 0
    fraud_rate: float = 0.0
    review_rate: float = 0.0
    avg_fraud_probability: float = 0.0
    model_version: str = "1.0.0"


class BiasReportResponse(BaseModel):
    """Response model for bias detection report."""

    is_fair: bool = True
    alerts: list[str] = []
    protected_attributes: list[str] = []
    attribute_details: dict = {}


class DriftReportResponse(BaseModel):
    """Response model for drift monitoring report."""

    drift_detected: bool = False
    drifted_features_count: int = 0
    drifted_features: list[str] = []
    psi_threshold: float = 0.20
    ks_threshold: float = 0.05
