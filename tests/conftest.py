"""Shared fixtures for fraud detection tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_transactions() -> pd.DataFrame:
    """Create a small synthetic transaction DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 50

    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:05d}" for i in range(n)],
            "timestamp": timestamps,
            "amount": rng.lognormal(3.5, 1.0, size=n).clip(1.0, 5000.0).round(2),
            "merchant_id": rng.choice(
                [f"MERCH-{i:03d}" for i in range(10)], size=n
            ),
            "customer_id": rng.choice(
                [f"CUST-{i:04d}" for i in range(5)], size=n
            ),
            "merchant_category": rng.choice(
                ["grocery", "electronics", "restaurant", "online"], size=n
            ),
            "latitude": rng.uniform(30.0, 45.0, size=n).round(4),
            "longitude": rng.uniform(-100.0, -70.0, size=n).round(4),
            "customer_age_group": rng.choice(
                ["18-25", "26-35", "36-50", "51-65", "65+"], size=n
            ),
            "customer_gender": rng.choice(["M", "F", "NB"], size=n),
            "customer_geo": rng.choice(
                ["north_america", "europe", "asia"], size=n
            ),
            "is_fraud": np.concatenate([np.zeros(45), np.ones(5)]).astype(int),
        }
    )
    return df


@pytest.fixture()
def feature_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Create a small feature matrix and label array for model testing."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 10

    X = rng.standard_normal((n_samples, n_features))
    y = np.concatenate(
        [np.zeros(190), np.ones(10)]
    ).astype(int)

    shuffle_idx = rng.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]
