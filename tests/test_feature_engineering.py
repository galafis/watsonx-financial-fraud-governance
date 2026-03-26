"""Tests for feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def setup_method(self) -> None:
        """Initialize FeatureEngineer with test configuration."""
        self.config = {
            "velocity": {"windows": [1, 6]},
            "amount": {"percentiles": [25, 50, 75]},
            "temporal": {"cyclical": True},
            "geo": {"max_distance_km": 500.0},
        }
        self.fe = FeatureEngineer(config=self.config)

    def _make_df(self, n: int = 20) -> pd.DataFrame:
        """Create a minimal DataFrame for testing."""
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "transaction_id": [f"TXN-{i}" for i in range(n)],
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
                "amount": rng.lognormal(3.5, 1.0, size=n).clip(1.0, 5000.0).round(2),
                "merchant_id": rng.choice(["M1", "M2", "M3"], size=n),
                "customer_id": rng.choice(["C1", "C2"], size=n),
            }
        )

    def test_transform_adds_temporal_features(self) -> None:
        """Temporal features including cyclical encodings are added."""
        df = self._make_df()
        result = self.fe.transform(df)

        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "is_night" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns

    def test_transform_adds_velocity_features(self) -> None:
        """Velocity features for configured windows are added."""
        df = self._make_df()
        result = self.fe.transform(df)

        for window in [1, 6]:
            assert f"velocity_count_{window}h" in result.columns
            assert f"velocity_sum_{window}h" in result.columns
            assert f"velocity_mean_{window}h" in result.columns
            assert f"velocity_std_{window}h" in result.columns

    def test_transform_adds_amount_statistics(self) -> None:
        """Customer-level amount statistics and z-scores are added."""
        df = self._make_df()
        result = self.fe.transform(df)

        assert "customer_amount_mean" in result.columns
        assert "customer_amount_std" in result.columns
        assert "amount_zscore" in result.columns
        assert "amount_to_max_ratio" in result.columns
        assert "log_amount" in result.columns

    def test_transform_adds_merchant_deviation(self) -> None:
        """Merchant deviation features are added."""
        df = self._make_df()
        result = self.fe.transform(df)

        assert "merchant_avg_amount" in result.columns
        assert "merchant_std_amount" in result.columns
        assert "merchant_txn_count" in result.columns
        assert "customer_unique_merchants" in result.columns
        assert "is_new_merchant" in result.columns

    def test_transform_adds_geo_features_without_coords(self) -> None:
        """Geo features default to zero when lat/lon are missing."""
        df = self._make_df()
        result = self.fe.transform(df)

        assert "geo_distance_km" in result.columns
        assert "geo_anomaly" in result.columns
        assert (result["geo_distance_km"] == 0.0).all()
        assert (result["geo_anomaly"] == 0).all()

    def test_transform_adds_geo_features_with_coords(self) -> None:
        """Geo distance is computed when lat/lon are present."""
        df = self._make_df(n=10)
        rng = np.random.default_rng(99)
        df["latitude"] = rng.uniform(30.0, 45.0, size=10)
        df["longitude"] = rng.uniform(-100.0, -70.0, size=10)

        result = self.fe.transform(df)

        assert "geo_distance_km" in result.columns
        assert result["geo_distance_km"].dtype == np.float64

    def test_transform_preserves_row_count(self) -> None:
        """Output has same number of rows as input."""
        df = self._make_df(n=30)
        result = self.fe.transform(df)
        assert len(result) == 30

    def test_cyclical_disabled(self) -> None:
        """Cyclical features are not added when disabled."""
        config = {**self.config, "temporal": {"cyclical": False}}
        fe = FeatureEngineer(config=config)
        df = self._make_df()
        result = fe.transform(df)

        assert "hour_sin" not in result.columns
        assert "hour_cos" not in result.columns

    def test_get_feature_names(self) -> None:
        """Feature name list includes all expected feature groups."""
        names = self.fe.get_feature_names()

        assert "amount" in names
        assert "log_amount" in names
        assert "hour" in names
        assert "is_weekend" in names
        assert "hour_sin" in names
        assert "velocity_count_1h" in names
        assert "velocity_sum_6h" in names
        assert "geo_distance_km" in names

    def test_haversine_known_distance(self) -> None:
        """Haversine distance for known coordinates is approximately correct."""
        # New York to London is approximately 5570 km
        dist = FeatureEngineer._haversine(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < dist < 5700

    def test_haversine_same_point(self) -> None:
        """Distance between same point is zero."""
        dist = FeatureEngineer._haversine(40.0, -74.0, 40.0, -74.0)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_velocity_first_transaction_is_zero(self) -> None:
        """First transaction for a customer should have zero velocity."""
        df = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2"],
                "timestamp": pd.to_datetime(
                    ["2024-01-01 10:00:00", "2024-01-01 10:30:00"], utc=True
                ),
                "amount": [100.0, 200.0],
                "merchant_id": ["M1", "M1"],
                "customer_id": ["C1", "C1"],
            }
        )
        result = self.fe.transform(df)
        assert result.iloc[0]["velocity_count_1h"] == 0

    def test_amount_zscore_single_transaction(self) -> None:
        """Z-score is 0 when customer has a single transaction (std=0)."""
        df = pd.DataFrame(
            {
                "transaction_id": ["T1"],
                "timestamp": pd.to_datetime(["2024-01-01 10:00:00"], utc=True),
                "amount": [500.0],
                "merchant_id": ["M1"],
                "customer_id": ["C1"],
            }
        )
        result = self.fe.transform(df)
        assert result.iloc[0]["amount_zscore"] == pytest.approx(0.0)
