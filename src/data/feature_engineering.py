"""Feature engineering for fraud detection.

Extracts temporal patterns, transaction velocity, amount statistics,
merchant category deviation, and geo-distance anomalies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class FeatureEngineer:
    """Generate fraud-detection features from raw transaction data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or settings.features_config
        self.velocity_windows: list[int] = self.config.get("velocity", {}).get(
            "windows", [1, 6, 24, 72]
        )
        self.amount_percentiles: list[int] = self.config.get("amount", {}).get(
            "percentiles", [25, 50, 75, 90, 95, 99]
        )
        self.use_cyclical: bool = self.config.get("temporal", {}).get("cyclical", True)
        self.geo_max_km: float = self.config.get("geo", {}).get("max_distance_km", 500.0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps.

        Args:
            df: Raw transaction DataFrame with at least columns:
                transaction_id, timestamp, amount, merchant_id, customer_id.

        Returns:
            DataFrame augmented with engineered features.
        """
        logger.info("feature_engineering_start", rows=len(df))
        df = df.copy().sort_values("timestamp").reset_index(drop=True)

        df = self._temporal_features(df)
        df = self._velocity_features(df)
        df = self._amount_statistics(df)
        df = self._merchant_deviation(df)
        df = self._geo_distance_features(df)

        logger.info("feature_engineering_done", features=len(df.columns))
        return df

    # ── Temporal Features ────────────────────────────────────────────────

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour-of-day, day-of-week, and cyclical encodings."""
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

        if self.use_cyclical:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        return df

    # ── Velocity Features ────────────────────────────────────────────────

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute transaction velocity over rolling windows per customer."""
        ts = pd.to_datetime(df["timestamp"])

        for window_h in self.velocity_windows:
            suffix = f"_{window_h}h"
            counts: list[int] = []
            sums: list[float] = []
            means: list[float] = []
            stds: list[float] = []

            for idx in range(len(df)):
                customer = df.iloc[idx]["customer_id"]
                current_ts = ts.iloc[idx]
                window_start = current_ts - pd.Timedelta(hours=window_h)

                mask = (df["customer_id"] == customer) & (ts >= window_start) & (ts < current_ts)
                window_amounts = df.loc[mask, "amount"]

                counts.append(len(window_amounts))
                sums.append(float(window_amounts.sum()) if len(window_amounts) > 0 else 0.0)
                means.append(float(window_amounts.mean()) if len(window_amounts) > 0 else 0.0)
                stds.append(float(window_amounts.std()) if len(window_amounts) > 1 else 0.0)

            df[f"velocity_count{suffix}"] = counts
            df[f"velocity_sum{suffix}"] = sums
            df[f"velocity_mean{suffix}"] = means
            df[f"velocity_std{suffix}"] = stds

        return df

    # ── Amount Statistics ────────────────────────────────────────────────

    def _amount_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute amount deviation from customer's historical statistics."""
        customer_stats = df.groupby("customer_id")["amount"].agg(["mean", "std", "min", "max"])
        customer_stats.columns = [
            "customer_amount_mean",
            "customer_amount_std",
            "customer_amount_min",
            "customer_amount_max",
        ]
        customer_stats["customer_amount_std"] = customer_stats["customer_amount_std"].fillna(0)

        df = df.merge(customer_stats, on="customer_id", how="left")
        df["amount_zscore"] = np.where(
            df["customer_amount_std"] > 0,
            (df["amount"] - df["customer_amount_mean"]) / df["customer_amount_std"],
            0.0,
        )
        df["amount_to_max_ratio"] = np.where(
            df["customer_amount_max"] > 0,
            df["amount"] / df["customer_amount_max"],
            1.0,
        )
        df["log_amount"] = np.log1p(df["amount"])

        return df

    # ── Merchant Category Deviation ──────────────────────────────────────

    def _merchant_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect unusual merchant categories for each customer."""
        merchant_stats = df.groupby("merchant_id")["amount"].agg(["mean", "std", "count"])
        merchant_stats.columns = [
            "merchant_avg_amount",
            "merchant_std_amount",
            "merchant_txn_count",
        ]
        merchant_stats["merchant_std_amount"] = merchant_stats["merchant_std_amount"].fillna(0)

        df = df.merge(merchant_stats, on="merchant_id", how="left")

        # Count unique merchants per customer
        customer_merchants = df.groupby("customer_id")["merchant_id"].nunique()
        customer_merchants.name = "customer_unique_merchants"
        df = df.merge(customer_merchants, on="customer_id", how="left")

        # Flag if this merchant is new for the customer
        customer_merchant_pairs = df.groupby(["customer_id", "merchant_id"]).cumcount()
        df["is_new_merchant"] = (customer_merchant_pairs == 0).astype(int)

        return df

    # ── Geo-Distance Features ────────────────────────────────────────────

    def _geo_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute geographic distance anomalies if lat/lon are available."""
        if "latitude" not in df.columns or "longitude" not in df.columns:
            df["geo_distance_km"] = 0.0
            df["geo_anomaly"] = 0
            return df

        distances: list[float] = []
        for idx in range(len(df)):
            customer = df.iloc[idx]["customer_id"]
            current_lat = df.iloc[idx]["latitude"]
            current_lon = df.iloc[idx]["longitude"]

            prev = df.iloc[:idx]
            prev_customer = prev[prev["customer_id"] == customer]

            if len(prev_customer) == 0:
                distances.append(0.0)
                continue

            last = prev_customer.iloc[-1]
            dist = self._haversine(current_lat, current_lon, last["latitude"], last["longitude"])
            distances.append(dist)

        df["geo_distance_km"] = distances
        df["geo_anomaly"] = (df["geo_distance_km"] > self.geo_max_km).astype(int)
        return df

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute Haversine distance in km between two lat/lon points."""
        r = 6371.0  # Earth radius in km
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        return float(2 * r * np.arcsin(np.sqrt(a)))

    def get_feature_names(self) -> list[str]:
        """Return the list of engineered feature names for model input."""
        base = [
            "amount",
            "log_amount",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_night",
            "amount_zscore",
            "amount_to_max_ratio",
            "customer_amount_mean",
            "customer_amount_std",
            "customer_amount_min",
            "customer_amount_max",
            "merchant_avg_amount",
            "merchant_std_amount",
            "merchant_txn_count",
            "customer_unique_merchants",
            "is_new_merchant",
            "geo_distance_km",
            "geo_anomaly",
        ]

        if self.use_cyclical:
            base.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])

        for w in self.velocity_windows:
            for agg in ["count", "sum", "mean", "std"]:
                base.append(f"velocity_{agg}_{w}h")

        return base
