"""Generate realistic synthetic transaction data for fraud detection training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class SyntheticTransactionGenerator:
    """Generate synthetic financial transaction data with configurable fraud rate."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or settings.synthetic_config
        self.n_transactions: int = cfg.get("n_transactions", 100_000)
        self.fraud_rate: float = cfg.get("fraud_rate", 0.02)
        self.n_merchants: int = cfg.get("n_merchants", 500)
        self.n_customers: int = cfg.get("n_customers", 10_000)
        self.seed: int = cfg.get("seed", 42)

    def generate(self, n_transactions: int | None = None) -> pd.DataFrame:
        """Generate a synthetic transaction dataset.

        Args:
            n_transactions: Override the default number of transactions.

        Returns:
            DataFrame with columns: transaction_id, timestamp, amount,
            merchant_id, customer_id, merchant_category, latitude, longitude,
            customer_age_group, customer_gender, customer_geo, is_fraud.
        """
        n = n_transactions or self.n_transactions
        rng = np.random.default_rng(self.seed)

        logger.info("generating_synthetic_data", n=n, fraud_rate=self.fraud_rate)

        # Generate base fields
        transaction_ids = [f"TXN-{i:08d}" for i in range(n)]
        timestamps = self._generate_timestamps(n, rng)
        customer_ids = rng.choice([f"CUST-{i:06d}" for i in range(self.n_customers)], size=n)
        merchant_ids = rng.choice([f"MERCH-{i:05d}" for i in range(self.n_merchants)], size=n)

        categories = ["grocery", "electronics", "gas_station", "restaurant", "online", "travel"]
        merchant_categories = rng.choice(categories, size=n)

        # Generate amounts — legitimate vs fraudulent
        n_fraud = int(n * self.fraud_rate)
        n_legit = n - n_fraud

        legit_amounts = rng.lognormal(mean=3.5, sigma=1.0, size=n_legit)
        legit_amounts = np.clip(legit_amounts, 1.0, 5000.0)

        fraud_amounts = rng.lognormal(mean=5.5, sigma=1.5, size=n_fraud)
        fraud_amounts = np.clip(fraud_amounts, 50.0, 50000.0)

        amounts = np.concatenate([legit_amounts, fraud_amounts])
        is_fraud = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

        # Shuffle
        shuffle_idx = rng.permutation(n)
        amounts = amounts[shuffle_idx]
        is_fraud = is_fraud[shuffle_idx]

        # Generate location data
        latitudes = rng.uniform(-33.0, 48.0, size=n)
        longitudes = rng.uniform(-120.0, 50.0, size=n)

        # Protected attributes for bias testing
        age_groups = rng.choice(["18-25", "26-35", "36-50", "51-65", "65+"], size=n)
        genders = rng.choice(["M", "F", "NB"], size=n, p=[0.48, 0.48, 0.04])
        geographies = rng.choice(
            ["north_america", "europe", "asia", "south_america", "africa"],
            size=n,
            p=[0.35, 0.30, 0.20, 0.10, 0.05],
        )

        df = pd.DataFrame(
            {
                "transaction_id": transaction_ids,
                "timestamp": timestamps,
                "amount": np.round(amounts, 2),
                "merchant_id": merchant_ids,
                "customer_id": customer_ids,
                "merchant_category": merchant_categories,
                "latitude": np.round(latitudes, 6),
                "longitude": np.round(longitudes, 6),
                "customer_age_group": age_groups,
                "customer_gender": genders,
                "customer_geo": geographies,
                "is_fraud": is_fraud.astype(int),
            }
        )

        logger.info(
            "synthetic_data_generated",
            total=len(df),
            fraud_count=int(df["is_fraud"].sum()),
            fraud_pct=round(df["is_fraud"].mean() * 100, 2),
        )
        return df

    def _generate_timestamps(self, n: int, rng: np.random.Generator) -> pd.DatetimeIndex:
        """Generate realistic timestamps over a 90-day window."""
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-03-31", tz="UTC")
        delta = (end - start).total_seconds()

        # Weight towards business hours
        raw_offsets = rng.uniform(0, delta, size=n)
        timestamps = pd.to_datetime(
            [start + pd.Timedelta(seconds=float(s)) for s in sorted(raw_offsets)], utc=True
        )
        return timestamps
