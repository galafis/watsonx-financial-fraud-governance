"""Transaction data ingestion from CSV files and Kafka streams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class TransactionLoader:
    """Load transaction data from CSV files or Kafka topics."""

    REQUIRED_COLUMNS = [
        "transaction_id",
        "timestamp",
        "amount",
        "merchant_id",
        "customer_id",
    ]

    def load_csv(self, file_path: str | Path) -> pd.DataFrame:
        """Load transactions from a CSV file.

        Args:
            file_path: Path to the CSV file containing transactions.

        Returns:
            DataFrame with validated and type-cast transaction data.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transaction file not found: {path}")

        logger.info("loading_csv", path=str(path))
        df = pd.read_csv(path)
        self._validate_columns(df)
        df = self._cast_types(df)
        logger.info("csv_loaded", rows=len(df), columns=list(df.columns))
        return df

    def load_from_kafka(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str = "fraud-detector",
        max_messages: int = 10000,
        timeout_ms: int = 5000,
    ) -> pd.DataFrame:
        """Load transactions from a Kafka topic.

        Args:
            bootstrap_servers: Kafka broker address(es).
            topic: Kafka topic name to consume from.
            group_id: Consumer group identifier.
            max_messages: Maximum number of messages to consume.
            timeout_ms: Poll timeout in milliseconds.

        Returns:
            DataFrame with consumed transactions.
        """
        from kafka import KafkaConsumer  # type: ignore[import-untyped]

        logger.info(
            "connecting_kafka",
            servers=bootstrap_servers,
            topic=topic,
            group=group_id,
        )

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=timeout_ms,
        )

        records: list[dict[str, Any]] = []
        for message in consumer:
            records.append(message.value)
            if len(records) >= max_messages:
                break

        consumer.close()

        if not records:
            logger.warning("kafka_no_messages", topic=topic)
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        df = pd.DataFrame(records)
        self._validate_columns(df)
        df = self._cast_types(df)
        logger.info("kafka_loaded", rows=len(df), topic=topic)
        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist in the DataFrame."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to appropriate types."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["amount"] = df["amount"].astype(np.float64)
        df["transaction_id"] = df["transaction_id"].astype(str)
        df["merchant_id"] = df["merchant_id"].astype(str)
        df["customer_id"] = df["customer_id"].astype(str)
        return df
