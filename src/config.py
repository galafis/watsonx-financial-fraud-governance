"""Application configuration loaded from environment variables and settings.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml_config() -> dict[str, Any]:
    """Load YAML configuration from config/settings.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml_config()


class WatsonxSettings(BaseSettings):
    """IBM Watsonx API connection settings."""

    api_key: str = Field(default="", alias="WATSONX_API_KEY")
    project_id: str = Field(default="", alias="WATSONX_PROJECT_ID")
    url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        alias="WATSONX_URL",
    )
    generation_model: str = Field(default="ibm/granite-13b-chat-v2")


class KafkaSettings(BaseSettings):
    """Apache Kafka connection settings."""

    bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    topic: str = Field(default="transactions", alias="KAFKA_TOPIC")
    group_id: str = Field(default="fraud-detector", alias="KAFKA_GROUP_ID")


class AppSettings(BaseSettings):
    """General application settings."""

    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8080, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")


class Settings:
    """Aggregated application settings."""

    def __init__(self) -> None:
        self.watsonx = WatsonxSettings()
        self.kafka = KafkaSettings()
        self.app = AppSettings()
        self.yaml = _yaml

    @property
    def model_config(self) -> dict[str, Any]:
        """Ensemble model configuration."""
        return self.yaml.get("model", {})

    @property
    def ensemble_config(self) -> dict[str, Any]:
        """Ensemble sub-model configuration."""
        return self.model_config.get("ensemble", {})

    @property
    def threshold_config(self) -> dict[str, Any]:
        """Decision threshold configuration."""
        return self.model_config.get("threshold", {})

    @property
    def features_config(self) -> dict[str, Any]:
        """Feature engineering configuration."""
        return self.yaml.get("features", {})

    @property
    def governance_config(self) -> dict[str, Any]:
        """Governance thresholds and parameters."""
        return self.yaml.get("governance", {})

    @property
    def synthetic_config(self) -> dict[str, Any]:
        """Synthetic data generation parameters."""
        return self.yaml.get("synthetic", {})

    @property
    def generation_params(self) -> dict[str, Any]:
        """Watsonx generation model parameters."""
        return self.yaml.get("watsonx", {}).get("generation", {}).get("parameters", {})


settings = Settings()
