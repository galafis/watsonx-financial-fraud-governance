"""Generate human-readable fraud alert narratives using IBM Watsonx Granite."""

from __future__ import annotations

from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.explainability.shap_explainer import ShapExplanation

logger = structlog.get_logger(__name__)

_NARRATIVE_PROMPT = """You are a fraud investigation assistant. Given the SHAP explanation
for a fraud detection alert, generate a clear, concise narrative for a human analyst.

Transaction ID: {transaction_id}
Fraud Probability: {fraud_probability:.1%}
Decision: {label}

Top risk factors (features increasing fraud score):
{positive_features}

Top mitigating factors (features decreasing fraud score):
{negative_features}

Write a 2-3 sentence narrative explaining WHY this transaction was flagged,
what specific risk indicators were detected, and recommended next steps.
Be specific and reference the actual feature values. Use professional language
suitable for a fraud analyst reviewing this alert."""


class NarrativeGenerator:
    """Generate natural-language fraud alert narratives via Watsonx Granite.

    Converts SHAP explanations into human-readable summaries that help
    fraud analysts understand model decisions and prioritize investigations.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._initialized = False

    def _initialize_model(self) -> None:
        """Lazy-initialize the Watsonx foundation model."""
        if self._initialized:
            return

        if not settings.watsonx.api_key:
            logger.warning("watsonx_api_key_not_set", msg="Using mock narrative generation")
            self._initialized = True
            return

        try:
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai import Credentials

            credentials = Credentials(
                url=settings.watsonx.url,
                api_key=settings.watsonx.api_key,
            )

            self._model = ModelInference(
                model_id=settings.watsonx.generation_model,
                credentials=credentials,
                project_id=settings.watsonx.project_id,
                params=settings.generation_params,
            )
            self._initialized = True
            logger.info("narrative_model_initialized", model=settings.watsonx.generation_model)
        except Exception as e:
            logger.error("narrative_model_init_failed", error=str(e))
            self._initialized = True  # Avoid retrying init

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def generate(
        self,
        explanation: ShapExplanation,
        fraud_probability: float,
        label: str,
    ) -> str:
        """Generate a fraud alert narrative from a SHAP explanation.

        Args:
            explanation: SHAP explanation for the transaction.
            fraud_probability: Ensemble fraud probability.
            label: Decision label (fraud/review/legitimate).

        Returns:
            Human-readable fraud alert narrative.
        """
        self._initialize_model()

        positive_features = "\n".join(
            f"  - {name}: contribution = {val:+.4f}"
            for name, val in explanation.top_positive_features
        ) or "  (none)"

        negative_features = "\n".join(
            f"  - {name}: contribution = {val:+.4f}"
            for name, val in explanation.top_negative_features
        ) or "  (none)"

        prompt = _NARRATIVE_PROMPT.format(
            transaction_id=explanation.transaction_id,
            fraud_probability=fraud_probability,
            label=label,
            positive_features=positive_features,
            negative_features=negative_features,
        )

        if self._model is None:
            return self._mock_narrative(explanation, fraud_probability, label)

        try:
            response = self._model.generate_text(prompt=prompt)
            narrative = response.strip()
            logger.info(
                "narrative_generated",
                txn_id=explanation.transaction_id,
                length=len(narrative),
            )
            return narrative
        except Exception as e:
            logger.error("narrative_generation_failed", error=str(e))
            return self._mock_narrative(explanation, fraud_probability, label)

    def _mock_narrative(
        self,
        explanation: ShapExplanation,
        fraud_probability: float,
        label: str,
    ) -> str:
        """Generate a template-based narrative when Watsonx is unavailable."""
        top_factors = [name for name, _ in explanation.top_positive_features[:3]]
        factors_str = ", ".join(top_factors) if top_factors else "no specific factors"

        return (
            f"Transaction {explanation.transaction_id} was classified as '{label}' "
            f"with a fraud probability of {fraud_probability:.1%}. "
            f"Key risk indicators include: {factors_str}. "
            f"Recommend manual review by fraud analyst for verification."
        )
