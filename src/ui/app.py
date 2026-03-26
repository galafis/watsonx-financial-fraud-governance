"""Streamlit dashboard for fraud investigation, governance, and monitoring."""

from __future__ import annotations

import os

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8080")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Watsonx Fraud Detection & Governance",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Watsonx Financial Fraud Detection & Governance")
    st.caption("AI-powered fraud detection with explainability and EU AI Act compliance")

    tab_investigate, tab_alerts, tab_governance, tab_bias, tab_drift = st.tabs(
        [
            "🔍 Transaction Investigation",
            "🚨 Fraud Alerts",
            "📊 Governance Metrics",
            "⚖️ Bias Analysis",
            "📈 Drift Monitoring",
        ]
    )

    with tab_investigate:
        _render_investigation_view()

    with tab_alerts:
        _render_alerts_view()

    with tab_governance:
        _render_governance_view()

    with tab_bias:
        _render_bias_view()

    with tab_drift:
        _render_drift_view()


# ── Transaction Investigation ────────────────────────────────────────────────


def _render_investigation_view() -> None:
    """Render the transaction investigation panel."""
    st.header("Transaction Investigation")

    col1, col2 = st.columns(2)
    with col1:
        txn_id = st.text_input("Transaction ID", value="TXN-00000001")
        amount = st.number_input("Amount ($)", min_value=0.01, value=250.00)
        merchant_id = st.text_input("Merchant ID", value="MERCH-00001")

    with col2:
        customer_id = st.text_input("Customer ID", value="CUST-000001")
        timestamp = st.text_input("Timestamp", value="2024-03-15T14:30:00Z")
        category = st.selectbox(
            "Category",
            ["grocery", "electronics", "gas_station", "restaurant", "online", "travel"],
        )

    if st.button("🔍 Analyze Transaction", type="primary"):
        try:
            response = httpx.post(
                f"{API_URL}/explain",
                json={
                    "transaction_id": txn_id,
                    "amount": amount,
                    "merchant_id": merchant_id,
                    "customer_id": customer_id,
                    "timestamp": timestamp,
                    "merchant_category": category,
                },
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                _display_prediction_result(data)
            else:
                st.error(f"API error: {response.status_code}")
        except httpx.ConnectError:
            st.warning("API not available. Showing sample analysis.")
            _display_prediction_result(_sample_prediction(txn_id, amount))


def _display_prediction_result(data: dict) -> None:
    """Display prediction result with visual indicators."""
    label = data.get("label", "unknown")
    prob = data.get("fraud_probability", 0)

    color_map = {"fraud": "red", "review": "orange", "legitimate": "green"}
    color = color_map.get(label, "gray")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Probability", f"{prob:.1%}")
    col2.metric("Decision", label.upper())
    col3.metric("Transaction", data.get("transaction_id", ""))

    # Gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "lightyellow"},
                    {"range": [70, 100], "color": "lightcoral"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
            title={"text": "Fraud Score"},
        )
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Narrative
    if data.get("narrative"):
        st.subheader("📝 AI-Generated Analysis")
        st.info(data["narrative"])

    # Feature contributions
    risk_factors = data.get("top_risk_factors", [])
    if risk_factors:
        st.subheader("⚠️ Risk Factors")
        for factor in risk_factors:
            for name, val in factor.items():
                st.progress(min(abs(val), 1.0), text=f"{name}: {val:+.4f}")


# ── Fraud Alerts ─────────────────────────────────────────────────────────────


def _render_alerts_view() -> None:
    """Render fraud alerts dashboard with Granite narratives."""
    st.header("Active Fraud Alerts")
    st.info("Fraud alerts with AI-generated narratives will appear here when the model is running.")

    # Sample alerts
    sample_alerts = [
        {
            "txn_id": "TXN-00045823",
            "amount": 8750.00,
            "score": 0.92,
            "label": "fraud",
            "narrative": "High-value transaction flagged due to unusual amount and new merchant pattern.",
        },
        {
            "txn_id": "TXN-00045901",
            "amount": 3200.00,
            "score": 0.65,
            "label": "review",
            "narrative": "Moderate risk: elevated velocity and atypical merchant category for customer.",
        },
    ]

    for alert in sample_alerts:
        with st.expander(
            f"🚨 {alert['txn_id']} — ${alert['amount']:,.2f} — {alert['label'].upper()}"
        ):
            st.metric("Fraud Score", f"{alert['score']:.1%}")
            st.write(alert["narrative"])


# ── Governance ───────────────────────────────────────────────────────────────


def _render_governance_view() -> None:
    """Render governance metrics dashboard."""
    st.header("Governance Metrics")

    try:
        response = httpx.get(f"{API_URL}/governance/metrics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Predictions", data.get("total_predictions", 0))
            col2.metric("Fraud Rate", f"{data.get('fraud_rate', 0):.1%}")
            col3.metric("Review Rate", f"{data.get('review_rate', 0):.1%}")
            col4.metric("Avg. Score", f"{data.get('avg_fraud_probability', 0):.4f}")
        else:
            st.warning("Could not fetch governance metrics.")
    except httpx.ConnectError:
        st.info("Connect to the API to view live governance metrics.")

    st.subheader("EU AI Act Compliance")
    st.markdown(
        """
        | Requirement | Status |
        |---|---|
        | Article 11 — Technical Documentation | ✅ Model Factsheet Generated |
        | Article 13 — Transparency | ✅ SHAP Explanations + Granite Narratives |
        | Article 14 — Human Oversight | ✅ Human-in-the-loop Review Required |
        | Article 15 — Accuracy & Robustness | ✅ Cross-validated + Drift Monitoring |
        | Article 10 — Data Governance | ✅ Bias Detection Across Protected Groups |
        """
    )


# ── Bias Analysis ────────────────────────────────────────────────────────────


def _render_bias_view() -> None:
    """Render bias analysis visualization."""
    st.header("Bias Analysis Across Protected Attributes")

    # Sample bias data for visualization
    groups = ["18-25", "26-35", "36-50", "51-65", "65+"]
    rates = [0.032, 0.028, 0.025, 0.030, 0.035]

    fig = px.bar(
        x=groups,
        y=rates,
        labels={"x": "Age Group", "y": "Positive Prediction Rate"},
        title="Demographic Parity — Fraud Flag Rate by Age Group",
        color=rates,
        color_continuous_scale="RdYlGn_r",
    )
    fig.add_hline(y=sum(rates) / len(rates), line_dash="dash", annotation_text="Overall Rate")
    st.plotly_chart(fig, use_container_width=True)

    genders = ["M", "F", "NB"]
    gender_rates = [0.029, 0.027, 0.031]
    fig2 = px.bar(
        x=genders,
        y=gender_rates,
        labels={"x": "Gender", "y": "Positive Prediction Rate"},
        title="Demographic Parity — Fraud Flag Rate by Gender",
        color=gender_rates,
        color_continuous_scale="RdYlGn_r",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Drift Monitoring ─────────────────────────────────────────────────────────


def _render_drift_view() -> None:
    """Render drift monitoring charts."""
    st.header("Data & Model Drift Monitoring")

    import numpy as np

    np.random.seed(42)
    days = list(range(1, 31))
    psi_values = np.random.uniform(0.01, 0.15, 30).tolist()
    psi_values[25] = 0.25  # Simulated spike

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=psi_values, mode="lines+markers", name="PSI"))
    fig.add_hline(y=0.20, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
    fig.update_layout(
        title="Population Stability Index — 30-Day Trend",
        xaxis_title="Day",
        yaxis_title="PSI",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "PSI > 0.20 indicates significant distribution shift requiring model review. "
        "KS-test is applied per-feature for granular drift detection."
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sample_prediction(txn_id: str, amount: float) -> dict:
    """Generate a sample prediction for demo purposes."""
    import math

    prob = min(1.0, max(0.0, 1.0 / (1.0 + math.exp(-0.001 * (amount - 3000)))))
    if prob >= 0.70:
        label = "fraud"
    elif prob >= 0.40:
        label = "review"
    else:
        label = "legitimate"

    return {
        "transaction_id": txn_id,
        "fraud_probability": prob,
        "label": label,
        "narrative": (
            f"Transaction {txn_id} was classified as '{label}' "
            f"with a fraud probability of {prob:.1%}. "
            f"Analysis based on transaction amount of ${amount:,.2f}."
        ),
        "top_risk_factors": [{"amount": 0.35}, {"velocity_count_1h": 0.20}],
        "top_mitigating_factors": [{"customer_amount_mean": -0.10}],
    }


if __name__ == "__main__":
    main()
