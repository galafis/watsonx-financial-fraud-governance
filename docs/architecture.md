# Architecture Overview

## System Architecture

The Watsonx Financial Fraud Detection system is organized into six layers: data ingestion, feature engineering, ensemble modeling, explainability, governance, and interface.

```mermaid
flowchart TB
    subgraph Ingestion["1. Data Ingestion"]
        CSV[CSV Files] --> Loader[TransactionLoader]
        Kafka[Apache Kafka] --> Loader
        SynGen[SyntheticTransactionGenerator] --> Loader
        Loader --> RawDF[Raw Transaction DataFrame]
    end

    subgraph Features["2. Feature Engineering"]
        RawDF --> FE[FeatureEngineer]
        FE --> Temporal[Temporal Features]
        FE --> Velocity[Velocity Features]
        FE --> Amount[Amount Statistics]
        FE --> Merchant[Merchant Deviation]
        FE --> Geo[Geo Distance]
        Temporal & Velocity & Amount & Merchant & Geo --> FeatureMatrix[Feature Matrix]
    end

    subgraph Model["3. Ensemble Model"]
        FeatureMatrix --> Trainer[FraudModelTrainer]
        Trainer --> CV[Stratified K-Fold CV]
        CV --> Ensemble[EnsembleFraudDetector]
        Ensemble --> XGB[XGBoost 40%]
        Ensemble --> LGB[LightGBM 40%]
        Ensemble --> IF[Isolation Forest 20%]
        XGB & LGB & IF --> Score[Weighted Fraud Score]
        Score --> Classify{Threshold}
        Classify -->|>= 0.70| Fraud[FRAUD]
        Classify -->|>= 0.40| Review[REVIEW]
        Classify -->|< 0.40| Legit[LEGITIMATE]
    end

    subgraph Explain["4. Explainability"]
        Score --> SHAP[ShapExplainer]
        SHAP --> TreeXGB[TreeExplainer XGBoost]
        SHAP --> TreeLGB[TreeExplainer LightGBM]
        TreeXGB & TreeLGB --> Contributions[Feature Contributions]
        Contributions --> Narrator[NarrativeGenerator]
        Narrator --> Granite[IBM Watsonx Granite LLM]
        Granite --> Narrative[Human-Readable Narrative]
    end

    subgraph Governance["5. Governance"]
        Score --> Bias[BiasDetector]
        Bias --> DP[Demographic Parity]
        Bias --> EO[Equalized Odds]
        Score --> Drift[DriftMonitor]
        Drift --> PSI[Population Stability Index]
        Drift --> KS[Kolmogorov-Smirnov Test]
        DP & EO & PSI & KS --> Factsheet[ModelFactsheet]
        Factsheet --> EUAI[EU AI Act Art. 11 Report]
    end

    subgraph Interface["6. Interface"]
        Score & Narrative & EUAI --> API[FastAPI REST API]
        API --> Dashboard[Streamlit Dashboard]
        API --> Swagger[OpenAPI / Swagger UI]
    end

    style Ingestion fill:#e3f2fd,stroke:#1565c0
    style Features fill:#f3e5f5,stroke:#6a1b9a
    style Model fill:#fff3e0,stroke:#e65100
    style Explain fill:#e8f5e9,stroke:#2e7d32
    style Governance fill:#fce4ec,stroke:#b71c1c
    style Interface fill:#f5f5f5,stroke:#424242
```

## Module Responsibilities

### Data Layer (`src/data/`)

| Module | Class | Responsibility |
|---|---|---|
| `ingestion.py` | `TransactionLoader` | Load transactions from CSV files or Apache Kafka topics with validation and type casting |
| `synthetic_generator.py` | `SyntheticTransactionGenerator` | Generate realistic synthetic transaction data with configurable fraud rate and protected attributes |
| `feature_engineering.py` | `FeatureEngineer` | Extract temporal patterns, velocity metrics, amount statistics, merchant deviation, and geo-distance features |

### Model Layer (`src/models/`)

| Module | Class | Responsibility |
|---|---|---|
| `ensemble.py` | `EnsembleFraudDetector` | Weighted ensemble combining XGBoost (40%), LightGBM (40%), and Isolation Forest (20%) |
| `ensemble.py` | `FraudPrediction` | Dataclass holding prediction result with probabilities and per-model scores |
| `anomaly.py` | `AnomalyDetector` | Isolation Forest wrapper with normalized anomaly scores in [0, 1] |
| `trainer.py` | `FraudModelTrainer` | Training pipeline with stratified k-fold cross-validation |

### Explainability Layer (`src/explainability/`)

| Module | Class | Responsibility |
|---|---|---|
| `shap_explainer.py` | `ShapExplainer` | SHAP TreeExplainer for XGBoost and LightGBM with weighted combination |
| `shap_explainer.py` | `ShapExplanation` | Dataclass with feature contributions and top positive/negative factors |
| `narrative_generator.py` | `NarrativeGenerator` | Generate human-readable fraud alert narratives using IBM Watsonx Granite |

### Governance Layer (`src/governance/`)

| Module | Class | Responsibility |
|---|---|---|
| `bias_detector.py` | `BiasDetector` | Evaluate demographic parity and equalized odds across protected attributes |
| `drift_monitor.py` | `DriftMonitor` | Monitor data drift using PSI and KS-test per feature |
| `factsheet.py` | `ModelFactsheet` | Generate EU AI Act Article 11 compliant model documentation |

### Interface Layer (`src/api/`, `src/ui/`)

| Module | Class | Responsibility |
|---|---|---|
| `routes.py` | FastAPI `app` | REST endpoints for prediction, explanation, and governance |
| `schemas.py` | Pydantic models | Request/response validation schemas |
| `app.py` | Streamlit `main` | Interactive dashboard with investigation, alerts, governance, bias, and drift views |

## Feature Engineering Pipeline

```mermaid
flowchart LR
    Raw[Raw Transaction] --> T[Temporal]
    Raw --> V[Velocity]
    Raw --> A[Amount]
    Raw --> M[Merchant]
    Raw --> G[Geographic]

    T --> T1[hour, day_of_week]
    T --> T2[is_weekend, is_night]
    T --> T3[hour_sin, hour_cos]
    T --> T4[dow_sin, dow_cos]

    V --> V1[count per 1h/6h/24h/72h]
    V --> V2[sum per window]
    V --> V3[mean per window]
    V --> V4[std per window]

    A --> A1[customer_amount_mean/std]
    A --> A2[amount_zscore]
    A --> A3[amount_to_max_ratio]
    A --> A4[log_amount]

    M --> M1[merchant_avg_amount]
    M --> M2[merchant_txn_count]
    M --> M3[customer_unique_merchants]
    M --> M4[is_new_merchant]

    G --> G1[geo_distance_km]
    G --> G2[geo_anomaly]
```

## Ensemble Decision Flow

```mermaid
flowchart LR
    Input[Feature Vector] --> XGB[XGBoost]
    Input --> LGB[LightGBM]
    Input --> IF[Isolation Forest]

    XGB -->|P_xgb| W1["x 0.40"]
    LGB -->|P_lgb| W2["x 0.40"]
    IF -->|P_if| W3["x 0.20"]

    W1 & W2 & W3 --> Sum[Weighted Sum]
    Sum --> P[Fraud Probability]

    P -->|">= 0.70"| F[FRAUD]
    P -->|">= 0.40"| R[REVIEW]
    P -->|"< 0.40"| L[LEGITIMATE]

    F --> Alert[Fraud Alert + SHAP + Narrative]
    R --> Queue[Review Queue]
    L --> Pass[Pass Through]
```

## Governance Compliance (EU AI Act)

| EU AI Act Article | Implementation |
|---|---|
| **Article 10 - Data Governance** | BiasDetector evaluates fairness across age, gender, geography |
| **Article 11 - Technical Documentation** | ModelFactsheet generates structured factsheet with all required metadata |
| **Article 13 - Transparency** | SHAP explanations + Granite LLM narratives for every flagged transaction |
| **Article 14 - Human Oversight** | Human-in-the-loop: all fraud-labeled transactions require analyst review |
| **Article 15 - Accuracy & Robustness** | Stratified CV metrics + continuous PSI/KS drift monitoring |

## Deployment Architecture

```mermaid
flowchart TB
    subgraph Docker["Docker Compose Stack"]
        ZK[Zookeeper :2181]
        K[Kafka :9092]
        API[FastAPI API :8080]
        UI[Streamlit UI :8501]
    end

    ZK --> K
    K --> API
    API --> UI

    Client[Client / Browser] --> UI
    Client --> API
    ExtSys[External Systems] --> K
```

The system runs as a Docker Compose stack with four services:
- **Zookeeper**: Kafka coordination
- **Kafka**: Real-time transaction streaming
- **API**: FastAPI fraud detection service (port 8080)
- **UI**: Streamlit governance dashboard (port 8501)

## Configuration

All model, feature, governance, and infrastructure parameters are centralized in `config/settings.yaml` and loaded through `src/config.py` using Pydantic settings with environment variable overrides.
