# ChurnIQ — MLOps Churn Prediction Pipeline

[![CI](https://github.com/donyvano/mlchurn/actions/workflows/ci.yml/badge.svg)](https://github.com/donyvano/mlchurn/actions/workflows/ci.yml)
[![Docker Build](https://github.com/donyvano/mlchurn/actions/workflows/docker-build.yml/badge.svg)](https://github.com/donyvano/mlchurn/actions/workflows/docker-build.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-0194E2.svg?logo=mlflow)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end MLOps pipeline for telecom customer churn prediction — from raw data ingestion to live inference, drift monitoring, and automated daily retraining. Built with production-grade tooling across the full ML lifecycle.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ChurnIQ — System Architecture                   │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
  │  Kaggle  │───▶│  data/      │───▶│  models/     │───▶│  MLflow     │
  │  Dataset │    │  ingest.py  │    │  train.py    │    │  Registry   │
  └──────────┘    │  validate   │    │  XGBoost     │    │  Staging ──▶│
                  └─────────────┘    │  LightGBM    │    │  Production │
                                     │  Optuna x50  │    └──────┬──────┘
  ┌──────────────────────────────┐   └──────────────┘           │
  │  Airflow DAG (daily @24h)    │                              │
  │  ingest → validate →         │   ┌──────────────┐          │
  │  preprocess → train →        │   │  FastAPI      │◀─────────┘
  │  evaluate → promote → notify │   │  POST /predict│
  └──────────────────────────────┘   │  GET  /health │
                                     └──────┬────────┘
  ┌──────────────────────────────┐          │
  │  Streamlit Dashboard         │◀─────────┘
  │  ├── Overview   (KPIs, ROC)  │
  │  ├── Predictions (live form) │   ┌──────────────┐
  │  ├── Monitoring  (PSI drift) │   │  GitHub       │
  │  └── Experiments (MLflow UI) │   │  Actions CI   │
  └──────────────────────────────┘   │  lint + test  │
                                     │  docker push  │
                                     └──────────────┘
```

---

## Quick Start

```bash
# 1. Install dependencies and download the dataset
make setup

# 2. Train XGBoost + LightGBM with Optuna tuning (tracked in MLflow)
make train

# 3. Launch all services (API + Dashboard + MLflow UI + Airflow)
make serve
```

Then open:
| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| ChurnIQ Dashboard | http://localhost:8501 |
| Airflow | http://localhost:8080 |

---

## Project Structure

```
mlchurn/
├── .github/workflows/
│   ├── ci.yml               # ruff + mypy + pytest on every push
│   └── docker-build.yml     # build & push images on merge to main
├── airflow/dags/
│   └── retrain_dag.py       # daily retraining DAG with 7 tasks
├── api/
│   ├── main.py              # FastAPI app with lifespan model loading
│   ├── schemas.py           # Pydantic v2 request/response schemas
│   ├── predictor.py         # MLflow model loader + inference engine
│   └── Dockerfile
├── dashboard/
│   ├── app.py               # Streamlit entry point
│   ├── pages/               # Overview / Predictions / Monitoring / Experiments
│   ├── components/          # kpi_cards, charts, sidebar
│   ├── styles/theme.py      # Dark palette, CSS injection
│   └── Dockerfile
├── data/
│   ├── ingest.py            # Download + schema validation
│   └── simulate_drift.py    # PSI drift simulation for demo
├── models/
│   ├── pipeline.py          # sklearn ColumnTransformer pipeline
│   ├── train.py             # Optuna tuning + MLflow logging
│   ├── evaluate.py          # Metrics + artefact generation
│   └── registry.py          # Staging → Production promotion logic
├── tests/
│   ├── test_pipeline.py     # Preprocessing unit tests
│   ├── test_api.py          # FastAPI endpoint tests (mocked model)
│   └── test_registry.py     # Promotion logic unit tests
├── docker-compose.yml
├── Makefile
└── pyproject.toml           # ruff + mypy + pytest config
```

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 843.40
  }'
```

**Response:**
```json
{
  "churn_probability": 0.7241,
  "churn_label": true,
  "confidence": "HIGH",
  "model_name": "churn-classifier",
  "model_version": "3",
  "prediction_id": "pred_a3f9c12e"
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_name": "churn-classifier",
  "model_version": "3",
  "mlflow_tracking_uri": "http://mlflow:5000"
}
```

### `GET /model-info`

```bash
curl http://localhost:8000/model-info
```

```json
{
  "model_name": "churn-classifier",
  "model_version": "3",
  "run_id": "abc123def456",
  "metrics": { "auc_roc": 0.874, "f1": 0.632, "precision": 0.671, "recall": 0.598 },
  "parameters": { "n_estimators": "300", "max_depth": "6" },
  "registered_at": "2024-01-15 10:22:00"
}
```

---

## Dashboard

<!-- Screenshot placeholder — run `make serve` and capture http://localhost:8501 -->

The ChurnIQ dashboard is a dark-themed Streamlit application with four pages:

- **Overview** — AUC-ROC, ROC curve, confusion matrix, feature importances
- **Predictions** — live inference form with churn probability gauge
- **Monitoring** — PSI drift scores per feature with 7-day timeline
- **Experiments** — full MLflow run history with AUC vs F1 scatter plot

---

## Development

```bash
# Run tests with coverage
make test

# Lint and format
make lint
make lint-fix

# Force promote best model to Production
make promote

# Generate drift simulation data
make simulate-drift

# Stop all services
make down

# Remove build artefacts
make clean
```

---

## ML Pipeline Details

| Component | Choice | Rationale |
|---|---|---|
| Models | XGBoost + LightGBM | Industry-standard gradient boosting; compared side-by-side |
| Tuning | Optuna (TPE sampler, 50 trials) | Bayesian search, pruning, reproducible seeds |
| Preprocessing | sklearn ColumnTransformer | No data leakage, serialisable, prod-identical transforms |
| Tracking | MLflow | Experiment tracking + model registry in one tool |
| Promotion | AUC-ROC delta ≥ 1% | Conservative; avoids noise-driven promotions |
| Drift | PSI per feature | Standard industry metric, interpretable thresholds |

---

## Contributing

1. Fork the repo and create a feature branch
2. Run `make lint` and `make test` before submitting
3. Open a PR — CI runs automatically

---

## License

MIT — see [LICENSE](LICENSE).
