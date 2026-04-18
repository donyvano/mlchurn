"""Integration tests for the FastAPI prediction endpoints."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.predictor import LoadedModel
from api.main import app, _model_state

VALID_PAYLOAD: dict[str, Any] = {
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
    "TotalCharges": 843.40,
}


@pytest.fixture()
def mock_loaded_model() -> LoadedModel:
    """Return a LoadedModel with a mocked internal model that returns fixed probabilities."""
    inner_model = MagicMock()
    inner_model.predict.return_value = [0.72]

    return LoadedModel(
        model=inner_model,
        model_name="churn-classifier",
        model_version="3",
        run_id="abc123",
        metrics={"auc_roc": 0.874, "f1": 0.632},
        parameters={"n_estimators": "300"},
        registered_at="2024-01-15 10:00:00",
    )


@pytest.fixture()
def client(mock_loaded_model: LoadedModel):
    """TestClient with a pre-injected mock model (bypasses MLflow at startup)."""
    _model_state["loaded"] = mock_loaded_model
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    _model_state.clear()


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_name" in data
        assert "model_version" in data

    def test_returns_503_without_model(self) -> None:
        _model_state.clear()
        with TestClient(app) as c:
            resp = c.get("/health")
        assert resp.status_code == 503


class TestModelInfoEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_metrics_present(self, client: TestClient) -> None:
        resp = client.get("/model-info")
        data = resp.json()
        assert "metrics" in data
        assert "auc_roc" in data["metrics"]


class TestPredictEndpoint:
    def test_valid_payload_returns_200(self, client: TestClient) -> None:
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client: TestClient) -> None:
        resp = client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert "churn_probability" in data
        assert "churn_label" in data
        assert "confidence" in data
        assert "prediction_id" in data
        assert "model_version" in data

    def test_probability_in_range(self, client: TestClient) -> None:
        resp = client.post("/predict", json=VALID_PAYLOAD)
        prob = resp.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_invalid_tenure_returns_422(self, client: TestClient) -> None:
        bad_payload = {**VALID_PAYLOAD, "tenure": -5}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_invalid_gender_returns_422(self, client: TestClient) -> None:
        bad_payload = {**VALID_PAYLOAD, "gender": "Unknown"}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_missing_field_returns_422(self, client: TestClient) -> None:
        incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422

    def test_high_probability_yields_correct_label(self, client: TestClient, mock_loaded_model: LoadedModel) -> None:
        mock_loaded_model.model.predict.return_value = [0.85]
        _model_state["loaded"] = mock_loaded_model
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["churn_label"] is True

    def test_low_probability_yields_correct_label(self, client: TestClient, mock_loaded_model: LoadedModel) -> None:
        mock_loaded_model.model.predict.return_value = [0.20]
        _model_state["loaded"] = mock_loaded_model
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["churn_label"] is False

    def test_confidence_high_for_extreme_probability(self, client: TestClient, mock_loaded_model: LoadedModel) -> None:
        mock_loaded_model.model.predict.return_value = [0.92]
        _model_state["loaded"] = mock_loaded_model
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["confidence"] == "HIGH"
