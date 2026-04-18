"""FastAPI application for churn prediction serving."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.predictor import LoadedModel, load_production_model, predict
from api.schemas import (
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(
    level=getattr(logging, os.getenv("API_LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_model_state: dict[str, LoadedModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the production model on startup and release on shutdown."""
    logger.info("Loading production model from MLflow registry...")
    try:
        _model_state["loaded"] = load_production_model()
        logger.info(
            "Model ready: %s v%s",
            _model_state["loaded"].model_name,
            _model_state["loaded"].model_version,
        )
    except RuntimeError as exc:
        logger.error("Failed to load model at startup: %s", exc)
    yield
    _model_state.clear()
    logger.info("Model unloaded — API shutting down")


app = FastAPI(
    title="ChurnIQ Prediction API",
    description="Production-grade churn prediction endpoint backed by MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _get_loaded_model() -> LoadedModel:
    """Retrieve the loaded model or raise 503 if unavailable."""
    model = _model_state.get("loaded")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Check MLflow registry and restart the API.",
        )
    return model


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health() -> HealthResponse:
    """Return API health status and loaded model metadata."""
    model = _get_loaded_model()
    return HealthResponse(
        status="ok",
        model_name=model.model_name,
        model_version=model.model_version,
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Ops"])
async def model_info() -> ModelInfoResponse:
    """Return full metadata and evaluation metrics of the Production model."""
    model = _get_loaded_model()
    return ModelInfoResponse(
        model_name=model.model_name,
        model_version=model.model_version,
        run_id=model.run_id,
        metrics=model.metrics,
        parameters=model.parameters,
        registered_at=model.registered_at,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    """Predict churn probability for a single customer.

    Accepts a JSON payload with all customer features and returns the predicted
    churn probability, binary label, and confidence tier.

    Args:
        customer: Validated customer feature payload.

    Returns:
        PredictionResponse with probability, label, and model metadata.
    """
    loaded = _get_loaded_model()

    features = customer.model_dump()
    features["InternetService"] = features["InternetService"].value if hasattr(
        features["InternetService"], "value"
    ) else features["InternetService"]
    features["Contract"] = features["Contract"].value if hasattr(
        features["Contract"], "value"
    ) else features["Contract"]
    features["PaymentMethod"] = features["PaymentMethod"].value if hasattr(
        features["PaymentMethod"], "value"
    ) else features["PaymentMethod"]

    try:
        prob, label, confidence, pred_id, _ = predict(loaded, features)
    except Exception as exc:
        logger.exception("Prediction failed for request")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_label=label,
        confidence=confidence,
        model_name=loaded.model_name,
        model_version=loaded.model_version,
        prediction_id=pred_id,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
