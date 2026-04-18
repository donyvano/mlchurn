"""MLflow model loader and inference engine for the churn prediction API."""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-classifier")
CONFIDENCE_HIGH_THRESHOLD = 0.7
CONFIDENCE_LOW_THRESHOLD = 0.3


@dataclass
class LoadedModel:
    """Container for a loaded MLflow model and its registry metadata."""

    model: Any
    model_name: str
    model_version: str
    run_id: str
    metrics: dict[str, float]
    parameters: dict[str, str]
    registered_at: Optional[str]


def load_production_model() -> LoadedModel:
    """Load the current Production model from the MLflow Model Registry.

    Fetches the latest Production version, loads it as a pyfunc model,
    and retrieves all associated metadata from the run.

    Returns:
        LoadedModel dataclass with model and metadata.

    Raises:
        RuntimeError: If no Production model is found in the registry.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError(
            f"Cannot connect to MLflow at {MLFLOW_TRACKING_URI}: {exc}"
        ) from exc

    if not versions:
        raise RuntimeError(
            f"No Production model found for '{MODEL_NAME}'. "
            "Run `make train && make promote` first."
        )

    latest = versions[0]
    model_uri = f"models:/{MODEL_NAME}/Production"

    logger.info(
        "Loading model '%s' v%s from %s", MODEL_NAME, latest.version, model_uri
    )
    model = mlflow.pyfunc.load_model(model_uri)

    run = client.get_run(latest.run_id)
    metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}
    parameters = {k: str(v) for k, v in run.data.params.items()}

    logger.info("Model loaded — AUC=%.4f", metrics.get("auc_roc", 0))

    return LoadedModel(
        model=model,
        model_name=MODEL_NAME,
        model_version=latest.version,
        run_id=latest.run_id,
        metrics=metrics,
        parameters=parameters,
        registered_at=latest.creation_timestamp
        and str(pd.Timestamp(latest.creation_timestamp, unit="ms")),
    )


def _compute_confidence(probability: float) -> str:
    """Map a churn probability to a confidence tier.

    Args:
        probability: Model output probability in [0, 1].

    Returns:
        "HIGH" if far from 0.5, "LOW" if close to 0.5, "MEDIUM" otherwise.
    """
    distance = abs(probability - 0.5)
    if distance >= 0.2:
        return "HIGH"
    if distance >= 0.1:
        return "MEDIUM"
    return "LOW"


def predict(
    loaded_model: LoadedModel,
    features: dict[str, Any],
) -> tuple[float, bool, str, str, float]:
    """Run inference on a single customer feature dict.

    Args:
        loaded_model: LoadedModel instance with the production model.
        features: Dict of feature_name → value matching the training schema.

    Returns:
        Tuple of (churn_probability, churn_label, confidence, prediction_id, latency_ms).
    """
    input_df = pd.DataFrame([features])

    t_start = time.perf_counter()
    proba_array = loaded_model.model.predict(input_df)
    latency_ms = (time.perf_counter() - t_start) * 1000

    churn_probability = float(proba_array[0] if proba_array.ndim == 1 else proba_array[0, 1])
    churn_label = churn_probability >= 0.5
    confidence = _compute_confidence(churn_probability)

    feature_hash = hashlib.sha256(str(sorted(features.items())).encode()).hexdigest()[:12]
    prediction_id = f"pred_{feature_hash}"

    logger.info(
        "prediction_id=%s churn_prob=%.4f label=%s confidence=%s latency_ms=%.1f",
        prediction_id,
        churn_probability,
        churn_label,
        confidence,
        latency_ms,
    )

    return churn_probability, churn_label, confidence, prediction_id, latency_ms
