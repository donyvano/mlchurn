"""Metrics computation and MLflow artifact logging for churn models."""

import io
import logging
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Compute the full suite of evaluation metrics on test data.

    Args:
        model: Fitted classifier with predict and predict_proba methods.
        X_test: Preprocessed test feature matrix.
        y_test: Binary test labels.

    Returns:
        Dict of metric_name → value with 4 decimal precision.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "log_loss": round(float(log_loss(y_test, y_proba)), 4),
    }


def _plot_roc_curve(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> plt.Figure:
    """Generate a styled ROC curve figure.

    Args:
        model: Fitted classifier.
        X_test: Test feature matrix.
        y_test: True binary labels.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0A0F1E")
    ax.set_facecolor("#0A0F1E")

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, color="#6366F1", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#475569", lw=1)

    ax.set_title("ROC Curve", color="white", fontsize=13, pad=12)
    ax.set_xlabel("False Positive Rate", color="#94A3B8")
    ax.set_ylabel("True Positive Rate", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#1E2640")
    ax.grid(color="#1E2640", linewidth=0.5)
    fig.tight_layout()
    return fig


def _plot_confusion_matrix(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> plt.Figure:
    """Generate a styled confusion matrix heatmap.

    Args:
        model: Fitted classifier.
        X_test: Test feature matrix.
        y_test: True binary labels.

    Returns:
        Matplotlib Figure object.
    """
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0A0F1E")
    ax.set_facecolor("#0A0F1E")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    ax.set_title("Confusion Matrix", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#94A3B8")
    ax.set_xlabel("Predicted", color="#94A3B8")
    ax.set_ylabel("Actual", color="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#1E2640")
    fig.tight_layout()
    return fig


def _plot_feature_importances(
    model: Any, feature_names: list[str], top_n: int = 20
) -> plt.Figure:
    """Generate a horizontal bar chart of the top-N feature importances.

    Args:
        model: Fitted classifier with feature_importances_ attribute.
        feature_names: List of feature name strings.
        top_n: Number of top features to display.

    Returns:
        Matplotlib Figure object.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0A0F1E")
    ax.set_facecolor("#0A0F1E")

    colors = plt.cm.cool(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(top_features, top_values, color=colors)

    ax.set_title("Feature Importances (Top 20)", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Importance Score", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#1E2640")
    ax.grid(axis="x", color="#1E2640", linewidth=0.5)
    fig.tight_layout()
    return fig


def _fig_to_mlflow_artifact(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to a PNG buffer and log it to MLflow.

    Args:
        fig: Figure to save.
        filename: Artifact filename within the MLflow run.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    mlflow.log_image(buf.read(), filename)
    plt.close(fig)


def log_artifacts_to_mlflow(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model_type: str,
    preprocessor: Any,
) -> None:
    """Log all visual artifacts and the preprocessor pipeline to the active MLflow run.

    Args:
        model: Fitted classifier.
        X_test: Test features (preprocessed).
        y_test: Test labels.
        feature_names: Human-readable feature names.
        model_type: "xgboost" or "lightgbm" (used in artifact naming).
        preprocessor: Fitted sklearn ColumnTransformer to log as artifact.
    """
    import tempfile
    import joblib
    import os

    roc_fig = _plot_roc_curve(model, X_test, y_test)
    _fig_to_mlflow_artifact(roc_fig, f"plots/{model_type}_roc_curve.png")

    cm_fig = _plot_confusion_matrix(model, X_test, y_test)
    _fig_to_mlflow_artifact(cm_fig, f"plots/{model_type}_confusion_matrix.png")

    fi_fig = _plot_feature_importances(model, feature_names)
    _fig_to_mlflow_artifact(fi_fig, f"plots/{model_type}_feature_importances.png")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline_path = os.path.join(tmp_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, pipeline_path)
        mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

    y_proba = model.predict_proba(X_test)[:, 1]
    predictions_df = pd.DataFrame({"y_true": y_test, "y_proba": y_proba})
    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_path = os.path.join(tmp_dir, "test_predictions.csv")
        predictions_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path, artifact_path="predictions")

    logger.info("Artifacts logged for %s", model_type)
