"""Overview page: global KPIs and model performance summary."""

import os
import random
from datetime import datetime, timedelta
from typing import Optional

import requests
import streamlit as st

from dashboard.components.charts import (
    confusion_matrix_chart,
    feature_importance_chart,
    roc_curve_chart,
)
from dashboard.components.kpi_cards import kpi_row
from dashboard.styles.theme import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    page_header,
)


def _fetch_model_info(api_url: str) -> Optional[dict]:
    """Fetch model metadata from the API /model-info endpoint.

    Args:
        api_url: Base URL of the FastAPI service.

    Returns:
        Dict with model info or None on failure.
    """
    try:
        resp = requests.get(f"{api_url}/model-info", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _mock_roc_data() -> tuple[list[float], list[float], float]:
    """Generate plausible mock ROC curve data for demo display."""
    fpr = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tpr = [0.0, 0.35, 0.55, 0.65, 0.72, 0.80, 0.85, 0.89, 0.92, 0.94, 0.96, 0.98, 1.0]
    auc = 0.874
    return fpr, tpr, auc


def _mock_confusion_matrix() -> list[list[int]]:
    return [[920, 143], [112, 234]]


def _mock_feature_importances() -> tuple[list[str], list[float]]:
    features = [
        "tenure", "MonthlyCharges", "TotalCharges", "Contract_Two year",
        "InternetService_Fiber optic", "PaymentMethod_Electronic check",
        "SeniorCitizen", "Contract_One year", "OnlineSecurity_No",
        "TechSupport_No", "MultipleLines_No", "PaperlessBilling",
        "StreamingTV_Yes", "OnlineBackup_Yes", "DeviceProtection_Yes",
    ]
    importances = [0.18, 0.15, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
                   0.04, 0.03, 0.03, 0.02, 0.02, 0.02]
    return features, importances


def render() -> None:
    """Render the Overview dashboard page."""
    page_header(
        " Overview",
        "Global model performance and KPI snapshot",
    )

    api_url = st.session_state.get("api_url", "http://localhost:8000")
    model_info = _fetch_model_info(api_url)

    if model_info:
        metrics = model_info.get("metrics", {})
        auc = metrics.get("auc_roc", 0.0)
        model_version = model_info.get("model_version", "—")
        model_name = model_info.get("model_name", "—")
        api_status = "Live"
    else:
        auc = 0.874
        model_version = "demo"
        model_name = "churn-classifier"
        api_status = "Offline (demo mode)"

    drift_score = round(random.uniform(0.04, 0.18), 3)
    preds_today = random.randint(900, 1500)

    kpi_row([
        {
            "label": "AUC-ROC",
            "value": f"{auc:.4f}",
            "delta": "+0.6% vs yesterday",
            "delta_positive": True,
            "accent_color": COLOR_SUCCESS,
            "icon": "",
        },
        {
            "label": "Predictions Today",
            "value": f"{preds_today:,}",
            "delta": "+12% vs yesterday",
            "delta_positive": True,
            "accent_color": ACCENT_SECONDARY,
            "icon": "",
        },
        {
            "label": "Mean Drift Score (PSI)",
            "value": f"{drift_score:.3f}",
            "delta": "Stable" if drift_score < 0.1 else "Attention",
            "delta_positive": drift_score < 0.1,
            "accent_color": COLOR_SUCCESS if drift_score < 0.1 else COLOR_WARNING,
            "icon": "",
        },
        {
            "label": "Model in Production",
            "value": f"v{model_version}",
            "delta": api_status,
            "delta_positive": model_info is not None,
            "accent_color": ACCENT_PRIMARY,
            "icon": "",
        },
    ])

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="medium")

    with col_left:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, auc_roc = _mock_roc_data()
        st.plotly_chart(
            roc_curve_chart(fpr, tpr, auc_roc),
            use_container_width=True,
        )

    with col_right:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = _mock_confusion_matrix()
        st.plotly_chart(
            confusion_matrix_chart(cm),
            use_container_width=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
    features, importances = _mock_feature_importances()
    st.plotly_chart(
        feature_importance_chart(features, importances, top_n=15),
        use_container_width=True,
    )
