"""Experiments page: MLflow run history and model comparison."""

import os
from typing import Optional

import mlflow
import pandas as pd
import streamlit as st

from dashboard.components.charts import metric_scatter_chart
from dashboard.components.kpi_cards import status_badge
from dashboard.styles.theme import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    COLOR_SUCCESS,
    page_header,
)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")


def _fetch_runs() -> Optional[pd.DataFrame]:
    """Query MLflow for all runs in the experiment.

    Returns:
        DataFrame of run data or None if MLflow is unreachable.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.auc_roc DESC"],
            max_results=50,
        )

        rows = []
        for run in runs:
            rows.append({
                "run_id": run.info.run_id[:8],
                "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
                "model_type": run.data.tags.get("model_type", "unknown"),
                "auc_roc": round(run.data.metrics.get("auc_roc", 0), 4),
                "f1": round(run.data.metrics.get("f1", 0), 4),
                "precision": round(run.data.metrics.get("precision", 0), 4),
                "recall": round(run.data.metrics.get("recall", 0), 4),
                "log_loss": round(run.data.metrics.get("log_loss", 0), 4),
                "status": run.info.status,
                "start_time": pd.Timestamp(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
            })
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def _mock_runs_df() -> pd.DataFrame:
    """Return a plausible mock runs dataframe for demo display."""
    return pd.DataFrame([
        {"run_id": "a1b2c3d4", "run_name": "xgboost-optuna", "model_type": "xgboost",
         "auc_roc": 0.874, "f1": 0.632, "precision": 0.671, "recall": 0.598,
         "log_loss": 0.412, "status": "Production", "start_time": "2024-01-15 10:22"},
        {"run_id": "e5f6g7h8", "run_name": "lightgbm-optuna", "model_type": "lightgbm",
         "auc_roc": 0.869, "f1": 0.621, "precision": 0.655, "recall": 0.591,
         "log_loss": 0.421, "status": "Staging", "start_time": "2024-01-15 11:05"},
        {"run_id": "i9j0k1l2", "run_name": "xgboost-optuna", "model_type": "xgboost",
         "auc_roc": 0.861, "f1": 0.609, "precision": 0.640, "recall": 0.581,
         "log_loss": 0.438, "status": "Archived", "start_time": "2024-01-14 09:18"},
        {"run_id": "m3n4o5p6", "run_name": "lightgbm-optuna", "model_type": "lightgbm",
         "auc_roc": 0.855, "f1": 0.598, "precision": 0.627, "recall": 0.572,
         "log_loss": 0.451, "status": "Archived", "start_time": "2024-01-14 10:47"},
    ])


def render() -> None:
    """Render the Experiments dashboard page."""
    page_header(" Experiments", "MLflow run history and model performance comparison")

    df = _fetch_runs()
    using_mock = df is None
    if using_mock:
        st.info(
            f"MLflow server unreachable at {MLFLOW_URI}. Displaying demo runs.",
            icon="",
        )
        df = _mock_runs_df()

    col_filter, col_sort = st.columns([2, 1])
    with col_filter:
        model_filter = st.multiselect(
            "Filter by model type",
            options=["xgboost", "lightgbm"],
            default=["xgboost", "lightgbm"],
        )
    with col_sort:
        sort_by = st.selectbox("Sort by", ["auc_roc", "f1", "precision", "recall"])

    filtered = df[df["model_type"].isin(model_filter)].sort_values(sort_by, ascending=False)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Run History</div>', unsafe_allow_html=True)

    table_html = """
    <table style='width:100%;border-collapse:collapse'>
      <thead>
        <tr style='background:#1A2138'>
          <th style='padding:0.6rem 1rem;text-align:left;color:#94A3B8;font-size:0.75rem;
                     text-transform:uppercase'>Run</th>
          <th style='padding:0.6rem 1rem;text-align:left;color:#94A3B8;font-size:0.75rem;
                     text-transform:uppercase'>Type</th>
          <th style='padding:0.6rem 1rem;text-align:right;color:#94A3B8;font-size:0.75rem;
                     text-transform:uppercase'>AUC-ROC</th>
          <th style='padding:0.6rem 1rem;text-align:right;color:#94A3B8;font-size:0.75rem;
                     text-transform:uppercase'>F1</th>
          <th style='padding:0.6rem 1rem;text-align:right;color:#94A3B8;font-size:0.75rem;
                     text-transform:uppercase'>Precision</th>
          <th style='padding:0.6rem 1rem;text-align:right;color:#94A3B8;font-size:0.75rem;
                     text-transform:uppercase'>Recall</th>
          <th style='padding:0.6rem 1rem;text-align:center;color:#94A3B8;font-size:0.75rem;
                     text-transform:uppercase'>Stage</th>
          <th style='padding:0.6rem 1rem;text-align:left;color:#94A3B8;font-size:0.75rem;
                     text-transform:uppercase'>Date</th>
        </tr>
      </thead>
      <tbody>
    """

    status_map = {
        "Production": ("Production", "success"),
        "Staging": ("Staging", "warning"),
        "Archived": ("Archived", "info"),
        "FINISHED": ("Finished", "info"),
        "RUNNING": ("Running", "warning"),
    }

    for _, row in filtered.iterrows():
        stage_text, stage_type = status_map.get(row["status"], (row["status"], "info"))
        badge = status_badge(stage_text, stage_type)
        type_color = ACCENT_PRIMARY if row["model_type"] == "xgboost" else ACCENT_SECONDARY
        table_html += f"""
        <tr style='border-top:1px solid #1E2640;background:#141929'>
          <td style='padding:0.6rem 1rem;font-family:monospace;color:#94A3B8;font-size:0.8rem'>
            {row['run_id']}</td>
          <td style='padding:0.6rem 1rem'>
            <span style='color:{type_color};font-weight:600;font-size:0.85rem'>
              {row['model_type']}</span></td>
          <td style='padding:0.6rem 1rem;text-align:right;color:#F1F5F9;font-family:monospace'>
            {row['auc_roc']:.4f}</td>
          <td style='padding:0.6rem 1rem;text-align:right;color:#94A3B8;font-family:monospace'>
            {row['f1']:.4f}</td>
          <td style='padding:0.6rem 1rem;text-align:right;color:#94A3B8;font-family:monospace'>
            {row['precision']:.4f}</td>
          <td style='padding:0.6rem 1rem;text-align:right;color:#94A3B8;font-family:monospace'>
            {row['recall']:.4f}</td>
          <td style='padding:0.6rem 1rem;text-align:center'>{badge}</td>
          <td style='padding:0.6rem 1rem;color:#94A3B8;font-size:0.8rem'>{row['start_time']}</td>
        </tr>
        """

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">AUC-ROC vs F1 — All Runs</div>', unsafe_allow_html=True)
    st.plotly_chart(metric_scatter_chart(filtered), use_container_width=True)
