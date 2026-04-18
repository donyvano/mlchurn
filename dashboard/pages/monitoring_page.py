"""Monitoring page: PSI drift detection and feature distribution comparison."""

import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.charts import feature_distribution_chart, psi_timeline_chart
from dashboard.components.kpi_cards import status_badge
from dashboard.styles.theme import (
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    page_header,
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute the Population Stability Index between two distributions.

    PSI < 0.1: stable, 0.1–0.2: moderate shift, > 0.2: significant drift.

    Args:
        expected: Reference (training) distribution values.
        actual: Current (live) distribution values.
        n_bins: Number of bins for discretisation.

    Returns:
        PSI score (float).
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        n_bins + 1,
    )

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def _load_reference_data() -> pd.DataFrame | None:
    """Try to load the raw training data for PSI reference distribution.

    Returns:
        DataFrame or None if data not available.
    """
    raw_path = Path("data/raw/telco_churn.csv")
    if not raw_path.exists():
        return None
    try:
        from data.ingest import load_raw_dataset
        return load_raw_dataset(raw_path)
    except Exception:
        return None


def _generate_mock_live(reference: pd.DataFrame, drift_strength: float = 0.15) -> pd.DataFrame:
    """Generate a drifted live dataset from the reference for demo purposes.

    Args:
        reference: Reference dataframe.
        drift_strength: Perturbation magnitude.

    Returns:
        Drifted dataframe.
    """
    rng = np.random.default_rng(99)
    live = reference.sample(n=500, replace=True, random_state=99).copy()
    for col in NUMERIC_FEATURES:
        shift = drift_strength * live[col].std()
        live[col] = (live[col] + rng.normal(shift, live[col].std() * 0.05, size=len(live))).clip(0)
    return live


def _build_psi_table(reference: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Compute PSI for all numeric features and return a summary dataframe.

    Args:
        reference: Reference (training) dataframe.
        live: Live (production) dataframe.

    Returns:
        DataFrame with columns: Feature, PSI, Status.
    """
    rows = []
    for col in NUMERIC_FEATURES:
        if col not in reference.columns or col not in live.columns:
            continue
        psi = _compute_psi(reference[col].to_numpy(), live[col].to_numpy())
        if psi < 0.1:
            status = " Stable"
            status_type = "success"
        elif psi < 0.2:
            status = " Attention"
            status_type = "warning"
        else:
            status = " Drift Detected"
            status_type = "danger"
        rows.append({"Feature": col, "PSI Score": psi, "_status": status, "_status_type": status_type})
    return pd.DataFrame(rows)


def _mock_psi_history(n_days: int = 7) -> tuple[list[str], list[float]]:
    """Generate a mock 7-day PSI history with a progressive trend.

    Args:
        n_days: Number of days to generate.

    Returns:
        Tuple of (date_strings, psi_values).
    """
    base = datetime.today()
    dates = [(base - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days - 1, -1, -1)]
    psi_values = [round(0.04 + i * 0.018 + random.uniform(-0.005, 0.005), 4) for i in range(n_days)]
    return dates, psi_values


def render() -> None:
    """Render the Monitoring dashboard page."""
    page_header(" Monitoring", "Data drift detection using Population Stability Index (PSI)")

    reference = _load_reference_data()
    if reference is None:
        st.info(
            "Reference data not found. Run `make setup` to download the dataset. "
            "Displaying demo distributions.",
            icon="",
        )
        rng = np.random.default_rng(0)
        reference = pd.DataFrame({
            "tenure": rng.integers(0, 72, 1000),
            "MonthlyCharges": rng.uniform(18, 120, 1000),
            "TotalCharges": rng.uniform(18, 8000, 1000),
            "Churn": rng.integers(0, 2, 1000),
        })

    live = _generate_mock_live(reference, drift_strength=0.15)
    psi_df = _build_psi_table(reference, live)

    mean_psi = psi_df["PSI Score"].mean()
    n_drifted = (psi_df["PSI Score"] > 0.2).sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean PSI", f"{mean_psi:.4f}", delta="↑ +0.008 vs yesterday")
    with col2:
        st.metric("Features Monitored", len(psi_df))
    with col3:
        st.metric("Drift Alerts", f"{n_drifted}", delta=None)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 3], gap="medium")

    with col_left:
        st.markdown('<div class="section-header">Feature Drift Summary</div>', unsafe_allow_html=True)
        table_html = "<table style='width:100%;border-collapse:collapse'>"
        table_html += """
        <tr style='background:#1A2138'>
          <th style='padding:0.6rem 1rem;text-align:left;color:#94A3B8;
                     font-size:0.75rem;text-transform:uppercase'>Feature</th>
          <th style='padding:0.6rem 1rem;text-align:right;color:#94A3B8;
                     font-size:0.75rem;text-transform:uppercase'>PSI</th>
          <th style='padding:0.6rem 1rem;text-align:center;color:#94A3B8;
                     font-size:0.75rem;text-transform:uppercase'>Status</th>
        </tr>
        """
        for _, row in psi_df.iterrows():
            badge_html = status_badge(row["_status"], row["_status_type"])
            table_html += f"""
            <tr style='border-top:1px solid #1E2640;background:#141929'>
              <td style='padding:0.6rem 1rem;color:#F1F5F9'>{row['Feature']}</td>
              <td style='padding:0.6rem 1rem;text-align:right;color:#94A3B8;font-family:monospace'>
                {row['PSI Score']:.4f}</td>
              <td style='padding:0.6rem 1rem;text-align:center'>{badge_html}</td>
            </tr>
            """
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">PSI Timeline (7 days)</div>', unsafe_allow_html=True)
        dates, psi_history = _mock_psi_history()
        st.plotly_chart(psi_timeline_chart(dates, psi_history), use_container_width=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Distribution Comparison</div>', unsafe_allow_html=True)

    selected_feature = st.selectbox("Select feature", NUMERIC_FEATURES)
    st.plotly_chart(
        feature_distribution_chart(
            train_values=reference[selected_feature].tolist(),
            live_values=live[selected_feature].tolist(),
            feature_name=selected_feature,
        ),
        use_container_width=True,
    )
