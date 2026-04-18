"""Sidebar navigation component for the ChurnIQ dashboard."""

import streamlit as st

from dashboard.styles.theme import (
    ACCENT_PRIMARY,
    BG_CARD,
    BORDER_SUBTLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

LOGO_HTML = f"""
<div style="padding:1.5rem 1rem 1rem;border-bottom:1px solid {BORDER_SUBTLE};margin-bottom:1rem">
  <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;
              color:{TEXT_PRIMARY};letter-spacing:-0.02em">
    Churn<span style="color:{ACCENT_PRIMARY}">IQ</span>
  </div>
  <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:0.2rem;
              text-transform:uppercase;letter-spacing:0.08em">
    MLOps Monitoring Platform
  </div>
</div>
"""

NAV_ITEMS = [
    ("Overview", "overview", ""),
    ("Predictions", "predictions", ""),
    ("Monitoring", "monitoring", ""),
    ("Experiments", "experiments", ""),
]


def render_sidebar(api_url: str = "http://localhost:8000") -> dict:
    """Render the dashboard sidebar and return user-selected filter values.

    Args:
        api_url: Base URL for the FastAPI backend.

    Returns:
        Dict with keys: api_url (str), auto_refresh (bool), refresh_interval (int).
    """
    with st.sidebar:
        st.markdown(LOGO_HTML, unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:0.7rem;text-transform:uppercase;'
            f'letter-spacing:0.08em;color:{TEXT_MUTED};margin-bottom:0.5rem">'
            f"Navigation</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:0.7rem;text-transform:uppercase;'
            f'letter-spacing:0.08em;color:{TEXT_MUTED};margin-bottom:0.5rem">'
            f"Settings</div>",
            unsafe_allow_html=True,
        )

        resolved_api_url = st.text_input(
            "API URL",
            value=api_url,
            help="Base URL of the FastAPI prediction service",
        )

        auto_refresh = st.toggle("Auto-refresh", value=False)
        refresh_interval = 30
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh interval (s)", min_value=10, max_value=120, value=30, step=10
            )

        st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.72rem;color:{TEXT_MUTED};padding:0.5rem 0">'
            f"ChurnIQ v1.0 · MLflow + XGBoost + LightGBM</div>",
            unsafe_allow_html=True,
        )

    return {
        "api_url": resolved_api_url,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
    }
