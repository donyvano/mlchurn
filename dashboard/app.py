"""ChurnIQ — Streamlit dashboard entry point."""

import os

import streamlit as st

st.set_page_config(
    page_title="ChurnIQ · MLOps Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.components.sidebar import render_sidebar
from dashboard.styles.theme import inject_css

inject_css()

API_URL = os.getenv("DASHBOARD_API_URL", "http://localhost:8000")
sidebar_state = render_sidebar(api_url=API_URL)

st.session_state["api_url"] = sidebar_state["api_url"]

if sidebar_state["auto_refresh"]:
    import time
    time.sleep(sidebar_state["refresh_interval"])
    st.rerun()

from dashboard.pages import (  # noqa: E402  (after set_page_config)
    overview_page,
    predictions_page,
    monitoring_page,
    experiments_page,
)

PAGES = {
    "Overview": overview_page.render,
    "Predictions": predictions_page.render,
    "Monitoring": monitoring_page.render,
    "Experiments": experiments_page.render,
}

tab_labels = list(PAGES.keys())
tabs = st.tabs(tab_labels)

for tab, (label, render_fn) in zip(tabs, PAGES.items()):
    with tab:
        render_fn()
