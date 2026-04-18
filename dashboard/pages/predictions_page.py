"""Predictions page: live inference form and prediction history."""

from datetime import datetime
from typing import Optional

import requests
import streamlit as st

from dashboard.components.charts import churn_gauge
from dashboard.styles.theme import (
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    page_header,
)

PREDICTION_HISTORY_KEY = "prediction_history"
MAX_HISTORY = 20


def _call_api(api_url: str, payload: dict) -> Optional[dict]:
    """POST the customer payload to the prediction API.

    Args:
        api_url: Base URL of the FastAPI service.
        payload: Customer feature dict.

    Returns:
        Response JSON dict or None on failure.
    """
    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API call failed: {exc}")
        return None


def _result_alert(probability: float) -> None:
    """Display a colour-coded alert based on churn probability.

    Args:
        probability: Churn probability in [0, 1].
    """
    pct = probability * 100
    if pct >= 70:
        st.markdown(
            f'<div class="alert-danger"> High churn risk — {pct:.1f}% probability. '
            f"Immediate retention action recommended.</div>",
            unsafe_allow_html=True,
        )
    elif pct >= 30:
        st.markdown(
            f'<div class="alert-warning"> Moderate churn risk — {pct:.1f}% probability. '
            f"Monitor and consider proactive outreach.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="alert-success"> Low churn risk — {pct:.1f}% probability. '
            f"Customer appears stable.</div>",
            unsafe_allow_html=True,
        )


def _build_form() -> Optional[dict]:
    """Render the customer input form and return the payload dict on submit.

    Returns:
        Feature dict if form was submitted, else None.
    """
    with st.form("predict_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            st.markdown("**Account**")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.35)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=843.40)

        with col3:
            st.markdown("**Services**")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button(" Predict Churn", use_container_width=True)

    if submitted:
        return {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }
    return None


def _render_history(history: list[dict]) -> None:
    """Display the last N predictions as a styled table.

    Args:
        history: List of prediction result dicts.
    """
    if not history:
        return

    st.markdown('<div class="section-header">Recent Predictions</div>', unsafe_allow_html=True)

    rows_html = ""
    for item in reversed(history[-MAX_HISTORY:]):
        pct = item["churn_probability"] * 100
        if pct >= 70:
            badge = f'<span style="color:#F43F5E;font-weight:600">High ({pct:.1f}%)</span>'
        elif pct >= 30:
            badge = f'<span style="color:#F59E0B;font-weight:600">Medium ({pct:.1f}%)</span>'
        else:
            badge = f'<span style="color:#10B981;font-weight:600">Low ({pct:.1f}%)</span>'

        rows_html += f"""
        <tr>
          <td style="color:#94A3B8;font-size:0.8rem">{item['timestamp']}</td>
          <td style="font-family:monospace;font-size:0.8rem">{item['prediction_id']}</td>
          <td>{badge}</td>
          <td style="color:#94A3B8">{item['confidence']}</td>
          <td style="color:#94A3B8">v{item['model_version']}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;background:#141929;
                      border-radius:10px;overflow:hidden">
          <thead>
            <tr style="background:#1A2138">
              <th style="padding:0.6rem 1rem;text-align:left;color:#94A3B8;
                         font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em">
                Timestamp</th>
              <th style="padding:0.6rem 1rem;text-align:left;color:#94A3B8;
                         font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em">
                ID</th>
              <th style="padding:0.6rem 1rem;text-align:left;color:#94A3B8;
                         font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em">
                Risk</th>
              <th style="padding:0.6rem 1rem;text-align:left;color:#94A3B8;
                         font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em">
                Confidence</th>
              <th style="padding:0.6rem 1rem;text-align:left;color:#94A3B8;
                         font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em">
                Model</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    """Render the Predictions dashboard page."""
    page_header(" Predictions", "Real-time churn inference for individual customers")

    api_url = st.session_state.get("api_url", "http://localhost:8000")

    if PREDICTION_HISTORY_KEY not in st.session_state:
        st.session_state[PREDICTION_HISTORY_KEY] = []

    payload = _build_form()

    if payload:
        with st.spinner("Running inference..."):
            result = _call_api(api_url, payload)

        if result:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            col_gauge, col_info = st.columns([2, 3], gap="large")

            with col_gauge:
                st.markdown('<div class="section-header">Churn Probability</div>', unsafe_allow_html=True)
                st.plotly_chart(churn_gauge(result["churn_probability"]), use_container_width=True)

            with col_info:
                st.markdown('<div class="section-header">Assessment</div>', unsafe_allow_html=True)
                _result_alert(result["churn_probability"])
                st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                st.json({
                    "Prediction ID": result["prediction_id"],
                    "Churn Probability": f"{result['churn_probability']:.2%}",
                    "Churn Label": "Will Churn" if result["churn_label"] else "Will Stay",
                    "Confidence": result["confidence"],
                    "Model": f"{result['model_name']} v{result['model_version']}",
                })

            st.session_state[PREDICTION_HISTORY_KEY].append({
                **result,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

    _render_history(st.session_state[PREDICTION_HISTORY_KEY])
