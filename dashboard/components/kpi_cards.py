"""Reusable KPI card components for the ChurnIQ dashboard."""

from typing import Optional

import streamlit as st

from dashboard.styles.theme import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    BG_CARD,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: Optional[bool] = None,
    accent_color: str = ACCENT_PRIMARY,
    icon: str = "",
) -> None:
    """Render a single KPI metric card via injected HTML.

    Args:
        label: Short uppercase label displayed above the value.
        value: Primary metric value (displayed large).
        delta: Optional delta string shown below the value.
        delta_positive: True = green delta, False = red, None = neutral.
        accent_color: CSS color for the left border accent.
        icon: Optional emoji/icon displayed before the label.
    """
    delta_class = ""
    if delta_positive is True:
        delta_class = "positive"
    elif delta_positive is False:
        delta_class = "negative"

    delta_html = ""
    if delta:
        arrow = "↑" if delta_positive else ("↓" if delta_positive is False else "→")
        delta_html = f'<div class="kpi-delta {delta_class}">{arrow} {delta}</div>'

    icon_html = f'<span style="margin-right:0.35rem">{icon}</span>' if icon else ""

    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{accent_color}">
          <div class="kpi-label">{icon_html}{label}</div>
          <div class="kpi-value">{value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(metrics: list[dict]) -> None:
    """Render a horizontal row of KPI cards in equal-width columns.

    Args:
        metrics: List of dicts with keys: label, value, delta (opt),
                 delta_positive (opt), accent_color (opt), icon (opt).

    Example::

        kpi_row([
            {"label": "AUC-ROC", "value": "0.8741", "delta": "+0.6% vs yesterday",
             "delta_positive": True, "accent_color": COLOR_SUCCESS},
            {"label": "Predictions Today", "value": "1,204", "icon": ""},
        ])
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            kpi_card(
                label=metric["label"],
                value=metric["value"],
                delta=metric.get("delta"),
                delta_positive=metric.get("delta_positive"),
                accent_color=metric.get("accent_color", ACCENT_PRIMARY),
                icon=metric.get("icon", ""),
            )


def status_badge(text: str, status: str) -> str:
    """Return an HTML badge string for inline use in st.markdown tables.

    Args:
        text: Badge text.
        status: "success" | "warning" | "danger" | "info".

    Returns:
        HTML string with a styled inline badge.
    """
    color_map = {
        "success": COLOR_SUCCESS,
        "warning": COLOR_WARNING,
        "danger": COLOR_DANGER,
        "info": ACCENT_SECONDARY,
    }
    color = color_map.get(status, ACCENT_PRIMARY)
    return (
        f'<span style="background:rgba({_hex_to_rgb(color)},0.15);color:{color};'
        f'border-radius:4px;padding:2px 8px;font-size:0.75rem;font-weight:600">'
        f"{text}</span>"
    )


def _hex_to_rgb(hex_color: str) -> str:
    """Convert a hex color string to an 'R,G,B' string for rgba() CSS.

    Args:
        hex_color: Hex color like '#6366F1'.

    Returns:
        Comma-separated RGB string, e.g. '99,102,241'.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r},{g},{b}"
