"""Plotly chart factory functions for the ChurnIQ dashboard."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure

from dashboard.styles.theme import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    BG_CARD,
    BORDER_SUBTLE,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
    PLOTLY_LAYOUT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def _base_fig(**kwargs) -> Figure:
    """Return a Figure with the shared dark layout applied."""
    fig = go.Figure(**kwargs)
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def roc_curve_chart(
    fpr: list[float], tpr: list[float], auc_score: float
) -> Figure:
    """Render a styled ROC curve with AUC annotation.

    Args:
        fpr: False positive rate values.
        tpr: True positive rate values.
        auc_score: Area under the ROC curve.

    Returns:
        Plotly Figure.
    """
    fig = _base_fig()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc_score:.4f})",
            line=dict(color=ACCENT_PRIMARY, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba(99,102,241,0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(color=TEXT_SECONDARY, width=1, dash="dot"),
        )
    )
    fig.update_layout(
        title=dict(text="ROC Curve", font=dict(color=TEXT_PRIMARY, size=14)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
    )
    return fig


def confusion_matrix_chart(cm: list[list[int]]) -> Figure:
    """Render a styled confusion matrix heatmap.

    Args:
        cm: 2×2 confusion matrix [[TN, FP], [FN, TP]].

    Returns:
        Plotly Figure.
    """
    labels = ["No Churn", "Churn"]
    z = cm
    text = [[str(v) for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color=TEXT_PRIMARY, size=18),
            colorscale=[[0, BG_CARD], [1, ACCENT_PRIMARY]],
            showscale=False,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Confusion Matrix", font=dict(color=TEXT_PRIMARY, size=14)),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def feature_importance_chart(
    feature_names: list[str], importances: list[float], top_n: int = 15
) -> Figure:
    """Render a horizontal bar chart of feature importances.

    Args:
        feature_names: Feature name strings.
        importances: Corresponding importance scores.
        top_n: Number of top features to display.

    Returns:
        Plotly Figure.
    """
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.nlargest(top_n, "importance").sort_values("importance")

    colors = [
        f"rgba(99,102,241,{0.4 + 0.6 * (i / len(df))})"
        for i in range(len(df))
    ]

    fig = _base_fig()
    fig.add_trace(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"Top {top_n} Feature Importances", font=dict(color=TEXT_PRIMARY, size=14)),
        xaxis_title="Importance Score",
        height=max(300, top_n * 28),
    )
    return fig


def churn_gauge(probability: float) -> Figure:
    """Render a gauge chart for a single churn probability.

    Args:
        probability: Churn probability in [0, 1].

    Returns:
        Plotly Figure.
    """
    pct = probability * 100
    if pct >= 70:
        bar_color = COLOR_DANGER
    elif pct >= 30:
        bar_color = COLOR_WARNING
    else:
        bar_color = COLOR_SUCCESS

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number=dict(suffix="%", font=dict(color=TEXT_PRIMARY, size=42)),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor=TEXT_SECONDARY,
                    tickfont=dict(color=TEXT_SECONDARY),
                ),
                bar=dict(color=bar_color, thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                bordercolor=BORDER_SUBTLE,
                steps=[
                    dict(range=[0, 30], color="rgba(16,185,129,0.1)"),
                    dict(range=[30, 70], color="rgba(245,158,11,0.1)"),
                    dict(range=[70, 100], color="rgba(244,63,94,0.1)"),
                ],
                threshold=dict(
                    line=dict(color=TEXT_SECONDARY, width=2),
                    thickness=0.8,
                    value=50,
                ),
            ),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def psi_timeline_chart(dates: list[str], psi_values: list[float]) -> Figure:
    """Render a PSI evolution timeline with threshold zones.

    Args:
        dates: ISO date strings.
        psi_values: Mean PSI values per date.

    Returns:
        Plotly Figure.
    """
    fig = _base_fig()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=psi_values,
            mode="lines+markers",
            name="Mean PSI",
            line=dict(color=ACCENT_SECONDARY, width=2),
            marker=dict(size=6, color=ACCENT_SECONDARY),
        )
    )
    for threshold, color, label in [
        (0.1, COLOR_WARNING, "Attention threshold"),
        (0.2, COLOR_DANGER, "Drift threshold"),
    ]:
        fig.add_hline(
            y=threshold,
            line=dict(color=color, dash="dash", width=1),
            annotation_text=label,
            annotation_font_color=color,
        )
    fig.update_layout(
        title=dict(text="PSI Over Time (7-day window)", font=dict(color=TEXT_PRIMARY, size=14)),
        xaxis_title="Date",
        yaxis_title="Population Stability Index",
    )
    return fig


def metric_scatter_chart(df: pd.DataFrame) -> Figure:
    """Scatter plot of AUC vs F1 across all MLflow runs.

    Args:
        df: DataFrame with columns: run_name, auc_roc, f1, model_type.

    Returns:
        Plotly Figure.
    """
    color_map = {"xgboost": ACCENT_PRIMARY, "lightgbm": ACCENT_SECONDARY}

    fig = _base_fig()
    for model_type, group in df.groupby("model_type"):
        fig.add_trace(
            go.Scatter(
                x=group["f1"],
                y=group["auc_roc"],
                mode="markers",
                name=model_type,
                marker=dict(
                    color=color_map.get(str(model_type), ACCENT_PRIMARY),
                    size=10,
                    line=dict(width=1, color=BORDER_SUBTLE),
                ),
                text=group["run_name"],
                hovertemplate="<b>%{text}</b><br>F1: %{x:.4f}<br>AUC: %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(text="AUC-ROC vs F1 across all runs", font=dict(color=TEXT_PRIMARY, size=14)),
        xaxis_title="F1 Score",
        yaxis_title="AUC-ROC",
    )
    return fig


def feature_distribution_chart(
    train_values: list[float],
    live_values: list[float],
    feature_name: str,
    n_bins: int = 30,
) -> Figure:
    """Overlay histogram comparing training vs live distributions for drift visualization.

    Args:
        train_values: Training set values for one feature.
        live_values: Live / production set values for the same feature.
        feature_name: Display name for the feature.
        n_bins: Number of histogram bins.

    Returns:
        Plotly Figure.
    """
    fig = _base_fig()
    for values, name, color in [
        (train_values, "Training", ACCENT_PRIMARY),
        (live_values, "Live / Production", ACCENT_SECONDARY),
    ]:
        fig.add_trace(
            go.Histogram(
                x=values,
                name=name,
                nbinsx=n_bins,
                marker_color=color,
                opacity=0.65,
                hovertemplate=f"{name}<br>{feature_name}: %{{x}}<br>Count: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="overlay",
        title=dict(text=f"Distribution: {feature_name}", font=dict(color=TEXT_PRIMARY, size=14)),
        xaxis_title=feature_name,
        yaxis_title="Count",
    )
    return fig
