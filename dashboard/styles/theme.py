"""Visual design constants and CSS injection for the ChurnIQ dashboard."""

# ── Colour palette ────────────────────────────────────────────────────────────
BG_DEEP = "#0A0F1E"
BG_CARD = "#141929"
BG_SURFACE = "#1A2138"
BORDER_SUBTLE = "#1E2640"

ACCENT_PRIMARY = "#6366F1"   # indigo electric
ACCENT_SECONDARY = "#22D3EE"  # cyan
COLOR_DANGER = "#F43F5E"     # coral red
COLOR_SUCCESS = "#10B981"    # emerald
COLOR_WARNING = "#F59E0B"    # amber

TEXT_PRIMARY = "#F1F5F9"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#475569"

# ── Plotly shared layout ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_SECONDARY, family="Inter, sans-serif"),
    xaxis=dict(
        gridcolor=BORDER_SUBTLE,
        linecolor=BORDER_SUBTLE,
        tickcolor=TEXT_MUTED,
        tickfont=dict(color=TEXT_SECONDARY),
    ),
    yaxis=dict(
        gridcolor=BORDER_SUBTLE,
        linecolor=BORDER_SUBTLE,
        tickcolor=TEXT_MUTED,
        tickfont=dict(color=TEXT_SECONDARY),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER_SUBTLE,
        font=dict(color=TEXT_SECONDARY),
    ),
    hoverlabel=dict(
        bgcolor=BG_SURFACE,
        bordercolor=BORDER_SUBTLE,
        font=dict(color=TEXT_PRIMARY),
    ),
    margin=dict(l=16, r=16, t=40, b=16),
)

# ── Global CSS ────────────────────────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG_DEEP};
    color: {TEXT_PRIMARY};
  }}

  /* Remove Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 1.5rem 2rem 2rem; max-width: 1400px; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: {BG_CARD};
    border-right: 1px solid {BORDER_SUBTLE};
  }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] p {{
    color: {TEXT_SECONDARY};
    font-size: 0.85rem;
  }}

  /* Metric overrides */
  [data-testid="stMetric"] {{
    background: {BG_CARD};
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border-left: 3px solid {ACCENT_PRIMARY};
  }}
  [data-testid="stMetricValue"] {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem !important;
    color: {TEXT_PRIMARY} !important;
  }}
  [data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: {BG_CARD};
    border-radius: 8px;
    padding: 2px;
    gap: 2px;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {TEXT_SECONDARY};
    border-radius: 6px;
    padding: 0.4rem 1rem;
    font-size: 0.875rem;
  }}
  .stTabs [aria-selected="true"] {{
    background: {ACCENT_PRIMARY} !important;
    color: white !important;
  }}

  /* Buttons */
  .stButton button {{
    background: {ACCENT_PRIMARY};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: opacity 0.2s;
  }}
  .stButton button:hover {{ opacity: 0.85; }}

  /* Inputs */
  .stTextInput input, .stNumberInput input, .stSelectbox > div > div {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    color: {TEXT_PRIMARY};
  }}

  /* Dataframes */
  .stDataFrame {{ border-radius: 10px; overflow: hidden; }}
  [data-testid="stDataFrame"] th {{
    background: {BG_SURFACE} !important;
    color: {TEXT_SECONDARY} !important;
    font-size: 0.8rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  [data-testid="stDataFrame"] td {{
    background: {BG_CARD} !important;
    color: {TEXT_PRIMARY} !important;
    font-size: 0.875rem !important;
  }}

  /* KPI card override (via st.markdown) */
  .kpi-card {{
    background: {BG_CARD};
    border-radius: 12px;
    padding: 1.25rem;
    border-left: 3px solid {ACCENT_PRIMARY};
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 0.5rem;
  }}
  .kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);
  }}
  .kpi-label {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {TEXT_SECONDARY};
    margin-bottom: 0.5rem;
  }}
  .kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1;
  }}
  .kpi-delta {{
    font-size: 0.78rem;
    margin-top: 0.4rem;
    color: {TEXT_MUTED};
  }}
  .kpi-delta.positive {{ color: {COLOR_SUCCESS}; }}
  .kpi-delta.negative {{ color: {COLOR_DANGER}; }}

  /* Alert boxes */
  .alert-danger {{
    background: rgba(244, 63, 94, 0.12);
    border: 1px solid {COLOR_DANGER};
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: {COLOR_DANGER};
    font-weight: 500;
  }}
  .alert-success {{
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid {COLOR_SUCCESS};
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: {COLOR_SUCCESS};
    font-weight: 500;
  }}
  .alert-warning {{
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid {COLOR_WARNING};
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: {COLOR_WARNING};
    font-weight: 500;
  }}

  /* Chart wrapper */
  .chart-wrapper {{
    background: {BG_CARD};
    border-radius: 14px;
    padding: 1rem;
    border: 1px solid {BORDER_SUBTLE};
  }}

  /* Section headers */
  .section-header {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {BORDER_SUBTLE};
  }}

  /* Divider */
  hr {{ border-color: {BORDER_SUBTLE}; margin: 1.5rem 0; }}
</style>
"""


def inject_css() -> None:
    """Inject global CSS into the Streamlit app.

    Must be called once per page at the top of each Streamlit script.
    """
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Render a styled page title block.

    Args:
        title: Main page heading.
        subtitle: Optional secondary description line.
    """
    import streamlit as st
    sub_html = f'<p style="color:{TEXT_SECONDARY};font-size:0.9rem;margin-top:0.25rem">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem">
          <h1 style="font-family:\'Space Grotesk\',sans-serif;font-size:1.75rem;
                     font-weight:700;color:{TEXT_PRIMARY};margin:0">{title}</h1>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
