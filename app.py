"""
app.py — Enhanced Streamlit Dashboard for India Semiconductor Growth Predictor
Run: streamlit run app.py
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Semiconductor Growth Predictor | Semicon India",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DIRECTORIES & DATA LOADERS ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "src" / "models"

# Coordinates for India Semiconductor Clusters
CLUSTER_COORDS = {
    "Sanand": {"lat": 22.9868, "lon": 72.3813, "state": "Gujarat", "region": "West"},
    "Dholera": {"lat": 22.2500, "lon": 72.1900, "state": "Gujarat", "region": "West"},
    "Surat": {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat", "region": "West"},
    "Morigaon": {"lat": 26.2500, "lon": 92.3400, "state": "Assam", "region": "North-East"},
    "Jewar": {"lat": 28.1287, "lon": 77.5559, "state": "Uttar Pradesh", "region": "North"},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha", "region": "East"},
    "Mohali": {"lat": 30.7046, "lon": 76.7179, "state": "Punjab", "region": "North"},
    "Andhra Pradesh": {"lat": 13.5300, "lon": 80.0300, "state": "Andhra Pradesh", "region": "South"},
    "Bhiwadi": {"lat": 28.2104, "lon": 76.8606, "state": "Rajasthan", "region": "North"},
}

@st.cache_data
def load_data():
    from src.preprocessing import (
        build_projects_clean, build_quarterly_investment,
        build_statetype_analysis, extract_non_ism_projects
    )
    df = build_projects_clean()
    q = build_quarterly_investment(df)
    st_ = build_statetype_analysis(df)
    non_ = extract_non_ism_projects()

    # Enrich df with geographical coordinates
    def get_coords(row):
        hub = row.get("hub_cluster")
        if hub in CLUSTER_COORDS:
            return pd.Series([CLUSTER_COORDS[hub]["lat"], CLUSTER_COORDS[hub]["lon"]])
        st_name = row.get("state")
        if st_name in CLUSTER_COORDS:
            return pd.Series([CLUSTER_COORDS[st_name]["lat"], CLUSTER_COORDS[st_name]["lon"]])
        return pd.Series([22.5, 78.9])

    df[["lat", "lon"]] = df.apply(get_coords, axis=1)
    return df, q, st_, non_

@st.cache_data
def load_scenarios():
    p = PROC / "market_scenarios.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def load_metrics():
    p = PROC / "model_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

df, q, st_agg, non_ism = load_data()
scenarios = load_scenarios()
metrics = load_metrics()

# ── THEMES CONFIGURATION ──────────────────────────────────────────────────────
THEMES = {
    "🌈 Electric Cyberpunk": {
        "bg": "#090d16",
        "card_bg": "rgba(16, 23, 41, 0.82)",
        "card_border": "rgba(0, 240, 255, 0.28)",
        "glow": "0 8px 32px rgba(0, 240, 255, 0.15)",
        "text": "#f8fafc",
        "subtext": "#94a3b8",
        "primary": "#00f0ff",
        "secondary": "#ff007f",
        "accent": "#ffe600",
        "gradient": "linear-gradient(135deg, #00f0ff 0%, #ff007f 100%)",
        "kpi_bg_1": "linear-gradient(135deg, #0f2042 0%, #00f0ff 120%)",
        "kpi_bg_2": "linear-gradient(135deg, #3d092c 0%, #ff007f 120%)",
        "kpi_bg_3": "linear-gradient(135deg, #3b2c04 0%, #ffe600 120%)",
        "kpi_bg_4": "linear-gradient(135deg, #1b0a38 0%, #a855f7 120%)",
        "palette": ["#00f0ff", "#ff007f", "#ffe600", "#a855f7", "#00ff88", "#ff6b35", "#38bdf8", "#f43f5e"],
        "plotly_template": "plotly_dark",
        "grid_color": "rgba(255, 255, 255, 0.08)",
    },
    "🌅 Sunset Aurora": {
        "bg": "#0d0b18",
        "card_bg": "rgba(24, 20, 42, 0.85)",
        "card_border": "rgba(255, 112, 67, 0.35)",
        "glow": "0 8px 32px rgba(255, 112, 67, 0.18)",
        "text": "#fdfbf7",
        "subtext": "#a8a29e",
        "primary": "#ff5e62",
        "secondary": "#ff9966",
        "accent": "#fbbf24",
        "gradient": "linear-gradient(135deg, #ff5e62 0%, #ff9966 50%, #c084fc 100%)",
        "kpi_bg_1": "linear-gradient(135deg, #43121d 0%, #ff5e62 120%)",
        "kpi_bg_2": "linear-gradient(135deg, #422006 0%, #ff9966 120%)",
        "kpi_bg_3": "linear-gradient(135deg, #31134a 0%, #c084fc 120%)",
        "kpi_bg_4": "linear-gradient(135deg, #451a03 0%, #fbbf24 120%)",
        "palette": ["#ff5e62", "#ff9966", "#c084fc", "#fbbf24", "#f43f5e", "#fb923c", "#818cf8", "#34d399"],
        "plotly_template": "plotly_dark",
        "grid_color": "rgba(255, 255, 255, 0.08)",
    },
    "🌿 Emerald Matrix": {
        "bg": "#06130e",
        "card_bg": "rgba(10, 30, 22, 0.85)",
        "card_border": "rgba(16, 185, 129, 0.35)",
        "glow": "0 8px 32px rgba(16, 185, 129, 0.16)",
        "text": "#f0fdf4",
        "subtext": "#9ca3af",
        "primary": "#10b981",
        "secondary": "#06b6d4",
        "accent": "#a3e635",
        "gradient": "linear-gradient(135deg, #10b981 0%, #06b6d4 100%)",
        "kpi_bg_1": "linear-gradient(135deg, #063c28 0%, #10b981 120%)",
        "kpi_bg_2": "linear-gradient(135deg, #063342 0%, #06b6d4 120%)",
        "kpi_bg_3": "linear-gradient(135deg, #243806 0%, #a3e635 120%)",
        "kpi_bg_4": "linear-gradient(135deg, #162444 0%, #38bdf8 120%)",
        "palette": ["#10b981", "#06b6d4", "#a3e635", "#38bdf8", "#34d399", "#22d3ee", "#eab308", "#f97316"],
        "plotly_template": "plotly_dark",
        "grid_color": "rgba(255, 255, 255, 0.08)",
    },
    "💎 Royal Sapphire": {
        "bg": "#0b1120",
        "card_bg": "rgba(15, 23, 42, 0.88)",
        "card_border": "rgba(56, 189, 248, 0.32)",
        "glow": "0 8px 32px rgba(56, 189, 248, 0.16)",
        "text": "#f8fafc",
        "subtext": "#94a3b8",
        "primary": "#38bdf8",
        "secondary": "#818cf8",
        "accent": "#f472b6",
        "gradient": "linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%)",
        "kpi_bg_1": "linear-gradient(135deg, #0c2d48 0%, #38bdf8 120%)",
        "kpi_bg_2": "linear-gradient(135deg, #1e1b4b 0%, #818cf8 120%)",
        "kpi_bg_3": "linear-gradient(135deg, #4a044e 0%, #f472b6 120%)",
        "kpi_bg_4": "linear-gradient(135deg, #064e3b 0%, #34d399 120%)",
        "palette": ["#38bdf8", "#818cf8", "#f472b6", "#34d399", "#fbbf24", "#c084fc", "#f87171", "#2dd4bf"],
        "plotly_template": "plotly_dark",
        "grid_color": "rgba(255, 255, 255, 0.08)",
    },
    "☀️ Radiant Light": {
        "bg": "#f8fafc",
        "card_bg": "rgba(255, 255, 255, 0.95)",
        "card_border": "rgba(99, 102, 241, 0.22)",
        "glow": "0 8px 24px rgba(99, 102, 241, 0.12)",
        "text": "#0f172a",
        "subtext": "#64748b",
        "primary": "#6366f1",
        "secondary": "#ec4899",
        "accent": "#f59e0b",
        "gradient": "linear-gradient(135deg, #6366f1 0%, #ec4899 50%, #f59e0b 100%)",
        "kpi_bg_1": "linear-gradient(135deg, #e0e7ff 0%, #6366f1 140%)",
        "kpi_bg_2": "linear-gradient(135deg, #fce7f3 0%, #ec4899 140%)",
        "kpi_bg_3": "linear-gradient(135deg, #fef3c7 0%, #f59e0b 140%)",
        "kpi_bg_4": "linear-gradient(135deg, #ccfbf1 0%, #14b8a6 140%)",
        "palette": ["#6366f1", "#ec4899", "#06b6d4", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444", "#14b8a6"],
        "plotly_template": "plotly_white",
        "grid_color": "rgba(0, 0, 0, 0.06)",
    }
}

# ── SIDEBAR CONTROLS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 16px 0;">
      <div style="font-size: 2.8rem; line-height: 1.1;">🔬⚡</div>
      <h2 style="margin: 6px 0 0 0; font-weight: 800; font-size: 1.25rem; letter-spacing: -0.5px;">
        India Semiconductor
      </h2>
      <span style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.8; font-weight: 600;">
        Growth & Capex Analytics
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎨 Theme Selector")
    chosen_theme_name = st.selectbox(
        "Choose Colorful Palette",
        list(THEMES.keys()),
        index=0,
        label_visibility="collapsed"
    )
    theme = THEMES[chosen_theme_name]

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Navigate to",
        [
            "📊 Executive Dashboard",
            "🗺️ Geo Hub & Cluster Map",
            "📈 Market Growth Forecaster",
            "🤖 ML Predictor & Sandbox",
            "⚖️ State Comparative Matrix",
            "🗂️ Interactive Project Showcase",
            "ℹ️ Methodology & About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Global Currency Unit")
    unit_mode = st.radio("Display Capex In:", ["₹ Crore (INR)", "$ Billion (USD)"], horizontal=True)
    inr_to_usd_rate = 84.0  # approximate ₹84 / USD

    def fmt_capex(crore_val, short=False):
        if pd.isna(crore_val):
            return "Undisclosed"
        if unit_mode == "$ Billion (USD)":
            usd_bn = (crore_val * 1e7) / (inr_to_usd_rate * 1e9)
            return f"${usd_bn:.2f}B" if short else f"${usd_bn:.3f} Billion"
        else:
            if short:
                if crore_val >= 1000:
                    return f"₹{crore_val/1000:.1f}k Cr"
                return f"₹{crore_val:,.0f} Cr"
            return f"₹{crore_val:,.0f} Crore"

    # Filters (for relevant pages)
    if page in ["📊 Executive Dashboard", "🗺️ Geo Hub & Cluster Map", "🗂️ Interactive Project Showcase"]:
        st.markdown("---")
        st.markdown("### 🔍 Quick Filters")
        states_all = sorted(df["state"].unique())
        sel_states = st.multiselect("States", states_all, default=states_all)
        
        fac_all = sorted(df["facility_type_canonical"].unique())
        sel_fac = st.multiselect("Facility Types", fac_all, default=fac_all)

        status_all = sorted(df["status"].unique())
        sel_status = st.multiselect("Status", status_all, default=status_all)
    else:
        sel_states = sorted(df["state"].unique())
        sel_fac = sorted(df["facility_type_canonical"].unique())
        sel_status = sorted(df["status"].unique())

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; text-align: center; opacity: 0.7; padding: 6px;">
      Semicon India Programme (ISM)<br>PIB Official Releases • 2023–2026
    </div>
    """, unsafe_allow_html=True)

# ── INJECT DYNAMIC CUSTOM CSS ────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
  
  html, body, [class*="css"] {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  }}
  
  .stApp {{
      background: {theme["bg"]};
      color: {theme["text"]};
  }}

  /* Top Hero Banner */
  .hero-banner {{
      background: {theme["gradient"]};
      padding: 24px 28px;
      border-radius: 18px;
      margin-bottom: 24px;
      box-shadow: {theme["glow"]};
      color: #ffffff;
      position: relative;
      overflow: hidden;
  }}
  .hero-banner::after {{
      content: "⚡ ISM 2.0";
      position: absolute;
      right: 20px;
      top: 16px;
      font-size: 0.8rem;
      font-weight: 800;
      letter-spacing: 1px;
      padding: 4px 12px;
      background: rgba(0, 0, 0, 0.35);
      border-radius: 30px;
      border: 1px solid rgba(255, 255, 255, 0.3);
  }}
  .hero-title {{
      font-size: 1.85rem;
      font-weight: 800;
      margin: 0;
      letter-spacing: -0.5px;
      line-height: 1.2;
  }}
  .hero-desc {{
      font-size: 0.95rem;
      opacity: 0.92;
      margin-top: 6px;
      max-width: 820px;
  }}

  /* KPI Card Styling */
  .kpi-container {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
  }}
  @media (max-width: 900px) {{
      .kpi-container {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  @media (max-width: 600px) {{
      .kpi-container {{ grid-template-columns: 1fr; }}
  }}

  .kpi-card {{
      border-radius: 16px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0,0,0,0.25);
      border: 1px solid rgba(255,255,255,0.12);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      color: #ffffff;
  }}
  .kpi-card:hover {{
      transform: translateY(-4px);
      box-shadow: {theme["glow"]};
      border-color: rgba(255,255,255,0.3);
  }}
  .kpi-card-1 {{ background: {theme["kpi_bg_1"]}; }}
  .kpi-card-2 {{ background: {theme["kpi_bg_2"]}; }}
  .kpi-card-3 {{ background: {theme["kpi_bg_3"]}; }}
  .kpi-card-4 {{ background: {theme["kpi_bg_4"]}; }}
  
  .kpi-icon-pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: rgba(255,255,255,0.2);
      margin-bottom: 10px;
      font-size: 1.2rem;
  }}
  .kpi-val {{
      font-size: 2rem;
      font-weight: 800;
      margin: 0;
      line-height: 1.1;
      letter-spacing: -0.5px;
  }}
  .kpi-lbl {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      font-weight: 600;
      opacity: 0.85;
      margin-top: 6px;
  }}
  .kpi-delta {{
      font-size: 0.75rem;
      margin-top: 4px;
      opacity: 0.95;
      font-weight: 500;
  }}

  /* Content Cards & Sections */
  .custom-card {{
      background: {theme["card_bg"]};
      border: 1px solid {theme["card_border"]};
      border-radius: 16px;
      padding: 20px 22px;
      margin-bottom: 20px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.18);
      backdrop-filter: blur(12px);
  }}
  .section-badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 1px;
      text-transform: uppercase;
      background: {theme["primary"]}22;
      color: {theme["primary"]};
      border: 1px solid {theme["primary"]}44;
      margin-bottom: 6px;
  }}
  .section-heading {{
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 12px;
      color: {theme["text"]};
      display: flex;
      align-items: center;
      gap: 8px;
  }}

  /* Badges */
  .badge-status {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 0.74rem;
      font-weight: 600;
  }}
  .badge-comm {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
  .badge-exec {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
  .badge-appr {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }}

  /* Sliders and Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 8px;
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 10px 10px 0 0;
      padding: 10px 18px;
      font-weight: 600;
  }}
</style>
""", unsafe_allow_html=True)

# Helper function to style Plotly charts consistently with the active theme
def polish_plot(fig, height=360, title=None):
    fig.update_layout(
        template=theme["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=theme["text"]),
        title=dict(
            text=title if title else "",
            font=dict(size=14, color=theme["text"], family="Plus Jakarta Sans, sans-serif"),
            x=0.01,
            y=0.98
        ) if title else None,
        margin=dict(l=15, r=15, t=35 if title else 15, b=15),
        hoverlabel=dict(
            bgcolor=theme["bg"],
            bordercolor=theme["primary"],
            font=dict(color=theme["text"], family="Plus Jakarta Sans, sans-serif")
        )
    )
    fig.update_xaxes(
        gridcolor=theme["grid_color"],
        zerolinecolor=theme["grid_color"],
        showline=True,
        linecolor=theme["grid_color"]
    )
    fig.update_yaxes(
        gridcolor=theme["grid_color"],
        zerolinecolor=theme["grid_color"],
        showline=True,
        linecolor=theme["grid_color"]
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 1. PAGE: EXECUTIVE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    # Hero Banner
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-title">🔬 India Semiconductor Growth Predictor</div>
      <div class="hero-desc">
        Comprehensive strategic analytics & intelligence dashboard tracking India's multi-billion dollar 
        semiconductor manufacturing, fabrication (Fab), and advanced packaging (ATMP/OSAT) ecosystem under 
        the India Semiconductor Mission (ISM).
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Filtered dataframe
    filt = df[
        df["state"].isin(sel_states) &
        df["facility_type_canonical"].isin(sel_fac) &
        df["status"].isin(sel_status)
    ]

    # Metrics calculation
    total_inv_known = filt.loc[filt["is_investment_known"], "investment_crore"].sum()
    n_proj = len(filt)
    n_states = filt["state"].nunique()
    n_commercial = len(filt[filt["status"].str.contains("Commercial", case=False, na=False)])
    pct_outlay = (total_inv_known / 203500.0) * 100.0

    # KPI Row
    st.markdown(f"""
    <div class="kpi-container">
      <div class="kpi-card kpi-card-1">
        <div class="kpi-icon-pill">💰</div>
        <div class="kpi-val">{fmt_capex(total_inv_known, short=True)}</div>
        <div class="kpi-lbl">Committed Capex</div>
        <div class="kpi-delta">⚡ {pct_outlay:.1f}% of ₹2.03L Cr ISM Outlay</div>
      </div>
      <div class="kpi-card kpi-card-2">
        <div class="kpi-icon-pill">🏭</div>
        <div class="kpi-val">{n_proj}</div>
        <div class="kpi-lbl">Approved Projects</div>
        <div class="kpi-delta">🎯 10 Known + 2 Undisclosed Capex</div>
      </div>
      <div class="kpi-card kpi-card-3">
        <div class="kpi-icon-pill">⚡</div>
        <div class="kpi-val">{n_commercial} Units</div>
        <div class="kpi-lbl">Commercial Production</div>
        <div class="kpi-delta">🚀 Sanand Hub (Micron, CG, Kaynes)</div>
      </div>
      <div class="kpi-card kpi-card-4">
        <div class="kpi-icon-pill">🗺️</div>
        <div class="kpi-val">{n_states} States</div>
        <div class="kpi-lbl">Geographic Spread</div>
        <div class="kpi-delta">🌐 Across 6 Specialized Tech Hubs</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ROW 1: Geographic Distribution & Hierarchical Breakdown ─────────────────
    r1c1, r1c2 = st.columns([5, 5])

    with r1c1:
        st.markdown("""
        <div class="section-badge">Capital Allocation</div>
        <div class="section-heading">🏛️ Investment Distribution by State</div>
        """, unsafe_allow_html=True)

        state_inv = (
            filt[filt["is_investment_known"]]
            .groupby("state")["investment_crore"]
            .sum()
            .reset_index()
            .sort_values("investment_crore", ascending=True)
        )
        if unit_mode == "$ Billion (USD)":
            state_inv["display_inv"] = (state_inv["investment_crore"] * 1e7) / (inr_to_usd_rate * 1e9)
            axis_title = "Committed Capex (USD $B)"
        else:
            state_inv["display_inv"] = state_inv["investment_crore"]
            axis_title = "Committed Capex (₹ Crore)"

        fig_st = px.bar(
            state_inv,
            x="display_inv",
            y="state",
            orientation="h",
            color="display_inv",
            color_continuous_scale=[[0, theme["primary"]], [0.5, theme["secondary"]], [1, theme["accent"]]],
            labels={"display_inv": axis_title, "state": "State"},
            text="display_inv"
        )
        if unit_mode == "$ Billion (USD)":
            fig_st.update_traces(
                texttemplate="$%{x:.2f}B",
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Capex: $%{x:.2f} Billion<extra></extra>"
            )
        else:
            fig_st.update_traces(
                texttemplate="₹%{x:,.0f} Cr",
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Capex: ₹%{x:,.0f} Crore<extra></extra>"
            )
        fig_st.update_layout(coloraxis_showscale=False)
        polish_plot(fig_st, height=340)
        st.plotly_chart(fig_st, use_container_width=True)

    with r1c2:
        st.markdown("""
        <div class="section-badge">Technology Spread</div>
        <div class="section-heading">🧩 Facility Type Capex Share</div>
        """, unsafe_allow_html=True)

        fac_inv = (
            filt[filt["is_investment_known"]]
            .groupby("facility_type_canonical")["investment_crore"]
            .sum()
            .reset_index()
        )
        fig_fac = px.pie(
            fac_inv,
            values="investment_crore",
            names="facility_type_canonical",
            hole=0.48,
            color_discrete_sequence=theme["palette"]
        )
        fig_fac.update_traces(
            textinfo="percent+label",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Share: %{percent}<br>Total: ₹%{value:,.0f} Cr<extra></extra>",
            marker=dict(line=dict(color=theme["bg"], width=2.5))
        )
        fig_fac.update_layout(
            showlegend=False,
            annotations=[dict(text="CAPEX<br>SPLIT", x=0.5, y=0.5, font_size=12, font_weight="bold", showarrow=False)]
        )
        polish_plot(fig_fac, height=340)
        st.plotly_chart(fig_fac, use_container_width=True)

    # ── ROW 2: Hierarchical Sunburst & Treemap Explorer ─────────────────────────
    st.markdown("""
    <div class="section-badge">Deep Drilldown</div>
    <div class="section-heading">🌳 Ecosystem Hierarchy: State ➔ Facility ➔ Company</div>
    """, unsafe_allow_html=True)

    tree_filt = filt[filt["is_investment_known"]].copy()
    tree_filt["display_val"] = tree_filt["investment_crore"]

    vis_mode = st.radio("Hierarchy Visualization Style:", ["🍩 Sunburst Ring", "🔲 Treemap Grid"], horizontal=True)

    if vis_mode == "🍩 Sunburst Ring":
        fig_tree = px.sunburst(
            tree_filt,
            path=["state", "facility_type_canonical", "company"],
            values="display_val",
            color="state",
            color_discrete_sequence=theme["palette"]
        )
    else:
        fig_tree = px.treemap(
            tree_filt,
            path=["state", "facility_type_canonical", "company"],
            values="display_val",
            color="facility_type_canonical",
            color_discrete_sequence=theme["palette"]
        )

    fig_tree.update_traces(
        hovertemplate="<b>%{label}</b><br>Capex: ₹%{value:,.0f} Cr<br>Parent: %{parent}<extra></extra>"
    )
    polish_plot(fig_tree, height=420)
    st.plotly_chart(fig_tree, use_container_width=True)

    # ── ROW 3: Quarterly Capex Timeline & Cumulative Surge ──────────────────────
    st.markdown("""
    <div class="section-badge">Temporal Momentum</div>
    <div class="section-heading">📈 Quarterly Approvals & Cumulative Capex Trajectory</div>
    """, unsafe_allow_html=True)

    fig_time = go.Figure()
    # Bar trace for count
    fig_time.add_trace(go.Bar(
        x=q["approval_quarter_label"],
        y=q["projects_approved"],
        name="Projects Approved",
        marker=dict(
            color=theme["primary"],
            line=dict(color=theme["accent"], width=1.5)
        ),
        text=q["projects_approved"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Projects Approved: %{y}<extra></extra>"
    ))

    # Scatter trace for cumulative capex
    cum_q = q.copy()
    cum_q.loc[~cum_q["investment_complete"], "cumulative_investment_crore"] = None
    y_cum = cum_q["cumulative_investment_crore"].div(1000)

    fig_time.add_trace(go.Scatter(
        x=cum_q["approval_quarter_label"],
        y=y_cum,
        name="Cumulative Capex (₹k Cr)",
        mode="lines+markers+text",
        line=dict(color=theme["secondary"], width=3.5, shape="spline"),
        marker=dict(size=10, color=theme["accent"], line=dict(color="#ffffff", width=2)),
        text=y_cum.apply(lambda v: f"₹{v:.1f}k Cr" if pd.notna(v) else ""),
        textposition="top center",
        yaxis="y2",
        connectgaps=False,
        hovertemplate="<b>%{x}</b><br>Cumulative Capex: ₹%{y:.1f}k Crore<extra></extra>"
    ))

    # Annotation for milestone
    fig_time.add_annotation(
        x="2024Q1", y=148.7, yref="y2",
        text="🔥 Tata Mega Fab (₹91.5k Cr) + TSAT (₹27k Cr)",
        showarrow=True, arrowhead=2, arrowsize=1, arrowcolor=theme["secondary"],
        ax=0, ay=-40, font=dict(color=theme["text"], size=10, weight="bold"),
        bgcolor=theme["card_bg"], bordercolor=theme["secondary"], borderwidth=1.5
    )

    fig_time.update_layout(
        yaxis=dict(title="Projects Approved (Count)", color=theme["primary"]),
        yaxis2=dict(
            title="Cumulative Capex (₹ Thousand Cr)",
            overlaying="y",
            side="right",
            color=theme["secondary"],
            gridcolor="rgba(0,0,0,0)"
        ),
        legend=dict(orientation="h", y=1.12, x=0.01),
        margin=dict(l=20, r=20, t=35, b=20)
    )
    polish_plot(fig_time, height=380)
    st.plotly_chart(fig_time, use_container_width=True)
    st.caption("ℹ️ Note: 2026Q2 (Crystal Matrix + Suchi Semicon) values are undisclosed in PIB releases and excluded from cumulative totals.")


# ═════════════════════════════════════════════════════════════════════════════
# 2. PAGE: GEO HUB & CLUSTER MAP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Geo Hub & Cluster Map":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-title">🗺️ India Semiconductor Geography & Cluster Map</div>
      <div class="hero-desc">
        Explore the spatial distribution of India's semiconductor mega-fabs, ATMP/OSAT hubs, compound semiconductor 
        foundries, and display driver fabrication lines across India.
      </div>
    </div>
    """, unsafe_allow_html=True)

    filt_map = df[
        df["state"].isin(sel_states) &
        df["facility_type_canonical"].isin(sel_fac) &
        df["status"].isin(sel_status)
    ].copy()

    # Calculate bubble sizing
    filt_map["bubble_size"] = filt_map["investment_crore"].fillna(2500)
    filt_map["bubble_size_scaled"] = np.sqrt(filt_map["bubble_size"]) * 1.5

    # Geo Scatter Map
    fig_map = px.scatter_geo(
        filt_map,
        lat="lat",
        lon="lon",
        color="facility_type_canonical",
        size="bubble_size_scaled",
        hover_name="company",
        hover_data={
            "lat": False,
            "lon": False,
            "bubble_size_scaled": False,
            "state": True,
            "project_location": True,
            "investment_crore": True,
            "status": True,
            "technology_or_product": True
        },
        color_discrete_sequence=theme["palette"],
        projection="natural earth"
    )

    fig_map.update_geos(
        center=dict(lat=22.5, lon=82.0),
        projection_scale=3.8,
        showland=True,
        landcolor=theme["card_bg"],
        showocean=True,
        oceancolor=theme["bg"],
        showcountries=True,
        countrycolor=theme["card_border"],
        showsubunits=True,
        subunitcolor=theme["card_border"],
        bgcolor="rgba(0,0,0,0)"
    )

    fig_map.update_traces(
        marker=dict(line=dict(width=1.5, color="#ffffff"), opacity=0.88),
        hovertemplate="<b>%{hovertext}</b><br>State: %{customdata[0]}<br>Location: %{customdata[1]}<br>Committed Capex: ₹%{customdata[2]:,.0f} Cr<br>Status: %{customdata[3]}<br>Focus: %{customdata[4]}<extra></extra>"
    )

    polish_plot(fig_map, height=520, title="Interactive India Semiconductor Hub Coordinates")
    st.plotly_chart(fig_map, use_container_width=True)

    # Detailed Cluster Breakdown
    st.markdown("""
    <div class="section-badge">Cluster Breakdown</div>
    <div class="section-heading">📍 Major Semiconductor Hubs & Clusters</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="custom-card">
          <h4 style="margin:0 0 8px 0; color:{theme["primary"]};">🏢 Sanand & Dholera (Gujarat)</h4>
          <p style="font-size:0.86rem; margin-bottom:6px;"><b>Role:</b> Mega Silicon Fab & Premier ATMP Cluster</p>
          <ul style="font-size:0.82rem; padding-left:16px; margin:0;">
            <li><b>Tata-PSMC Fab:</b> ₹91,526 Cr (50k WSPM)</li>
            <li><b>Micron ATMP:</b> ₹22,516 Cr (Operational)</li>
            <li><b>CG Semi:</b> ₹7,584 Cr (OSAT)</li>
            <li><b>Kaynes Semicon:</b> ₹3,307 Cr (OSAT)</li>
            <li><b>Crystal Matrix:</b> GaN Epi & Micro-LED</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="custom-card">
          <h4 style="margin:0 0 8px 0; color:{theme["secondary"]};">🏗️ Morigaon (Assam)</h4>
          <p style="font-size:0.86rem; margin-bottom:6px;"><b>Role:</b> High-Volume Northeast Packaging Gateway</p>
          <ul style="font-size:0.82rem; padding-left:16px; margin:0;">
            <li><b>Tata Semiconductor (TSAT):</b> ₹27,120 Cr</li>
            <li><b>Capacity:</b> 48 Million units/day</li>
            <li><b>Technologies:</b> Flip-chip, Wire-bond, SiP</li>
            <li><b>Strategic Impact:</b> Transforming Assam into high-tech manufacturing capital</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="custom-card">
          <h4 style="margin:0 0 8px 0; color:{theme["accent"]};">🧪 Bhubaneswar & Jewar & Mohali</h4>
          <p style="font-size:0.86rem; margin-bottom:6px;"><b>Role:</b> Compound Semi, Display & Discrete Power</p>
          <ul style="font-size:0.82rem; padding-left:16px; margin:0;">
            <li><b>SiCSem (Bhubaneswar):</b> ₹2,066 Cr (SiC Fab)</li>
            <li><b>3D Glass Solutions (Odisha):</b> ₹1,943 Cr (3DHI)</li>
            <li><b>HCL-Foxconn (Jewar, UP):</b> ₹3,706 Cr (DDIC)</li>
            <li><b>CDIL (Mohali, Punjab):</b> ₹117 Cr (SiC Power)</li>
            <li><b>ASIP (Andhra Pradesh):</b> ₹480 Cr (SiP)</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3. PAGE: MARKET GROWTH FORECASTER & WHAT-IF SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Market Growth Forecaster":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-title">📈 India Semiconductor Market Growth Forecaster</div>
      <div class="hero-desc">
        Interactive parametric multi-scenario growth modelling based on NITI Aayog ISM Roadmap, 
        IESA market estimates, and dynamic fiscal incentive simulators.
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_scen, tab_whatif = st.tabs(["📊 Market Projection Scenarios", "💡 What-If Incentive & Subsidy Calculator"])

    with tab_scen:
        sc1, sc2, sc3 = st.columns(3)
        base_usd = sc1.number_input("Base Market Size 2023 (USD $B)", 30.0, 80.0, 44.0, 1.0)
        user_cagr = sc2.slider("Custom Scenario CAGR %", 5, 35, 20) / 100.0
        end_year = sc3.slider("Projection Horizon", 2026, 2036, 2032)

        years = list(range(2023, end_year + 1))
        scen_data = {
            "Year": years,
            "Conservative (15% CAGR)": [base_usd * (1.15 ** (y - 2023)) for y in years],
            "Base ISM Roadmap (19% CAGR)": [base_usd * (1.19 ** (y - 2023)) for y in years],
            "Optimistic Boom (22% CAGR)": [base_usd * (1.22 ** (y - 2023)) for y in years],
            f"Custom Scenario ({int(user_cagr*100)}% CAGR)": [base_usd * ((1 + user_cagr) ** (y - 2023)) for y in years],
        }
        sdf = pd.DataFrame(scen_data)

        # Plot multi-scenario fan chart
        fig_scen = go.Figure()
        
        # Shaded confidence band between conservative and optimistic
        fig_scen.add_trace(go.Scatter(
            x=years + years[::-1],
            y=scen_data["Optimistic Boom (22% CAGR)"] + scen_data["Conservative (15% CAGR)"][::-1],
            fill="toself",
            fillcolor=f"{theme['primary']}15",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Analyst Projection Range (15%–22%)"
        ))

        line_colors = {
            "Conservative": theme["palette"][3],
            "Base": theme["palette"][0],
            "Optimistic": theme["palette"][1],
            "Custom": theme["accent"]
        }

        for col in sdf.columns[1:]:
            key = col.split(" ")[0]
            width = 3.5 if key == "Custom" else 2.5
            dash = "dash" if key == "Conservative" else ("dot" if key == "Custom" else "solid")
            fig_scen.add_trace(go.Scatter(
                x=sdf["Year"],
                y=sdf[col].round(1),
                name=col,
                mode="lines+markers",
                line=dict(color=line_colors.get(key, theme["primary"]), width=width, dash=dash),
                marker=dict(size=7),
                hovertemplate="<b>%{x}</b><br>" + col + ": $%{y:.1f} Billion<extra></extra>"
            ))

        # Milestone lines
        fig_scen.add_hline(
            y=103.4, line_dash="dashdot", line_color=theme["secondary"],
            annotation_text="🎯 IESA/IBEF 2030 Target: $103.4B",
            annotation_font_color=theme["secondary"],
            annotation_position="top left"
        )
        fig_scen.add_hline(
            y=55.0, line_dash="dot", line_color=theme["primary"],
            annotation_text="⚡ 2026 Interim Target: $55.0B",
            annotation_font_color=theme["primary"],
            annotation_position="bottom left"
        )

        fig_scen.update_layout(
            yaxis_title="Market Size (USD $ Billion)",
            xaxis_title="Calendar Year",
            legend=dict(orientation="h", y=1.12, x=0.01),
            hovermode="x unified"
        )
        polish_plot(fig_scen, height=440)
        st.plotly_chart(fig_scen, use_container_width=True)

        # Growth Comparison Table
        st.markdown("### 📋 Projected Market Valuation (USD $B & ₹ Lakh Cr)")
        disp_table = sdf.copy()
        for col in disp_table.columns[1:]:
            disp_table[col] = disp_table[col].apply(lambda v: f"${v:.1f}B (₹{v*inr_to_usd_rate/100:.2f}L Cr)")
        st.dataframe(disp_table.set_index("Year"), use_container_width=True)

    with tab_whatif:
        st.markdown("""
        <div class="custom-card">
          <h4 style="margin:0 0 10px 0;">💡 Semiconductor Capex & Subsidy Incentive Simulator</h4>
          <p style="font-size:0.86rem; opacity:0.85;">
            Under the Semicon India Mission (ISM), the Central Government provides <b>50% fiscal support</b> on pari-passu basis 
            for approved Silicon Fabs, Compound Fabs, and ATMP/OSAT units. State governments typically provide an additional <b>15% to 25%</b> top-up subsidy.
          </p>
        </div>
        """, unsafe_allow_html=True)

        wi1, wi2, wi3 = st.columns(3)
        sim_capex = wi1.number_input("Proposed Project Capex (₹ Crore)", 500, 150000, 20000, 500)
        sim_central_pct = wi2.slider("Central Government Subsidy %", 25, 50, 50) / 100.0
        sim_state_pct = wi3.slider("State Top-Up Subsidy %", 10, 30, 20) / 100.0

        central_subsidy = sim_capex * sim_central_pct
        state_subsidy = sim_capex * sim_state_pct
        private_equity = sim_capex - (central_subsidy + state_subsidy)
        total_incentive = central_subsidy + state_subsidy
        incentive_ratio = (total_incentive / sim_capex) * 100.0

        # Gauge & Donut
        g_col1, g_col2 = st.columns([5, 5])
        with g_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=incentive_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Total Government Fiscal Support (%)", 'font': {'size': 15, 'color': theme["text"]}},
                delta={'reference': 50.0, 'increasing': {'color': theme["primary"]}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': theme["text"]},
                    'bar': {'color': theme["secondary"]},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': theme["card_border"],
                    'steps': [
                        {'range': [0, 50], 'color': f"{theme['primary']}22"},
                        {'range': [50, 75], 'color': f"{theme['secondary']}44"},
                        {'range': [75, 100], 'color': f"{theme['accent']}66"},
                    ],
                    'threshold': {
                        'line': {'color': theme["accent"], 'width': 4},
                        'thickness': 0.75,
                        'value': 70.0
                    }
                }
            ))
            polish_plot(fig_gauge, height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g_col2:
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Central Subsidy (ISM)", "State Subsidy Top-up", "Private Capex / Equity"],
                values=[central_subsidy, state_subsidy, max(0, private_equity)],
                hole=0.52,
                marker=dict(colors=[theme["primary"], theme["secondary"], theme["accent"]],
                            line=dict(color=theme["bg"], width=2.5))
            )])
            fig_donut.update_traces(
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.0f} Crore<br>Share: %{percent}<extra></extra>"
            )
            polish_plot(fig_donut, height=320, title="Financing Split Breakdown")
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown(f"""
        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; margin-top:10px;">
          <div style="background:{theme['card_bg']}; padding:14px; border-radius:12px; border:1px solid {theme['card_border']}; text-align:center;">
            <div style="font-size:0.8rem; opacity:0.8;">Central Government Outlay</div>
            <div style="font-size:1.4rem; font-weight:800; color:{theme['primary']};">₹{central_subsidy:,.0f} Cr</div>
          </div>
          <div style="background:{theme['card_bg']}; padding:14px; border-radius:12px; border:1px solid {theme['card_border']}; text-align:center;">
            <div style="font-size:0.8rem; opacity:0.8;">State Government Outlay</div>
            <div style="font-size:1.4rem; font-weight:800; color:{theme['secondary']};">₹{state_subsidy:,.0f} Cr</div>
          </div>
          <div style="background:{theme['card_bg']}; padding:14px; border-radius:12px; border:1px solid {theme['card_border']}; text-align:center;">
            <div style="font-size:0.8rem; opacity:0.8;">Promoter Equity & Debt</div>
            <div style="font-size:1.4rem; font-weight:800; color:{theme['accent']};">₹{private_equity:,.0f} Cr</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 4. PAGE: ML PREDICTOR & SANDBOX
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Predictor & Sandbox":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-title">🤖 Machine Learning Capex Predictor</div>
      <div class="hero-desc">
        Supervised machine learning model trained with Leave-One-Out Cross-Validation (LOO-CV) 
        evaluating semiconductor project capex requirements across states and facility types.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(245, 158, 11, 0.12); border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; font-size: 0.88rem;">
      ⚠️ <b>Methodological Small-Sample Warning:</b> The official ISM dataset comprises 12 approved projects, 
      of which 10 have disclosed capital expenditures. The model uses LOO-CV for transparent, honest evaluation. 
      Predictions are illustrative for ecosystem planning and comparative sensitivity analysis.
    </div>
    """, unsafe_allow_html=True)

    import joblib
    model_path = MODELS_DIR / "investment_model.pkl"
    enc_path = MODELS_DIR / "encoders.pkl"

    if model_path.exists() and enc_path.exists():
        enc = joblib.load(enc_path)
        model = joblib.load(model_path)
        from src.models import predict_investment

        pcol1, pcol2 = st.columns([5, 5])
        with pcol1:
            st.markdown("""
            <div class="section-badge">Simulator</div>
            <div class="section-heading">🎛️ Project Parameter Controls</div>
            """, unsafe_allow_html=True)

            in_state = st.selectbox("Destination State", sorted(enc["state"].classes_))
            in_fac = st.selectbox("Facility Type", sorted(enc["facility_type_canonical"].classes_))
            in_fy = st.slider("Target Fiscal Year of Approval", 2024, 2035, 2027)

            pred_val = predict_investment(in_state, in_fac, int(in_fy), model, enc)
            pred_usd = (pred_val * 1e7) / (inr_to_usd_rate * 1e9)

            st.markdown(f"""
            <div style="background:{theme['kpi_bg_1']}; border-radius:14px; padding:20px; color:#ffffff; margin-top:16px; box-shadow:{theme['glow']};">
              <div style="font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; opacity:0.9;">
                Estimated Committed Capex
              </div>
              <div style="font-size:2.4rem; font-weight:800; margin:6px 0;">
                ₹{pred_val:,.0f} Cr
              </div>
              <div style="font-size:1.05rem; opacity:0.95;">
                ≈ <b>${pred_usd:.2f} Billion USD</b> (at ₹84/$)
              </div>
            </div>
            """, unsafe_allow_html=True)

        with pcol2:
            st.markdown("""
            <div class="section-badge">Gauge Meter</div>
            <div class="section-heading">🎯 Investment Intensity Gauge</div>
            """, unsafe_allow_html=True)

            fig_pred_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_val,
                title={'text': "Predicted Capex (₹ Crore)", 'font': {'size': 14, 'color': theme["text"]}},
                gauge={
                    'axis': {'range': [0, 100000], 'tickwidth': 1, 'tickcolor': theme["text"]},
                    'bar': {'color': theme["secondary"]},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 1.5,
                    'bordercolor': theme["card_border"],
                    'steps': [
                        {'range': [0, 10000], 'color': f"{theme['primary']}25"},
                        {'range': [10000, 30000], 'color': f"{theme['secondary']}35"},
                        {'range': [30000, 100000], 'color': f"{theme['accent']}45"},
                    ]
                }
            ))
            polish_plot(fig_pred_gauge, height=300)
            st.plotly_chart(fig_pred_gauge, use_container_width=True)

        # Cross-facility comparison chart for the selected state
        st.markdown(f"""
        <div class="section-badge">Comparative Sensitivity</div>
        <div class="section-heading">📊 Capex Comparison Across All Facility Types for {in_state} (FY{in_fy})</div>
        """, unsafe_allow_html=True)

        fac_preds = []
        for f in enc["facility_type_canonical"].classes_:
            val = predict_investment(in_state, f, int(in_fy), model, enc)
            fac_preds.append({"Facility Type": f, "Estimated Capex (₹ Cr)": val})
        df_fac_preds = pd.DataFrame(fac_preds).sort_values("Estimated Capex (₹ Cr)", ascending=True)

        fig_fac_comp = px.bar(
            df_fac_preds,
            x="Estimated Capex (₹ Cr)",
            y="Facility Type",
            orientation="h",
            color="Estimated Capex (₹ Cr)",
            color_continuous_scale=[[0, theme["primary"]], [1, theme["secondary"]]],
            text="Estimated Capex (₹ Cr)"
        )
        fig_fac_comp.update_traces(
            texttemplate="₹%{x:,.0f} Cr",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Predicted Capex: ₹%{x:,.0f} Cr<extra></extra>"
        )
        fig_fac_comp.update_layout(coloraxis_showscale=False)
        polish_plot(fig_fac_comp, height=300)
        st.plotly_chart(fig_fac_comp, use_container_width=True)

        # Feature Importances & Model Evaluation
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            <div class="section-badge">Interpretability</div>
            <div class="section-heading">🧬 Model Feature Importances</div>
            """, unsafe_allow_html=True)
            if metrics and metrics.get("feature_importances"):
                fi_df = pd.DataFrame(list(metrics["feature_importances"].items()), columns=["Feature", "Importance"])
                fi_df["Feature Label"] = fi_df["Feature"].map({
                    "facility_type_canonical_enc": "Facility Type (Fab / OSAT)",
                    "approval_fiscal_year": "Approval Fiscal Year",
                    "state_enc": "State Location"
                }).fillna(fi_df["Feature"])
                fig_fi = px.bar(
                    fi_df.sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature Label",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=[[0, theme["primary"]], [1, theme["accent"]]],
                    text="Importance"
                )
                fig_fi.update_traces(texttemplate="%{x:.1%}", textposition="outside")
                fig_fi.update_layout(coloraxis_showscale=False)
                polish_plot(fig_fi, height=240)
                st.plotly_chart(fig_fi, use_container_width=True)

        with m2:
            st.markdown("""
            <div class="section-badge">Model Benchmark</div>
            <div class="section-heading">🏆 Cross-Validation Performance (LOO-CV)</div>
            """, unsafe_allow_html=True)
            if metrics and "model_results" in metrics:
                res_df = pd.DataFrame([
                    {"Model": m, "LOO-MAE (₹ Cr)": r["loo_mae_crore"], "LOO-R²": r["loo_r2"]}
                    for m, r in metrics["model_results"].items()
                ])
                fig_res = px.bar(
                    res_df,
                    x="Model",
                    y="LOO-MAE (₹ Cr)",
                    color="Model",
                    color_discrete_sequence=theme["palette"],
                    text="LOO-MAE (₹ Cr)"
                )
                fig_res.update_traces(texttemplate="₹%{y:,.0f} Cr", textposition="outside")
                polish_plot(fig_res, height=240)
                st.plotly_chart(fig_res, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# 5. PAGE: STATE COMPARATIVE MATRIX
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ State Comparative Matrix":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-title">⚖️ State Semiconductor Ecosystem Comparative Matrix</div>
      <div class="hero-desc">
        Benchmark states head-to-head across committed capex, project counts, technology diversity, 
        and cluster maturity scores.
      </div>
    </div>
    """, unsafe_allow_html=True)

    all_states = sorted(df["state"].unique())
    selected_comp_states = st.multiselect(
        "Select States for Head-to-Head Comparison:",
        all_states,
        default=["Gujarat", "Assam", "Odisha", "Uttar Pradesh"]
    )

    if not selected_comp_states:
        st.warning("Please select at least one state to display comparisons.")
    else:
        # Aggregated metrics per selected state
        state_metrics = []
        for st_name in selected_comp_states:
            st_df = df[df["state"] == st_name]
            known_capex = st_df.loc[st_df["is_investment_known"], "investment_crore"].sum()
            p_count = len(st_df)
            fac_div = st_df["facility_type_canonical"].nunique()
            comm_units = len(st_df[st_df["status"].str.contains("Commercial", case=False, na=False)])
            state_metrics.append({
                "State": st_name,
                "Capex (₹ Cr)": known_capex,
                "Projects": p_count,
                "Facility Types": fac_div,
                "Commercial Units": comm_units
            })
        df_comp = pd.DataFrame(state_metrics)

        # Radar / Spider Chart
        st.markdown("""
        <div class="section-badge">Radar Comparison</div>
        <div class="section-heading">🕸️ Multidimensional Semiconductor Capability Radar</div>
        """, unsafe_allow_html=True)

        categories = ["Capex Scale", "Project Volume", "Tech Diversity", "Operational Readiness"]
        fig_radar = go.Figure()

        # Normalized scores (0-100)
        max_capex = max(df["investment_crore"].sum(), 1)
        for _, row in df_comp.iterrows():
            capex_score = min(100, (row["Capex (₹ Cr)"] / 100000.0) * 100)
            proj_score = min(100, (row["Projects"] / 6.0) * 100)
            tech_score = min(100, (row["Facility Types"] / 4.0) * 100)
            comm_score = min(100, (row["Commercial Units"] / 3.0) * 100)

            fig_radar.add_trace(go.Scatterpolar(
                r=[capex_score, proj_score, tech_score, comm_score, capex_score],
                theta=categories + [categories[0]],
                fill="toself",
                name=row["State"]
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=theme["grid_color"]),
                bgcolor="rgba(0,0,0,0)"
            ),
            legend=dict(orientation="h", y=1.12, x=0.01)
        )
        polish_plot(fig_radar, height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

        # Comparative Metrics Table
        st.markdown("### 📊 Side-by-Side State Performance Metrics")
        st.dataframe(df_comp.set_index("State"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# 6. PAGE: INTERACTIVE PROJECT SHOWCASE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Interactive Project Showcase":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-title">🗂️ Interactive Project Showcase & Raw Data Explorer</div>
      <div class="hero-desc">
        Explore detailed technical specs, corporate JV structures, announced capacities, 
        and official PIB cabinet release records for all semiconductor facilities.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Search query
    search_q = st.text_input("🔍 Search by Company, Technology, City, or Product Keyword:", "")

    filt_showcase = df[
        df["state"].isin(sel_states) &
        df["facility_type_canonical"].isin(sel_fac) &
        df["status"].isin(sel_status)
    ].copy()

    if search_q.strip():
        q_lower = search_q.strip().lower()
        filt_showcase = filt_showcase[
            filt_showcase["company"].str.lower().str.contains(q_lower) |
            filt_showcase["technology_or_product"].str.lower().str.contains(q_lower) |
            filt_showcase["project_location"].str.lower().str.contains(q_lower)
        ]

    st.markdown(f"**Found {len(filt_showcase)} projects matching criteria.**")

    view_type = st.radio("Display Format:", ["✨ Visual Cards Showcase", "📊 Detailed Data Table"], horizontal=True)

    if view_type == "✨ Visual Cards Showcase":
        for _, proj in filt_showcase.iterrows():
            st_badge = "badge-comm" if "Commercial" in proj["status"] else ("badge-exec" if "execution" in proj["status"] else "badge-appr")
            capex_str = fmt_capex(proj["investment_crore"])

            st.markdown(f"""
            <div class="custom-card">
              <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                <div>
                  <span class="badge-status {st_badge}">{proj["status"]}</span>
                  <span style="font-size:0.75rem; opacity:0.75; margin-left:8px;">ID: #{proj["project_id"]} • {proj["approval_period"]}</span>
                  <h3 style="margin:6px 0 2px 0; color:{theme["text"]}; font-weight:800; font-size:1.25rem;">
                    {proj["company"]}
                  </h3>
                  <div style="font-size:0.86rem; color:{theme["primary"]}; font-weight:600;">
                    📍 {proj["project_location"]} • {proj["facility_type_canonical"]}
                  </div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:1.5rem; font-weight:800; color:{theme["secondary"]};">
                    {capex_str}
                  </div>
                  <div style="font-size:0.75rem; opacity:0.8;">Committed Capex</div>
                </div>
              </div>
              <div style="margin-top:12px; font-size:0.86rem; border-top:1px solid {theme['card_border']}; padding-top:10px;">
                <p style="margin:3px 0;"><b>⚡ Technology & Products:</b> {proj["technology_or_product"]}</p>
                <p style="margin:3px 0;"><b>📦 Capacity:</b> {proj["capacity"]}</p>
                <div style="margin-top:8px;">
                  <a href="{proj['source_url']}" target="_blank" style="color:{theme['primary']}; text-decoration:none; font-size:0.8rem; font-weight:600;">
                    🔗 View Official PIB Press Release ↗
                  </a>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.dataframe(filt_showcase, use_container_width=True)

    # Download Buttons
    st.markdown("---")
    st.markdown("### 📥 Export Cleaned Datasets")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "📄 Download ISM Projects (CSV)",
        df.to_csv(index=False),
        file_name="ism_projects_clean.csv",
        mime="text/csv"
    )
    d2.download_button(
        "📈 Download Quarterly Capex (CSV)",
        q.to_csv(index=False),
        file_name="quarterly_investment.csv",
        mime="text/csv"
    )
    d3.download_button(
        "📦 Download State & Facility Aggregates (CSV)",
        st_agg.to_csv(index=False),
        file_name="state_facility_aggregates.csv",
        mime="text/csv"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7. PAGE: METHODOLOGY & ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Methodology & About":
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-title">ℹ️ Methodology, Data Provenance & Architecture</div>
      <div class="hero-desc">
        End-to-end data engineering, machine learning modeling, and analytical governance for 
        India's semiconductor manufacturing ecosystem.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔬 Project Overview
    The **India Semiconductor Growth Predictor** is a rigorous data science dashboard tracking 
    capital expenditure, geographic hub clustering, technology adoption, and growth trajectories under the 
    **Semicon India Programme** (administered by the India Semiconductor Mission - ISM, MeitY).

    **Author:** Nidhi Chougule  
    **Data Provenance:** Press Information Bureau (PIB), Ministry of Electronics and IT (MeitY), NITI Aayog, IESA.

    ---
    ### 🏛️ Official Data Pipeline & Verification
    | Data Source | Entity Covered | Coverage / Metric | Link |
    | :--- | :--- | :--- | :--- |
    | **PIB / Cabinet Approval** | 12 Approved ISM Units | Capex, Capacity, Facility Type | [pib.gov.in](https://www.pib.gov.in) |
    | **NITI Aayog** | Roadmap to 2030 | 19% Industry CAGR Forecast | [niti.gov.in](https://niti.gov.in) |
    | **IESA & IBEF** | Indian Market Size | $44B (2023) to $103.4B (2030) | [iesaonline.org](https://iesaonline.org) |
    | **MeitY / ISM** | Programme Outlay | ₹2,03,500 Crore Phase 1+2 | [ism.gov.in](https://ism.gov.in) |

    ---
    ### 🛡️ Rigorous Analytical Standards
    1. **Strict Date Separation**: Official approval dates are extracted strictly from `approval_period` (Month-Year) specified in cabinet notifications. Data retrieval dates are never conflated with approval dates.
    2. **Handling Missing Values**: Projects with undisclosed capex (such as Crystal Matrix and Suchi Semicon in May 2026) are preserved as `NaN` rather than imputed arbitrarily.
    3. **Small-Sample Machine Learning**: Leave-One-Out Cross-Validation (LOO-CV) is enforced over k-fold CV to provide unbiased estimation given the 10 disclosed data points.
    4. **Scenario Analysis Labeling**: Market projections are explicitly differentiated as parametric scenario simulations using published CAGR parameters rather than black-box extrapolations.

    ---
    ### 📂 GitHub Repository
    Explore the codebase, notebooks, and models on GitHub:  
    👉 **[github.com/nidhichougule/india-semiconductor-growth-predictor](https://github.com/nidhichougule/india-semiconductor-growth-predictor)**
    """)
