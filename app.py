"""
app.py — Streamlit dashboard for India Semiconductor Growth Predictor
Run: streamlit run app.py
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Semiconductor Growth Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "src" / "models"

@st.cache_data
def load_data():
    from src.preprocessing import (
        build_projects_clean, build_quarterly_investment,
        build_statetype_analysis, extract_non_ism_projects
    )
    df   = build_projects_clean()
    q    = build_quarterly_investment(df)
    st_  = build_statetype_analysis(df)
    non_ = extract_non_ism_projects()
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0f1117; }
  .kpi-box {
      background: linear-gradient(135deg,#1e3a5f,#2e86ab);
      border-radius: 12px; padding: 18px 22px;
      text-align: center; color: white;
      box-shadow: 0 4px 15px rgba(46,134,171,0.25);
  }
  .kpi-value { font-size: 2.2rem; font-weight: 800; margin: 0; }
  .kpi-label { font-size: 0.82rem; opacity: 0.85; margin-top: 4px; }
  .warn-box {
      background: rgba(241,143,1,0.12); border-left: 4px solid #f18f01;
      padding: 10px 14px; border-radius: 6px; font-size: 0.84rem;
      margin-bottom: 10px;
  }
  .section-title { font-size: 1.25rem; font-weight: 700;
      border-left: 4px solid #2e86ab; padding-left: 10px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ── load ──────────────────────────────────────────────────────────────────────
df, q, st_agg, non_ism = load_data()
scenarios = load_scenarios()
metrics   = load_metrics()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg", width=60)
    st.title("🔬 India Semiconductor\nGrowth Predictor")
    st.caption("Data: PIB / India Semiconductor Mission (ISM)")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Overview", "📈 Market Scenarios",
                                  "🤖 ML Model", "🗂️ Raw Data", "ℹ️ About"])
    st.markdown("---")
    # Filters (overview)
    if page == "📊 Overview":
        states_all = sorted(df["state"].unique())
        sel_states = st.multiselect("Filter by state", states_all, default=states_all)
        fac_all = sorted(df["facility_type_canonical"].unique())
        sel_fac = st.multiselect("Filter by facility type", fac_all, default=fac_all)
    st.markdown("---")
    st.caption("© 2026 Nidhi Chougule | MSc Data Science")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("## 🔬 India Semiconductor Growth Predictor")
    st.markdown("### ISM-Approved Projects Dashboard  *(PIB official data)*")

    filt = df[df["state"].isin(sel_states) & df["facility_type_canonical"].isin(sel_fac)]

    # KPIs
    total_inv  = filt.loc[filt["is_investment_known"], "investment_crore"].sum()
    n_proj     = len(filt)
    n_states   = filt["state"].nunique()
    n_fac      = filt["facility_type_canonical"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{n_proj}", "ISM-Approved Projects"),
        (c2, f"Rs.{total_inv/1000:.1f}k cr", "Committed Capex"),
        (c3, f"{n_states}", "States Covered"),
        (c4, f"{n_fac}", "Facility Types"),
    ]:
        col.markdown(f"""<div class="kpi-box">
          <p class="kpi-value">{val}</p>
          <p class="kpi-label">{lbl}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: investment by state | facility type pie
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown('<p class="section-title">Investment by State</p>', unsafe_allow_html=True)
        state_inv = (
            filt[filt["is_investment_known"]]
            .groupby("state")["investment_crore"].sum()
            .reset_index()
            .sort_values("investment_crore", ascending=True)
        )
        fig = px.bar(state_inv, x="investment_crore", y="state", orientation="h",
                     color="investment_crore", color_continuous_scale="Blues",
                     labels={"investment_crore": "Rs. crore", "state": "State"})
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_showscale=False, height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<p class="section-title">Investment by Facility Type</p>', unsafe_allow_html=True)
        fac_inv = (
            filt[filt["is_investment_known"]]
            .groupby("facility_type_canonical")["investment_crore"].sum()
            .reset_index()
        )
        fig2 = px.pie(fac_inv, values="investment_crore", names="facility_type_canonical",
                      color_discrete_sequence=px.colors.qualitative.Bold, hole=0.4)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320,
                           paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                           legend=dict(font_size=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: quarterly timeline
    st.markdown('<p class="section-title">Quarterly Approvals & Cumulative Investment</p>',
                unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=q["approval_quarter_label"], y=q["projects_approved"],
        name="Projects approved", marker_color="#2e86ab",
        text=q["projects_approved"], textposition="outside"
    ))
    cum_q = q.copy()
    cum_q.loc[~cum_q["investment_complete"], "cumulative_investment_crore"] = None
    fig3.add_trace(go.Scatter(
        x=cum_q["approval_quarter_label"],
        y=cum_q["cumulative_investment_crore"].div(1000),
        name="Cumulative capex (Rs. thousand crore)",
        mode="lines+markers", line=dict(color="#f18f01", width=3),
        yaxis="y2", connectgaps=False
    ))
    fig3.update_layout(
        yaxis=dict(title="Projects approved", color="#2e86ab"),
        yaxis2=dict(title="Cumulative capex (Rs. thousand cr)", overlaying="y",
                    side="right", color="#f18f01"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=0, r=0, t=30, b=0), height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("⚠️ 2026Q2 (Crystal Matrix + Suchi Semicon): investment not disclosed — excluded from cumulative.")

    # Row 3: projects by state bar + status
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.markdown('<p class="section-title">Projects by State</p>', unsafe_allow_html=True)
        pc = filt["state"].value_counts().reset_index()
        pc.columns = ["state", "count"]
        fig4 = px.bar(pc, x="state", y="count", color="count",
                      color_continuous_scale="Teal", text="count",
                      labels={"count": "Projects", "state": "State"})
        fig4.update_layout(showlegend=False, coloraxis_showscale=False, height=300,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    with r3c2:
        st.markdown('<p class="section-title">Project Status Distribution</p>', unsafe_allow_html=True)
        sc = filt["status"].value_counts().reset_index()
        sc.columns = ["status", "count"]
        fig5 = px.pie(sc, values="count", names="status", hole=0.45,
                      color_discrete_sequence=["#2e86ab","#f18f01","#44bba4","#c73e1d"])
        fig5.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300,
                           paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MARKET SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Market Scenarios":
    st.markdown("## 📈 India Semiconductor Market Growth Scenarios")
    st.markdown("""<div class="warn-box">
    <b>Scenario Analysis — NOT ML prediction.</b> Projections use published CAGR rates from
    NITI Aayog ISM Roadmap and IESA/IBEF estimates. Input assumptions can be adjusted below.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    base_usd = c1.number_input("Base market size 2023 (USD billion)", 30.0, 80.0, 44.0, 1.0)
    cagr_base = c2.slider("Base CAGR % (NITI Aayog)", 10, 30, 19) / 100
    end_year  = c3.slider("Projection end year", 2026, 2035, 2030)

    years = list(range(2023, end_year + 1))
    scen_df = pd.DataFrame({
        "Year": years,
        f"Base ({int(cagr_base*100)}% CAGR)": [base_usd * (1 + cagr_base) ** (y - 2023) for y in years],
        "Optimistic (22% CAGR)": [base_usd * 1.22 ** (y - 2023) for y in years],
        "Conservative (15% CAGR)": [base_usd * 1.15 ** (y - 2023) for y in years],
    })

    fig = go.Figure()
    colors = {"Base": "#2e86ab", "Optimistic": "#44bba4", "Conservative": "#f18f01"}
    for col in scen_df.columns[1:]:
        key = col.split(" ")[0]
        fig.add_trace(go.Scatter(
            x=scen_df["Year"], y=scen_df[col].round(1),
            name=col, mode="lines+markers",
            line=dict(color=colors.get(key, "#aaa"), width=2.5),
        ))
    fig.add_hline(y=103.4, line_dash="dash", line_color="#c73e1d",
                  annotation_text="IESA/IBEF 2030 target: $103.4B",
                  annotation_font_color="#c73e1d")
    fig.update_layout(
        yaxis_title="Market Size (USD billion)", xaxis_title="Year",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=0, r=0, t=30, b=0), height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scenario table
    st.markdown("### Projected Market Size (USD billion)")
    disp = scen_df.copy()
    for c in disp.columns[1:]:
        disp[c] = disp[c].apply(lambda v: f"${v:.1f}B")
    st.dataframe(disp.set_index("Year"), use_container_width=True)

    st.markdown("""
    **Sources:** NITI Aayog ISM Roadmap 2022, IESA India Electronics & Semiconductor Association,
    IBEF Sector Report 2024. Base market size from IESA FY2023 estimate.
    """)

    # ISM committed capex box
    st.markdown("---")
    st.markdown("### ISM Programme Committed Capex")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total ISM approvals", "12 projects")
    k2.metric("Known committed capex", "Rs. 1,60,365 crore")
    k3.metric("Phase 1+2 programme outlay", "Rs. 2,03,500 crore")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ML MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 ML Model":
    st.markdown("## 🤖 Investment Prediction Model")
    st.markdown("""<div class="warn-box">
    <b>Small-sample warning.</b> Only 10 projects have disclosed investment values.
    Model is <b>illustrative</b> — trained with Leave-One-Out CV for honest evaluation.
    Do not use for investment decisions.
    </div>""", unsafe_allow_html=True)

    if metrics is None:
        st.info("Run `python -m src.models` first to train the model.")
    else:
        # Model metrics
        st.markdown("### Model Performance (Leave-One-Out Cross-Validation, n=10)")
        best = metrics["best_model"]
        res  = metrics["model_results"]
        rows = []
        for m, r in res.items():
            rows.append({"Model": m,
                         "LOO-MAE (Rs. crore)": f"{r['loo_mae_crore']:,.0f}",
                         "LOO-R²": f"{r['loo_r2']:.3f}",
                         "Selected": "✅ Best" if m == best else ""})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption(f"**Best model:** {best}  |  n_training={metrics['n_training']}  |  "
                   f"Missing investment projects excluded: {metrics['n_missing_investment']}")

        # Feature importances
        if metrics.get("feature_importances"):
            st.markdown("### Feature Importances")
            fi_df = pd.DataFrame(list(metrics["feature_importances"].items()),
                                 columns=["Feature", "Importance"])
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="Blues")
            fig_fi.update_layout(showlegend=False, coloraxis_showscale=False, height=220,
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 font_color="white", margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_fi, use_container_width=True)

        # Prediction tool
        st.markdown("### Predict Investment for a New Project")
        st.caption("Illustrative prediction only. Output depends on training data patterns.")

        from src.models import predict_investment
        import joblib
        model_path = MODELS_DIR / "investment_model.pkl"
        enc_path   = MODELS_DIR / "encoders.pkl"
        if model_path.exists() and enc_path.exists():
            enc = joblib.load(enc_path)
            pc1, pc2, pc3 = st.columns(3)
            sel_state = pc1.selectbox("State", sorted(enc["state"].classes_))
            sel_fac   = pc2.selectbox("Facility type", sorted(enc["facility_type_canonical"].classes_))
            sel_fy    = pc3.number_input("Fiscal year", 2025, 2035, 2027)
            if st.button("Predict", type="primary"):
                pred = predict_investment(sel_state, sel_fac, int(sel_fy),
                                          joblib.load(model_path), enc)
                st.success(f"Predicted investment: **Rs. {pred:,.0f} crore** ≈ Rs. {pred/1000:.1f}k crore")
                st.caption("⚠️ This is a model estimate based on 10 training examples. High uncertainty.")

    # Quarterly series limitation
    st.markdown("---")
    st.markdown("### Time-Series Forecasting Limitation")
    st.markdown("""
    The ISM quarterly approval series has only **6 non-empty quarters** over 13 quarters
    (54% gap rate). This is far below the ~30 observations needed for ARIMA/ML time-series.

    **Recommended alternative data** (not yet collected):
    - **DGCI&S/DGFT HS-8542** monthly import/export data (electronic integrated circuits)
    - Source: https://tradestat.commerce.gov.in
    - Provides ~180 monthly observations from FY2010 — suitable for ARIMA
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗂️ Raw Data":
    st.markdown("## 🗂️ Project Data Explorer")

    tab1, tab2, tab3 = st.tabs(["ISM Projects", "Quarterly Summary", "State × Facility"])

    with tab1:
        st.markdown("**All 12 ISM-approved projects** (data/processed/projects_clean.csv)")
        cols_show = ["project_id","company","state","facility_type_canonical",
                     "investment_crore","is_investment_known","approval_period",
                     "approval_quarter_label","approval_fiscal_year","status"]
        st.dataframe(df[cols_show], use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False),
                           file_name="ism_projects_clean.csv", mime="text/csv")

    with tab2:
        st.markdown("**Quarterly investment series** — 6 quarters with approvals")
        st.dataframe(q, use_container_width=True, hide_index=True)
        st.caption("investment_complete=False means investment not disclosed for ≥1 project in that quarter.")

    with tab3:
        st.markdown("**State × Facility type aggregation**")
        st.dataframe(st_agg, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
    ### India Semiconductor Growth Predictor

    An end-to-end data science project analysing India's semiconductor ecosystem
    under the Semicon India Programme (India Semiconductor Mission).

    **Author:** Nidhi Chougule | MSc Data Science

    ---
    ### Data Sources
    | Source | Description | URL |
    |--------|-------------|-----|
    | PIB / ISM | 12 ISM cabinet-approved projects | pib.gov.in |
    | NITI Aayog | 19% CAGR assumption | niti.gov.in |
    | IESA / IBEF | 2030 market target $103.4B | ibef.org |
    | MeitY / ISM | Programme outlay Rs. 2,03,500 cr | ism.gov.in |

    All investment figures are **committed capital expenditure** per PIB press releases —
    not realised disbursements.

    ---
    ### Methodology Notes
    - **Approval dates**: from `approval_period` (month-year) in File 1 only.
      PIB `official_source_date` is data-collection date, never used as approval date.
    - **Missing investments** (Crystal Matrix, Suchi Semicon): preserved as NaN, not imputed.
    - **ML model**: Random Forest with LOO-CV, n=10. Illustrative only.
    - **Market projections**: parametric scenario analysis using published CAGR.
      NOT trained ML output.
    - **Quarterly forecasting**: not feasible — only 6 data points. Documented in
      `docs/ml_feasibility_note.md`.

    ---
    ### GitHub Repository
    [github.com/nidhichougule/india-semiconductor-growth-predictor](https://github.com/nidhichougule/india-semiconductor-growth-predictor)
    """)
