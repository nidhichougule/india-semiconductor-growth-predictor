"""
src/eda.py
==========
Exploratory Data Analysis for India Semiconductor Growth Predictor.
Generates charts saved to docs/charts/.

Run: python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing import build_projects_clean, build_quarterly_investment, build_statetype_analysis

CHARTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Colour palette ---
COLORS = ["#1e3a5f", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d",
          "#3b1f2b", "#44bba4", "#e94f37"]
ACCENT = "#2e86ab"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig_investment_by_state(df: pd.DataFrame) -> str:
    """Bar chart: total known investment by state."""
    state_inv = (
        df[df["is_investment_known"]]
        .groupby("state")["investment_crore"]
        .sum()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(state_inv.index, state_inv.values / 1000, color=ACCENT)
    ax.set_xlabel("Investment (Rs. thousand crore)", fontsize=11)
    ax.set_title("ISM-Approved Investment by State\n(known capex only)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, state_inv.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"Rs.{val/1000:.0f}k cr", va="center", fontsize=9)
    fig.tight_layout()
    out = CHARTS_DIR / "01_investment_by_state.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")
    return str(out)


def fig_investment_by_facility(df: pd.DataFrame) -> str:
    """Pie chart: investment by canonical facility type."""
    fac = (
        df[df["is_investment_known"]]
        .groupby("facility_type_canonical")["investment_crore"]
        .sum()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        fac.values, labels=None, autopct="%1.1f%%",
        colors=COLORS[:len(fac)], startangle=140,
        pctdistance=0.78, wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(wedges, fac.index, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=9, frameon=False)
    ax.set_title("Investment Share by Facility Type\n(Rs. crore, known capex)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = CHARTS_DIR / "02_investment_by_facility.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")
    return str(out)


def fig_quarterly_timeline(q_df: pd.DataFrame) -> str:
    """Timeline: projects approved and investment per quarter."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(q_df))
    labels = q_df["approval_quarter_label"].tolist()

    # Bar: project count
    ax1.bar(x, q_df["projects_approved"], color=ACCENT, alpha=0.7, label="Projects approved")
    ax1.set_ylabel("Projects approved", fontsize=11, color=ACCENT)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.tick_params(axis="y", labelcolor=ACCENT)

    # Line: cumulative investment (right axis)
    ax2 = ax1.twinx()
    cumvals = q_df["cumulative_investment_crore"].tolist()
    known_x = [i for i, v in enumerate(cumvals) if not pd.isna(v)]
    known_y = [v / 1000 for v in cumvals if not pd.isna(v)]
    ax2.plot(known_x, known_y, "o-", color=COLORS[2], linewidth=2.5,
             markersize=7, label="Cumulative investment")
    ax2.set_ylabel("Cumulative investment (Rs. thousand crore)", fontsize=11, color=COLORS[2])
    ax2.tick_params(axis="y", labelcolor=COLORS[2])
    ax2.spines["right"].set_visible(True)

    # Mark incomplete quarter
    incomplete = q_df[~q_df["investment_complete"]]
    for _, row in incomplete.iterrows():
        idx = labels.index(row["approval_quarter_label"])
        ax1.bar(idx, row["projects_approved"], color=COLORS[4], alpha=0.8)

    patch_known = mpatches.Patch(color=ACCENT, alpha=0.7, label="Projects (complete inv.)")
    patch_incomplete = mpatches.Patch(color=COLORS[4], alpha=0.8, label="Projects (inv. not disclosed)")
    line_cum = plt.Line2D([0], [0], color=COLORS[2], linewidth=2, marker="o", label="Cumulative investment")
    ax1.legend(handles=[patch_known, patch_incomplete, line_cum], loc="upper left", fontsize=9, frameon=False)

    ax1.set_title("ISM Project Approvals and Cumulative Investment by Quarter", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = CHARTS_DIR / "03_quarterly_timeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")
    return str(out)


def fig_project_count_by_state(df: pd.DataFrame) -> str:
    """Count of ISM projects per state."""
    counts = df["state"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(counts.index, counts.values, color=[COLORS[i % len(COLORS)] for i in range(len(counts))])
    ax.set_xlabel("Number of projects", fontsize=11)
    ax.set_title("ISM Projects by State", fontsize=13, fontweight="bold")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.05, i, str(v), va="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = CHARTS_DIR / "04_projects_by_state.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")
    return str(out)


def fig_facility_type_count(df: pd.DataFrame) -> str:
    """Count of projects by canonical facility type."""
    counts = df["facility_type_canonical"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(counts.index, counts.values, color=COLORS[3])
    ax.set_xlabel("Number of projects", fontsize=11)
    ax.set_title("ISM Projects by Facility Type", fontsize=13, fontweight="bold")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.05, i, str(v), va="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = CHARTS_DIR / "05_facility_type_count.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")
    return str(out)


def fig_market_projection() -> str:
    """Parametric India semiconductor market projection (NITI Aayog / IESA inputs)."""
    base_year = 2023
    base_market_usd = 44.0  # IESA estimate FY2023, USD billion
    cagr = 0.19             # 19% CAGR per NITI Aayog ISM Roadmap
    years = list(range(2023, 2031))
    market = [base_market_usd * (1 + cagr) ** (y - base_year) for y in years]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(years, market, alpha=0.18, color=ACCENT)
    ax.plot(years, market, "o-", color=ACCENT, linewidth=2.5, markersize=8)
    for y, m in zip(years, market):
        ax.annotate(f"${m:.0f}B", (y, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5)
    ax.axhline(y=103.4, color=COLORS[2], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(2026, 106, "IESA/IBEF 2030 target: $103.4B", color=COLORS[2], fontsize=9)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Market size (USD billion)", fontsize=11)
    ax.set_title("India Semiconductor Market Projection (19% CAGR)\n"
                 "Source: NITI Aayog ISM Roadmap — parametric, not ML-derived",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 220)
    fig.tight_layout()
    out = CHARTS_DIR / "06_market_projection.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")
    return str(out)


def fig_investment_heatmap(st_df: pd.DataFrame) -> str:
    """Heatmap: investment by state x facility type."""
    pivot = st_df.pivot_table(
        index="state", columns="facility_type_canonical",
        values="total_investment_crore", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values / 1000, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Investment (Rs. thousand crore)", fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val > 0:
                ax.text(j, i, f"{val/1000:.0f}k", ha="center", va="center", fontsize=8,
                        color="white" if val > 50000 else "black")
    ax.set_title("Investment Heat Map: State x Facility Type (Rs. crore)", fontsize=12, fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    out = CHARTS_DIR / "07_investment_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")
    return str(out)


def run_all():
    print("Generating EDA charts...")
    df = build_projects_clean()
    q  = build_quarterly_investment(df)
    st = build_statetype_analysis(df)

    fig_investment_by_state(df)
    fig_investment_by_facility(df)
    fig_quarterly_timeline(q)
    fig_project_count_by_state(df)
    fig_facility_type_count(df)
    fig_market_projection()
    fig_investment_heatmap(st)
    print(f"Done. Charts in {CHARTS_DIR}")


if __name__ == "__main__":
    run_all()
