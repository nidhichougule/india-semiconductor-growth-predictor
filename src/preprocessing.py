"""
preprocessing.py
================
Cleans, normalises, and produces three clearly-scoped output
datasets for the India Semiconductor Growth Predictor project.

OUTPUTS
-------
1. projects_clean.csv       — Project-level analysis (12 ISM approvals)
2. statetype_analysis.csv   — State × facility-type aggregation
3. quarterly_investment.csv — Cumulative investment by quarter
                              (5 non-empty rows, 10 projects with known ₹)

DATA INTEGRITY RULES enforced here
-----------------------------------
- Missing investment values (Crystal Matrix, Suchi Semicon) are
  retained as NaN — never filled with guesses.
- approval_date_parsed is set to the FIRST DAY of the stated
  approval month (month-granularity is the finest available).
- Sahasra Semiconductors (SPECS, not ISM) is separated and never
  silently merged into ISM aggregates.
- The quarterly series is built from 10 projects with known ₹ only;
  2026Q2 is produced with a NaN investment sum and flagged
  `investment_complete = False`.
- No synthetic rows are added to increase sample size.

Author: Nidhi Chougule
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.parser import parse as dateparse

from src.data_loader import load_raw_projects, load_raw_hubs

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical facility-type mapping
# (normalises the heterogeneous strings in the raw files)
# ---------------------------------------------------------------------------
_FACILITY_TYPE_MAP = {
    "atmp": "ATMP/OSAT",
    "osat": "ATMP/OSAT",
    "atmp / osat": "ATMP/OSAT",
    "semiconductor fab": "Semiconductor Fab",
    "compound semiconductor fab + packaging": "Compound Semiconductor Fab",
    "compound semiconductor fab + atmp": "Compound Semiconductor Fab",
    "semiconductor manufacturing / ddic": "DDIC / Display",
    "advanced packaging / 3d heterogeneous integration": "Advanced Packaging",
    "advanced packaging / 3dhi": "Advanced Packaging",
    "semiconductor packaging": "Advanced Packaging",
    "advanced semiconductor packaging": "Advanced Packaging",   # File 2 ASIP variant
    "semiconductor manufacturing / discrete semiconductors": "Discrete Semiconductor",
    "discrete semiconductor manufacturing": "Discrete Semiconductor",
}


def _normalise_facility_type(raw: str) -> str:
    """Map raw facility_type string to a canonical category."""
    key = str(raw).strip().lower()
    return _FACILITY_TYPE_MAP.get(key, raw.strip())


# ---------------------------------------------------------------------------
# Investment-units check
# ---------------------------------------------------------------------------
# Both raw files use "investment_crore" header and values are verified as
# Indian Rupees (crore).  No unit conversion is needed; we just document this.
INVESTMENT_UNIT = "INR crore (₹ crore)"
INVESTMENT_NOTE = (
    "Values are committed/approved capital expenditure as declared in "
    "PIB press releases — NOT realized/disbursed investment, NOT revenue."
)


# ---------------------------------------------------------------------------
# Dataset 1: Project-level clean dataset (all 12 ISM approvals)
# ---------------------------------------------------------------------------

def build_projects_clean() -> pd.DataFrame:
    """
    Clean and normalise File 1 (12 ISM-approved projects).

    Steps
    -----
    1. Normalise column names (snake_case, strip whitespace).
    2. Parse approval_period (Month Year string) -> approval_date (first of month).
    3. Derive fiscal_year and approval_quarter from approval_date.
    4. Normalise facility_type to canonical categories.
    5. Preserve investment_crore NaN for 2 projects where not disclosed.
    6. Add is_investment_known boolean flag.
    7. Add data_classification column (always 'ISM Approved' for this file).

    Returns
    -------
    pd.DataFrame with 12 rows.
    """
    df = load_raw_projects()

    # 1. Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2. Parse approval_period -> approval_date (first of stated month).
    #    Raises ValueError clearly if a value cannot be parsed, rather than
    #    silently producing NaT and corrupting downstream aggregates.
    def _parse_approval_period(raw: str) -> pd.Timestamp:
        try:
            return pd.Timestamp(dateparse("1 " + str(raw).strip()))
        except Exception as exc:
            raise ValueError(
                f"Cannot parse approval_period value '{raw}' as a month-year "
                f"string: {exc}"
            ) from exc

    df["approval_date"] = df["approval_period"].apply(_parse_approval_period)
    df["approval_date"] = pd.to_datetime(df["approval_date"])

    # 3. Fiscal year (India: Apr–Mar); quarter within fiscal year
    df["approval_quarter_period"] = df["approval_date"].dt.to_period("Q")
    df["approval_quarter_label"] = df["approval_quarter_period"].astype(str)

    # Fiscal year: if month >= 4, FY = year+1, else FY = year
    df["approval_fiscal_year"] = df["approval_date"].apply(
        lambda d: d.year + 1 if d.month >= 4 else d.year
    )

    # 4. Normalise facility_type
    df["facility_type_canonical"] = df["facility_type"].apply(_normalise_facility_type)

    # 5 & 6. Investment completeness flag
    df["is_investment_known"] = df["investment_crore"].notna()

    # 7. Classification
    df["data_classification"] = "ISM Approved"

    # 8. State: strip whitespace
    df["state"] = df["state"].str.strip()

    # Reorder columns for clarity
    col_order = [
        "project_id", "company", "project_location", "state",
        "facility_type", "facility_type_canonical",
        "technology_or_product",
        "investment_crore", "is_investment_known",
        "capacity",
        "approval_period", "approval_date",
        "approval_quarter_label", "approval_fiscal_year",
        "status", "data_classification",
        "source_url", "hub_cluster",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    return df


# ---------------------------------------------------------------------------
# Dataset 2: State × Facility-type aggregation
# ---------------------------------------------------------------------------

def build_statetype_analysis(projects_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate known-investment projects by state and canonical facility type.

    Parameters
    ----------
    projects_df : pd.DataFrame, optional
        Output of build_projects_clean().  If None, builds it fresh.

    Returns
    -------
    pd.DataFrame with columns:
        state, facility_type_canonical, project_count,
        total_investment_crore, projects_with_known_investment,
        investment_coverage_pct
    """
    if projects_df is None:
        projects_df = build_projects_clean()

    agg = (
        projects_df
        .groupby(["state", "facility_type_canonical"], as_index=False)
        .agg(
            project_count=("project_id", "count"),
            total_investment_crore=("investment_crore", "sum"),
            projects_with_known_investment=("is_investment_known", "sum"),
        )
    )
    agg["investment_coverage_pct"] = (
        agg["projects_with_known_investment"] / agg["project_count"] * 100
    ).round(1)
    agg = agg.sort_values("total_investment_crore", ascending=False).reset_index(drop=True)
    return agg


# ---------------------------------------------------------------------------
# Dataset 3: Quarterly cumulative investment series
# ---------------------------------------------------------------------------

def build_quarterly_investment(projects_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build a quarterly cumulative investment table from ISM-approved projects.

    DATA LIMITATIONS (documented, not hidden)
    -----------------------------------------
    - Only 10 of 12 projects have a disclosed investment value.
    - Crystal Matrix Limited (CML) and Suchi Semicon (SSPL), both approved
      May 2026 (2026Q2), have no investment figure in the source data.
    - 2026Q2 row is included with investment_crore_in_quarter = NaN and
      investment_complete = False to preserve the approval event.
    - The resulting series has 6 non-empty quarters over a 13-quarter span
      (Jun 2023 – May 2026), with 7 gap quarters.
    - This series is TOO SPARSE for ARIMA/ML time-series forecasting.
      See docs/ml_feasibility_note.md for the full assessment.

    Parameters
    ----------
    projects_df : pd.DataFrame, optional
        Output of build_projects_clean().

    Returns
    -------
    pd.DataFrame with columns:
        quarter_period, quarter_label, projects_approved,
        investment_crore_in_quarter, cumulative_investment_crore,
        investment_complete
    """
    if projects_df is None:
        projects_df = build_projects_clean()

    q_grp = (
        projects_df
        .groupby("approval_quarter_label", as_index=False)
        .agg(
            projects_approved=("project_id", "count"),
            investment_crore_in_quarter=("investment_crore", "sum"),
            projects_with_investment=("is_investment_known", "sum"),
        )
        .sort_values("approval_quarter_label")
    )

    # Flag quarters where NOT all project investments are known
    q_grp["investment_complete"] = (
        q_grp["projects_with_investment"] == q_grp["projects_approved"]
    )

    # groupby .sum() treats NaN as 0 by default; restore NaN where the quarter
    # is incomplete (i.e. not all projects in that quarter have a known investment).
    q_grp.loc[~q_grp["investment_complete"], "investment_crore_in_quarter"] = np.nan

    # Cumulative investment — NaN-propagating.
    # Once an incomplete (NaN) quarter is reached, the cumulative is also NaN,
    # clearly flagging that we cannot claim a total for any subsequent period.
    # pandas Expanding.sum(skipna=...) was removed in 2.x; implement manually.
    cumvals = []
    running = 0.0
    has_nan = False
    for v in q_grp["investment_crore_in_quarter"]:
        if has_nan or pd.isna(v):
            has_nan = True
            cumvals.append(np.nan)
        else:
            running += v
            cumvals.append(running)
    q_grp["cumulative_investment_crore"] = cumvals

    q_grp = q_grp.drop(columns=["projects_with_investment"])
    return q_grp.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Non-ISM project: Sahasra Semiconductors
# ---------------------------------------------------------------------------

def extract_non_ism_projects() -> pd.DataFrame:
    """
    Return records from File 2 that are NOT ISM-approved projects.

    Currently: 1 record — Sahasra Semiconductors Pvt. Ltd.
    (SPECS + EMC funded, operational facility).

    These must NEVER be merged into ISM aggregates without explicit labelling.
    """
    df = load_raw_hubs()
    non_ism = df[df["classification"] != "Approved ISM project"].copy()
    return non_ism.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save all processed datasets
# ---------------------------------------------------------------------------

def save_all(verbose: bool = True) -> dict:
    """
    Build and save all three processed datasets to data/processed/.

    Returns
    -------
    dict mapping dataset name -> saved Path
    """
    saved = {}

    projects = build_projects_clean()
    p1 = _PROCESSED_DIR / "projects_clean.csv"
    projects.to_csv(p1, index=False)
    saved["projects_clean"] = p1
    if verbose:
        print(f"[OK] projects_clean.csv        -> {projects.shape[0]} rows, {projects.shape[1]} cols")

    statetype = build_statetype_analysis(projects)
    p2 = _PROCESSED_DIR / "statetype_analysis.csv"
    statetype.to_csv(p2, index=False)
    saved["statetype_analysis"] = p2
    if verbose:
        print(f"[OK] statetype_analysis.csv    -> {statetype.shape[0]} rows, {statetype.shape[1]} cols")

    quarterly = build_quarterly_investment(projects)
    p3 = _PROCESSED_DIR / "quarterly_investment.csv"
    quarterly.to_csv(p3, index=False)
    saved["quarterly_investment"] = p3
    if verbose:
        print(f"[OK] quarterly_investment.csv  -> {quarterly.shape[0]} rows, {quarterly.shape[1]} cols")
        print(f"     NOTE: {quarterly['investment_complete'].sum()} of {len(quarterly)} quarters have complete investment data.")

    non_ism = extract_non_ism_projects()
    p4 = _PROCESSED_DIR / "non_ism_projects.csv"
    non_ism.to_csv(p4, index=False)
    saved["non_ism_projects"] = p4
    if verbose:
        print(f"[OK] non_ism_projects.csv      -> {non_ism.shape[0]} rows (excluded from ISM aggregates)")

    return saved


if __name__ == "__main__":
    print("Running preprocessing pipeline...")
    save_all(verbose=True)
    print("\nDone. Check data/processed/ for output files.")
