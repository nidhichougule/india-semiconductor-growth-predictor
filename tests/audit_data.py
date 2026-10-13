"""
Phase 2 Data Audit Script
=========================
Run from the repository root:
    python tests/audit_data.py

Outputs a full structural audit of both raw CSV files, cross-file
entity matching results, date/quarter analysis, investment completeness,
and an ML feasibility verdict.

DATA INTEGRITY NOTES
--------------------
- approval_period (File 1) is the ONLY field used for project approval dates.
  It contains month-year strings (e.g. "June 2023") derived directly from PIB
  cabinet-approval press releases.
- official_source_date (File 2) is the PIB data-COLLECTION date, NOT the
  project approval date, and is NEVER used as an approval date here.
- Month-year strings are parsed to the first day of that month.  No intra-month
  precision is claimed or manufactured.
- investment_crore values are committed/approved capital expenditure (capex)
  as stated in PIB press releases, NOT realised disbursements or revenue.
- Cross-file entity matching uses a transparent lookup table, not fuzzy heuristics.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dateutil.parser import parse as dateparse, ParserError

# ---- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"

FILE1 = RAW / "india_approved_semiconductor_projects_government.csv"
FILE2 = RAW / "india_semiconductor_projects_and_hubs_2026_government_sources.csv"

SEP  = "=" * 68
SEP2 = "-" * 68

# ---- Load -------------------------------------------------------------------
print(SEP)
print("LOADING RAW FILES")
print(SEP)
for fpath in (FILE1, FILE2):
    if not fpath.exists():
        print(f"FATAL: file not found: {fpath}", file=sys.stderr)
        sys.exit(1)
    print(f"  OK  {fpath.name}")

f1 = pd.read_csv(FILE1)
f2 = pd.read_csv(FILE2)
print(f"\n  File 1 shape: {f1.shape[0]} rows x {f1.shape[1]} cols")
print(f"  File 2 shape: {f2.shape[0]} rows x {f2.shape[1]} cols")

# ---- Per-file structural audit ----------------------------------------------
def file_audit(df, label):
    print()
    print(SEP)
    print(f"STRUCTURAL AUDIT -- {label}")
    print(SEP)

    print(f"\n  Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    # Column names, dtypes, missing, unique
    print(f"  {'Column':<45} {'dtype':<12} {'missing':>7}  {'unique':>6}")
    print(f"  {SEP2}")
    for col in df.columns:
        n_miss   = int(df[col].isna().sum())
        n_unique = int(df[col].nunique(dropna=False))
        dtype    = str(df[col].dtype)
        flag     = "  << HAS MISSING" if n_miss > 0 else ""
        print(f"  {col:<45} {dtype:<12} {n_miss:>7}  {n_unique:>6}{flag}")

    # Duplicate rows
    n_dup = int(df.duplicated().sum())
    print(f"\n  Duplicate rows (exact): {n_dup}")

    # Numeric summary
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print(f"\n  Numeric summary:")
        print(df[num_cols].describe().to_string())

file_audit(f1, "File 1 -- india_approved_semiconductor_projects_government.csv")
file_audit(f2, "File 2 -- india_semiconductor_projects_and_hubs_2026_government_sources.csv")

# ---- Investment unit clarification ------------------------------------------
print()
print(SEP)
print("INVESTMENT UNIT CLARIFICATION")
print(SEP)
print("""
  Column name : investment_crore (both files)
  Unit        : Indian Rupees -- crore (1 crore = 10 million INR)
  Measure     : Committed / approved capital expenditure (capex) as
                declared in the relevant PIB press release.
  NOT         : Realised disbursement, revenue, or profit.
  Source      : PIB / India Semiconductor Mission (ISM) cabinet approvals.

  Minor value discrepancy for 3D Glass Solutions Inc.:
    File 1  investment_crore = 1943       (rounded, from PRID=2247814)
    File 2  investment_crore = 1943.53    (precise figure, from PRID=2253549)
  -> File 2 is the more precise source for this project.  Preprocessing uses
     File 1 values (as-is) and documents this discrepancy in the provenance
     manifest.  No blending of the two figures is performed.
""")

# ---- Cross-file entity matching ---------------------------------------------
print(SEP)
print("CROSS-FILE ENTITY MATCHING (transparent lookup table)")
print(SEP)
print("""
  Matching strategy: manually reviewed, case-normalised company name lookup.
  NOT fuzzy / probabilistic.  Each mapping is verifiable against PIB releases.
  Notes on name differences are documented inline.
""")

# Canonical mapping: File 1 project_id -> File 2 id, with name-diff notes
ENTITY_MAP = {
    # f1_id : (f2_id, note)
    1:  (1,  "Exact match (Micron Technology Inc.)"),
    2:  (2,  "File 1 includes TSAT subsidiary; File 2 uses TEPL only -- same legal entity"),
    3:  (3,  "File 2 lists TEPL (Morigaon); File 1 as TSAT -- same group, diff subsidiary"),
    4:  (4,  "File 1: 'CG Power / CG Semi'; File 2: CG Power Ltd -- same entity, CG Semi is trade name"),
    5:  (5,  "File 1: 'Kaynes / Kaynes Semicon'; File 2: Kaynes Technology India Ltd"),
    6:  (6,  "File 1: 'HCL-Foxconn JV (Vama Sundari)'; File 2: Vama Sundari / Foxconn India"),
    7:  (8,  "File 1 id=7 is SiCSem; File 2 id=8 is SiCSem -- ordering differs"),
    8:  (7,  "File 1 id=8 is 3D Glass Solutions; File 2 id=7 is 3D Glass / HIPSPL"),
    9:  (9,  "Exact match (CDIL), minor capacity rounding"),
    10: (10, "Exact match (ASIP)"),
    11: (11, "Exact match (CML) -- investment NaN in both"),
    12: (12, "Exact match (SSPL) -- investment NaN in both"),
}

print(f"  {'F1 id':<7} {'F2 id':<7} {'F1 company (truncated)':<40} {'Match note'}")
print(f"  {'-'*6}  {'-'*6}  {'-'*40}  {'-'*35}")
for f1_id, (f2_id, note) in ENTITY_MAP.items():
    f1_name = f1.loc[f1["project_id"] == f1_id, "company"].values
    f1_name_str = (f1_name[0][:38] + "..") if f1_name.size and len(f1_name[0]) > 40 else (f1_name[0] if f1_name.size else "?")
    print(f"  {f1_id:<7} {f2_id:<7} {f1_name_str:<40}  {note[:70]}")

# Sahasra -- only in File 2
sahasra = f2[f2["company"].str.contains("Sahasra", na=False)]
print(f"\n  File 2 only (NOT in File 1):  Sahasra Semiconductors Pvt. Ltd.")
print(f"    id={sahasra['id'].values[0]}  classification={sahasra['classification'].values[0]}")
print(f"    programme={sahasra['government_programme'].values[0]}")
print(f"    NOTE: SPECS+EMC funded -- NEVER an ISM approval.  Excluded from all ISM aggregates.")

# ---- Approval date audit ----------------------------------------------------
print()
print(SEP)
print("APPROVAL DATE AUDIT (File 1 -- approval_period field ONLY)")
print(SEP)
print("""
  Source: approval_period column in File 1 (month-year string from cabinet approval).
  PIB official_source_date in File 2 is the data-collection date, NOT used here.
  No exact day is inferred; all dates set to 1st of the stated month.
""")

print("  Raw approval_period values and their frequency:")
for v, cnt in f1["approval_period"].value_counts().sort_index().items():
    print(f"    {cnt:>2} project(s)   [{v}]")

# Parse dates with error capture
parse_errors = []
def safe_parse(s):
    try:
        return dateparse("1 " + str(s).strip())
    except ParserError as e:
        parse_errors.append((s, str(e)))
        return pd.NaT

f1["_approval_date"] = f1["approval_period"].apply(safe_parse)
f1["_approval_date"] = pd.to_datetime(f1["_approval_date"])

if parse_errors:
    print(f"\n  PARSE WARNINGS ({len(parse_errors)}):")
    for val, err in parse_errors:
        print(f"    [{val}] -> {err}")
else:
    print("\n  Parse warnings: none -- all approval_period values parsed cleanly.")

f1["_quarter"] = f1["_approval_date"].dt.to_period("Q")

print(f"\n  Date range:  {f1['_approval_date'].min().strftime('%b %Y')}  to  "
      f"{f1['_approval_date'].max().strftime('%b %Y')}")

all_q    = pd.period_range(f1["_quarter"].min(), f1["_quarter"].max(), freq="Q")
active_q = set(f1["_quarter"].astype(str))
empty_q  = [str(q) for q in all_q if str(q) not in active_q]

print(f"  Total calendar quarters in range:  {len(all_q)}")
print(f"  Quarters WITH >=1 approval:        {len(active_q)}")
print(f"  Quarters WITH ZERO approvals:      {len(empty_q)}")
print(f"  Empty quarters: {empty_q}")
print()
print("  NOTE: 'Zero approvals' = no ISM cabinet approval in that quarter.")
print("        This is an OBSERVED ABSENCE, not a data-collection gap.")
print("        No interpolation is applied.")

# Quarter detail
print("\n  Quarterly breakdown:")
q_sum = (
    f1.groupby("_quarter")
    .agg(
        projects       = ("company", "count"),
        inv_known      = ("investment_crore", lambda x: x.notna().sum()),
        inv_sum_crore  = ("investment_crore", "sum"),
    )
    .reset_index()
)
q_sum.columns = ["quarter", "projects_approved", "projects_with_inv", "investment_crore_sum"]
print(q_sum.to_string(index=False))

# ---- Investment completeness ------------------------------------------------
print()
print(SEP)
print("INVESTMENT COMPLETENESS")
print(SEP)
known   = f1["investment_crore"].notna()
missing = ~known

print(f"\n  Known investment:   {known.sum():>2} / {len(f1)} projects")
print(f"  Missing investment: {missing.sum():>2} / {len(f1)} projects")
print(f"  Known sum (Rs. crore): {f1.loc[known, 'investment_crore'].sum():>12,.0f}")
print()
print("  Projects with MISSING investment_crore:")
for _, r in f1[missing].iterrows():
    print(f"    [{r['company'][:60]}]  approval_period=[{r['approval_period']}]")

print()
print("  Unit: INR crore (committed capex per PIB press release)")
print("  Values are NOT realised disbursements, revenues, or profits.")

# ---- ML feasibility verdict -------------------------------------------------
print()
print(SEP)
print("ML / FORECASTING FEASIBILITY VERDICT")
print(SEP)

n_non_empty_q  = len(active_q)
n_known_inv_q  = int((q_sum["projects_with_inv"] == q_sum["projects_approved"]).sum())
gap_pct        = len(empty_q) / len(all_q) * 100

print(f"""
  Metric                                    Value
  ----------------------------------------- ------------------------------------------
  Total ISM project approvals               {len(f1)}
  Quarters with >=1 approval                {n_non_empty_q}  (out of {len(all_q)} in range)
  Quarters with ZERO approvals              {len(empty_q)}  ({gap_pct:.0f}% gap rate)
  Quarters with COMPLETE investment         {n_known_inv_q}
  Projects with MISSING investment          {missing.sum()}  (Crystal Matrix, Suchi Semicon)
  Observations for quarterly ML             {n_non_empty_q}  non-empty rows (max)

  VERDICT: INSUFFICIENT for ML/ARIMA quarterly time-series forecasting.
  -----------------------------------------------------------------------
  A minimum of ~20-30 consecutive quarterly observations is typically
  required for ARIMA or ML time-series methods to produce reliable
  forecasts.  Fabricating intermediate observations would violate data
  integrity.

  RECOMMENDED ALTERNATIVE:
    DGCI&S / DGFT monthly HS-8542 trade statistics
    (Harmonised System code for Electronic Integrated Circuits)
    Source : https://tradestat.commerce.gov.in  (official GoI portal)
    Metric : monthly import/export value & volume, from FY2010 onward
    -> Provides 150+ monthly observations -- suitable for ARIMA/ML.
""")
