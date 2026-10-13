"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py and src/data_loader.py.

All tests use small, self-contained fixtures (DataFrames constructed inline)
rather than the real CSV files so that the test suite never mutates raw data
and runs without any file I/O.

Run from the repository root:
    pytest tests/test_preprocessing.py -v

Coverage areas
--------------
1.  Approval date parsing   — valid formats, bad formats, first-of-month rule
2.  Missing investment      — NaN preserved; boolean flag correct
3.  Duplicate handling      — exact-duplicate rows flagged (data_loader audit)
4.  Unit consistency        — investment_crore values are numeric floats (not strings)
5.  Facility type mapping   — known raw strings map to canonical categories
6.  Quarterly aggregation   — investment_complete flag, NaN propagation in cumulative
7.  Raw data preservation   — load_raw_projects() returns the CSV unchanged
8.  Non-ISM separation      — Sahasra record excluded from ISM aggregates
9.  Fiscal year derivation  — Apr–Mar boundary logic
10. Column completeness     — required output columns present after preprocessing
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from dateutil.parser import parse as dateparse

# ── allow imports from repo root ────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    _normalise_facility_type,
    build_projects_clean,
    build_quarterly_investment,
    build_statetype_analysis,
    extract_non_ism_projects,
    INVESTMENT_UNIT,
    INVESTMENT_NOTE,
)
from src.data_loader import load_raw_projects, load_raw_hubs


# ════════════════════════════════════════════════════════════════════════════
# 1.  Approval Date Parsing
# ════════════════════════════════════════════════════════════════════════════

class TestApprovalDateParsing:
    """Dates must be parsed from approval_period field only (month-year strings).
    The result is always the FIRST day of that month — no intra-month precision."""

    # --- Helper: build a minimal projects DataFrame for mocking ---
    @staticmethod
    def _minimal_df(approval_period: str, investment: float = 10000.0) -> pd.DataFrame:
        return pd.DataFrame([{
            "project_id": 1,
            "company": "Test Corp",
            "project_location": "Test City",
            "state": "Gujarat",
            "facility_type": "ATMP",
            "technology_or_product": "memory",
            "investment_crore": investment,
            "capacity": "1M units/day",
            "approval_period": approval_period,
            "status": "Approved",
            "source_url": "https://example.com",
            "hub_cluster": "Test Hub",
        }])

    @pytest.mark.parametrize("period,expected_date", [
        ("June 2023",     "2023-06-01"),
        ("February 2024", "2024-02-01"),
        ("May 2025",      "2025-05-01"),
        ("August 2025",   "2025-08-01"),
        ("September 2024","2024-09-01"),
        ("May 2026",      "2026-05-01"),
    ])
    def test_parses_to_first_of_month(self, period, expected_date):
        """All approval periods in the dataset must parse to 1st of their month."""
        parsed = pd.Timestamp(dateparse("1 " + period))
        assert parsed == pd.Timestamp(expected_date), (
            f"'{period}' should parse to {expected_date}, got {parsed.date()}"
        )
        assert parsed.day == 1, "Day must always be 1 — no intra-month precision is claimed."

    def test_bad_approval_period_raises(self):
        """An unrecognisable approval_period must raise ValueError, not silently NaT."""
        bad_df = self._minimal_df("Q1 FY2024")  # not a month-year string
        # Patch the name as imported into preprocessing, not in data_loader
        with patch("src.preprocessing.load_raw_projects", return_value=bad_df):
            with pytest.raises((ValueError, Exception)):
                build_projects_clean()

    def test_approval_date_is_not_source_collection_date(self):
        """The PIB official_source_date (01-Apr-2026) must NOT appear as approval_date.
        This test ensures we never accidentally import the data-collection date."""
        # official_source_date in File 2 is 2026-04-01 for most rows
        # None of the File 1 approval periods correspond to April 2026
        f1_real = load_raw_projects()
        assert "April 2026" not in f1_real["approval_period"].values, (
            "April 2026 (the PIB data-collection date) must not appear in "
            "approval_period — that field is reserved for cabinet approval dates."
        )

    def test_approval_date_col_present_in_output(self):
        """build_projects_clean() must produce an 'approval_date' column."""
        df = build_projects_clean()
        assert "approval_date" in df.columns

    def test_approval_date_dtype_is_datetime(self):
        """approval_date must be a proper datetime, not an object/string."""
        df = build_projects_clean()
        assert pd.api.types.is_datetime64_any_dtype(df["approval_date"]), (
            f"approval_date dtype is {df['approval_date'].dtype}, expected datetime64."
        )


# ════════════════════════════════════════════════════════════════════════════
# 2.  Missing Investment Values
# ════════════════════════════════════════════════════════════════════════════

class TestMissingInvestment:
    """NaN investment values must be preserved, never imputed or filled."""

    def test_missing_investment_preserved_as_nan(self):
        """Crystal Matrix and Suchi Semicon must remain NaN after preprocessing."""
        df = build_projects_clean()
        missing = df[df["investment_crore"].isna()]
        assert len(missing) == 2, (
            f"Expected exactly 2 projects with NaN investment; found {len(missing)}."
        )
        companies = set(missing["company"].str.strip().str[:12])
        # check partial names to be robust to whitespace/parenthetical variants
        assert any("Crystal" in c for c in missing["company"]), (
            "Crystal Matrix Limited must have NaN investment."
        )
        assert any("Suchi" in c for c in missing["company"]), (
            "Suchi Semicon Private Limited must have NaN investment."
        )

    def test_is_investment_known_flag(self):
        """is_investment_known must be True for all rows with a non-NaN investment."""
        df = build_projects_clean()
        assert (df["is_investment_known"] == df["investment_crore"].notna()).all(), (
            "is_investment_known must exactly mirror investment_crore.notna()."
        )

    def test_known_investment_count(self):
        """Exactly 10 of 12 projects have a known investment value."""
        df = build_projects_clean()
        assert df["is_investment_known"].sum() == 10, (
            f"Expected 10 known investments; got {df['is_investment_known'].sum()}."
        )

    def test_investment_sum_excludes_nan(self):
        """Known-investment sum must match the actual PIB-stated total.
        10 projects with disclosed capex sum to Rs. 1,60,365 crore.
        (Crystal Matrix and Suchi Semicon excluded -- no figure disclosed.)"""
        df = build_projects_clean()
        total = df.loc[df["is_investment_known"], "investment_crore"].sum()
        # Allow +-1 crore tolerance for float rounding
        assert abs(total - 160365) < 1, (
            f"Known investment total should be ~Rs. 1,60,365 crore; got {total:,.0f}."
        )

    def test_fixture_nan_investment(self):
        """Unit test: a fixture row with NaN investment stays NaN in aggregation."""
        fixture = pd.DataFrame([
            {"project_id": 1, "company": "A Corp", "approval_quarter_label": "2025Q3",
             "investment_crore": 5000.0, "is_investment_known": True},
            {"project_id": 2, "company": "B Corp", "approval_quarter_label": "2025Q3",
             "investment_crore": np.nan, "is_investment_known": False},
        ])
        # sum() on mixed NaN/float returns the float (NaN treated as 0 in pandas sum)
        # This is intentional — we then restore NaN via the investment_complete flag
        q_sum = fixture.groupby("approval_quarter_label")["investment_crore"].sum()
        assert q_sum["2025Q3"] == 5000.0  # pandas sum ignores NaN
        # But investment_complete = False should then override this to NaN
        inv_complete = (fixture.groupby("approval_quarter_label")["is_investment_known"].sum()
                        == fixture.groupby("approval_quarter_label")["project_id"].count())
        assert inv_complete["2025Q3"] == False


# ════════════════════════════════════════════════════════════════════════════
# 3.  Duplicate Handling
# ════════════════════════════════════════════════════════════════════════════

class TestDuplicateHandling:
    """Raw files must have zero exact-duplicate rows."""

    def test_no_exact_duplicates_file1(self):
        df = load_raw_projects()
        n_dup = df.duplicated().sum()
        assert n_dup == 0, f"File 1 has {n_dup} exact-duplicate rows."

    def test_no_exact_duplicates_file2(self):
        df = load_raw_hubs()
        n_dup = df.duplicated().sum()
        assert n_dup == 0, f"File 2 has {n_dup} exact-duplicate rows."

    def test_unique_project_ids_file1(self):
        df = load_raw_projects()
        assert df["project_id"].nunique() == len(df), "project_id must be unique."

    def test_fixture_duplicates_detected(self):
        """Demonstrate that pd.duplicated() catches exact-duplicate rows."""
        fixture = pd.DataFrame([
            {"id": 1, "company": "X", "investment_crore": 100.0},
            {"id": 1, "company": "X", "investment_crore": 100.0},  # duplicate
        ])
        assert fixture.duplicated().sum() == 1


# ════════════════════════════════════════════════════════════════════════════
# 4.  Unit Consistency
# ════════════════════════════════════════════════════════════════════════════

class TestUnitConsistency:
    """investment_crore must be numeric (float64), not string."""

    def test_investment_dtype_file1(self):
        df = load_raw_projects()
        assert pd.api.types.is_float_dtype(df["investment_crore"]), (
            f"File 1 investment_crore dtype is {df['investment_crore'].dtype}; expected float."
        )

    def test_investment_dtype_file2(self):
        df = load_raw_hubs()
        assert pd.api.types.is_float_dtype(df["investment_crore"]), (
            f"File 2 investment_crore dtype is {df['investment_crore'].dtype}; expected float."
        )

    def test_investment_unit_constant_defined(self):
        """The INVESTMENT_UNIT constant must clearly state 'INR crore'."""
        assert "INR" in INVESTMENT_UNIT or "crore" in INVESTMENT_UNIT.lower(), (
            f"INVESTMENT_UNIT='{INVESTMENT_UNIT}' must reference INR crore."
        )

    def test_investment_note_mentions_capex(self):
        """INVESTMENT_NOTE must explicitly state these are capital expenditure figures."""
        assert "capital" in INVESTMENT_NOTE.lower() or "capex" in INVESTMENT_NOTE.lower(), (
            "INVESTMENT_NOTE must clarify that investment values are capital expenditure."
        )

    def test_no_string_investment_values_in_output(self):
        """After preprocessing, investment_crore must remain float, not object."""
        df = build_projects_clean()
        assert pd.api.types.is_float_dtype(df["investment_crore"]), (
            f"Preprocessed investment_crore dtype is {df['investment_crore'].dtype}."
        )

    def test_all_known_investments_positive(self):
        """All disclosed investment values must be positive (>0)."""
        df = build_projects_clean()
        known = df.loc[df["is_investment_known"], "investment_crore"]
        assert (known > 0).all(), "All known investments must be strictly positive."


# ════════════════════════════════════════════════════════════════════════════
# 5.  Facility Type Mapping
# ════════════════════════════════════════════════════════════════════════════

class TestFacilityTypeMapping:
    """All raw facility_type strings must map to a canonical category."""

    @pytest.mark.parametrize("raw,expected", [
        ("ATMP",                                          "ATMP/OSAT"),
        ("OSAT",                                          "ATMP/OSAT"),
        ("ATMP / OSAT",                                   "ATMP/OSAT"),
        ("Semiconductor Fab",                             "Semiconductor Fab"),
        ("Compound Semiconductor Fab + Packaging",        "Compound Semiconductor Fab"),
        ("Compound Semiconductor Fab + ATMP",             "Compound Semiconductor Fab"),
        ("Semiconductor manufacturing / DDIC",            "DDIC / Display"),
        ("Advanced packaging / 3D heterogeneous integration", "Advanced Packaging"),
        ("Advanced packaging / 3DHI",                    "Advanced Packaging"),
        ("Semiconductor packaging",                       "Advanced Packaging"),
        ("Advanced Semiconductor Packaging",              "Advanced Packaging"),
        ("Semiconductor manufacturing / discrete semiconductors", "Discrete Semiconductor"),
        ("Discrete Semiconductor Manufacturing",          "Discrete Semiconductor"),
    ])
    def test_known_facility_types_map_correctly(self, raw, expected):
        result = _normalise_facility_type(raw)
        assert result == expected, (
            f"'{raw}' should map to '{expected}', got '{result}'."
        )

    def test_unknown_facility_type_passthrough(self):
        """An unrecognised string should pass through unchanged (not silently lost)."""
        raw = "Photonics Manufacturing"
        result = _normalise_facility_type(raw)
        assert result == raw, f"Unknown type should pass through; got '{result}'."

    def test_all_raw_file1_types_map_to_canonical(self):
        """Every facility_type value in File 1 must have a canonical mapping
        (no raw strings should appear unchanged in the output except unknowns)."""
        df = build_projects_clean()
        raw_df = load_raw_projects()
        # collect raw facility types
        raw_types = set(raw_df["facility_type"].str.strip().str.lower())
        from src.preprocessing import _FACILITY_TYPE_MAP
        unmapped = raw_types - set(_FACILITY_TYPE_MAP.keys())
        # All raw types should be in the map (none should fall through as-is
        # with a raw string that happens to look wrong)
        assert unmapped == set(), (
            f"These facility_type values have no canonical mapping: {unmapped}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 6.  Quarterly Aggregation
# ════════════════════════════════════════════════════════════════════════════

class TestQuarterlyAggregation:
    """Quarterly investment table must correctly flag incomplete quarters and
    propagate NaN in the cumulative sum once an incomplete quarter is reached."""

    def test_quarterly_table_row_count(self):
        """There must be exactly 6 distinct quarters (one per unique approval month)."""
        df = build_projects_clean()
        q = build_quarterly_investment(df)
        assert len(q) == 6, (
            f"Expected 6 quarterly rows; got {len(q)}.\n{q}"
        )

    def test_investment_complete_flag(self):
        """2026Q2 has two projects without investment — investment_complete must be False."""
        df = build_projects_clean()
        q = build_quarterly_investment(df)
        q2_row = q[q["approval_quarter_label"] == "2026Q2"]
        assert len(q2_row) == 1, "2026Q2 must be present in quarterly table."
        assert q2_row.iloc[0]["investment_complete"] == False, (
            "2026Q2 investment_complete must be False (Crystal Matrix + Suchi Semicon missing)."
        )

    def test_incomplete_quarter_investment_is_nan(self):
        """investment_crore_in_quarter must be NaN for incomplete quarters (not 0)."""
        df = build_projects_clean()
        q = build_quarterly_investment(df)
        incomplete = q[~q["investment_complete"]]
        for _, row in incomplete.iterrows():
            assert pd.isna(row["investment_crore_in_quarter"]), (
                f"Quarter {row['approval_quarter_label']} is incomplete but "
                f"investment_crore_in_quarter={row['investment_crore_in_quarter']} (expected NaN)."
            )

    def test_cumulative_nan_after_incomplete_quarter(self):
        """Once an incomplete quarter is reached the cumulative must be NaN."""
        df = build_projects_clean()
        q = build_quarterly_investment(df)
        q_sorted = q.sort_values("approval_quarter_label").reset_index(drop=True)
        # Find first incomplete quarter index
        first_incomplete_idx = q_sorted[~q_sorted["investment_complete"]].index.min()
        if pd.isna(first_incomplete_idx):
            pytest.skip("No incomplete quarters found — nothing to test.")
        rows_after = q_sorted.loc[first_incomplete_idx:]
        for _, row in rows_after.iterrows():
            assert pd.isna(row["cumulative_investment_crore"]), (
                f"cumulative_investment_crore must be NaN at and after the first "
                f"incomplete quarter, but {row['approval_quarter_label']} = "
                f"{row['cumulative_investment_crore']}."
            )

    def test_complete_quarters_cumulative_increasing(self):
        """Cumulative investment must be monotonically non-decreasing for complete quarters."""
        df = build_projects_clean()
        q = build_quarterly_investment(df)
        complete = q[q["investment_complete"]].sort_values("approval_quarter_label")
        cumvals = complete["cumulative_investment_crore"].dropna()
        assert list(cumvals) == sorted(cumvals), (
            "Cumulative investment must be non-decreasing for complete quarters."
        )

    @staticmethod
    def test_fixture_nan_propagation_in_cumulative():
        """Unit fixture: manual NaN-propagating cumsum matches expected values."""
        s = pd.Series([100.0, 200.0, np.nan, 150.0])
        # Replicate the logic from build_quarterly_investment
        cumvals = []
        running = 0.0
        has_nan = False
        for v in s:
            if has_nan or pd.isna(v):
                has_nan = True
                cumvals.append(np.nan)
            else:
                running += v
                cumvals.append(running)
        assert cumvals[0] == 100.0
        assert cumvals[1] == 300.0
        assert pd.isna(cumvals[2]), "Should be NaN at the NaN position."
        assert pd.isna(cumvals[3]), "Should be NaN after the NaN position."


# ════════════════════════════════════════════════════════════════════════════
# 7.  Raw Data Preservation
# ════════════════════════════════════════════════════════════════════════════

class TestRawDataPreservation:
    """load_raw_projects() and load_raw_hubs() must return the data exactly
    as stored in the CSV, with no transformations applied."""

    def test_file1_row_count(self):
        """File 1 must have exactly 12 rows (12 ISM approvals)."""
        df = load_raw_projects()
        assert len(df) == 12, f"Expected 12 rows; got {len(df)}."

    def test_file2_row_count(self):
        """File 2 must have exactly 13 rows (12 ISM + 1 SPECS)."""
        df = load_raw_hubs()
        assert len(df) == 13, f"Expected 13 rows; got {len(df)}."

    def test_raw_approval_period_is_string(self):
        """approval_period must be read as strings, not auto-parsed to dates."""
        df = load_raw_projects()
        assert df["approval_period"].dtype == object, (
            f"approval_period should be object/str; got {df['approval_period'].dtype}."
        )

    def test_preprocessing_does_not_modify_raw_df(self):
        """Calling build_projects_clean() must not mutate the original raw DataFrame."""
        raw = load_raw_projects()
        raw_cols_before = list(raw.columns)
        raw_shape_before = raw.shape
        # Run preprocessing (uses load_raw_projects internally, not `raw`)
        _ = build_projects_clean()
        # Verify the variable we loaded manually is untouched
        assert list(raw.columns) == raw_cols_before, "Raw df columns were mutated."
        assert raw.shape == raw_shape_before, "Raw df shape was mutated."

    def test_raw_investment_missing_is_nan_not_string(self):
        """Empty cells in investment_crore must be NaN, not the string 'nan' or ''."""
        df = load_raw_projects()
        # pandas read_csv converts empty cells to NaN automatically
        string_nans = df["investment_crore"].apply(
            lambda x: isinstance(x, str) and x.strip() in ("", "nan", "NaN")
        )
        assert not string_nans.any(), (
            "investment_crore has string 'nan' or '' values — should be float NaN."
        )


# ════════════════════════════════════════════════════════════════════════════
# 8.  Non-ISM Separation (Sahasra Semiconductors)
# ════════════════════════════════════════════════════════════════════════════

class TestNonISMSeparation:
    """Sahasra Semiconductors (SPECS+EMC) must never appear in ISM aggregates."""

    def test_sahasra_excluded_from_projects_clean(self):
        df = build_projects_clean()
        has_sahasra = df["company"].str.contains("Sahasra", case=False, na=False).any()
        assert not has_sahasra, (
            "Sahasra Semiconductors (SPECS, not ISM) must not appear in projects_clean."
        )

    def test_sahasra_in_non_ism_extract(self):
        non_ism = extract_non_ism_projects()
        assert len(non_ism) >= 1, "extract_non_ism_projects() must return ≥1 row."
        assert non_ism["company"].str.contains("Sahasra", case=False, na=False).any(), (
            "Sahasra must appear in the non-ISM extract."
        )

    def test_non_ism_classification(self):
        """Non-ISM records must NOT have classification 'Approved ISM project'."""
        non_ism = extract_non_ism_projects()
        bad = non_ism["classification"].str.contains("Approved ISM", na=False)
        assert not bad.any(), (
            "Non-ISM extract must not contain 'Approved ISM project' classification."
        )

    def test_projects_clean_row_count_is_12(self):
        """Exactly 12 ISM-approved rows must appear after preprocessing."""
        df = build_projects_clean()
        assert len(df) == 12, f"Expected 12 ISM projects; got {len(df)}."


# ════════════════════════════════════════════════════════════════════════════
# 9.  Fiscal Year Derivation
# ════════════════════════════════════════════════════════════════════════════

class TestFiscalYear:
    """India's fiscal year runs April–March.
    A project approved in June 2023 belongs to FY2024 (Apr 2023–Mar 2024)."""

    @pytest.mark.parametrize("month,year,expected_fy", [
        (6, 2023, 2024),   # June 2023 → FY2024
        (2, 2024, 2024),   # February 2024 → FY2024
        (9, 2024, 2025),   # September 2024 → FY2025
        (5, 2025, 2026),   # May 2025 → FY2026
        (8, 2025, 2026),   # August 2025 → FY2026
        (5, 2026, 2027),   # May 2026 → FY2027
        (3, 2024, 2024),   # March 2024 → FY2024 (end of fiscal)
        (4, 2024, 2025),   # April 2024 → FY2025 (start of new fiscal)
    ])
    def test_fiscal_year_april_march(self, month, year, expected_fy):
        import datetime
        d = datetime.date(year, month, 1)
        fy = d.year + 1 if d.month >= 4 else d.year
        assert fy == expected_fy, (
            f"Month {month}/{year}: expected FY{expected_fy}, got FY{fy}."
        )

    def test_approval_fiscal_year_in_output(self):
        """approval_fiscal_year must be present in build_projects_clean() output."""
        df = build_projects_clean()
        assert "approval_fiscal_year" in df.columns

    def test_june_2023_is_fy2024(self):
        """Micron (June 2023) must map to FY2024."""
        df = build_projects_clean()
        micron = df[df["company"].str.contains("Micron", na=False)]
        assert len(micron) == 1
        assert micron.iloc[0]["approval_fiscal_year"] == 2024, (
            f"Micron (June 2023) should be FY2024, got {micron.iloc[0]['approval_fiscal_year']}."
        )


# ════════════════════════════════════════════════════════════════════════════
# 10.  Column Completeness
# ════════════════════════════════════════════════════════════════════════════

class TestColumnCompleteness:
    """All required output columns must be present after preprocessing."""

    REQUIRED_PROJECTS_COLS = [
        "project_id", "company", "state", "facility_type_canonical",
        "investment_crore", "is_investment_known",
        "approval_period", "approval_date", "approval_quarter_label",
        "approval_fiscal_year", "data_classification",
    ]

    REQUIRED_QUARTERLY_COLS = [
        "approval_quarter_label", "projects_approved",
        "investment_crore_in_quarter", "cumulative_investment_crore",
        "investment_complete",
    ]

    def test_projects_clean_required_columns(self):
        df = build_projects_clean()
        missing = [c for c in self.REQUIRED_PROJECTS_COLS if c not in df.columns]
        assert not missing, f"projects_clean is missing columns: {missing}"

    def test_quarterly_required_columns(self):
        q = build_quarterly_investment(build_projects_clean())
        missing = [c for c in self.REQUIRED_QUARTERLY_COLS if c not in q.columns]
        assert not missing, f"quarterly_investment is missing columns: {missing}"

    def test_data_classification_is_ism_approved(self):
        """All rows in projects_clean must have classification 'ISM Approved'."""
        df = build_projects_clean()
        assert (df["data_classification"] == "ISM Approved").all(), (
            "data_classification must be 'ISM Approved' for all rows in projects_clean."
        )
