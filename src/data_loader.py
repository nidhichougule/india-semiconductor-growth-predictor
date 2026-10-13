"""
data_loader.py
==============
Loads the raw verified government CSV files for the
India Semiconductor Growth Predictor project.

Raw files are treated as read-only.  This module performs NO
transformation—it only reads and validates that the expected
files are present and have the expected minimum shape.

Usage
-----
    from src.data_loader import load_raw_projects, load_raw_hubs

Author: Nidhi Chougule
Data sources: PIB / India Semiconductor Mission (ISM) — see data/raw/data_source_manifest.md
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _REPO_ROOT / "data" / "raw"

FILE1 = _RAW_DIR / "india_approved_semiconductor_projects_government.csv"
FILE2 = _RAW_DIR / "india_semiconductor_projects_and_hubs_2026_government_sources.csv"

# Expected column sets (used for validation)
_EXPECTED_COLS_F1 = {
    "project_id", "company", "project_location", "state",
    "facility_type", "technology_or_product", "investment_crore",
    "capacity", "approval_period", "status", "source_url", "hub_cluster",
}

_EXPECTED_COLS_F2 = {
    "id", "company", "city_or_location", "state",
    "facility_type", "technology_or_product", "investment_crore",
    "capacity", "classification", "status", "government_programme",
    "official_source_date", "official_source_url", "hub_relevance",
}


def load_raw_projects(path: Path = FILE1) -> pd.DataFrame:
    """
    Load File 1: ISM-approved semiconductor projects (12 rows).

    Parameters
    ----------
    path : Path
        Path to the CSV.  Defaults to the committed raw file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with dtypes as-read (investment_crore as float64,
        approval_period as object/string).

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at ``path``.
    ValueError
        If expected columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            "Run 'git lfs pull' or restore from data/raw/ in the repository."
        )

    df = pd.read_csv(path)

    missing_cols = _EXPECTED_COLS_F1 - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"File 1 is missing expected columns: {missing_cols}"
        )

    return df


def load_raw_hubs(path: Path = FILE2) -> pd.DataFrame:
    """
    Load File 2: ISM projects + hubs + SPECS-funded facilities (13 rows).

    NOTE: This file contains 12 ISM-approved projects (overlapping with
    File 1) PLUS 1 SPECS-funded facility (Sahasra Semiconductors).
    The `classification` column distinguishes them.

    Parameters
    ----------
    path : Path
        Path to the CSV.  Defaults to the committed raw file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame.

    Raises
    ------
    FileNotFoundError, ValueError
        Same as load_raw_projects.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}"
        )

    df = pd.read_csv(path)

    missing_cols = _EXPECTED_COLS_F2 - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"File 2 is missing expected columns: {missing_cols}"
        )

    return df


def describe_raw(df: pd.DataFrame, label: str = "") -> None:
    """Print a concise audit summary for a raw DataFrame."""
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Shape:          {df.shape}")
    print(f"  Columns:        {list(df.columns)}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"  Duplicates:     {df.duplicated().sum()}")
