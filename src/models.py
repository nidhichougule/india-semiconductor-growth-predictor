"""
src/models.py
=============
ML modelling for India Semiconductor Growth Predictor.

Given the small ISM dataset (n=12, 10 with known investment), this module:
  1. Trains a Random Forest regressor to predict investment_crore from
     facility_type + state features (illustrative; LOO-CV used for honest eval).
  2. Provides a parametric market-growth scenario model using published CAGR
     inputs — clearly labelled as scenario analysis, NOT ML prediction.
  3. Saves model artefacts and metrics to data/processed/.

Limitations are logged and surfaced in the dashboard.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")

from src.preprocessing import build_projects_clean

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame):
    """
    Encode state and facility_type_canonical as integers.
    Returns X (array), y (array), encoders dict, and feature names list.
    Only rows with known investment are used.
    """
    known = df[df["is_investment_known"]].copy().reset_index(drop=True)
    encoders = {}
    for col in ["state", "facility_type_canonical"]:
        le = LabelEncoder()
        known[col + "_enc"] = le.fit_transform(known[col])
        encoders[col] = le

    features = ["state_enc", "facility_type_canonical_enc", "approval_fiscal_year"]
    X = known[features].values.astype(float)
    y = known["investment_crore"].values
    return X, y, encoders, features, known


# ---------------------------------------------------------------------------
# Model 1: Investment predictor (LOO-CV on n=10)
# ---------------------------------------------------------------------------

def train_investment_model(df: pd.DataFrame = None):
    """
    Train a Random Forest to predict investment_crore.
    Uses Leave-One-Out CV (honest eval for tiny datasets).
    Returns metrics dict and fitted model.
    """
    if df is None:
        df = build_projects_clean()
    X, y, encoders, features, known = build_features(df)
    n = len(y)

    # Try three models, pick the best by LOO-MAE
    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(alpha=1.0),
    }

    loo = LeaveOneOut()
    results = {}
    for name, model in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo_preds = []
            for train_idx, test_idx in loo.split(X):
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])[0]
                loo_preds.append(pred)
        mae = mean_absolute_error(y, loo_preds)
        # R² on LOO predictions
        ss_res = sum((y[i] - loo_preds[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - np.mean(y)) ** 2 for i in range(n))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results[name] = {"loo_mae_crore": round(mae, 0), "loo_r2": round(r2, 4)}

    # Best model by MAE
    best_name = min(results, key=lambda k: results[k]["loo_mae_crore"])
    best_model = candidates[best_name]
    best_model.fit(X, y)  # refit on all data

    # Feature importances (RF/GB only)
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        for f, imp in zip(features, best_model.feature_importances_):
            feature_importances[f] = round(float(imp), 4)

    metrics = {
        "n_training": n,
        "n_total_projects": 12,
        "n_missing_investment": 2,
        "best_model": best_name,
        "model_results": results,
        "feature_importances": feature_importances,
        "warning": (
            f"SMALL SAMPLE WARNING: n={n} is far below what is needed for reliable ML. "
            "LOO-CV metrics are optimistic estimates. This model is ILLUSTRATIVE only. "
            "Do not use for investment decisions."
        ),
        "features_used": features,
        "target": "investment_crore",
        "unit": "INR crore",
    }

    # Save model and encoders
    joblib.dump(best_model, MODELS_DIR / "investment_model.pkl")
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    with open(PROCESSED / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return best_model, encoders, metrics


def predict_investment(state: str, facility_type: str, fiscal_year: int,
                       model=None, encoders=None) -> float:
    """Predict investment for a given state/facility/FY."""
    if model is None:
        model = joblib.load(MODELS_DIR / "investment_model.pkl")
    if encoders is None:
        encoders = joblib.load(MODELS_DIR / "encoders.pkl")

    state_enc = encoders["state"].transform([state])[0]
    fac_enc   = encoders["facility_type_canonical"].transform([facility_type])[0]
    X = np.array([[state_enc, fac_enc, fiscal_year]])
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# Model 2: Market growth scenario (parametric, published inputs)
# ---------------------------------------------------------------------------

SCENARIO_PARAMS = {
    "base_year": 2023,
    "base_market_usd_bn": 44.0,   # IESA FY2023 estimate
    "cagr_niti_aayog": 0.19,      # 19% CAGR per NITI Aayog ISM Roadmap
    "target_2030_iesa": 103.4,    # USD billion, IESA/IBEF consensus
    "inr_per_usd": 84.0,          # approximate FY2025 rate
    "ism_committed_crore": 160365,# Rs. crore committed capex across 10 ISM projects
    "source": "NITI Aayog ISM Roadmap, IESA, IBEF (2024-25)",
}


def build_market_scenarios() -> pd.DataFrame:
    """
    Build three parametric market scenarios from 2023 to 2030:
      - Base: 19% CAGR (NITI Aayog)
      - Optimistic: 22% CAGR
      - Conservative: 15% CAGR
    Returns DataFrame with year and USD-billion projections.
    Note: These are scenario projections, NOT trained ML predictions.
    """
    years = list(range(2023, 2031))
    base = SCENARIO_PARAMS["base_market_usd_bn"]
    df = pd.DataFrame({"year": years})
    for label, cagr in [("base_19pct", 0.19), ("optimistic_22pct", 0.22), ("conservative_15pct", 0.15)]:
        df[label] = [base * (1 + cagr) ** (y - 2023) for y in years]

    out = PROCESSED / "market_scenarios.csv"
    df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_all():
    print("Training investment model...")
    df = build_projects_clean()
    model, encoders, metrics = train_investment_model(df)
    print(f"  Best model: {metrics['best_model']}")
    print(f"  LOO-MAE: Rs. {metrics['model_results'][metrics['best_model']]['loo_mae_crore']:,.0f} crore")
    print(f"  LOO-R2:  {metrics['model_results'][metrics['best_model']]['loo_r2']}")
    print(f"  WARNING: {metrics['warning'][:80]}...")

    print("Building market scenarios...")
    scen = build_market_scenarios()
    print(f"  2030 base projection: ${scen['base_19pct'].iloc[-1]:.1f}B")

    print("Done. Artefacts in data/processed/ and src/models/")


if __name__ == "__main__":
    run_all()
