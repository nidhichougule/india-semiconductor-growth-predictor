# India Semiconductor Growth Predictor 🔬

An end-to-end data science project analysing India's semiconductor ecosystem under the **Semicon India Programme (India Semiconductor Mission — ISM)**. Built as part of an MSc Data Science college project.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)](https://streamlit.io)
[![Data](https://img.shields.io/badge/Data-PIB%20%2F%20ISM-green)](https://pib.gov.in)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/nidhichougule/india-semiconductor-growth-predictor.git
cd india-semiconductor-growth-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run preprocessing (saves CSVs to data/processed/)
python -m src.preprocessing

# 4. Train ML model (saves model + metrics to src/models/ and data/processed/)
python -m src.models

# 5. Generate EDA charts (saved to docs/charts/)
python -m src.eda

# 6. Launch the Streamlit dashboard
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
india-semiconductor-growth-predictor/
│
├── app.py                          # Streamlit dashboard (7 interactive pages with 5 colorful themes)
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── raw/
│   │   ├── india_approved_semiconductor_projects_government.csv       # File 1 (12 ISM projects)
│   │   ├── india_semiconductor_projects_and_hubs_2026_government_sources.csv  # File 2 (13 records)
│   │   └── data_source_manifest.md                                    # Provenance of raw files
│   ├── processed/
│   │   ├── projects_clean.csv          # Cleaned ISM project data (12 rows)
│   │   ├── statetype_analysis.csv      # State × facility-type aggregation
│   │   ├── quarterly_investment.csv    # Quarterly series (6 rows, NaN flagged)
│   │   ├── non_ism_projects.csv        # Sahasra Semiconductors (SPECS, excluded from ISM agg)
│   │   ├── market_scenarios.csv        # Parametric market growth projections
│   │   ├── model_metrics.json          # ML model LOO-CV metrics
│   │   └── PROVENANCE.md               # Processed data provenance manifest
│   └── interim/                        # Intermediate files (ignored)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Load raw CSVs (read-only; validates expected columns)
│   ├── preprocessing.py    # Clean, normalise, produce processed datasets
│   ├── eda.py              # EDA charts (saved as PNGs to docs/charts/)
│   ├── models.py           # ML investment predictor + market scenario builder
│   ├── features/           # (future) feature engineering
│   ├── models/             # Saved model artefacts (investment_model.pkl, encoders.pkl)
│   └── visualization/      # (future) additional viz helpers
│
├── tests/
│   ├── audit_data.py           # Full data audit (run: python tests/audit_data.py)
│   └── test_preprocessing.py  # 68 unit tests (pytest)
│
├── docs/
│   ├── charts/                 # EDA charts (PNG)
│   └── ml_feasibility_note.md # Why quarterly time-series ML is not feasible
│
└── notebooks/                  # (future) Jupyter exploration notebooks
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPIs, investment by state/facility, quarterly timeline, status distribution |
| **Market Scenarios** | Parametric CAGR projections (adjustable sliders) |
| **ML Model** | Investment predictor metrics (LOO-CV), feature importances, prediction tool |
| **Raw Data** | Browse all processed tables; download CSV |
| **About** | Data sources, methodology, limitations |

---

## 📋 Data Sources

| File | Source | Rows | Notes |
|------|--------|------|-------|
| `india_approved_semiconductor_projects_government.csv` | PIB / ISM (cabinet approvals) | 12 | All ISM-approved projects |
| `india_semiconductor_projects_and_hubs_2026_government_sources.csv` | PIB (multiple PRIDs) | 13 | 12 ISM + Sahasra (SPECS) |

All investment figures = **committed / approved capital expenditure** per PIB press releases.  
NOT realised disbursements, revenue, or profit.

Market projection inputs:
- **19% CAGR**: NITI Aayog ISM Roadmap (2022)
- **$103.4B 2030 target**: IESA / IBEF Sector Report
- **Rs. 2,03,500 crore programme outlay**: MeitY / ISM

---

## 🤖 ML Approach & Honest Limitations

### Investment Predictor
- **Target**: `investment_crore` (approved capex)
- **Features**: state, facility type, fiscal year
- **n = 10** (2 projects have no disclosed investment)
- **Evaluation**: Leave-One-Out CV (honest for tiny datasets)
- **Best model**: Random Forest / Gradient Boosting (selected by LOO-MAE)
- ⚠️ **This model is illustrative.** Do not use for investment decisions.

### Quarterly Time-Series — Not Feasible
Only 6 non-empty quarters over 13 quarters (54% gap rate). ARIMA requires ~30+ observations.
No synthetic data was manufactured. See `docs/ml_feasibility_note.md`.

---

## 🧪 Running Tests

```bash
pytest tests/test_preprocessing.py -v
# 68 tests covering: date parsing, missing investments, duplicates,
# unit consistency, facility mapping, quarterly aggregation, raw data preservation
```

---

## 📦 Requirements

See `requirements.txt`. Key packages: `pandas`, `scikit-learn`, `streamlit`, `plotly`, `matplotlib`.

---

## 📜 Licence

Data: [Government of India Open Data Licence (GODL) v1.0](https://data.gov.in/government-open-data-license-india)  
Code: MIT Licence

---

*Author: Nidhi Chougule 
