# ML Feasibility Assessment — Quarterly Investment Series

## India Semiconductor Growth Predictor
**Author:** Nidhi Chougule  
**Date:** 13-Oct-2026  
**Phase:** 2 — Data Audit & Preprocessing

---

## Summary

> **The ISM approval quarterly series is insufficient for ML or ARIMA time-series forecasting.**
> No model will be trained on this series. No synthetic observations will be manufactured.

---

## Why the Series Is Too Sparse

The ISM approval data (File 1) produces the following quarterly series:

| Quarter | Projects Approved | Investment ₹ crore | Complete? |
|---------|-------------------|--------------------|-----------|
| 2023Q2  | 1                 | 22,516             | ✅ Yes     |
| 2024Q1  | 3                 | 1,26,230           | ✅ Yes     |
| 2024Q3  | 1                 | 3,307              | ✅ Yes     |
| 2025Q2  | 1                 | 3,706              | ✅ Yes     |
| 2025Q3  | 4                 | 4,606              | ✅ Yes     |
| 2026Q2  | 2                 | NaN                | ❌ No      |

**Key metrics:**

| Metric | Value |
|--------|-------|
| Non-empty quarters | 6 |
| Total quarters in range (Jun 2023–May 2026) | 13 |
| Gap quarters (zero approvals) | 7 (54% gap rate) |
| Complete-investment quarters | 5 |

### Minimum sample size for time-series methods

| Method | Typical minimum observations |
|--------|------------------------------|
| ARIMA / SARIMA | 30–50 stationary observations |
| Exponential Smoothing (Holt-Winters) | 20–30 |
| LSTM / GRU (deep learning) | 100+ |
| Simple linear regression | ~20 (marginal) |

**6 non-empty quarters (5 with complete investment) is far below any threshold.**

### Structural issues beyond sample size

1. **Irregular spacing**: Approvals are event-driven (cabinet decisions), not periodic.  
   Gap quarters represent genuine absence of approval events, not measurement error.

2. **Mixed batch sizes**: Quarter 2024Q1 has 3 approvals (worth ₹1.26 lakh crore);  
   2023Q2 has 1 approval (₹22,516 crore). The variance is dominated by a single large project.

3. **No leading indicators in this dataset**: The series contains only past approval events,  
   not pipeline data, application counts, or programme budget utilisation rates.

4. **Incomplete tail**: The final observed quarter (2026Q2) has missing investment values,  
   making the cumulative total unreliable.

---

## What Will Be Built Instead

### Phase 3 approach (pending separate data collection)

The quarterly ISM approval series will be used as:
- **Scenario parameters** (e.g. total committed capex = ₹1,60,365 crore)
- **Reference points** for capacity and state distribution analysis
- **Narrative context** in the dashboard

For actual forecasting, the recommended alternative is **DGCI&S/DGFT monthly trade data**:

| Property | Value |
|----------|-------|
| **Source** | https://tradestat.commerce.gov.in (GoI official trade portal) |
| **HS Code** | 8542 — Electronic Integrated Circuits |
| **Metrics** | Monthly import value (USD/INR), export value, volume |
| **Availability** | Monthly from FY2010 onward |
| **Observations** | ~180 monthly data points (April 2010–March 2025) |
| **Suitability** | ✅ Sufficient for ARIMA, exponential smoothing, ML regression |

This data would allow forecasting of **India's semiconductor import/export trajectory**,  
which is a more meaningful and well-defined forecasting target than ISM approval timing.

---

## Decision: No Model in Phase 2

- ✅ Project-level analysis complete (`projects_clean.csv`)
- ✅ State × facility aggregation complete (`statetype_analysis.csv`)
- ✅ Quarterly series documented with honest NaN and completeness flags
- ❌ No ARIMA / ML model trained — data is insufficient
- ❌ No synthetic/interpolated observations added

**Next step (Phase 3):** Collect DGCI&S HS-8542 trade data and document its provenance  
before beginning any forecasting work.

---

*This document must be updated if new ISM approval data becomes available or if  
the forecasting target is redefined.*
