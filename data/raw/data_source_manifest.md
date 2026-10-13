# Data Source Manifest
## India Semiconductor Growth Predictor

This file documents the provenance, access date, and licence for every raw data file
in this directory. **All entries have been manually verified against their source URLs.**
No fabricated or interpolated data is present in this directory.

---

## File 1: `india_approved_semiconductor_projects_government.csv`

| Field | Value |
|-------|-------|
| **Filename** | `india_approved_semiconductor_projects_government.csv` |
| **Description** | All 12 ISM-approved semiconductor projects with investment (₹ crore), capacity, state, facility type, approval period, and current status |
| **Rows** | 12 project records |
| **Primary Source** | Press Information Bureau (PIB), Government of India |
| **Source URL (batch)** | https://www.pib.gov.in/PressReleasePage.aspx?PRID=2247814 |
| **Source URL (May 2026)** | https://www.pib.gov.in/PressReleasePage.aspx?PRID=2258116 |
| **Programme** | India Semiconductor Mission (ISM) / Semicon India Programme |
| **Data Collector** | Nidhi Chougule (manual extraction from PIB releases) |
| **Date Collected** | 05-Oct-2026 |
| **Date of Latest Source** | May 2026 (PRID=2258116) |
| **Licence** | Government of India Open Data Licence (GODL) v1.0 — free to use with attribution |
| **Verification Status** | ✅ Every row's source_url column links to its specific PIB release |

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `project_id` | int | Sequential identifier |
| `company` | string | Company / JV name as per cabinet approval |
| `project_location` | string | City and district |
| `state` | string | Indian state |
| `facility_type` | string | Fab / ATMP / OSAT / Compound Semiconductor / Advanced Packaging |
| `technology_or_product` | string | Specific technology or product manufactured |
| `investment_crore` | float | Approved capital investment in Indian Rupees (crore) |
| `capacity` | string | Stated production capacity |
| `approval_period` | string | Month and year of cabinet/ISM approval |
| `status` | string | Current project status (as of source date) |
| `source_url` | string | PIB press release URL for this specific project |
| `hub_cluster` | string | Geographic semiconductor hub/cluster name |

---

## File 2: `india_semiconductor_projects_and_hubs_2026_government_sources.csv`

| Field | Value |
|-------|-------|
| **Filename** | `india_semiconductor_projects_and_hubs_2026_government_sources.csv` |
| **Description** | Enriched version of ISM project data with additional hub relevance classification, official programme name, and precise source dates |
| **Rows** | 13 records (12 ISM + 1 SPECS-funded facility: Sahasra Semiconductors) |
| **Primary Source** | PIB; also PIB PRID=2253549 (3D Glass Solutions) and PRID=2261509 (Sahasra) |
| **Source URLs** | https://pib.gov.in/PressReleasePage.aspx?PRID=2247814, 2253549, 2258119, 2261509 |
| **Programme** | India Semiconductor Mission (ISM) / SPECS / EMC |
| **Data Collector** | Nidhi Chougule (manual extraction from PIB releases) |
| **Date Collected** | 05-Oct-2026 |
| **Date of Latest Source** | 15-May-2026 (Sahasra, PRID=2261509) |
| **Licence** | Government of India Open Data Licence (GODL) v1.0 — free to use with attribution |
| **Verification Status** | ✅ Every row verified against named PIB release |

### Additional Columns (beyond File 1)

| Column | Type | Description |
|--------|------|-------------|
| `classification` | string | Whether it is an ISM approval or SPECS-funded |
| `government_programme` | string | Semicon India Programme / ISM / SPECS / EMC |
| `official_source_date` | date | Date of the PIB release used |
| `official_source_url` | string | Exact PIB URL |
| `hub_relevance` | string | Whether site is a cluster hub or project-only location |

---

## Market Forecast Context (not a file — used in modelling phase)

The following figures from credible sources will be used as **parametric inputs** to
the market demand model. They will be clearly labelled as model assumptions, not ground truth.

| Metric | Value | Source | URL |
|--------|-------|--------|-----|
| India semiconductor demand CAGR | 19% | NITI Aayog / ISM Roadmap | https://niti.gov.in |
| Market size 2030 (IESA/IBEF) | USD 103.4 Billion | IESA / IBEF | https://ibef.org |
| Market size 2030 (consensus) | USD 90–108 Billion | NITI Aayog / UBS | — |
| Semicon India Programme Phase 1 outlay | ₹76,000 crore | MeitY / ISM | https://ism.gov.in |
| Semicon India Programme Phase 2 outlay | ₹1,27,500 crore | MeitY / ISM | https://ism.gov.in |
| Total committed investment (all ISM projects) | ₹1,60,365 crore | PIB, May 2026 | https://pib.gov.in/PRID=2258116 |

---

*Last updated: 13-Oct-2026 by Nidhi Chougule*
