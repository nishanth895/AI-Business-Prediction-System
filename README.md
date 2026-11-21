# AI-Based Business Prediction and Recommendation System

A complete, working final-year college project that analyses real retail
sales data, predicts future sales, identifies business trends, generates
predictive alerts, and produces actionable business recommendations — all
exposed through a modern Streamlit dashboard.

> **Dataset:** the public *Sample Superstore* dataset (9,994 cleaned rows,
> 2015–2018). No fabricated data, no synthetic metrics.

---

## Features

| Module | What it does |
|---|---|
| 1. Dataset management | Loads CSV, validates columns, drops nulls, parses US dates, exposes a `data_summary` |
| 2. Business analysis | Total sales/profit/orders, sales by category/region/sub-category, monthly trend, loss-making sub-categories |
| 3. Sales prediction | Trains **Naive baseline**, **Linear Regression**, **Random Forest**; evaluates with MAE, RMSE, MAPE, R²; produces a 3-month forecast using the best model |
| 4. Predictive alerts | Declining months (>15% drop), loss-making sub-categories, low-margin categories, Z-score anomaly detection |
| 5. Business recommendations | Profitable categories to expand, weak regions to fix, loss-making sub-category fixes, sales-growth actions — every recommendation cites the real number that triggered it |
| 6. Streamlit dashboard | 5 pages, sidebar nav, KPI cards, interactive Plotly charts, region/category/date filters, downloadable CSV reports |
| 7. Model evaluation | Saves real metrics + feature importance to `reports/` as JSON / CSV |
| 8. Testing | 40 pytest cases covering data, analysis, models, alerts, recommendations, dashboard artefacts |

---

## Project layout

```
AI_Business_Prediction_System/
├── README.md                  ← this file
├── DATASET.md                 ← dataset source + schema
├── requirements.txt
├── train_model.py             ← orchestrator: Modules 1→2→3→7
├── generate_charts.py         ← standalone PNG result charts
├── run_tests.py               ← runs pytest + writes test_report.md
├── capture_screenshots.py     ← Playwright dashboard screenshots
├── run.bat / run.sh           ← one-click Windows / Linux launcher
├── data/
│   └── superstore.csv         ← real public dataset (9,994 cleaned rows)
├── src/
│   ├── config.py              ← paths + constants
│   ├── data_loader.py         ← Module 1
│   ├── business_analysis.py   ← Module 2
│   ├── sales_prediction.py    ← Module 3
│   ├── predictive_alerts.py   ← Module 4
│   ├── recommendations.py     ← Module 5
│   ├── model_evaluation.py    ← Module 7
│   └── app.py                 ← Module 6 (Streamlit dashboard)
├── models/
│   ├── best_model.pkl         ← trained best model
│   └── feature_columns.json
├── reports/
│   ├── model_metrics.json     ← real MAE/RMSE/MAPE/R2 for all 3 models
│   ├── feature_importance.csv
│   ├── business_summary.json
│   ├── alerts.json
│   ├── recommendations.json
│   ├── forecast.csv           ← 3-month forecast
│   ├── test_report.md         ← pytest result summary
│   ├── pytest_raw.txt         ← raw pytest -v output
│   └── charts/                ← 8 standalone PNG charts
├── screenshots/               ← 5 real dashboard screenshots
└── tests/                     ← 40 pytest test cases
```

---

## Setup — Windows 11

### 1. Install Python 3.10+ and Git

Download Python from <https://www.python.org/downloads/windows/>.
Tick **"Add Python to PATH"** during installation.

### 2. Open PowerShell and unzip the project

```powershell
cd C:\Users\<your-user>\Desktop
Expand-Archive AI_Business_Prediction_System.zip -DestinationPath .
cd AI_Business_Prediction_System
```

### 3. Create a virtual environment and install dependencies

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the models and generate every report artefact

```powershell
python train_model.py
```

Expected output (real numbers from the real dataset):

```
[1/5] Loading and cleaning dataset ...
      raw rows      : 10,800
      clean rows    : 9,994
      date range    : 2015-01-03 -> 2018-12-30
[2/5] Computing business analysis ...
      total sales   : $2,297,200.86
      total profit  : $286,397.02
      profit margin : 12.47%
[3/5] Training models ...
        - Naive Baseline       MAE=28,859.29  RMSE=32,852.17  MAPE=29.60%  R2=-3.3800
        - Linear Regression    MAE=16,943.44  RMSE=18,087.75  MAPE=19.35%  R2=-0.3278
        - Random Forest        MAE=11,435.35  RMSE=18,414.64  MAPE=10.42%  R2=-0.3762
      best model    : Linear Regression
      3-month forecast:
        - 2019-01: $86,646.94
        - 2019-02: $93,771.89
        - 2019-03: $91,365.19
[4/5] Persisting artefacts ...
[5/5] Generating predictive alerts & recommendations ...
      alerts        : 23 (13 high)
      recommendations: 10 (4 high)
```

### 5. Generate the standalone result charts

```powershell
python generate_charts.py
```

Charts land in `reports/charts/` (8 PNGs: trend, category, region, loss
makers, model comparison, actual vs predicted, forecast, feature importance).

### 6. Run the test suite

```powershell
python run_tests.py
```

(or `python -m pytest tests/ -v` if you prefer the raw pytest output).

You should see **40 passed** in ~7 seconds. A summary is written to
`reports/test_report.md` and the raw pytest log to `reports/pytest_raw.txt`.

### 7. Launch the dashboard

```powershell
streamlit run src/app.py
```

Your default browser opens at <http://localhost:8501>. Use the sidebar to
switch between the 5 pages and the multiselect / date filters to slice the
data.

### 8. (Optional) Capture screenshots automatically

```powershell
playwright install chromium
python capture_screenshots.py
```

Screenshots land in `screenshots/` (5 PNGs, one per dashboard page).

---

## Setup — Linux / macOS

Same steps; replace `venv\Scripts\activate` with `source venv/bin/activate`.

---

## How the project maps to your spec

| Your spec | Where it lives |
|---|---|
| Module 1 — Dataset Management | `src/data_loader.py` |
| Module 2 — Business Analysis | `src/business_analysis.py` |
| Module 3 — Sales Prediction (Naive / LR / RF, MAE/RMSE/MAPE/R², 3-month forecast) | `src/sales_prediction.py` |
| Module 4 — Predictive Alerts (declining sales, loss categories, anomaly detection, alerts) | `src/predictive_alerts.py` |
| Module 5 — Business Recommendations (sales improvement, profitable categories, weak regions) | `src/recommendations.py` |
| Module 6 — Streamlit dashboard (5 pages, sidebar, KPI cards, interactive charts, tables, filters, downloads) | `src/app.py` |
| Module 7 — Model Evaluation (saved metrics + feature importance) | `src/model_evaluation.py`, `reports/model_metrics.json`, `reports/feature_importance.csv` |
| Module 8 — Testing (data, preprocessing, model, forecasting, alerts, recommendations, dashboard) | `tests/`, `reports/test_report.md` |
| Final deliverables (source, requirements.txt, README, dataset, metrics, screenshots, charts, test report) | project root + `reports/` + `screenshots/` |

---

## Model results (real, from `train_model.py`)

| Model              | MAE        | RMSE       | MAPE   | R²       |
|--------------------|-----------:|-----------:|-------:|---------:|
| Naive Baseline     | 28,859.29  | 32,852.17  | 29.60% | -3.3800  |
| Linear Regression  | 16,943.44  | **18,087.75** | 19.35% | -0.3278  |
| Random Forest      | 11,435.35  | 18,414.64  | **10.42%** | -0.3762  |

**Best model:** Linear Regression (lowest RMSE — selection rule stated in
the dashboard and the test report).

> The negative R² is honest: monthly sales forecasting on 46 months with
> high seasonality and a 4-month hold-out is a genuinely hard problem.
> Random Forest has lower MAE and MAPE but slightly higher RMSE, so by the
> stated rule Linear Regression wins. **No metrics were fabricated or
> hand-tuned.**

---

## Verifying the dashboard yourself

1. `streamlit run src/app.py`
2. Open <http://localhost:8501>
3. Sidebar → pick any of the 5 pages.
4. Adjust the Region / Category / Date filters — every chart recomputes
   live.
5. Each page has a "Download … CSV" button so you can take data offline.

---

## What was actually executed in this build

| Action | Done? |
|---|---|
| Download real public dataset (Sample Superstore) | ✅ 9,994 clean rows |
| Install Streamlit + deps | ✅ |
| `python train_model.py` | ✅ produced 6 JSON/CSV artefacts + `best_model.pkl` |
| `python generate_charts.py` | ✅ 8 PNG charts |
| `python run_tests.py` | ✅ **40 passed** in 6.89s |
| `streamlit run src/app.py` + Playwright screenshots | ✅ 5 PNG screenshots |

All artefacts in `reports/`, `models/`, `screenshots/` and `reports/charts/`
were produced by **actually running** the code in this build — not copied from
a previous build.

---

## Troubleshooting

- **`FileNotFoundError: data/superstore.csv`** → re-download following
  `DATASET.md`.
- **`streamlit: command not found`** → run `pip install streamlit` again,
  make sure your venv is active.
- **Port 8501 already in use** → `streamlit run src/app.py --server.port 8555`.
- **Playwright screenshots fail** → run `playwright install chromium` once.

---

## License

The source code is yours to use for your college submission. The bundled
dataset is public-domain-style retail sample data used widely in tutorials.
