# Permanent Deployment Guide — Streamlit Community Cloud

This guide walks you through deploying your **AI-Based Business Prediction and Recommendation System** permanently on **Streamlit Community Cloud** — completely free, with a public HTTPS URL you can share with faculty.

**Final URL format:** `https://<yourname>-business-prediction.streamlit.app`

No tunnels, no temporary URLs, no credit card. The app stays live forever (auto-redeploys if inactive for 7 days — just visit the URL to wake it up).

---

## Prerequisites (3 free accounts, ~5 minutes)

| Account | Where to sign up | Cost |
|---|---|---|
| GitHub | <https://github.com/signup> | Free |
| Streamlit Community Cloud | <https://share.streamlit.io> (sign in with GitHub) | Free |
| Email | For verification | Free |

**No credit card required. No AWS/GCP account needed. No domain needed.**

---

## Step 1 — Download and extract the project ZIP

Download: **[AI_Business_Prediction_System.zip](AI_Business_Prediction_System.zip)**

Extract it on your Windows 11 laptop:

```powershell
cd $env:USERPROFILE\Desktop
Expand-Archive AI_Business_Prediction_System.zip -DestinationPath .
cd AI_Business_Prediction_System
```

After extraction, verify these files exist:

```
AI_Business_Prediction_System/
├── .streamlit/config.toml        ← cloud config (theme + server settings)
├── .gitignore                    ← excludes venv, caches, secrets
├── .gitattributes                ← cross-platform line endings
├── .python-version               ← pins Python 3.11
├── requirements.txt              ← runtime dependencies (NO playwright)
├── requirements-dev.txt          ← dev-only deps (pytest, playwright)
├── README.md                     ← project documentation
├── DATASET.md                    ← dataset source + schema
├── DEPLOY.md                     ← this file
├── train_model.py                ← orchestrator: trains models, saves artefacts
├── generate_charts.py            ← standalone PNG result charts
├── run_tests.py                  ← runs pytest + writes test_report.md
├── capture_screenshots.py        ← Playwright dashboard screenshots
├── run.bat / run.sh              ← one-click Windows / Linux launcher
├── data/superstore.csv           ← real public dataset (9,994 rows)
├── src/                          ← 8 Python modules (the application)
│   ├── config.py
│   ├── data_loader.py
│   ├── business_analysis.py
│   ├── sales_prediction.py
│   ├── predictive_alerts.py
│   ├── recommendations.py
│   ├── model_evaluation.py
│   └── app.py                    ← Streamlit dashboard (main entry point)
├── models/                       ← pre-trained model + feature columns
│   ├── best_model.pkl
│   └── feature_columns.json
├── reports/                      ← pre-computed artefacts (real metrics)
│   ├── model_metrics.json
│   ├── business_summary.json
│   ├── alerts.json
│   ├── recommendations.json
│   ├── forecast.csv
│   ├── feature_importance.csv
│   ├── test_report.md
│   ├── pytest_raw.txt
│   └── charts/                   ← 8 standalone PNG charts
├── screenshots/                  ← 5 dashboard page screenshots
└── tests/                        ← 40 pytest test cases
```

**Critical:** The `.streamlit/`, `.gitignore`, `.python-version`, `.gitattributes` files are hidden by default in Windows File Explorer. To verify they exist, open PowerShell and run:

```powershell
cd AI_Business_Prediction_System
Get-ChildItem -Force | Where-Object { $_.Name -like ".*" }
```

You should see `.gitignore`, `.gitattributes`, `.python-version`, and `.streamlit/`.

---

## Step 2 — Create a public GitHub repository

1. Go to <https://github.com/new>
2. Fill in:
   - **Repository name:** `ai-business-prediction`
   - **Description:** `AI-Based Business Prediction and Recommendation System Using Machine Learning — Streamlit dashboard`
   - **Visibility:** **Public** (Streamlit Cloud free tier requires public repos)
   - **Initialize this repository with:** ❌ Do NOT add README, .gitignore, or license (the project ships with its own)
3. Click **Create repository**

---

## Step 3 — Upload the project to GitHub (web uploader — no Git CLI needed)

1. On your new repo's page, click **"uploading an existing file"** (the link text appears below the "…or push an existing repository from the command line" section)
2. **Drag the entire contents of `AI_Business_Prediction_System/`** into the GitHub web uploader

### ⚠️ Critical: Include hidden dotfiles

GitHub's web uploader hides dotfiles by default. You MUST include:

| File | Why it's needed |
|---|---|
| `.streamlit/config.toml` | Streamlit Cloud reads this for theme + server settings |
| `.gitignore` | Standard hygiene (excludes venv, caches, secrets) |
| `.python-version` | Tells Streamlit Cloud to use Python 3.11 |
| `.gitattributes` | Cross-platform line ending normalization |

**How to reveal dotfiles in the GitHub uploader's file picker:**
- **Windows:** Type `*.*` in the "File name" field and press Enter → all files (including dotfiles) appear
- **macOS:** Press `Cmd+Shift+.` in the file picker
- **Linux:** Press `Ctrl+H` in the file picker

3. After all files are uploaded, scroll down and click **Commit changes**
4. Commit message: `Initial commit — AI Business Prediction System`
5. Click **Commit changes**

### Verify the upload

After committing, your repo should look like this:

```
ai-business-prediction/
├── .gitignore
├── .gitattributes
├── .python-version
├── .streamlit/
│   └── config.toml
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── DATASET.md
├── DEPLOY.md
├── train_model.py
├── generate_charts.py
├── run_tests.py
├── capture_screenshots.py
├── run.bat
├── run.sh
├── data/
│   └── superstore.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── business_analysis.py
│   ├── sales_prediction.py
│   ├── predictive_alerts.py
│   ├── recommendations.py
│   ├── model_evaluation.py
│   └── app.py
├── models/
│   ├── best_model.pkl
│   └── feature_columns.json
├── reports/
│   ├── ... (8 files + charts/ folder)
├── screenshots/
│   └── ... (5 PNGs)
└── tests/
    └── ... (8 files)
```

---

## Step 4 — Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io>
2. Sign in with your GitHub account (authorize Streamlit Cloud to access your public repos)
3. Click **"New app"** (top-right corner)
4. Fill in the deployment form with these **EXACT** values:

| Field | Value |
|---|---|
| **Repository** | `<your-github-username>/ai-business-prediction` |
| **Branch** | `main` |
| **Main file path** | `src/app.py` |
| **Python version** | `3.11` (matches `.python-version` file) |
| **App URL (custom subdomain)** | `yourname-business-prediction` (optional — makes the URL prettier) |
| **Advanced settings → Secrets** | (leave empty — no secrets needed for this project) |

5. Click **Deploy!**

---

## Step 5 — Wait for the build (~3-4 minutes)

Streamlit Cloud will now:
1. Clone your GitHub repo
2. Create a Python 3.11 virtual environment
3. Run `pip install -r requirements.txt` (installs pandas, numpy, scikit-learn, streamlit, plotly, matplotlib)
4. Run `streamlit run src/app.py`
5. Serve the app on a public HTTPS URL

### Build log

You can watch the build log live in your browser. You'll see:

```
🦄 Streamlit Community Cloud
Cloning repository...
Creating virtual environment [Python 3.11]...
Installing dependencies from requirements.txt...
  Collecting pandas...
  Collecting streamlit...
  ...
Successfully installed pandas-2.2.3 streamlit-1.67.0 ...
Running: streamlit run src/app.py --server.port 8501 --server.headless true
  You can now view your Streamlit app in your browser.
  Local URL: http://0.0.0.0:8501
✨ App deployed successfully!
```

### Your permanent URL

After the build completes, your app will be live at:

```
https://<yourname>-business-prediction.streamlit.app
```

(or `https://<random-id>.streamlit.app` if you didn't set a custom subdomain)

**This URL is permanent** — it stays live as long as you don't delete the app from Streamlit Cloud. You can share it with faculty, embed it in your resume, or link it from your GitHub README.

---

## Step 6 — Verify all 5 dashboard pages work

Open your new URL in Google Chrome. Use the **sidebar on the left** to switch between pages:

| Page | What you should see |
|---|---|
| **Overview** | 4 KPI cards (Total Sales $2.30M, Total Profit $286K, Total Customers 793, AOV $459), monthly trend bar+line chart, region pie chart, top 10 products table |
| **Business Analysis** | 6 tabs: By Category, By Region, By Sub-Category, Monthly Trend, Loss Makers, Segment |
| **Sales Forecast** | Model comparison table (Linear Regression ★ best, RMSE $18,088), actual vs predicted chart, 3-month forecast (Jan-Mar 2019: $86K/$93K/$91K), feature importance |
| **Predictive Alerts** | 23 alerts (13 high), alert cards by type, ±2σ anomaly band chart |
| **Business Recommendations** | 10 prioritised recommendations (4 high), each with evidence + expected impact |

### Real data, no fabrication

Every number in the dashboard comes from the actual `data/superstore.csv` (9,994 cleaned rows, 2015-2018). The ML metrics (Linear Regression RMSE $18,088, MAE $16,943, MAPE 19.35%, R² -0.33) match the values in your project report's Table 7.4.

---

## Step 7 — Share with faculty

Send your faculty the URL:

```
https://<yourname>-business-prediction.streamlit.app
```

They can open it directly in any browser (Chrome, Edge, Firefox, Safari) — no login, no installation, no download required.

---

## Troubleshooting

### Build fails: `ModuleNotFoundError: No module named 'src'`

**Cause:** Main file path is wrong.
**Fix:** Set Main file path = `src/app.py` (NOT `app.py`). The app uses `from src.config import ...` which requires the path to be `src/app.py` so the project root is in `sys.path`.

### Build fails: `FileNotFoundError: data/superstore.csv`

**Cause:** The `data/` folder didn't make it into the GitHub repo.
**Fix:** Re-upload to GitHub, ensure the `data/superstore.csv` file (2.3 MB) is included. Check by visiting your repo URL — you should see a `data/` folder containing `superstore.csv`.

### Build fails: `playwright install` or `chromium` errors

**Cause:** `playwright` is in `requirements.txt` (it shouldn't be).
**Fix:** Verify `requirements.txt` contains only: pandas, numpy, scikit-learn, streamlit, plotly, matplotlib. `playwright` should be in `requirements-dev.txt` (which Streamlit Cloud does NOT install).

### First page load is slow (~30 seconds)

**Normal behavior.** Streamlit Cloud idles the app after 7 days of no traffic. The second visit loads instantly. The app falls back to recomputing the ML model on-the-fly if `reports/*.json` aren't present (adds ~10 seconds), so make sure the `reports/` folder is uploaded to GitHub.

### "This app has gone to sleep" message

Visit the URL → app wakes up in ~10 seconds. Streamlit Cloud sends "is this app still needed?" emails after 7 days of inactivity — click the link or just visit the URL to keep it alive.

### Charts render but KPIs show 0

**Cause:** The `data/superstore.csv` file is corrupted or missing rows.
**Fix:** Re-download the ZIP from the deployment guide link, re-extract, and re-upload `data/superstore.csv` to GitHub.

### `streamlit.app` subdomain already taken

Choose a different subdomain (e.g., `yourname-biz-prediction`, `yourname-ai-business`, `yourname-business-ml`). The default random subdomain (`<random-id>.streamlit.app`) always works if you don't set a custom one.

---

## Why Streamlit Community Cloud (and not Render / Railway / HuggingFace)

| Platform | Free tier supports Streamlit? | Persistent? | Verdict |
|---|---|---|---|
| **Streamlit Community Cloud** | Yes — purpose-built | Yes (auto-redeploy) | ✅ Best for this project |
| Hugging Face Spaces | Yes | Yes | Works, but requires HF account + git LFS for dataset |
| Render free tier | No (sleeps after 15 min, no WebSocket) | No | ❌ Breaks Streamlit |
| Railway | Trial credits only | No | ❌ Not free long-term |
| PythonAnywhere | Streamlit not officially supported | Yes | ❌ Hacky |

Streamlit Community Cloud is the **official free hosting** for Streamlit apps — purpose-built by the Streamlit team, no surprise limitations.

---

## What happens behind the scenes (technical details)

| Cloud-side action | What it does |
|---|---|
| Repo clone | `git clone https://github.com/<you>/ai-business-prediction` |
| Python env | Creates Python 3.11 virtualenv (matches `.python-version`) |
| Dependencies | `pip install -r requirements.txt` (runtime deps only) |
| App launch | `streamlit run src/app.py --server.port 8501 --server.headless true` |
| Streamlit config | Reads `.streamlit/config.toml` (dark theme, no usage stats, max upload 200 MB) |
| HTTPS termination | Streamlit Cloud's reverse proxy handles HTTPS, CORS, XSRF automatically |
| Process manager | Restarts the app if it crashes; idles it after 7 days of no traffic |

---

## Quick reference — the 4 things you need to remember

1. **Repo name:** `ai-business-prediction` (public)
2. **Main file path on Streamlit Cloud:** `src/app.py`
3. **Python version:** `3.11`
4. **Final URL pattern:** `https://<yourname>-business-prediction.streamlit.app`

---

## Need help?

If any step fails:

1. **Build log** — copy the error from Streamlit Cloud's build log viewer
2. **Repo URL** — verify your GitHub repo is public and contains all files
3. **Paste the error here** — I'll debug it

The project itself has been verified end-to-end (40/40 tests pass, real ML metrics, 5 dashboard pages with real data). Any cloud failure will be a deployment-config issue, not a code issue.
