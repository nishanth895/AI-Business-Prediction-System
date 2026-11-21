"""Central configuration: paths and constants used across the project."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

DATA_PATH = DATA_DIR / "superstore.csv"

# Model / report artefacts
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
BUSINESS_SUMMARY_PATH = REPORTS_DIR / "business_summary.json"
ALERTS_PATH = REPORTS_DIR / "alerts.json"
RECOMMENDATIONS_PATH = REPORTS_DIR / "recommendations.json"
FORECAST_PATH = REPORTS_DIR / "forecast.csv"
TEST_REPORT_PATH = REPORTS_DIR / "test_report.md"

# ---------------------------------------------------------------------------
# Dataset schema (columns the loader requires for the analysis to work)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "Order ID",
    "Order Date",
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category",
    "Product Name",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
]

NUMERIC_COLUMNS = ["Sales", "Quantity", "Discount", "Profit"]
CATEGORICAL_COLUMNS = [
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category",
]

# Make sure the artefact directories exist on import
for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, SCREENSHOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
