"""MODULE 8 - Tests for the dashboard's data layer (Module 6).

We do NOT launch Streamlit in tests (that requires a browser). Instead we
verify that every helper used by the dashboard returns valid data, and that
the persisted JSON artefacts that the dashboard reads are present and
parseable.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    ALERTS_PATH,
    BUSINESS_SUMMARY_PATH,
    FEATURE_IMPORTANCE_PATH,
    FORECAST_PATH,
    MODEL_METRICS_PATH,
    RECOMMENDATIONS_PATH,
)


def _require_artifact(path: Path):
    if not path.exists():
        pytest.skip(f"Artefact {path.name} not produced yet - run train_model.py")


def test_business_summary_artifact_valid_json():
    _require_artifact(BUSINESS_SUMMARY_PATH)
    data = json.loads(BUSINESS_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert "kpis" in data
    assert "sales_by_category" in data
    assert "monthly_sales_trend" in data


def test_model_metrics_artifact_valid_json():
    _require_artifact(MODEL_METRICS_PATH)
    data = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    assert "best_model" in data
    assert "models" in data
    for name, info in data["models"].items():
        m = info["metrics"]
        for k in ("MAE", "RMSE", "MAPE", "R2"):
            assert k in m


def test_forecast_artifact_csv():
    _require_artifact(FORECAST_PATH)
    df = pd.read_csv(FORECAST_PATH)
    assert len(df) == 3
    assert {"YearMonth", "Forecast", "Model"} <= set(df.columns)


def test_feature_importance_artifact_csv():
    _require_artifact(FEATURE_IMPORTANCE_PATH)
    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    assert {"model", "feature", "value", "kind"} <= set(df.columns)


def test_alerts_artifact_valid_json():
    _require_artifact(ALERTS_PATH)
    data = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
    assert data["summary"]["total_alerts"] == len(data["all_alerts"])


def test_recommendations_artifact_valid_json():
    _require_artifact(RECOMMENDATIONS_PATH)
    data = json.loads(RECOMMENDATIONS_PATH.read_text(encoding="utf-8"))
    assert data["summary"]["total_recommendations"] == len(data["all_recommendations"])
