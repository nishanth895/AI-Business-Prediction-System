"""MODULE 7 - Model Evaluation.

Saves real model evaluation metrics, forecast results and feature importance
to ``reports/`` as JSON / CSV so they can be inspected independently of the
dashboard.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import (
    BEST_MODEL_PATH,
    BUSINESS_SUMMARY_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_IMPORTANCE_PATH,
    LABEL_ENCODERS_PATH,
    MODEL_METRICS_PATH,
    ALERTS_PATH,
    RECOMMENDATIONS_PATH,
    FORECAST_PATH,
)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def save_model_metrics(comparison: Dict[str, object], best_model_name: str, path: Path = MODEL_METRICS_PATH) -> None:
    """Persist model comparison metrics + the chosen model name."""
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "best_model": best_model_name,
        "feature_columns": comparison["feature_columns"],
        "test_index": comparison["test_index"],
        "test_actual": comparison["test_actual"],
        "models": {
            name: {
                "metrics": m["metrics"],
                "predictions": m.get("predictions", []),
            }
            for name, m in comparison["models"].items()
        },
        "selection_rule": "Lowest RMSE, tie broken by lowest MAE",
    }
    _write_json(path, payload)


def save_feature_importance(comparison: Dict[str, object], path: Path = FEATURE_IMPORTANCE_PATH) -> None:
    """Persist feature importance for tree-based models + coefficients for LR."""
    rows = []
    for name, info in comparison["models"].items():
        if "feature_importance" in info:
            for feat, val in info["feature_importance"].items():
                rows.append({"model": name, "feature": feat, "value": float(val), "kind": "importance"})
        if "coefficients" in info:
            for feat, val in info["coefficients"].items():
                rows.append({"model": name, "feature": feat, "value": float(val), "kind": "coefficient"})
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def save_forecast(forecast_df: pd.DataFrame, path: Path = FORECAST_PATH) -> None:
    forecast_df.to_csv(path, index=False)


def save_business_summary(summary: Dict[str, object], path: Path = BUSINESS_SUMMARY_PATH) -> None:
    _write_json(path, summary)


def save_alerts(alerts: Dict[str, object], path: Path = ALERTS_PATH) -> None:
    _write_json(path, alerts)


def save_recommendations(recommendations: Dict[str, object], path: Path = RECOMMENDATIONS_PATH) -> None:
    _write_json(path, recommendations)


def save_best_model(model, best_model_name: str, feature_columns: list[str],
                    path: Path = BEST_MODEL_PATH,
                    feature_columns_path: Path = FEATURE_COLUMNS_PATH) -> None:
    """Pickle the best model and persist the feature column list."""
    # Naive baseline is small enough to pickle, but we also save its last value
    with path.open("wb") as fh:
        pickle.dump({"model": model, "name": best_model_name}, fh)
    with feature_columns_path.open("w", encoding="utf-8") as fh:
        json.dump(feature_columns, fh, indent=2)


def load_best_model(path: Path = BEST_MODEL_PATH):
    """Reload the persisted best model."""
    with path.open("rb") as fh:
        return pickle.load(fh)
