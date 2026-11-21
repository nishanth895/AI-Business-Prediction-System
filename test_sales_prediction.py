"""MODULE 8 - Tests for sales prediction (Module 3)."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_and_clean
from src.sales_prediction import (
    NaiveBaseline,
    build_monthly_dataset,
    compute_metrics,
    forecast_next_n_months,
    train_and_compare,
)


def test_build_monthly_dataset_shape():
    df, _ = load_and_clean()
    m = build_monthly_dataset(df)
    assert "Sales" in m.columns
    assert "time_index" in m.columns
    assert "sales_lag_1" in m.columns
    # No NaNs after the engineered features (we drop the warmup rows)
    assert m[["Sales", "time_index", "sales_lag_1", "sales_lag_2", "sales_roll_3"]].isna().sum().sum() == 0
    # Should cover ~4 years (2015-2018), so at least 40 months after lag warmup
    assert len(m) >= 30


def test_compute_metrics_values():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 290])
    m = compute_metrics(y_true, y_pred)
    assert m["MAE"] > 0
    assert m["RMSE"] >= m["MAE"]
    assert m["MAPE"] > 0
    assert m["R2"] <= 1.0  # R² can be negative


def test_naive_baseline_persistence():
    bl = NaiveBaseline().fit(pd.DataFrame({"x": [1, 2, 3]}), pd.Series([10, 20, 30]))
    preds = bl.predict(pd.DataFrame({"x": [4, 5, 6, 7]}))
    assert np.allclose(preds, [30, 30, 30, 30])


def test_train_and_compare_returns_three_models():
    df, _ = load_and_clean()
    monthly = build_monthly_dataset(df)
    result = train_and_compare(monthly)
    assert set(result["models"].keys()) == {"Naive Baseline", "Linear Regression", "Random Forest"}
    assert result["best_model_name"] in result["models"]
    for name, info in result["models"].items():
        m = info["metrics"]
        assert m["MAE"] >= 0
        assert m["RMSE"] >= 0
        assert m["MAPE"] >= 0


def test_forecast_next_n_months_returns_three_months():
    df, _ = load_and_clean()
    monthly = build_monthly_dataset(df)
    result = train_and_compare(monthly)
    fc = forecast_next_n_months(monthly, result["best_model"],
                                result["best_model_name"], n=3)
    assert len(fc) == 3
    assert "YearMonth" in fc.columns
    assert "Forecast" in fc.columns
    assert (fc["Forecast"] > 0).all()


def test_forecast_with_short_history_returns_empty():
    # Build a tiny monthly frame
    short = pd.DataFrame({
        "YearMonth": pd.date_range("2018-01-01", periods=3, freq="MS"),
        "Sales": [1000, 2000, 1500],
        "Profit": [100, 200, 150],
        "Quantity": [10, 20, 15],
        "Orders": [1, 2, 1],
        "AvgDiscount": [0.0, 0.1, 0.05],
        "time_index": [0, 1, 2],
        "month_num": [1, 2, 3],
        "quarter": [1, 1, 1],
        "sales_lag_1": [np.nan, 1000, 2000],
        "sales_lag_2": [np.nan, np.nan, 1000],
        "sales_roll_3": [np.nan, np.nan, 1500],
    }).dropna()
    bl = NaiveBaseline().fit(short, short["Sales"])
    fc = forecast_next_n_months(short, bl, "Naive Baseline", n=3)
    assert fc.empty  # not enough history
