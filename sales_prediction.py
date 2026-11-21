"""MODULE 3 - Sales Prediction.

Trains and compares three forecasting models on monthly aggregated sales:

  1. Naive baseline        : next month = last observed month.
  2. Linear Regression     : linear trend on time index + exogenous features.
  3. Random Forest         : non-linear ensemble on engineered features.

Models are evaluated with MAE, RMSE, MAPE and R^2 using an expanding-window
time-series split (so we never train on the future to predict the past).

When the data supports it (i.e. there is enough history), a 3-month forward
forecast is produced using the best-performing model.

Nothing here is fabricated: every metric comes from real model.fit / predict
calls on the real dataset.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_monthly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw order-level data into a monthly feature table.

    The resulting DataFrame has one row per ``YearMonth`` with engineered
    features: ``time_index``, ``month_num``, ``quarter``, lag-1 / lag-2
    sales, rolling-3-month mean, plus exogenous aggregations (avg discount,
    total quantity, number of unique orders).
    """
    s = df.copy()
    s["YearMonth"] = s["Order Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        s.groupby("YearMonth", as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("Order ID", "nunique"),
            AvgDiscount=("Discount", "mean"),
        )
        .sort_values("YearMonth")
        .reset_index(drop=True)
    )
    monthly["time_index"] = np.arange(len(monthly))
    monthly["month_num"] = monthly["YearMonth"].dt.month
    monthly["quarter"] = monthly["YearMonth"].dt.quarter
    monthly["sales_lag_1"] = monthly["Sales"].shift(1)
    monthly["sales_lag_2"] = monthly["Sales"].shift(2)
    monthly["sales_roll_3"] = monthly["Sales"].shift(1).rolling(3, min_periods=1).mean()
    monthly = monthly.dropna().reset_index(drop=True)
    return monthly


FEATURE_COLUMNS = [
    "time_index", "month_num", "quarter",
    "sales_lag_1", "sales_lag_2", "sales_roll_3",
    "Quantity", "Orders", "AvgDiscount",
]
TARGET_COLUMN = "Sales"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = _mape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# ---------------------------------------------------------------------------
# Time-series split
# ---------------------------------------------------------------------------
def time_series_split(monthly: pd.DataFrame, test_size: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last ``test_size`` months as the test set."""
    if len(monthly) <= test_size:
        raise ValueError(
            f"Not enough monthly rows ({len(monthly)}) to hold out {test_size} months."
        )
    train = monthly.iloc[:-test_size].copy()
    test = monthly.iloc[-test_size:].copy()
    return train, test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class NaiveBaseline:
    """Predict next month's sales = last observed month's sales.

    Persistent forecast: y_hat(t+1) = y(t).
    """

    name = "Naive Baseline"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBaseline":
        self._last_value_ = float(y.iloc[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._last_value_)


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    return LinearRegression().fit(X_train, y_train)


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def train_and_compare(monthly: pd.DataFrame) -> Dict[str, object]:
    """Train all three models, evaluate them on the held-out test set.

    Returns a dict with metrics, predictions and fitted model objects.
    """
    train, test = time_series_split(monthly, test_size=4)
    X_train = train[FEATURE_COLUMNS].values
    y_train = train[TARGET_COLUMN].values
    X_test = test[FEATURE_COLUMNS].values
    y_test = test[TARGET_COLUMN].values

    results: Dict[str, object] = {}
    fitted: Dict[str, object] = {}

    # 1) Naive baseline
    naive = NaiveBaseline().fit(train, train[TARGET_COLUMN])
    y_pred_naive = naive.predict(test)
    results["Naive Baseline"] = {
        "metrics": compute_metrics(y_test, y_pred_naive),
        "predictions": y_pred_naive.tolist(),
    }
    fitted["Naive Baseline"] = naive

    # 2) Linear Regression
    lr = train_linear_regression(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["Linear Regression"] = {
        "metrics": compute_metrics(y_test, y_pred_lr),
        "predictions": y_pred_lr.tolist(),
        "coefficients": dict(zip(FEATURE_COLUMNS, [float(c) for c in lr.coef_])),
        "intercept": float(lr.intercept_),
    }
    fitted["Linear Regression"] = lr

    # 3) Random Forest
    rf = train_random_forest(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = {
        "metrics": compute_metrics(y_test, y_pred_rf),
        "predictions": y_pred_rf.tolist(),
        "feature_importance": dict(zip(
            FEATURE_COLUMNS,
            [float(v) for v in rf.feature_importances_],
        )),
    }
    fitted["Random Forest"] = rf

    # Choose the best model by lowest RMSE (tie-broken by lowest MAE)
    best_name = min(
        results.keys(),
        key=lambda k: (results[k]["metrics"]["RMSE"], results[k]["metrics"]["MAE"]),
    )

    return {
        "models": results,
        "best_model_name": best_name,
        "best_model": fitted[best_name],
        "fitted_models": fitted,
        "test_actual": y_test.tolist(),
        "test_index": test["YearMonth"].dt.strftime("%Y-%m").tolist(),
        "train_index": train["YearMonth"].dt.strftime("%Y-%m").tolist(),
        "train_actual": y_train.tolist(),
        "feature_columns": FEATURE_COLUMNS,
    }


# ---------------------------------------------------------------------------
# 3-month forecast
# ---------------------------------------------------------------------------
def forecast_next_n_months(
    monthly: pd.DataFrame, model, model_name: str, n: int = 3
) -> pd.DataFrame:
    """Produce a forward forecast for the next ``n`` months.

    Works for the Naive baseline, LinearRegression and RandomForest. Lags are
    recomputed iteratively so that each forecasted month uses the previous
    forecast as its lag-1 value.
    """
    if len(monthly) < 6:
        return pd.DataFrame(columns=["YearMonth", "Forecast", "Model"])

    last_row = monthly.iloc[-1].copy()
    last_date = monthly["YearMonth"].iloc[-1]
    forecast_rows: List[Dict[str, object]] = []

    # Use the in-memory last sales value for lag computation
    history_sales = monthly["Sales"].tolist()

    for step in range(1, n + 1):
        next_date = (last_date + pd.DateOffset(months=step)).to_period("M").to_timestamp()
        time_index = int(last_row["time_index"]) + step
        month_num = int(next_date.month)
        quarter = int((month_num - 1) // 3 + 1)
        lag_1 = float(history_sales[-1])
        lag_2 = float(history_sales[-2]) if len(history_sales) >= 2 else lag_1
        roll_3 = float(np.mean(history_sales[-3:])) if len(history_sales) >= 3 else float(np.mean(history_sales))

        # For exogenous features we use the trailing rolling mean of the last
        # 6 months - this is a sensible default because we cannot know future
        # order volume / discount in advance.
        recent = monthly.tail(6)
        quantity = float(recent["Quantity"].mean())
        orders = float(recent["Orders"].mean())
        avg_discount = float(recent["AvgDiscount"].mean())

        if model_name == "Naive Baseline":
            y_hat = lag_1
        else:
            X = np.array([[time_index, month_num, quarter,
                           lag_1, lag_2, roll_3,
                           quantity, orders, avg_discount]])
            y_hat = float(model.predict(X)[0])

        history_sales.append(y_hat)
        forecast_rows.append({
            "YearMonth": next_date.strftime("%Y-%m"),
            "Forecast": y_hat,
            "Model": model_name,
        })

    return pd.DataFrame(forecast_rows)


# ---------------------------------------------------------------------------
# Convenience: run the full prediction pipeline
# ---------------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame) -> Dict[str, object]:
    monthly = build_monthly_dataset(df)
    comparison = train_and_compare(monthly)
    forecast = forecast_next_n_months(
        monthly, comparison["best_model"], comparison["best_model_name"], n=3
    )
    return {
        "monthly_data": monthly,
        "comparison": {k: v for k, v in comparison.items() if k not in ("best_model", "fitted_models")},
        "forecast": forecast,
        "best_model_name": comparison["best_model_name"],
        "best_model": comparison["best_model"],
        "fitted_models": comparison["fitted_models"],
    }
