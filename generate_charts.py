"""generate_charts.py

Produce standalone PNG result charts under ``reports/charts/`` so they can be
embedded into the project report. Uses only the real dataset and the real
model artefacts produced by ``train_model.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.business_analysis import (
    loss_making_subcategories,
    monthly_sales_trend,
    sales_by_category,
    sales_by_region,
    sales_by_subcategory,
)
from src.config import (
    FEATURE_IMPORTANCE_PATH,
    FORECAST_PATH,
    MODEL_METRICS_PATH,
    REPORTS_DIR,
)
from src.data_loader import load_and_clean

CHARTS_DIR = REPORTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.3,
    "font.size": 10,
})


def chart_monthly_trend(df) -> None:
    trend = monthly_sales_trend(df)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(trend["YearMonth"], trend["Sales"], label="Sales", color="#4e79a7")
    ax.plot(trend["YearMonth"], trend["Profit"], label="Profit",
            color="#e15759", marker="o", linewidth=2)
    ax.set_title("Monthly Sales & Profit Trend (2015–2018)")
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")
    ax.legend(loc="upper left")
    fig.savefig(CHARTS_DIR / "01_monthly_trend.png")
    plt.close(fig)


def chart_sales_by_category(df) -> None:
    g = sales_by_category(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(g["Category"], g["Sales"], color="#4e79a7", label="Sales")
    ax.barh(g["Category"], g["Profit"], color="#59a14f", label="Profit")
    ax.set_title("Sales & Profit by Category")
    ax.set_xlabel("Amount ($)")
    ax.legend()
    fig.savefig(CHARTS_DIR / "02_sales_by_category.png")
    plt.close(fig)


def chart_sales_by_region(df) -> None:
    g = sales_by_region(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(g))
    w = 0.35
    ax.bar([i - w/2 for i in x], g["Sales"], width=w, label="Sales", color="#4e79a7")
    ax.bar([i + w/2 for i in x], g["Profit"], width=w, label="Profit", color="#f28e2b")
    ax.set_xticks(list(x)); ax.set_xticklabels(g["Region"])
    ax.set_title("Sales & Profit by Region")
    ax.set_ylabel("Amount ($)")
    ax.legend()
    fig.savefig(CHARTS_DIR / "03_sales_by_region.png")
    plt.close(fig)


def chart_loss_makers(df) -> None:
    losses = loss_making_subcategories(df)
    if losses.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e15759" if p < 0 else "#59a14f" for p in losses["Profit"]]
    ax.barh(losses["Sub-Category"], losses["Profit"], color=colors)
    ax.set_title("Loss-making Sub-Categories (Total Profit < $0)")
    ax.set_xlabel("Total Profit ($)")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.savefig(CHARTS_DIR / "04_loss_makers.png")
    plt.close(fig)


def chart_model_comparison() -> None:
    if not MODEL_METRICS_PATH.exists():
        return
    data = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    rows = []
    for name, info in data["models"].items():
        m = info["metrics"]
        rows.append({"Model": name, "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE": m["MAPE"]})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
        ax.bar(df["Model"], df[metric], color=["#4e79a7", "#f28e2b", "#59a14f"])
        ax.set_title(f"{metric} (lower is better)")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Model Comparison on Held-out Test Set", y=1.02)
    fig.savefig(CHARTS_DIR / "05_model_comparison.png")
    plt.close(fig)


def chart_actual_vs_predicted() -> None:
    if not MODEL_METRICS_PATH.exists():
        return
    data = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    test_idx = data["test_index"]
    test_actual = data["test_actual"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(test_idx, test_actual, marker="o", linewidth=2, label="Actual", color="#4e79a7")
    for name, info in data["models"].items():
        ax.plot(test_idx, info["predictions"], marker="x", linestyle="--",
                label=name)
    ax.set_title("Actual vs Predicted Monthly Sales (held-out test)")
    ax.set_xlabel("Month"); ax.set_ylabel("Sales ($)")
    ax.legend()
    fig.savefig(CHARTS_DIR / "06_actual_vs_predicted.png")
    plt.close(fig)


def chart_forecast(df) -> None:
    if not FORECAST_PATH.exists():
        return
    fc = pd.read_csv(FORECAST_PATH)
    hist = monthly_sales_trend(df)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(hist["YearMonth"], hist["Sales"], color="#4e79a7", label="Historical Sales")
    ax.plot(fc["YearMonth"], fc["Forecast"], marker="o", linewidth=2,
            color="#e15759", label=f"3-month Forecast ({fc['Model'].iloc[0]})")
    ax.set_title("Historical Sales with 3-Month Forward Forecast")
    ax.set_xlabel("Month"); ax.set_ylabel("Sales ($)")
    ax.legend()
    fig.savefig(CHARTS_DIR / "07_forecast.png")
    plt.close(fig)


def chart_feature_importance() -> None:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return
    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    if fi.empty:
        return
    data = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    best = data["best_model"]
    sub = fi[fi["model"] == best].copy()
    if sub.empty:
        sub = fi
    sub = sub.reindex(sub["value"].abs().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4e79a7" if v >= 0 else "#e15759" for v in sub["value"]]
    ax.barh(sub["feature"], sub["value"], color=colors)
    ax.set_title(f"Feature Importance / Coefficients ({best})")
    ax.set_xlabel("Importance / Coefficient value")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.savefig(CHARTS_DIR / "08_feature_importance.png")
    plt.close(fig)


def main() -> None:
    df, _ = load_and_clean()
    chart_monthly_trend(df)
    chart_sales_by_category(df)
    chart_sales_by_region(df)
    chart_loss_makers(df)
    chart_model_comparison()
    chart_actual_vs_predicted()
    chart_forecast(df)
    chart_feature_importance()
    print(f"Charts written to {CHARTS_DIR}")
    for p in sorted(CHARTS_DIR.glob("*.png")):
        print(f"  - {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
