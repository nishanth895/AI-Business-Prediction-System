"""train_model.py

Orchestrator that runs the full training pipeline:

  Module 1 - load + clean the dataset
  Module 2 - compute business summary
  Module 3 - train Naive / LinearRegression / RandomForest, evaluate, forecast
  Module 7 - persist metrics, feature importance, forecast, best model

Run from the project root:

    python train_model.py

The script prints a concise summary of what it produced and where the
artefacts live. Nothing is fabricated - every metric comes from real model
fit / predict calls on the real dataset.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow ``python train_model.py`` from project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.business_analysis import build_business_summary
from src.data_loader import load_and_clean
from src.model_evaluation import (
    save_best_model,
    save_business_summary,
    save_feature_importance,
    save_forecast,
    save_alerts,
    save_recommendations,
    save_model_metrics,
)
from src.predictive_alerts import generate_alerts
from src.recommendations import build_recommendations
from src.sales_prediction import build_monthly_dataset, train_and_compare, forecast_next_n_months


def main() -> int:
    start = time.time()
    print("=" * 78)
    print("AI-BASED BUSINESS PREDICTION AND RECOMMENDATION SYSTEM")
    print("Training pipeline")
    print("=" * 78)

    # ---- Module 1: load + clean --------------------------------------------
    print("\n[1/5] Loading and cleaning dataset ...")
    df, summary = load_and_clean()
    print(f"      raw rows      : {summary['raw_rows']:,}")
    print(f"      clean rows    : {summary['rows']:,}")
    print(f"      rows dropped  : {summary['drop_report']['__rows_dropped__']:,}")
    print(f"      date range    : {summary['date_range']['start']} -> {summary['date_range']['end']}")
    print(f"      columns       : {summary['columns']}")

    # ---- Module 2: business analysis ---------------------------------------
    print("\n[2/5] Computing business analysis ...")
    biz = build_business_summary(df)
    kpi = biz["kpis"]
    print(f"      total sales   : ${kpi['Total Sales ($)']:,.2f}")
    print(f"      total profit  : ${kpi['Total Profit ($)']:,.2f}")
    print(f"      total orders  : {kpi['Total Orders']:,}")
    print(f"      profit margin : {kpi['Profit Margin (%)']:.2f}%")
    save_business_summary(biz)
    print(f"      saved -> reports/business_summary.json")

    # ---- Module 3: prediction ----------------------------------------------
    print("\n[3/5] Building monthly dataset and training models ...")
    monthly = build_monthly_dataset(df)
    print(f"      monthly rows  : {len(monthly)}")
    comparison = train_and_compare(monthly)
    print("      metrics:")
    for name, info in comparison["models"].items():
        m = info["metrics"]
        print(f"        - {name:<20} MAE={m['MAE']:<12,.2f}  "
              f"RMSE={m['RMSE']:<12,.2f}  MAPE={m['MAPE']:<7.2f}%  R2={m['R2']:.4f}")
    print(f"      best model    : {comparison['best_model_name']}")

    forecast = forecast_next_n_months(
        monthly,
        comparison["best_model"],
        comparison["best_model_name"],
        n=3,
    )
    print("      3-month forecast:")
    for _, row in forecast.iterrows():
        print(f"        - {row['YearMonth']}: ${row['Forecast']:,.2f}  ({row['Model']})")

    # ---- Module 7: persist artefacts ---------------------------------------
    print("\n[4/5] Persisting artefacts ...")
    save_model_metrics(comparison, comparison["best_model_name"])
    save_feature_importance(comparison)
    save_forecast(forecast)
    save_best_model(
        comparison["best_model"],
        comparison["best_model_name"],
        comparison["feature_columns"],
    )
    print("      saved -> reports/model_metrics.json")
    print("      saved -> reports/feature_importance.csv")
    print("      saved -> reports/forecast.csv")
    print("      saved -> models/best_model.pkl")

    # ---- Alerts + recommendations (real data) ------------------------------
    print("\n[5/5] Generating predictive alerts & recommendations ...")
    alerts = generate_alerts(df)
    recs = build_recommendations(df)
    save_alerts(alerts)
    save_recommendations(recs)
    print(f"      alerts        : {alerts['summary']['total_alerts']} "
          f"({alerts['summary']['high_severity']} high)")
    print(f"      recommendations: {recs['summary']['total_recommendations']} "
          f"({recs['summary']['high']} high)")

    elapsed = time.time() - start
    print("\n" + "=" * 78)
    print(f"Training pipeline complete in {elapsed:.2f}s.")
    print("Inspect reports/ for JSON artefacts and models/best_model.pkl.")
    print("Run the dashboard with:  streamlit run src/app.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
