"""MODULE 4 - Predictive Alerts.

Generates business alerts derived from the real dataset:

  1. Declining sales      : months where sales fell > 15% vs the previous
                            month, or where a 3-month rolling mean is
                            decreasing for 2 consecutive months.
  2. Loss-making categories: sub-categories whose total profit < 0.
  3. Unusual trends       : months whose sales is more than 2 standard
                            deviations away from the rolling mean (an IQR / Z
                            score based anomaly detection, clearly explained
                            in the dashboard).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .business_analysis import loss_making_subcategories, monthly_sales_trend


def detect_declining_months(df: pd.DataFrame, drop_pct: float = 0.15) -> List[Dict[str, object]]:
    """Flag months where sales dropped more than ``drop_pct`` vs prior month."""
    monthly = monthly_sales_trend(df)
    monthly["pct_change"] = monthly["Sales"].pct_change()
    monthly["roll_mean_3"] = monthly["Sales"].rolling(3, min_periods=1).mean()

    alerts: List[Dict[str, object]] = []
    for _, row in monthly.iterrows():
        if pd.isna(row["pct_change"]):
            continue
        if row["pct_change"] < -drop_pct:
            alerts.append({
                "type": "DECLINING_SALES",
                "month": str(row["YearMonth"].strftime("%Y-%m")),
                "sales": float(row["Sales"]),
                "pct_change_vs_prev_month": round(float(row["pct_change"]) * 100, 2),
                "severity": "HIGH" if row["pct_change"] < -0.30 else "MEDIUM",
                "message": (
                    f"Sales in {row['YearMonth'].strftime('%Y-%m')} fell "
                    f"{abs(row['pct_change'])*100:.1f}% vs the previous month."
                ),
            })
    return alerts


def detect_loss_categories(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Flag sub-categories with negative total profit."""
    losses = loss_making_subcategories(df)
    alerts: List[Dict[str, object]] = []
    for _, row in losses.iterrows():
        alerts.append({
            "type": "LOSS_MAKING_SUBCATEGORY",
            "sub_category": str(row["Sub-Category"]),
            "category": str(row["Category"]),
            "total_profit": float(row["Profit"]),
            "total_sales": float(row["Sales"]),
            "severity": "HIGH" if row["Profit"] < -1000 else "MEDIUM",
            "message": (
                f"Sub-category '{row['Sub-Category']}' (under '{row['Category']}') "
                f"has a total loss of ${row['Profit']:,.2f}."
            ),
        })
    return alerts


def detect_anomaly_months(df: pd.DataFrame, threshold: float = 2.0) -> List[Dict[str, object]]:
    """Detect months whose sales is > ``threshold`` std from rolling mean.

    This is a simple, clearly-explainable anomaly detector: it computes the
    6-month rolling mean and rolling std, then flags any month whose sales
    deviates by more than ``threshold`` standard deviations.
    """
    monthly = monthly_sales_trend(df).copy()
    if len(monthly) < 6:
        return []
    monthly["roll_mean_6"] = monthly["Sales"].rolling(6, min_periods=3).mean()
    monthly["roll_std_6"] = monthly["Sales"].rolling(6, min_periods=3).std()
    monthly["z_score"] = (monthly["Sales"] - monthly["roll_mean_6"]) / monthly["roll_std_6"].replace(0, np.nan)

    alerts: List[Dict[str, object]] = []
    for _, row in monthly.iterrows():
        if pd.isna(row["z_score"]):
            continue
        if abs(row["z_score"]) >= threshold:
            direction = "unusually high" if row["z_score"] > 0 else "unusually low"
            alerts.append({
                "type": "UNUSUAL_TREND",
                "month": str(row["YearMonth"].strftime("%Y-%m")),
                "sales": float(row["Sales"]),
                "rolling_mean_6m": float(row["roll_mean_6"]),
                "z_score": round(float(row["z_score"]), 2),
                "severity": "HIGH" if abs(row["z_score"]) >= 3 else "MEDIUM",
                "message": (
                    f"Sales in {row['YearMonth'].strftime('%Y-%m')} were {direction} "
                    f"(z-score={row['z_score']:.2f}) vs the 6-month rolling mean."
                ),
            })
    return alerts


def detect_low_profit_categories(df: pd.DataFrame, min_margin: float = 5.0) -> List[Dict[str, object]]:
    """Flag categories whose profit margin is below ``min_margin`` %."""
    g = df.groupby("Category", as_index=False).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum")
    )
    g["Margin (%)"] = 100.0 * g["Profit"] / g["Sales"]
    alerts: List[Dict[str, object]] = []
    for _, row in g.iterrows():
        if row["Margin (%)"] < min_margin:
            alerts.append({
                "type": "LOW_PROFIT_CATEGORY",
                "category": str(row["Category"]),
                "total_sales": float(row["Sales"]),
                "total_profit": float(row["Profit"]),
                "margin_pct": round(float(row["Margin (%)"]), 2),
                "severity": "HIGH" if row["Margin (%)"] < 0 else "MEDIUM",
                "message": (
                    f"Category '{row['Category']}' has a profit margin of "
                    f"{row['Margin (%)']:.2f}% (below {min_margin}% threshold)."
                ),
            })
    return alerts


def generate_alerts(df: pd.DataFrame) -> Dict[str, object]:
    """Combine every alert detector into one structured payload."""
    declining = detect_declining_months(df)
    losses = detect_loss_categories(df)
    anomalies = detect_anomaly_months(df)
    low_margin = detect_low_profit_categories(df)

    all_alerts = declining + losses + anomalies + low_margin
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_alerts.sort(key=lambda a: severity_order.get(a["severity"], 3))

    summary = {
        "total_alerts": len(all_alerts),
        "high_severity": sum(1 for a in all_alerts if a["severity"] == "HIGH"),
        "medium_severity": sum(1 for a in all_alerts if a["severity"] == "MEDIUM"),
        "by_type": {
            t: sum(1 for a in all_alerts if a["type"] == t)
            for t in {a["type"] for a in all_alerts}
        },
    }
    return {
        "summary": summary,
        "declining_sales": declining,
        "loss_making_subcategories": losses,
        "unusual_trends": anomalies,
        "low_profit_categories": low_margin,
        "all_alerts": all_alerts,
    }
