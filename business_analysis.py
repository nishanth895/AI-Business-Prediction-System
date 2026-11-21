"""MODULE 2 - Business Analysis.

Computes the headline KPIs and sliced aggregations used by the dashboard and
the recommendation engine. Every function returns real numbers computed from
the cleaned dataset - no fabricated values.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def total_sales(df: pd.DataFrame) -> float:
    return float(df["Sales"].sum())


def total_profit(df: pd.DataFrame) -> float:
    return float(df["Profit"].sum())


def total_orders(df: pd.DataFrame) -> int:
    return int(df["Order ID"].nunique())


def total_customers(df: pd.DataFrame) -> int:
    return int(df["Customer ID"].nunique()) if "Customer ID" in df.columns else 0


def kpis(df: pd.DataFrame) -> Dict[str, float | int]:
    """Return the headline KPIs used by the dashboard header."""
    return {
        "Total Sales ($)": total_sales(df),
        "Total Profit ($)": total_profit(df),
        "Total Orders": total_orders(df),
        "Total Customers": total_customers(df),
        "Avg Order Value ($)": float(total_sales(df) / max(total_orders(df), 1)),
        "Profit Margin (%)": float(
            100.0 * total_profit(df) / total_sales(df) if total_sales(df) else 0.0
        ),
    }


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Sales & profit aggregated by ``Category``."""
    g = (
        df.groupby("Category", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"), Quantity=("Quantity", "sum"))
        .sort_values("Sales", ascending=False)
    )
    g["Profit Margin (%)"] = (100.0 * g["Profit"] / g["Sales"]).round(2)
    return g.reset_index(drop=True)


def sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("Region", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"))
        .sort_values("Sales", ascending=False)
    )
    g["Profit Margin (%)"] = (100.0 * g["Profit"] / g["Sales"]).round(2)
    return g.reset_index(drop=True)


def sales_by_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["Category", "Sub-Category"], as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Quantity=("Quantity", "sum"))
        .sort_values("Sales", ascending=False)
    )
    return g.reset_index(drop=True)


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly total sales & profit across the whole dataset."""
    s = df.copy()
    s["YearMonth"] = s["Order Date"].dt.to_period("M").dt.to_timestamp()
    g = (
        s.groupby("YearMonth", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"))
        .sort_values("YearMonth")
        .reset_index(drop=True)
    )
    return g


def loss_making_subcategories(df: pd.DataFrame) -> pd.DataFrame:
    """Sub-categories whose total profit is negative."""
    g = sales_by_subcategory(df)
    return g[g["Profit"] < 0].sort_values("Profit").reset_index(drop=True)


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("Product Name", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Quantity=("Quantity", "sum"))
        .sort_values("Sales", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def segment_performance(df: pd.DataFrame) -> pd.DataFrame:
    if "Segment" not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby("Segment", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"))
        .sort_values("Sales", ascending=False)
    )
    g["Profit Margin (%)"] = (100.0 * g["Profit"] / g["Sales"]).round(2)
    return g.reset_index(drop=True)


def build_business_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Bundle every aggregation into one dict for downstream consumers."""
    return {
        "kpis": kpis(df),
        "sales_by_category": sales_by_category(df).to_dict(orient="records"),
        "sales_by_region": sales_by_region(df).to_dict(orient="records"),
        "sales_by_subcategory": sales_by_subcategory(df).to_dict(orient="records"),
        "monthly_sales_trend": monthly_sales_trend(df).to_dict(orient="records"),
        "loss_making_subcategories": loss_making_subcategories(df).to_dict(orient="records"),
        "top_products": top_products(df).to_dict(orient="records"),
        "segment_performance": segment_performance(df).to_dict(orient="records"),
    }
