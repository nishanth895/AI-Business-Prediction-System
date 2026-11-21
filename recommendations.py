"""MODULE 5 - Business Recommendations.

Turns the dataset analysis into actionable, prioritised recommendations.
Every recommendation is derived from real computed numbers - it cites the
figure that triggered it, so the user can verify each recommendation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .business_analysis import (
    loss_making_subcategories,
    monthly_sales_trend,
    sales_by_category,
    sales_by_region,
    sales_by_subcategory,
)


def recommend_profitable_categories(df: pd.DataFrame, top_n: int = 3) -> List[Dict[str, object]]:
    """Rank categories by absolute profit and recommend the top performers."""
    g = sales_by_category(df).sort_values("Profit", ascending=False)
    recs: List[Dict[str, object]] = []
    for rank, (_, row) in enumerate(g.head(top_n).iterrows(), start=1):
        recs.append({
            "priority": "HIGH" if rank == 1 else "MEDIUM",
            "category": str(row["Category"]),
            "action": f"Expand inventory and marketing for '{row['Category']}'",
            "evidence": (
                f"Total profit ${row['Profit']:,.2f} on sales ${row['Sales']:,.2f} "
                f"(margin {row['Profit Margin (%)']:.2f}%)."
            ),
            "expected_impact": "Increase overall profit by reinforcing the strongest margin engine.",
        })
    return recs


def flag_weak_regions(df: pd.DataFrame, min_margin: float = 10.0) -> List[Dict[str, object]]:
    """Recommend restructuring for regions whose margin is below ``min_margin``."""
    g = sales_by_region(df)
    weak = g[g["Profit Margin (%)"] < min_margin].sort_values("Profit Margin (%)")
    recs: List[Dict[str, object]] = []
    for _, row in weak.iterrows():
        recs.append({
            "priority": "HIGH" if row["Profit Margin (%)"] < 5 else "MEDIUM",
            "region": str(row["Region"]),
            "action": (
                f"Review pricing & discount strategy in {row['Region']} region"
            ),
            "evidence": (
                f"Sales ${row['Sales']:,.2f}, profit ${row['Profit']:,.2f} "
                f"(margin {row['Profit Margin (%)']:.2f}%)."
            ),
            "expected_impact": (
                "Improving this region's margin to the 10% benchmark would add "
                f"${0.10 * row['Sales'] - row['Profit']:,.2f} in profit."
            ),
        })
    return recs


def recommend_discontinue_or_fix_loss_makers(df: pd.DataFrame) -> List[Dict[str, object]]:
    """For loss-making sub-categories, recommend pricing/discount review."""
    losses = loss_making_subcategories(df)
    recs: List[Dict[str, object]] = []
    for _, row in losses.iterrows():
        avg_disc = df[df["Sub-Category"] == row["Sub-Category"]]["Discount"].mean()
        recs.append({
            "priority": "HIGH" if row["Profit"] < -2000 else "MEDIUM",
            "sub_category": str(row["Sub-Category"]),
            "category": str(row["Category"]),
            "action": (
                f"Reduce average discount (currently {avg_disc*100:.1f}%) and "
                f"review supplier cost for '{row['Sub-Category']}'"
            ),
            "evidence": (
                f"Loss of ${row['Profit']:,.2f} on sales ${row['Sales']:,.2f}."
            ),
            "expected_impact": (
                "Bringing this sub-category back to break-even would lift total profit."
            ),
        })
    return recs


def recommend_sales_growth(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Suggest sales-growth actions derived from monthly trend analysis."""
    monthly = monthly_sales_trend(df)
    recs: List[Dict[str, object]] = []

    # Identify best-selling month historically
    if not monthly.empty:
        best_row = monthly.loc[monthly["Sales"].idxmax()]
        recs.append({
            "priority": "MEDIUM",
            "action": (
                f"Replicate the marketing mix used in {best_row['YearMonth'].strftime('%Y-%m')}, "
                f"the best-performing month (${best_row['Sales']:,.2f})."
            ),
            "evidence": (
                f"Peak monthly sales occurred in "
                f"{best_row['YearMonth'].strftime('%Y-%m')}."
            ),
            "expected_impact": "Seasonal replication can lift future months by ~10-15%.",
        })

    # Discount / margin insight
    avg_disc = float(df["Discount"].mean())
    high_disc = df[df["Discount"] > 0.3]
    if len(high_disc) > 0:
        lost_profit = float(high_disc["Profit"].sum())
        recs.append({
            "priority": "HIGH" if lost_profit < 0 else "MEDIUM",
            "action": (
                f"Tighten discount policy - {len(high_disc)} orders had > 30% discount "
                f"(avg discount {avg_disc*100:.1f}%)."
            ),
            "evidence": (
                f"Combined profit on high-discount orders: ${lost_profit:,.2f}."
            ),
            "expected_impact": "Lowering discounts on these items can directly improve profit.",
        })

    # Bundle recommendation - top product + best margin sub-cat
    sub = sales_by_subcategory(df)
    if not sub.empty:
        top_volume = sub.sort_values("Quantity", ascending=False).iloc[0]
        best_margin = sub[sub["Profit"] > 0].sort_values(
            "Profit", ascending=False
        ).iloc[0] if (sub["Profit"] > 0).any() else None
        if best_margin is not None:
            recs.append({
                "priority": "LOW",
                "action": (
                    f"Bundle high-volume '{top_volume['Sub-Category']}' with "
                    f"high-margin '{best_margin['Sub-Category']}'."
                ),
                "evidence": (
                    f"{top_volume['Sub-Category']} sold "
                    f"{int(top_volume['Quantity'])} units; "
                    f"{best_margin['Sub-Category']} earned "
                    f"${best_margin['Profit']:,.2f} profit."
                ),
                "expected_impact": "Cross-sell lifts both volume and blended margin.",
            })
    return recs


def build_recommendations(df: pd.DataFrame) -> Dict[str, object]:
    profitable = recommend_profitable_categories(df)
    weak_regions = flag_weak_regions(df)
    fix_losses = recommend_discontinue_or_fix_loss_makers(df)
    grow = recommend_sales_growth(df)

    all_recs = profitable + weak_regions + fix_losses + grow
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_recs.sort(key=lambda r: priority_order.get(r.get("priority", "LOW"), 3))

    return {
        "summary": {
            "total_recommendations": len(all_recs),
            "high": sum(1 for r in all_recs if r.get("priority") == "HIGH"),
            "medium": sum(1 for r in all_recs if r.get("priority") == "MEDIUM"),
            "low": sum(1 for r in all_recs if r.get("priority") == "LOW"),
        },
        "profitable_categories": profitable,
        "weak_regions": weak_regions,
        "loss_making_fixes": fix_losses,
        "sales_growth": grow,
        "all_recommendations": all_recs,
    }
