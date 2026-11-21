"""MODULE 8 - Tests for business analysis (Module 2)."""
import pandas as pd

from src.business_analysis import (
    build_business_summary,
    kpis,
    loss_making_subcategories,
    monthly_sales_trend,
    sales_by_category,
    sales_by_region,
    sales_by_subcategory,
    segment_performance,
    top_products,
)
from src.data_loader import load_and_clean


def test_kpis_are_finite_and_positive():
    df, _ = load_and_clean()
    k = kpis(df)
    assert k["Total Sales ($)"] > 0
    assert k["Total Orders"] > 0
    assert k["Total Customers"] > 0
    assert -100 < k["Profit Margin (%)"] < 100


def test_sales_by_category_three_categories():
    df, _ = load_and_clean()
    g = sales_by_category(df)
    assert len(g) == 3  # Furniture, Office Supplies, Technology
    assert set(g["Category"]) == {"Furniture", "Office Supplies", "Technology"}
    assert (g["Sales"] > 0).all()


def test_sales_by_region_four_regions():
    df, _ = load_and_clean()
    g = sales_by_region(df)
    assert set(g["Region"]) == {"South", "West", "Central", "East"}
    assert len(g) == 4


def test_monthly_sales_trend_is_monotonic_in_time():
    df, _ = load_and_clean()
    t = monthly_sales_trend(df)
    assert t["YearMonth"].is_monotonic_increasing
    assert (t["Sales"] >= 0).all()


def test_loss_making_subcategories_returns_dataframe():
    df, _ = load_and_clean()
    losses = loss_making_subcategories(df)
    assert isinstance(losses, pd.DataFrame)
    if not losses.empty:
        assert (losses["Profit"] < 0).all()


def test_top_products_returns_top_n():
    df, _ = load_and_clean()
    top = top_products(df, n=5)
    assert len(top) == 5
    assert top["Sales"].is_monotonic_decreasing


def test_segment_performance_returns_dataframe():
    df, _ = load_and_clean()
    seg = segment_performance(df)
    assert not seg.empty
    assert "Segment" in seg.columns


def test_build_business_summary_keys():
    df, _ = load_and_clean()
    summary = build_business_summary(df)
    for key in ["kpis", "sales_by_category", "sales_by_region",
                "sales_by_subcategory", "monthly_sales_trend",
                "loss_making_subcategories", "top_products",
                "segment_performance"]:
        assert key in summary
