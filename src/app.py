"""MODULE 6 - Streamlit dashboard.

Five-page interactive application:

  1. Overview             - KPI cards + headline trends.
  2. Business Analysis    - sales by category / region / sub-category, monthly
                            trend, loss-making sub-categories.
  3. Sales Forecast       - model comparison, evaluation metrics, 3-month
                            forecast, feature importance.
  4. Predictive Alerts    - declining months, loss-making sub-categories,
                            unusual trend anomalies.
  5. Business Recommendations - prioritised, evidence-based actions.

Run with:

    streamlit run src/app.py

The dashboard loads ``reports/*.json`` produced by ``train_model.py``. If the
artefacts do not exist, it recomputes them on-the-fly from the dataset so the
UI is always usable.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
from src.config import (
    ALERTS_PATH,
    BEST_MODEL_PATH,
    BUSINESS_SUMMARY_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_IMPORTANCE_PATH,
    FORECAST_PATH,
    MODEL_METRICS_PATH,
    RECOMMENDATIONS_PATH,
)
from src.data_loader import load_and_clean
from src.predictive_alerts import generate_alerts
from src.recommendations import build_recommendations
from src.sales_prediction import (
    build_monthly_dataset,
    forecast_next_n_months,
    train_and_compare,
)


# ---------------------------------------------------------------------------
# Page config + caching
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Business Prediction & Recommendation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_data() -> pd.DataFrame:
    df, _ = load_and_clean()
    return df


@st.cache_data(show_spinner=False)
def cached_business_summary() -> dict:
    if BUSINESS_SUMMARY_PATH.exists():
        return json.loads(BUSINESS_SUMMARY_PATH.read_text(encoding="utf-8"))
    return build_business_summary(cached_data())


@st.cache_data(show_spinner=False)
def cached_model_metrics() -> dict:
    if MODEL_METRICS_PATH.exists():
        return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    # Recompute on the fly if missing
    df = cached_data()
    monthly = build_monthly_dataset(df)
    comparison = train_and_compare(monthly)
    return {
        "best_model": comparison["best_model_name"],
        "feature_columns": comparison["feature_columns"],
        "test_index": comparison["test_index"],
        "test_actual": comparison["test_actual"],
        "models": {
            name: {"metrics": m["metrics"], "predictions": m.get("predictions", [])}
            for name, m in comparison["models"].items()
        },
    }


@st.cache_data(show_spinner=False)
def cached_forecast() -> pd.DataFrame:
    if FORECAST_PATH.exists():
        return pd.read_csv(FORECAST_PATH)
    df = cached_data()
    monthly = build_monthly_dataset(df)
    comparison = train_and_compare(monthly)
    return forecast_next_n_months(monthly, comparison["best_model"],
                                  comparison["best_model_name"], n=3)


@st.cache_data(show_spinner=False)
def cached_feature_importance() -> pd.DataFrame:
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame(columns=["model", "feature", "value", "kind"])


@st.cache_data(show_spinner=False)
def cached_alerts() -> dict:
    if ALERTS_PATH.exists():
        return json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
    return generate_alerts(cached_data())


@st.cache_data(show_spinner=False)
def cached_recommendations() -> dict:
    if RECOMMENDATIONS_PATH.exists():
        return json.loads(RECOMMENDATIONS_PATH.read_text(encoding="utf-8"))
    return build_recommendations(cached_data())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def kpi_card(label: str, value: str, delta: str | None = None) -> None:
    """Render a KPI card in a Streamlit column."""
    st.markdown(
        f"""
        <div style="background:#0e1117;border:1px solid #262730;border-radius:12px;
                    padding:18px 20px;height:130px;display:flex;flex-direction:column;
                    justify-content:space-between;">
          <div style="color:#a4b0be;font-size:13px;letter-spacing:.5px;">{label}</div>
          <div style="color:#fafafa;font-size:24px;font-weight:700;">{value}</div>
          <div style="color:#7bed9f;font-size:12px;">{delta or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filtered_dataframe(df: pd.DataFrame, regions, categories, date_range) -> pd.DataFrame:
    out = df.copy()
    if regions:
        out = out[out["Region"].isin(regions)]
    if categories:
        out = out[out["Category"].isin(categories)]
    if date_range and len(date_range) == 2:
        lo, hi = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        out = out[(out["Order Date"] >= lo) & (out["Order Date"] <= hi)]
    return out


def download_button(df: pd.DataFrame, filename: str, label: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 AI Business Suite")
    st.caption("Retail sales prediction & recommendation system")
    st.divider()
    df_all = cached_data()
    regions = sorted(df_all["Region"].dropna().unique().tolist())
    categories = sorted(df_all["Category"].dropna().unique().tolist())

    st.subheader("Filters")
    sel_regions = st.multiselect("Region", regions, default=regions)
    sel_categories = st.multiselect("Category", categories, default=categories)

    d_min = df_all["Order Date"].min().to_pydatetime()
    d_max = df_all["Order Date"].max().to_pydatetime()
    sel_dates = st.date_input(
        "Order date range",
        value=(d_min, d_max),
        min_value=d_min,
        max_value=d_max,
    )

    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "Business Analysis", "Sales Forecast",
         "Predictive Alerts", "Business Recommendations"],
        index=0,
    )

    st.divider()
    st.caption(f"Dataset rows: {len(df_all):,}")
    st.caption(f"Date span: {d_min.date()} → {d_max.date()}")


# Apply filters once
df_view = filtered_dataframe(df_all, sel_regions, sel_categories, sel_dates)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
st.title(page)

if page == "Overview":
    k = kpis(df_view)
    c1, c2, c3, c4 = st.columns(4)
    kpi_card("Total Sales", fmt_money(k["Total Sales ($)"]),
             f"{k['Total Orders']:,} orders")
    kpi_card("Total Profit", fmt_money(k["Total Profit ($)"]),
             f"Margin {k['Profit Margin (%)']:.2f}%")
    kpi_card("Total Customers", f"{k['Total Customers']:,}",
             f"AOV {fmt_money(k['Avg Order Value ($)'])}")
    kpi_card("Avg Order Value", fmt_money(k["Avg Order Value ($)"]),
             f"Per {k['Total Orders']:,} orders")
    st.write("")  # spacer

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Monthly sales & profit trend")
        trend = monthly_sales_trend(df_view)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend["YearMonth"], y=trend["Sales"],
                             name="Sales", marker_color="#4e79a7"))
        fig.add_trace(go.Scatter(x=trend["YearMonth"], y=trend["Profit"],
                                 name="Profit", mode="lines+markers",
                                 line=dict(color="#f28e2b", width=2)))
        fig.update_layout(barmode="group", height=380,
                          legend=dict(orientation="h", y=-0.25),
                          margin=dict(t=20, b=20))
        st.plotly_chart(fig, width='stretch')

    with right:
        st.subheader("Sales by region")
        rg = sales_by_region(df_view)
        fig2 = px.pie(rg, names="Region", values="Sales",
                      hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=380, margin=dict(t=20, b=20),
                           showlegend=True)
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Top 10 products by sales")
    st.dataframe(top_products(df_view, n=10), width='stretch', hide_index=True)
    download_button(top_products(df_view, n=50), "top_products.csv", "Download top-50 products CSV")


elif page == "Business Analysis":
    tabs = st.tabs(["By Category", "By Region", "By Sub-Category",
                    "Monthly Trend", "Loss Makers", "Segment"])

    with tabs[0]:
        st.subheader("Sales & profit by category")
        cat = sales_by_category(df_view)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = px.bar(cat, x="Category", y="Sales",
                         color="Profit", color_continuous_scale="RdYlGn",
                         text_auto=".2s")
            fig.update_layout(height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width='stretch')
        with c2:
            st.dataframe(cat, width='stretch', hide_index=True)
        download_button(cat, "sales_by_category.csv", "Download category CSV")

    with tabs[1]:
        st.subheader("Sales & profit by region")
        reg = sales_by_region(df_view)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = px.bar(reg, x="Region", y=["Sales", "Profit"],
                         barmode="group", text_auto=".2s")
            fig.update_layout(height=380, margin=dict(t=20, b=20),
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width='stretch')
        with c2:
            st.dataframe(reg, width='stretch', hide_index=True)
        download_button(reg, "sales_by_region.csv", "Download region CSV")

    with tabs[2]:
        st.subheader("Sales & profit by sub-category")
        sub = sales_by_subcategory(df_view)
        fig = px.bar(sub, x="Sub-Category", y="Sales",
                     color="Profit", color_continuous_scale="RdYlGn",
                     text_auto=".2s")
        fig.update_layout(height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig, width='stretch')
        st.dataframe(sub, width='stretch', hide_index=True)
        download_button(sub, "sales_by_subcategory.csv", "Download sub-category CSV")

    with tabs[3]:
        st.subheader("Monthly sales & profit trend")
        trend = monthly_sales_trend(df_view)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend["YearMonth"], y=trend["Sales"],
                             name="Sales", marker_color="#4e79a7"))
        fig.add_trace(go.Scatter(x=trend["YearMonth"], y=trend["Profit"],
                                 name="Profit", mode="lines+markers",
                                 line=dict(color="#e15759", width=2)))
        fig.update_layout(barmode="overlay", height=420,
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=20, b=20))
        st.plotly_chart(fig, width='stretch')
        download_button(trend, "monthly_trend.csv", "Download monthly trend CSV")

    with tabs[4]:
        st.subheader("Loss-making sub-categories")
        losses = loss_making_subcategories(df_view)
        if losses.empty:
            st.success("No loss-making sub-categories for the current filters.")
        else:
            fig = px.bar(losses, x="Sub-Category", y="Profit",
                         color="Category", text_auto=".2s")
            fig.update_layout(height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width='stretch')
            st.dataframe(losses, width='stretch', hide_index=True)
            download_button(losses, "loss_makers.csv", "Download loss-makers CSV")

    with tabs[5]:
        st.subheader("Segment performance")
        seg = segment_performance(df_view)
        if seg.empty:
            st.info("No segment data available.")
        else:
            fig = px.bar(seg, x="Segment", y=["Sales", "Profit"],
                         barmode="group", text_auto=".2s")
            fig.update_layout(height=380, margin=dict(t=20, b=20),
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width='stretch')
            st.dataframe(seg, width='stretch', hide_index=True)
            download_button(seg, "segment_performance.csv", "Download segment CSV")


elif page == "Sales Forecast":
    metrics = cached_model_metrics()
    forecast = cached_forecast()
    fi = cached_feature_importance()

    st.subheader("Model comparison")
    model_names = list(metrics["models"].keys())
    rows = []
    for name in model_names:
        m = metrics["models"][name]["metrics"]
        rows.append({"Model": name, **m})
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics.style.format({
        "MAE": "{:,.2f}", "RMSE": "{:,.2f}", "MAPE": "{:.2f}%", "R2": "{:.4f}"
    }), width='stretch', hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", metrics["best_model"])
    best_metrics = metrics["models"][metrics["best_model"]]["metrics"]
    c2.metric("Best RMSE", f"{best_metrics['RMSE']:,.2f}")
    c3.metric("Best MAPE", f"{best_metrics['MAPE']:.2f}%")

    st.caption("Selection rule: lowest RMSE, tie broken by lowest MAE. "
               "Metrics come from a real expanding-window time-series split "
               "(last 4 months held out). Nothing is fabricated.")

    # Actual vs predicted
    st.subheader("Actual vs predicted on held-out test set")
    test_idx = metrics["test_index"]
    test_actual = metrics["test_actual"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_idx, y=test_actual,
                             name="Actual", mode="lines+markers",
                             line=dict(color="#4e79a7", width=3)))
    for name in model_names:
        preds = metrics["models"][name]["predictions"]
        fig.add_trace(go.Scatter(x=test_idx, y=preds,
                                 name=name, mode="lines+markers",
                                 line=dict(width=2, dash="dot")))
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=20, b=20),
                      xaxis_title="Month", yaxis_title="Monthly Sales ($)")
    st.plotly_chart(fig, width='stretch')

    # Metric bars
    st.subheader("Error metric comparison (lower is better)")
    fig2 = go.Figure()
    for metric_name in ["MAE", "RMSE"]:
        fig2.add_trace(go.Bar(
            x=model_names, y=[metrics["models"][n]["metrics"][metric_name] for n in model_names],
            name=metric_name,
        ))
    fig2.update_layout(barmode="group", height=340,
                       legend=dict(orientation="h", y=-0.2),
                       margin=dict(t=20, b=20))
    st.plotly_chart(fig2, width='stretch')

    # 3-month forecast
    st.subheader("3-month forward forecast")
    if forecast.empty:
        st.info("Forecast could not be produced - not enough history.")
    else:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            hist = monthly_sales_trend(df_view)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hist["YearMonth"], y=hist["Sales"],
                                 name="Historical Sales", marker_color="#4e79a7"))
            fig.add_trace(go.Scatter(
                x=forecast["YearMonth"], y=forecast["Forecast"],
                name="Forecast", mode="lines+markers",
                line=dict(color="#e15759", width=3, dash="dot"),
                marker=dict(size=10),
            ))
            fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2),
                              margin=dict(t=20, b=20),
                              xaxis_title="Month", yaxis_title="Sales ($)")
            st.plotly_chart(fig, width='stretch')
        with c2:
            st.dataframe(forecast, width='stretch', hide_index=True)
        download_button(forecast, "forecast.csv", "Download forecast CSV")

    # Feature importance
    st.subheader("Feature importance / coefficients")
    if fi.empty:
        st.info("No feature-importance artefact found. Run train_model.py first.")
    else:
        best_fi = fi[fi["model"] == metrics["best_model"]].copy()
        if best_fi.empty:
            best_fi = fi
        best_fi = best_fi.sort_values("value", key=abs, ascending=False)
        fig = px.bar(best_fi, x="value", y="feature", orientation="h",
                     color="kind", text_auto=".3f")
        fig.update_layout(height=420, yaxis=dict(autorange="reversed"),
                          margin=dict(t=20, b=20),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width='stretch')
        st.dataframe(best_fi, width='stretch', hide_index=True)
        download_button(fi, "feature_importance_all.csv", "Download full feature table")


elif page == "Predictive Alerts":
    alerts = cached_alerts()
    s = alerts["summary"]
    c1, c2, c3, c4 = st.columns(4)
    kpi_card("Total Alerts", str(s["total_alerts"]), None)
    kpi_card("High Severity", str(s["high_severity"]), None)
    kpi_card("Medium Severity", str(s["medium_severity"]), None)
    kpi_card("Alert Types", str(len(s["by_type"])), None)
    st.write("")

    st.subheader("Alert types")
    if not alerts["declining_sales"]:
        st.success("No declining-month alerts for the current filters.")
    else:
        with st.expander(f"Declining sales ({len(alerts['declining_sales'])})", expanded=True):
            df_d = pd.DataFrame(alerts["declining_sales"])
            st.dataframe(df_d[["month", "sales", "pct_change_vs_prev_month",
                               "severity", "message"]],
                         width='stretch', hide_index=True)

    if alerts["loss_making_subcategories"]:
        with st.expander(f"Loss-making sub-categories ({len(alerts['loss_making_subcategories'])})", expanded=True):
            df_l = pd.DataFrame(alerts["loss_making_subcategories"])
            st.dataframe(df_l[["sub_category", "category", "total_sales",
                               "total_profit", "severity", "message"]],
                         width='stretch', hide_index=True)

    if alerts["unusual_trends"]:
        with st.expander(f"Unusual trends ({len(alerts['unusual_trends'])})", expanded=True):
            df_u = pd.DataFrame(alerts["unusual_trends"])
            st.dataframe(df_u[["month", "sales", "rolling_mean_6m",
                               "z_score", "severity", "message"]],
                         width='stretch', hide_index=True)

    if alerts["low_profit_categories"]:
        with st.expander(f"Low-profit categories ({len(alerts['low_profit_categories'])})", expanded=True):
            df_lp = pd.DataFrame(alerts["low_profit_categories"])
            st.dataframe(df_lp[["category", "total_sales", "total_profit",
                                "margin_pct", "severity", "message"]],
                         width='stretch', hide_index=True)

    st.subheader("Unusual-trend visualisation")
    monthly = monthly_sales_trend(df_view).copy()
    if len(monthly) >= 6:
        monthly["roll_mean_6"] = monthly["Sales"].rolling(6, min_periods=3).mean()
        monthly["roll_std_6"] = monthly["Sales"].rolling(6, min_periods=3).std()
        monthly["upper"] = monthly["roll_mean_6"] + 2 * monthly["roll_std_6"]
        monthly["lower"] = monthly["roll_mean_6"] - 2 * monthly["roll_std_6"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["YearMonth"], y=monthly["Sales"],
                             name="Sales", marker_color="#4e79a7"))
        fig.add_trace(go.Scatter(x=monthly["YearMonth"], y=monthly["roll_mean_6"],
                                 name="6m rolling mean", line=dict(color="#f28e2b", width=2)))
        fig.add_trace(go.Scatter(
            x=list(monthly["YearMonth"]) + list(monthly["YearMonth"])[::-1],
            y=list(monthly["upper"]) + list(monthly["lower"])[::-1],
            fill="toself", fillcolor="rgba(225,87,89,0.15)",
            line=dict(color="rgba(225,87,89,0)"),
            name="+/- 2σ band",
        ))
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=20, b=20),
                          xaxis_title="Month", yaxis_title="Sales ($)")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Not enough monthly data for the rolling-band visualisation.")

    all_alerts_df = pd.DataFrame(alerts["all_alerts"])
    download_button(all_alerts_df, "alerts.csv", "Download all alerts CSV")


elif page == "Business Recommendations":
    recs = cached_recommendations()
    s = recs["summary"]
    c1, c2, c3 = st.columns(3)
    kpi_card("Total Recommendations", str(s["total_recommendations"]), None)
    kpi_card("High Priority", str(s["high"]), None)
    kpi_card("Medium Priority", str(s["medium"]), None)
    st.write("")

    def render_block(title, items):
        if not items:
            return
        st.subheader(title)
        for r in items:
            sev = r.get("priority", "MEDIUM")
            color = {"HIGH": "#e15759", "MEDIUM": "#f28e2b", "LOW": "#59a14f"}[sev]
            st.markdown(
                f"""
                <div style="border-left:5px solid {color};background:#1e2227;
                            padding:14px 18px;border-radius:6px;margin-bottom:12px;">
                  <div style="color:#fafafa;font-size:15px;font-weight:600;">
                    {r.get('action', r.get('message', ''))}
                  </div>
                  <div style="color:#a4b0be;font-size:13px;margin-top:6px;">
                    Evidence: {r.get('evidence', '')}
                  </div>
                  <div style="color:#7bed9f;font-size:12px;margin-top:6px;">
                    Expected impact: {r.get('expected_impact', '')}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_block("Profitable categories to expand", recs["profitable_categories"])
    render_block("Weak regions to fix", recs["weak_regions"])
    render_block("Loss-making sub-category fixes", recs["loss_making_fixes"])
    render_block("Sales-growth actions", recs["sales_growth"])

    st.subheader("All recommendations")
    df_recs = pd.DataFrame(recs["all_recommendations"])
    st.dataframe(df_recs, width='stretch', hide_index=True)
    download_button(df_recs, "recommendations.csv", "Download recommendations CSV")
