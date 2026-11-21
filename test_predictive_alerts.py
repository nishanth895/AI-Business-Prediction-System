"""MODULE 8 - Tests for predictive alerts (Module 4)."""
from src.data_loader import load_and_clean
from src.predictive_alerts import (
    detect_anomaly_months,
    detect_declining_months,
    detect_loss_categories,
    detect_low_profit_categories,
    generate_alerts,
)


def test_generate_alerts_returns_summary_and_lists():
    df, _ = load_and_clean()
    alerts = generate_alerts(df)
    assert "summary" in alerts
    assert "declining_sales" in alerts
    assert "loss_making_subcategories" in alerts
    assert "unusual_trends" in alerts
    assert "low_profit_categories" in alerts
    assert alerts["summary"]["total_alerts"] == len(alerts["all_alerts"])


def test_detect_loss_categories_returns_negative_profit_only():
    df, _ = load_and_clean()
    losses = detect_loss_categories(df)
    for a in losses:
        assert a["total_profit"] < 0
        assert a["type"] == "LOSS_MAKING_SUBCATEGORY"


def test_detect_declining_months_negative_pct_change():
    df, _ = load_and_clean()
    declines = detect_declining_months(df)
    for a in declines:
        assert a["pct_change_vs_prev_month"] < 0
        assert a["type"] == "DECLINING_SALES"


def test_detect_anomaly_months_has_zscore():
    df, _ = load_and_clean()
    anomalies = detect_anomaly_months(df)
    for a in anomalies:
        assert "z_score" in a
        assert abs(a["z_score"]) >= 2.0
        assert a["type"] == "UNUSUAL_TREND"


def test_detect_low_profit_categories_threshold():
    df, _ = load_and_clean()
    cats = detect_low_profit_categories(df, min_margin=5.0)
    for c in cats:
        assert c["margin_pct"] < 5.0


def test_alert_severity_values_valid():
    df, _ = load_and_clean()
    alerts = generate_alerts(df)
    for a in alerts["all_alerts"]:
        assert a["severity"] in {"HIGH", "MEDIUM", "LOW"}
        assert "message" in a
        assert "type" in a
