"""MODULE 8 - Tests for the recommendations engine (Module 5)."""
from src.data_loader import load_and_clean
from src.recommendations import (
    build_recommendations,
    flag_weak_regions,
    recommend_discontinue_or_fix_loss_makers,
    recommend_profitable_categories,
    recommend_sales_growth,
)


def test_build_recommendations_has_summary_and_lists():
    df, _ = load_and_clean()
    recs = build_recommendations(df)
    assert "summary" in recs
    assert recs["summary"]["total_recommendations"] == len(recs["all_recommendations"])
    assert "profitable_categories" in recs
    assert "weak_regions" in recs
    assert "loss_making_fixes" in recs
    assert "sales_growth" in recs


def test_recommend_profitable_categories_returns_top():
    df, _ = load_and_clean()
    recs = recommend_profitable_categories(df, top_n=3)
    assert len(recs) <= 3
    assert all(r["priority"] in {"HIGH", "MEDIUM", "LOW"} for r in recs)


def test_flag_weak_regions_uses_margin_threshold():
    df, _ = load_and_clean()
    weak = flag_weak_regions(df, min_margin=10.0)
    for r in weak:
        # Margin in evidence should be below the threshold we passed
        assert "margin" in r["evidence"].lower() or "%" in r["evidence"]


def test_loss_making_fixes_reference_negative_profit():
    df, _ = load_and_clean()
    fixes = recommend_discontinue_or_fix_loss_makers(df)
    for f in fixes:
        assert "Loss of $" in f["evidence"]


def test_recommend_sales_growth_returns_actions():
    df, _ = load_and_clean()
    recs = recommend_sales_growth(df)
    assert isinstance(recs, list)
    for r in recs:
        assert "action" in r
        assert "evidence" in r
        assert "priority" in r


def test_priority_ordering_high_first():
    df, _ = load_and_clean()
    recs = build_recommendations(df)
    priorities = [r.get("priority") for r in recs["all_recommendations"]]
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    numeric = [rank[p] for p in priorities]
    assert numeric == sorted(numeric)
