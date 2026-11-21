"""MODULE 8 - Tests for the dataset loader (Module 1)."""
import pandas as pd
import pytest

from src.data_loader import (
    clean_dataset,
    convert_dates,
    data_summary,
    handle_missing_values,
    load_and_clean,
    load_dataset,
    validate_columns,
)


def test_load_dataset_returns_dataframe():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_required_columns_present():
    df = load_dataset()
    # Should NOT raise
    validate_columns(df)


def test_validate_columns_raises_on_missing():
    df = pd.DataFrame({"Sales": [1, 2]})
    with pytest.raises(ValueError):
        validate_columns(df)


def test_clean_dataset_drops_nulls_and_parses_dates():
    df = load_dataset()
    cleaned, drop_report = clean_dataset(df)
    assert len(cleaned) <= len(df)
    assert cleaned["Order Date"].dtype.kind == "M"  # datetime64
    assert cleaned["Sales"].dtype.kind == "f"
    assert cleaned["Profit"].dtype.kind == "f"
    # Critical columns must have no nulls after cleaning
    for col in ["Order Date", "Region", "Category", "Sub-Category", "Sales", "Profit"]:
        assert cleaned[col].isna().sum() == 0, f"Column {col} still has nulls"


def test_convert_dates_with_us_format():
    df = pd.DataFrame({"Order Date": ["1/16/2017", "4/15/2018", "12/31/2018"],
                       "Ship Date": ["1/20/2017", "4/20/2018", "1/3/2019"]})
    out = convert_dates(df)
    assert out["Order Date"].dtype.kind == "M"
    assert out["Ship Date"].dtype.kind == "M"
    assert out["Order Date"].iloc[0] == pd.Timestamp("2017-01-16")


def test_handle_missing_values_returns_drop_report():
    df = load_dataset()
    cleaned, report = handle_missing_values(df)
    assert "__rows_dropped__" in report
    assert report["__rows_dropped__"] >= 0


def test_data_summary_keys():
    df, _ = load_and_clean()
    summary = data_summary(df)
    assert "rows" in summary
    assert "columns" in summary
    assert "date_range" in summary
    assert summary["rows"] > 0


def test_load_and_clean_pipeline():
    df, summary = load_and_clean()
    assert summary["raw_rows"] >= summary["rows"]
    assert summary["rows"] > 1000  # real dataset should have thousands of rows
