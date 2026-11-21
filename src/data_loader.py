"""MODULE 1 - Dataset Management.

Responsibilities:
  * Load the CSV dataset from ``data/superstore.csv``.
  * Validate that all required columns are present.
  * Handle missing values.
  * Convert ``Order Date`` / ``Ship Date`` into proper ``datetime`` objects.
  * Provide helpers that expose rows, columns and a data summary.

The functions in this module never fabricate data. If a required column is
missing they raise ``ValueError`` so the caller can surface a clear message.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import REQUIRED_COLUMNS, DATA_PATH


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame.

    Parameters
    ----------
    path:
        Optional path to a CSV file. Defaults to ``data/superstore.csv``.

    Returns
    -------
    pandas.DataFrame
        Raw, unmodified frame (only basic column-name stripping applied).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Place a real retail sales CSV "
            "there (see DATASET.md)."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> None:
    """Ensure all required columns exist in ``df``.

    Raises ``ValueError`` listing the missing columns if any are absent.
    """
    required = required or REQUIRED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop rows that miss critical analytical fields.

    A row is dropped only when one of the analytically-critical columns
    (Order Date, Region, Category, Sub-Category, Sales, Profit, Quantity) is
    NaN. Numeric columns are kept as-is otherwise (no imputation, because we
    must not fabricate values).

    Returns the cleaned DataFrame and a dict describing how many rows were
    dropped per column.
    """
    critical = [
        "Order ID", "Order Date", "Region", "Category", "Sub-Category",
        "Sales", "Quantity", "Profit",
    ]
    before = len(df)
    per_col = {c: int(df[c].isna().sum()) for c in critical if c in df.columns}
    cleaned = df.dropna(subset=[c for c in critical if c in df.columns]).copy()
    per_col["__rows_dropped__"] = before - len(cleaned)
    return cleaned, per_col


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``Order Date`` and ``Ship Date`` to proper datetime dtype.

    The Sample Superstore dataset uses US-style ``MM/DD/YYYY`` dates, so we
    explicitly pass ``dayfirst=False`` (pandas' default) to avoid ambiguity.
    """
    for col in ("Order Date", "Ship Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=False, errors="coerce")
    # Drop any rows where order date could not be parsed
    if "Order Date" in df.columns:
        df = df[df["Order Date"].notna()].copy()
    return df


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Run the full cleaning pipeline and return the clean frame + report.

    Steps:
        1. Validate required columns.
        2. Strip whitespace from text columns.
        3. Coerce numeric columns to numeric dtype.
        4. Drop rows with missing critical fields.
        5. Convert date columns.
    """
    validate_columns(df)
    text_cols = df.select_dtypes(include="object").columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    for c in ("Sales", "Quantity", "Discount", "Profit"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df, drop_report = handle_missing_values(df)
    df = convert_dates(df)
    df = df.reset_index(drop=True)
    return df, drop_report


def data_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Return a human-readable summary of the dataset."""
    summary: Dict[str, object] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "date_range": {
            "start": str(df["Order Date"].min()) if "Order Date" in df else None,
            "end": str(df["Order Date"].max()) if "Order Date" in df else None,
        },
    }
    if "Sales" in df.columns:
        summary["sales_total"] = float(df["Sales"].sum())
        summary["sales_mean"] = float(df["Sales"].mean())
    if "Profit" in df.columns:
        summary["profit_total"] = float(df["Profit"].sum())
    return summary


def load_and_clean(path: str | Path = DATA_PATH) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """One-shot helper: load, clean and summarise the dataset."""
    raw = load_dataset(path)
    cleaned, drop_report = clean_dataset(raw)
    summary = data_summary(cleaned)
    summary["drop_report"] = drop_report
    summary["raw_rows"] = int(len(raw))
    return cleaned, summary
