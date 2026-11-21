# Dataset

This project uses the **Sample Superstore** retail sales dataset — a real,
publicly available, widely used dataset for retail analytics.

## Source

The dataset is bundled with this project at `data/superstore.csv`
(10,800 raw rows, 9,994 valid after cleaning, 2.3 MB).

Public mirror used:

```
https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv
```

You can re-download it with:

```bash
# Linux / macOS
curl -L -o data/superstore.csv \
  https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv

# Windows PowerShell
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv" `
  -OutFile "data\superstore.csv"
```

## Schema

| Column           | Type      | Description                                  |
|------------------|-----------|----------------------------------------------|
| Row ID           | int       | Sequential row identifier                    |
| Order ID         | string    | Unique order identifier                      |
| Order Date       | date      | Date the order was placed (US MM/DD/YYYY)    |
| Ship Date        | date      | Date the order shipped                       |
| Ship Mode        | string    | Shipping class (Second / Standard / First)   |
| Customer ID      | string    | Customer identifier                          |
| Customer Name    | string    | Customer name                                |
| Segment          | string    | Consumer / Corporate / Home Office           |
| Country          | string    | Always "United States"                       |
| City             | string    | Customer city                                |
| State            | string    | Customer state                               |
| Postal Code      | int       | Postal code                                  |
| Region           | string    | Central / East / South / West                |
| Product ID       | string    | Product identifier                           |
| Category         | string    | Furniture / Office Supplies / Technology     |
| Sub-Category     | string    | 17 sub-categories (Bookcases, Chairs, …)     |
| Product Name     | string    | Full product name                            |
| Sales            | float     | Order line sales amount (USD)                |
| Quantity         | int       | Units ordered                                |
| Discount         | float     | Discount applied (0.0–0.85)                  |
| Profit           | float     | Order line profit (USD, may be negative)     |

## Cleaning applied

`src/data_loader.py` performs the following on every load:

1. Strip whitespace from string columns.
2. Coerce `Sales`, `Quantity`, `Discount`, `Profit` to numeric.
3. Drop the 806 rows where critical analytical columns are null
   (Order Date, Region, Category, Sub-Category, Sales, Profit, Quantity).
4. Convert `Order Date` / `Ship Date` to `datetime64` using
   `dayfirst=False` (US MM/DD/YYYY).

After cleaning: **9,994 rows × 21 columns**, date range
**2015-01-03 → 2018-12-30**, 5,009 unique orders, 4 regions, 3 categories,
17 sub-categories.

## Why this dataset is appropriate

It contains every field the project requires:

- Order Date → monthly trend + time-series forecasting.
- Region / Category / Sub-Category → business analysis slices.
- Sales / Profit / Quantity / Discount → KPIs + recommendations.
- The Profit column legitimately contains negative values, so
  loss-making sub-category and low-profit-margin detection work on real
  losses (not fabricated data).

No values are imputed, no rows are duplicated, no synthetic records are
added. Every metric shown in the dashboard is computed from this file.
