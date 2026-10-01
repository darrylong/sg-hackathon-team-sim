"""The "address book" for Problem 05: every other script imports paths, constants and loaders from here.

Usage in another script (in the same src/ folder):
    from config import load_sales, REGIONS, RANDOM_STATE

Run this file directly to check that everything it points to exists:
    .venv/bin/python submission_template_vip/05_channel_analytics/src/config.py
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project root and paths
# ---------------------------------------------------------------------------
# This file is <root>/submission_template_vip/05_channel_analytics/src/config.py,
# so 3 folders up is the hpe-hackathon folder - works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Input data (read-only - never write into SG_Hackathon_Pack/).
PACK_DIR = PROJECT_ROOT / "SG_Hackathon_Pack" / "05_channel_analytics"
SALES_CSV = PACK_DIR / "data" / "partner_sales.csv"
MASTER_CSV = PACK_DIR / "data" / "partner_master.csv"
TARGETS_CSV = PACK_DIR / "data" / "targets.csv"
SAMPLE_AT_RISK_CSV = PACK_DIR / "artifacts" / "sample_submission_at_risk.csv"
SAMPLE_FORECAST_CSV = PACK_DIR / "artifacts" / "sample_submission_forecast.csv"

# Our outputs. OUTPUT_DIR is found from this file's own location (src/ -> 05_channel_analytics/),
# so it keeps working if the team folder is renamed from "submission_template_vip".
OUTPUT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = OUTPUT_DIR / "src" / "reports"
MODELS_DIR = OUTPUT_DIR / "src" / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIONS = ["APJ", "EMEA", "AMS"]
FORECAST_QUARTER = "2026Q3"  # the quarter we predict (Jul-Sep 2026)
LAST_QUARTER = "2026Q2"      # the last quarter we have data for
RANDOM_STATE = 42

# Exact column order the graders expect, read from their sample files.
AT_RISK_COLUMNS = list(pd.read_csv(SAMPLE_AT_RISK_CSV, nrows=0).columns)
FORECAST_COLUMNS = list(pd.read_csv(SAMPLE_FORECAST_CSV, nrows=0).columns)

# An order line this many times bigger than the 99.99th percentile is treated as an outlier.
OUTLIER_MULTIPLIER = 10
OUTLIER_PERCENTILE = 0.9999


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def to_quarter(dates):
    """Turn dates into quarter labels like "2026Q2"."""
    return pd.to_datetime(dates).dt.to_period("Q").astype(str)


def outlier_threshold(revenue):
    """Revenue above which ONE order line is treated as an outlier.

    WHY computed, not hard-coded: the README warns about one extreme line (~$37.8M)
    that would distort trends and forecasts. 10x the 99.99th percentile sits far
    above every normal large deal but well below that line - and adapts if data changes.
    """
    return OUTLIER_MULTIPLIER * revenue.quantile(OUTLIER_PERCENTILE)


def load_sales(exclude_outliers=True, verbose=True):
    """Load order lines with parsed dates and a quarter column.

    exclude_outliers=True  -> drop lines above outlier_threshold() (use for trends/forecasts)
    exclude_outliers=False -> keep them, flagged in an "is_outlier" column
    """
    sales = pd.read_csv(SALES_CSV, parse_dates=["order_date"])
    sales["quarter"] = to_quarter(sales["order_date"])
    threshold = outlier_threshold(sales["revenue_usd"])
    sales["is_outlier"] = sales["revenue_usd"] > threshold

    outliers = sales[sales["is_outlier"]]
    if verbose and len(outliers):
        action = "Excluded" if exclude_outliers else "Flagged (kept)"
        print(f"{action} {len(outliers)} outlier line(s) above ${threshold:,.0f}:")
        for row in outliers.itertuples():
            print(f"  {row.order_date:%Y-%m-%d}  {row.partner_id}  {row.region}  "
                  f"{row.product_family}  ${row.revenue_usd:,.0f}")
    if exclude_outliers:
        sales = sales[~sales["is_outlier"]].copy()
    return sales


def load_master():
    """Partner master with parsed onboarded_date and the onboarding quarter."""
    master = pd.read_csv(MASTER_CSV, parse_dates=["onboarded_date"])
    master["onboarded_quarter"] = to_quarter(master["onboarded_date"])
    return master


def load_targets():
    """Quarterly targets, with "2024-Q1" rewritten as "2024Q1" to match the sales quarter column."""
    targets = pd.read_csv(TARGETS_CSV)
    targets["quarter"] = targets["quarter"].str.replace("-", "", regex=False)
    return targets


if __name__ == "__main__":
    print("Paths:")
    paths = {
        "PROJECT_ROOT": PROJECT_ROOT, "SALES_CSV": SALES_CSV, "MASTER_CSV": MASTER_CSV,
        "TARGETS_CSV": TARGETS_CSV, "SAMPLE_AT_RISK_CSV": SAMPLE_AT_RISK_CSV,
        "SAMPLE_FORECAST_CSV": SAMPLE_FORECAST_CSV, "OUTPUT_DIR": OUTPUT_DIR,
        "REPORTS_DIR": REPORTS_DIR, "MODELS_DIR": MODELS_DIR,
    }
    for name, path in paths.items():
        shown = path if path == PROJECT_ROOT else path.relative_to(PROJECT_ROOT)
        print(f"  {'✅' if path.exists() else '❌'} {name:<20} {shown}")

    print(f"\nREGIONS: {REGIONS}   FORECAST_QUARTER: {FORECAST_QUARTER}   LAST_QUARTER: {LAST_QUARTER}")
    print(f"AT_RISK_COLUMNS: {AT_RISK_COLUMNS}")
    print(f"FORECAST_COLUMNS: {FORECAST_COLUMNS}")
    print(f"RANDOM_STATE: {RANDOM_STATE}\n")

    raw = pd.read_csv(SALES_CSV, usecols=["revenue_usd"])["revenue_usd"]
    print(f"Outlier threshold: {OUTLIER_MULTIPLIER} x 99.99th percentile "
          f"(${raw.quantile(OUTLIER_PERCENTILE):,.0f}) = ${outlier_threshold(raw):,.0f}")
    sales = load_sales(exclude_outliers=True)
    print(f"Kept {len(sales):,} of {len(raw):,} order lines; "
          f"quarters {sales['quarter'].min()} .. {sales['quarter'].max()}")
