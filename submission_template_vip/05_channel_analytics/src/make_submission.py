"""Write and validate the two Problem 05 submission files.

Inputs (run these first):  health.py -> reports/partner_health.csv
                           forecast.py -> reports/forecast.csv
Outputs (in OUTPUT_DIR):   submission_at_risk.csv, submission_forecast.csv

Run:
    .venv/bin/python submission_template_vip/05_channel_analytics/src/make_submission.py
"""

import sys

import pandas as pd

from config import (
    AT_RISK_COLUMNS,
    FORECAST_COLUMNS,
    OUTPUT_DIR,
    REGIONS,
    REPORTS_DIR,
    SAMPLE_AT_RISK_CSV,
    SAMPLE_FORECAST_CSV,
    load_master,
)

AT_RISK_PATH = OUTPUT_DIR / "submission_at_risk.csv"
FORECAST_PATH = OUTPUT_DIR / "submission_forecast.csv"
REGION_ORDER = REGIONS + ["ALL"]
SUM_TOLERANCE = 0.005  # ALL may differ from the sum of regions by at most 0.5%


def build_at_risk(master):
    """One row per master partner, in master order, with exactly the graded columns."""
    health = pd.read_csv(REPORTS_DIR / "partner_health.csv")
    sub = master[["partner_id"]].merge(health[["partner_id", "health_score", "at_risk"]],
                                       on="partner_id", how="left")
    sub["at_risk"] = sub["at_risk"].astype("Int64")  # stays integer; missing would show up in validation
    return sub[AT_RISK_COLUMNS]


def build_forecast():
    """Exactly 4 rows (APJ, EMEA, AMS, ALL) with the graded column names."""
    forecast = pd.read_csv(REPORTS_DIR / "forecast.csv").set_index("region").reindex(REGION_ORDER)
    sub = pd.DataFrame({
        "region": REGION_ORDER,
        "forecast_revenue_usd": forecast["forecast"].round(0).to_numpy(),
        "lo80_usd": forecast["lo80"].round(0).to_numpy(),
        "hi80_usd": forecast["hi80"].round(0).to_numpy(),
    })
    return sub[FORECAST_COLUMNS]


def check(results, name, passed):
    print(f"  {'✅' if passed else '❌'} {name}")
    results.append(bool(passed))


def validate_at_risk(master):
    print(f"\nValidating {AT_RISK_PATH.name}:")
    sub = pd.read_csv(AT_RISK_PATH)
    sample_columns = list(pd.read_csv(SAMPLE_AT_RISK_CSV, nrows=0).columns)
    ids, master_ids = sub["partner_id"], set(master["partner_id"])
    results = []
    check(results, "exact columns and order (same as sample file)", list(sub.columns) == sample_columns)
    check(results, f"one row per master partner ({len(sub):,} rows vs {len(master_ids):,})", len(sub) == len(master_ids))
    check(results, "every master partner_id present", master_ids <= set(ids))
    check(results, "no unknown partner_ids", set(ids) <= master_ids)
    check(results, "no duplicate partner_ids", not ids.duplicated().any())
    check(results, "no missing values", not sub.isna().any().any())
    check(results, "at_risk is 0 or 1", sub["at_risk"].isin([0, 1]).all())
    check(results, "health_score is numeric", pd.api.types.is_numeric_dtype(sub["health_score"]))
    print(f"     ({int(sub['at_risk'].sum())} partners flagged, {sub['at_risk'].mean():.2%})")
    return all(results)


def validate_forecast():
    print(f"\nValidating {FORECAST_PATH.name}:")
    sub = pd.read_csv(FORECAST_PATH)
    sample_columns = list(pd.read_csv(SAMPLE_FORECAST_CSV, nrows=0).columns)
    numbers = sub[["forecast_revenue_usd", "lo80_usd", "hi80_usd"]]
    results = []
    check(results, "exact columns and order (same as sample file)", list(sub.columns) == sample_columns)
    check(results, f"exactly the 4 regions in order {REGION_ORDER}", list(sub["region"]) == REGION_ORDER)
    check(results, "no missing values", not sub.isna().any().any())
    check(results, "lo80 <= forecast <= hi80 on every row",
          ((sub["lo80_usd"] <= sub["forecast_revenue_usd"]) & (sub["forecast_revenue_usd"] <= sub["hi80_usd"])).all())
    check(results, "all numbers positive", (numbers > 0).all().all())
    regions_sum = sub.loc[sub["region"].isin(REGIONS), "forecast_revenue_usd"].sum()
    all_value = sub.loc[sub["region"] == "ALL", "forecast_revenue_usd"].iloc[0]
    gap = abs(all_value / regions_sum - 1)
    check(results, f"ALL equals the sum of regions (difference {gap:.4%} <= {SUM_TOLERANCE:.1%})", gap <= SUM_TOLERANCE)
    print(sub.to_string(index=False))
    return all(results)


def main():
    master = load_master()

    print("Writing submissions...")
    build_at_risk(master).to_csv(AT_RISK_PATH, index=False, encoding="utf-8")
    print(f"  Wrote {AT_RISK_PATH.name}")
    build_forecast().to_csv(FORECAST_PATH, index=False, encoding="utf-8")
    print(f"  Wrote {FORECAST_PATH.name}")

    at_risk_ok = validate_at_risk(master)
    forecast_ok = validate_forecast()
    if not (at_risk_ok and forecast_ok):
        sys.exit("\n❌ Validation FAILED - fix the ❌ items above before handing in.")
    print("\n✅ Both Problem 05 submission files pass every check.")


if __name__ == "__main__":
    main()
