"""Build the partner x quarter panel that the health model and dashboard use.

WHY a panel: the health score compares each partner with its OWN history
(trend, gaps, attainment). That needs one tidy row per partner per quarter,
including the quarters with zero revenue (those zeros are the signal!).

Run:
    .venv/bin/python submission_template_vip/05_channel_analytics/src/data_prep.py
Writes src/reports/panel.csv and src/reports/monthly_revenue.csv
"""

import pandas as pd

from config import LAST_QUARTER, REPORTS_DIR, load_master, load_sales, load_targets

DATA_START = pd.Timestamp("2024-01-01")
FIRST_QUARTER = "2024Q1"
MASTER_FIELDS = ["region", "tier", "partner_type", "country", "onboarded_date"]


def quarter_end(quarter):
    """Last day of a quarter label, e.g. "2026Q2" -> 2026-06-30."""
    return pd.Period(quarter, freq="Q").end_time.normalize()


def all_quarters():
    return [str(q) for q in pd.period_range(FIRST_QUARTER, LAST_QUARTER, freq="Q")]


def panel_skeleton(master):
    """One row per partner per quarter, from its onboarding quarter (or 2024Q1) to the last quarter."""
    quarters = all_quarters()
    rows = [(pid, q) for pid, start in zip(master["partner_id"], master["onboarded_quarter"].clip(lower=FIRST_QUARTER))
            for q in quarters if q >= start]
    return pd.DataFrame(rows, columns=["partner_id", "quarter"])


def quarterly_sales_stats(sales):
    """Revenue, orders, activity and mix per partner-quarter (only quarters with orders)."""
    sales = sales.assign(month=sales["order_date"].dt.to_period("M"),
                         margin_x_rev=sales["margin_pct"] * sales["revenue_usd"])
    grouped = sales.groupby(["partner_id", "quarter"])
    stats = grouped.agg(
        revenue=("revenue_usd", "sum"),
        n_orders=("revenue_usd", "size"),        # order lines
        n_active_months=("month", "nunique"),
        n_product_families=("product_family", "nunique"),
        margin_x_rev=("margin_x_rev", "sum"),
        last_order_date=("order_date", "max"),
    ).reset_index()
    # Revenue-weighted margin: a $500k deal at 10% matters more than a $5k deal at 25%.
    stats["margin_pct"] = stats["margin_x_rev"] / stats["revenue"]
    return stats.drop(columns="margin_x_rev")


def add_days_since_last_order(panel, master):
    """Days from the last order (up to that quarter's end) to the quarter end.

    The last order date is carried forward through empty quarters. If a partner has
    no order in the data yet, we count from its onboarding date (or 2024-01-01,
    the start of our data, for older partners - a lower bound).
    """
    panel = panel.sort_values(["partner_id", "quarter"])
    last = panel.groupby("partner_id")["last_order_date"].ffill()
    start = panel["partner_id"].map(master.set_index("partner_id")["onboarded_date"]).clip(lower=DATA_START)
    ends = panel["quarter"].map({q: quarter_end(q) for q in panel["quarter"].unique()})
    panel["days_since_last_order"] = (ends - last.fillna(start)).dt.days.clip(lower=0)
    return panel.drop(columns="last_order_date")


def build_panel(sales=None, master=None, targets=None):
    """The full partner x quarter panel (see module docstring)."""
    sales = load_sales(exclude_outliers=True) if sales is None else sales
    master = load_master() if master is None else master
    targets = load_targets() if targets is None else targets

    panel = panel_skeleton(master).merge(quarterly_sales_stats(sales), on=["partner_id", "quarter"], how="left")
    # No orders in a quarter = real zeros (not missing data).
    for column in ["revenue", "n_orders", "n_active_months", "n_product_families"]:
        panel[column] = panel[column].fillna(0)

    panel = panel.merge(targets, on=["partner_id", "quarter"], how="left")
    panel["attainment"] = panel["revenue"] / panel["target_usd"]
    panel = add_days_since_last_order(panel, master)

    master_idx = master.set_index("partner_id")
    onboarded_q = panel["partner_id"].map(master_idx["onboarded_quarter"])
    panel["tenure_quarters"] = [
        (pd.Period(q, freq="Q") - pd.Period(o, freq="Q")).n for q, o in zip(panel["quarter"], onboarded_q)
    ]
    for field in MASTER_FIELDS:
        panel[field] = panel["partner_id"].map(master_idx[field])
    return panel.reset_index(drop=True)


def build_monthly(sales=None, master=None):
    """Monthly revenue per partner (partner x month, zeros filled) for short-term trends."""
    sales = load_sales(exclude_outliers=True, verbose=False) if sales is None else sales
    master = load_master() if master is None else master
    months = pd.period_range(DATA_START, quarter_end(LAST_QUARTER), freq="M")
    monthly = (sales.assign(month=sales["order_date"].dt.to_period("M"))
               .pivot_table(index="partner_id", columns="month", values="revenue_usd", aggfunc="sum", fill_value=0))
    monthly = monthly.reindex(index=master["partner_id"], columns=months, fill_value=0)
    monthly.columns = monthly.columns.astype(str)
    return monthly


if __name__ == "__main__":
    print("Building partner x quarter panel...")
    sales = load_sales(exclude_outliers=True)
    master = load_master()
    panel = build_panel(sales, master)
    panel.to_csv(REPORTS_DIR / "panel.csv", index=False)
    print(f"  panel shape: {panel.shape} ({panel['partner_id'].nunique():,} partners, "
          f"{panel['quarter'].nunique()} quarters) -> panel.csv")
    print(f"  zero-revenue partner-quarters: {(panel['revenue'] == 0).sum():,}")

    monthly = build_monthly(sales, master)
    monthly.to_csv(REPORTS_DIR / "monthly_revenue.csv")
    print(f"  monthly shape: {monthly.shape} -> monthly_revenue.csv")

    print("\n5 example rows:")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(panel.sample(5, random_state=42).to_string(index=False))
