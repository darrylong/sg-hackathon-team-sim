"""Exploratory data analysis (EDA) for Problem 05 - partner channel analytics.

WHY: before building a health score and a forecast we need to know how revenue
moves (seasonality, outliers), where it comes from (regions, tiers, top partners),
and what "leaving the channel" looked like in the past - because that history is
what our at-risk model will learn from.

Run:
    .venv/bin/python submission_template_vip/05_channel_analytics/src/explore.py

Outputs (in src/reports/eda/): one PNG per question + findings.txt
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files only - no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LAST_QUARTER, REGIONS, REPORTS_DIR, load_master, load_sales, load_targets

EDA_DIR = REPORTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# A partner's quarter is "token" if revenue is below this share of its own average quarter.
TOKEN_SHARE = 0.05
LAST_4_QUARTERS = ["2025Q3", "2025Q4", "2026Q1", "2026Q2"]

_LOG_LINES = []
SAVED_CHARTS = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(text=""):
    """Print a line AND keep it for findings.txt."""
    print(text)
    _LOG_LINES.append(str(text))


def section(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


def save_chart(fig, filename):
    fig.tight_layout()
    fig.savefig(EDA_DIR / filename, dpi=120)
    plt.close(fig)
    SAVED_CHARTS.append(filename)
    log(f"[chart saved] {filename}")


def millions(x):
    return f"${x / 1e6:,.1f}M"


def weighted_margin(df):
    """Revenue-weighted margin %: big deals count more than small ones."""
    return (df["revenue_usd"] * df["margin_pct"]).sum() / df["revenue_usd"].sum()


def quarter_offset(quarter, n):
    """Quarter label n quarters after (or before, if negative) the given one."""
    return str(pd.Period(quarter, freq="Q") + n)


# ---------------------------------------------------------------------------
# Partner x quarter tables, shared by sections 6-8
# ---------------------------------------------------------------------------
def partner_quarter_tables(sales, master):
    """Revenue and order-line counts per partner per quarter (0 when no orders),
    plus each partner's token threshold."""
    quarters = sorted(sales["quarter"].unique())
    revenue = sales.pivot_table(index="partner_id", columns="quarter", values="revenue_usd",
                                aggfunc="sum", fill_value=0)
    orders = sales.pivot_table(index="partner_id", columns="quarter", values="revenue_usd",
                               aggfunc="count", fill_value=0)
    # Every partner in the master gets a row, even with no sales at all.
    revenue = revenue.reindex(index=master["partner_id"], columns=quarters, fill_value=0)
    orders = orders.reindex(index=master["partner_id"], columns=quarters, fill_value=0)

    # Average only over the quarters the partner existed (from onboarding, or 2024Q1 if earlier),
    # so a partner that joined in 2026Q1 isn't judged against 10 quarters of zeros.
    onboarded = master.set_index("partner_id")["onboarded_quarter"].clip(lower=quarters[0])
    existed = pd.DataFrame({q: onboarded <= q for q in quarters})
    average = revenue.where(existed).mean(axis=1)
    token = TOKEN_SHARE * average
    return quarters, revenue, orders, token, existed


def leaver_flags(revenue, token, quarters):
    """For each quarter Q (from the 2nd on): who was active in Q-1 and left in Q.

    active  = revenue above the partner's token level
    leaver  = active in Q-1, then no or only token orders in Q
    """
    above = revenue.gt(token, axis=0)
    active_prev, left = {}, {}
    for prev_q, q in zip(quarters[:-1], quarters[1:]):
        active_prev[q] = above[prev_q]
        left[q] = above[prev_q] & ~above[q]
    return pd.DataFrame(active_prev), pd.DataFrame(left)


# ---------------------------------------------------------------------------
# 1. Joins and data quality
# ---------------------------------------------------------------------------
def check_joins(sales, master, targets):
    """WHY: every later table joins these files. A missing partner or a region
    mismatch would silently drop or misplace revenue."""
    section("1. JOINS & DATA QUALITY")
    unknown = set(sales["partner_id"]) - set(master["partner_id"])
    no_sales = set(master["partner_id"]) - set(sales["partner_id"])
    log(f"Partners in master: {len(master):,}; in sales: {sales['partner_id'].nunique():,}")
    log(f"Sales partner_ids missing from master: {len(unknown)}")
    log(f"Master partners with no sales at all: {len(no_sales)}")

    pairs = sales[["partner_id", "region", "tier"]].drop_duplicates()
    merged = pairs.merge(master[["partner_id", "region", "tier"]], on="partner_id", suffixes=("_sales", "_master"))
    log(f"Region mismatches (sales vs master): {(merged['region_sales'] != merged['region_master']).sum()}")
    log(f"Tier mismatches (sales vs master):   {(merged['tier_sales'] != merged['tier_master']).sum()}")

    # Targets should exist for every quarter from onboarding (or 2024Q1) to 2026Q2.
    quarters = sorted(sales["quarter"].unique())
    expected = [(p, q) for p, oq in zip(master["partner_id"], master["onboarded_quarter"])
                for q in quarters if q >= oq]
    have = set(zip(targets["partner_id"], targets["quarter"]))
    missing = len(set(expected) - have)
    extra = len(have - set(expected))
    log(f"Targets: {len(targets):,} rows for {targets['partner_id'].nunique():,} partners; "
        f"expected partner-quarters {len(expected):,}, missing {missing}, unexpected {extra}")
    return {"unknown": len(unknown), "no_sales": len(no_sales), "targets_missing": missing}


# ---------------------------------------------------------------------------
# 2. Quarterly revenue with and without the outlier
# ---------------------------------------------------------------------------
def revenue_trend(sales_all):
    """WHY: one ~$37.8M order line can make a quarter look like a boom. We plot both
    versions to show how much it would distort any trend or forecast."""
    section("2. QUARTERLY REVENUE BY REGION - with and without the outlier line")
    with_outlier = sales_all.pivot_table(index="quarter", columns="region", values="revenue_usd", aggfunc="sum")
    without = sales_all[~sales_all["is_outlier"]].pivot_table(
        index="quarter", columns="region", values="revenue_usd", aggfunc="sum")
    with_outlier, without = with_outlier[REGIONS], without[REGIONS]
    log("Revenue without the outlier ($M):")
    log((without / 1e6).round(2).to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    for region, color in zip(REGIONS, ["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(with_outlier.index, with_outlier[region] / 1e6, linestyle="--", color=color, alpha=0.5,
                label=f"{region} with outlier")
        ax.plot(without.index, without[region] / 1e6, marker="o", color=color, label=f"{region} without outlier")
    ax.set_ylabel("revenue ($M)")
    ax.set_title("Quarterly revenue by region - dashed lines include the one outlier order")
    ax.legend(ncol=2, fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    save_chart(fig, "02_quarterly_revenue_by_region.png")

    outlier = sales_all[sales_all["is_outlier"]].iloc[0]
    region, quarter = outlier["region"], outlier["quarter"]
    impact = with_outlier.loc[quarter, region] / without.loc[quarter, region] - 1
    log(f"\nOutlier: {millions(outlier['revenue_usd'])} on {outlier['order_date']:%Y-%m-%d} "
        f"({outlier['partner_id']}, {region}, {outlier['product_family']})")
    log(f"It inflates {region} {quarter} revenue by {impact:.1%} "
        f"({millions(without.loc[quarter, region])} -> {millions(with_outlier.loc[quarter, region])})")
    return {"region": region, "quarter": quarter, "impact": impact, "without": without}


# ---------------------------------------------------------------------------
# 3. Seasonality
# ---------------------------------------------------------------------------
def seasonality(quarterly):
    """WHY: if Q3 always dips after Q2, a naive "next quarter = last quarter"
    forecast for 2026Q3 would be too high. We measure the dip for each region."""
    section("3. SEASONALITY - quarter-on-quarter % change")
    table = quarterly.copy()
    table["ALL"] = table.sum(axis=1)
    rows = []
    for year in ["2024", "2025"]:
        for frm, to in [("Q2", "Q3"), ("Q3", "Q4")]:
            change = table.loc[f"{year}{to}"] / table.loc[f"{year}{frm}"] - 1
            rows.append(pd.Series(change * 100, name=f"{year} {frm}->{to}"))
    result = pd.DataFrame(rows).round(1)
    log("% change (outlier excluded):")
    log(result.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    result.T.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("% change")
    ax.set_title("Seasonality: Q2->Q3 dips, Q3->Q4 recovers")
    ax.tick_params(axis="x", rotation=0)
    save_chart(fig, "03_seasonality.png")
    return result


# ---------------------------------------------------------------------------
# 4. Performance by segment, last 4 quarters
# ---------------------------------------------------------------------------
def segment_performance(sales, master, targets):
    """WHY: tells us where revenue and margin come from, and which segments
    hit their targets - the dashboard's 'performance by partner, region, tier, product'."""
    section(f"4. PERFORMANCE BY SEGMENT - last 4 quarters ({LAST_4_QUARTERS[0]}..{LAST_4_QUARTERS[-1]})")
    recent = sales[sales["quarter"].isin(LAST_4_QUARTERS)].merge(
        master[["partner_id", "partner_type"]], on="partner_id")
    recent_targets = targets[targets["quarter"].isin(LAST_4_QUARTERS)].merge(
        master[["partner_id", "region", "tier", "partner_type"]], on="partner_id")

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, column in zip(axes.flat, ["region", "tier", "partner_type", "product_family"]):
        grouped = recent.groupby(column)
        table = pd.DataFrame({
            "revenue_$M": grouped["revenue_usd"].sum() / 1e6,
            "share_%": grouped["revenue_usd"].sum() / recent["revenue_usd"].sum() * 100,
            "margin_%": grouped.apply(weighted_margin, include_groups=False),
        })
        if column != "product_family":  # targets are per partner, not per product
            table["attainment_%"] = (grouped["revenue_usd"].sum()
                                     / recent_targets.groupby(column)["target_usd"].sum() * 100)
        table = table.sort_values("revenue_$M", ascending=False).round(1)
        log(f"\nby {column}:")
        log(table.to_string())
        if column == "product_family":
            log("  (no attainment: targets are set per partner, not per product)")
        results[column] = table

        table["revenue_$M"].plot.bar(ax=ax, color="steelblue")
        ax.set_title(f"Revenue by {column} (last 4 quarters)")
        ax.set_ylabel("$M")
        ax.tick_params(axis="x", rotation=0)
        if "attainment_%" in table:
            twin = ax.twinx()
            twin.plot(range(len(table)), table["attainment_%"], "o-", color="crimson")
            twin.set_ylabel("target attainment %", color="crimson")
    save_chart(fig, "04_segment_performance.png")
    return results


# ---------------------------------------------------------------------------
# 5. Concentration
# ---------------------------------------------------------------------------
def concentration(sales, master):
    """WHY: if a few partners make most of the revenue, losing one of them is a
    big risk - and the at-risk list should be read with revenue at stake in mind."""
    section(f"5. CONCENTRATION - last 4 quarters")
    recent = sales[sales["quarter"].isin(LAST_4_QUARTERS)]
    by_partner = recent.groupby(["region", "partner_id"])["revenue_usd"].sum()
    rows = []
    for region in REGIONS:
        region_rev = by_partner.loc[region].sort_values(ascending=False)
        rows.append({"region": region, "partners": len(region_rev),
                     "top10_share_%": round(region_rev.head(10).sum() / region_rev.sum() * 100, 1)})
    table = pd.DataFrame(rows).set_index("region")
    log(table.to_string())

    top10 = (recent.groupby("partner_id")["revenue_usd"].sum().nlargest(10).rename("revenue_usd")
             .reset_index().merge(master[["partner_id", "partner_name", "region", "tier", "partner_type"]]))
    top10["share_%"] = (top10["revenue_usd"] / recent["revenue_usd"].sum() * 100).round(2)
    top10["revenue_usd"] = top10["revenue_usd"].map(millions)
    log("\nTop 10 partners overall:")
    log(top10.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    table["top10_share_%"].plot.bar(ax=ax, color="darkorange")
    ax.set_ylabel("% of region revenue")
    ax.set_title("Share of region revenue from its top 10 partners")
    ax.tick_params(axis="x", rotation=0)
    save_chart(fig, "05_concentration.png")
    return table


# ---------------------------------------------------------------------------
# 6. Partner activity
# ---------------------------------------------------------------------------
def partner_activity(master, quarters, revenue, token, existed):
    """WHY: shows how the partner base grows (onboarding) and how many existing
    partners go quiet each quarter - the raw material for 'leaving'."""
    section("6. PARTNER ACTIVITY PER QUARTER")
    log(f"Definition: a quarter is 'token' if revenue < {TOKEN_SHARE:.0%} of that partner's own average "
        "quarterly revenue (averaged over the quarters since it was onboarded, within the data window).")
    onboarded = master["onboarded_quarter"].value_counts()
    rows = []
    for q in quarters:
        exists = existed[q]
        rev = revenue[q]
        rows.append({
            "quarter": q,
            "onboarded_partners": int(exists.sum()),
            "newly_onboarded": int(onboarded.get(q, 0)),
            "active (any orders)": int((rev > 0).sum()),
            "zero revenue": int(((rev == 0) & exists).sum()),
            "token (>0 but < threshold)": int(((rev > 0) & (rev < token)).sum()),
        })
    table = pd.DataFrame(rows).set_index("quarter")
    log(table.to_string())
    log(f"\nPartners onboarded before 2024: {int((master['onboarded_date'] < '2024-01-01').sum()):,}; "
        f"during the window: {int((master['onboarded_date'] >= '2024-01-01').sum()):,}")

    fig, ax = plt.subplots(figsize=(11, 5))
    table[["onboarded_partners", "active (any orders)"]].plot(ax=ax, marker="o")
    ax2 = ax.twinx()
    table[["zero revenue", "token (>0 but < threshold)"]].plot.bar(ax=ax2, alpha=0.3, color=["gray", "crimson"])
    ax.set_ylabel("partners")
    ax2.set_ylabel("zero / token partners")
    ax.set_title("Partner base, active partners, and quiet partners per quarter")
    ax.tick_params(axis="x", rotation=45)
    save_chart(fig, "06_partner_activity.png")
    return table


# ---------------------------------------------------------------------------
# 7. Historical leavers
# ---------------------------------------------------------------------------
def historical_leavers(master, quarters, revenue, token):
    """WHY: there are no churn labels. We build them by 'time travel': a partner
    active in Q-1 that places no or only token orders in Q is a leaver in Q.
    These labels are what the at-risk model will learn from."""
    section("7. HISTORICAL LEAVERS - active in Q-1, none or token orders in Q")
    active_prev, left = leaver_flags(revenue, token, quarters)
    region = master.set_index("partner_id")["region"]

    rows = []
    for q in left.columns:
        row = {"quarter": q}
        for r in REGIONS + ["ALL"]:
            mask = (region == r) if r != "ALL" else pd.Series(True, index=region.index)
            base = active_prev[q] & mask
            row[f"{r}_rate_%"] = round(left[q][base].mean() * 100, 2)
        row["ALL_leavers"] = int(left[q].sum())
        row["ALL_active_prev"] = int(active_prev[q].sum())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("quarter")
    log(table.to_string())

    # Do leavers come back later? If many do, "leaving" is partly a pause.
    above = revenue.gt(token, axis=0)
    returned = total = 0
    for q in left.columns:
        later = [c for c in quarters if c > q]
        if not later:
            continue
        ids = left.index[left[q]]
        total += len(ids)
        returned += above.loc[ids, later].any(axis=1).sum()
    return_rate = returned / total if total else np.nan
    log(f"\nLeavers (2024Q2..2026Q1) who were active again in a later quarter: {returned} of {total} "
        f"({return_rate:.0%})")

    fig, ax = plt.subplots(figsize=(10, 5))
    table[[f"{r}_rate_%" for r in REGIONS + ["ALL"]]].plot(ax=ax, marker="o")
    ax.set_ylabel("% of last quarter's active partners who left")
    ax.set_title("Historical leaver rate by quarter and region")
    ax.tick_params(axis="x", rotation=45)
    save_chart(fig, "07_leaver_rate.png")
    return {"table": table, "active_prev": active_prev, "left": left, "return_rate": return_rate}


# ---------------------------------------------------------------------------
# 8. Fading before leaving
# ---------------------------------------------------------------------------
def fading_pattern(quarters, revenue, orders, existed, leavers):
    """WHY: if leavers already shrink in the quarters before they leave, that decline
    is an early-warning signal - exactly what a health score can learn.
    We also compare HOW OFTEN leavers order: rare, lumpy buyers have empty quarters more easily."""
    section("8. FADING PATTERN - the 4 quarters before leaving (indexed, Q-4 = 100)")
    active_prev, left = leavers["active_prev"], leavers["left"]
    offsets = [-4, -3, -2, -1, 0]
    sums = {g: {m: np.zeros(len(offsets)) for m in ("revenue", "orders")} for g in ("leavers", "stayers")}
    events = {"leavers": 0, "stayers": 0}

    # Order frequency: average order lines per quarter while onboarded, for every leaver/stayer event.
    avg_orders = orders.where(existed).mean(axis=1)
    freq = {"leavers": [], "stayers": []}
    for q in left.columns:
        freq["leavers"] += list(avg_orders[left[q]])
        freq["stayers"] += list(avg_orders[active_prev[q] & ~left[q]])
    freq_table = pd.DataFrame({g: pd.Series(v).describe(percentiles=[0.25, 0.5, 0.75])[["25%", "50%", "75%"]]
                               for g, v in freq.items()}).T.round(1)
    freq_table.columns = ["p25 orders/quarter", "median orders/quarter", "p75 orders/quarter"]
    log("How often do they order? (average order lines per quarter, all leaver/stayer events):")
    log(freq_table.to_string())

    # Only quarters with 4 full quarters of history before them.
    for q in [q for q in left.columns if quarter_offset(q, -4) >= quarters[0]]:
        cols = [quarter_offset(q, o) for o in offsets]
        for group, ids in (("leavers", left[q]), ("stayers", active_prev[q] & ~left[q])):
            sums[group]["revenue"] += revenue.loc[ids, cols].sum().to_numpy()
            sums[group]["orders"] += orders.loc[ids, cols].sum().to_numpy()
            events[group] += int(ids.sum())

    labels = ["Q-4", "Q-3", "Q-2", "Q-1", "Q (leave)"]
    table = pd.DataFrame({f"{g} {m}": sums[g][m] / sums[g][m][0] * 100
                          for g in ("leavers", "stayers") for m in ("revenue", "orders")}, index=labels).round(1)
    log(f"\nIndexed revenue/orders (Q-4 = 100). Leaver events: {events['leavers']:,}; "
        f"stayer events: {events['stayers']:,}")
    log(table.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["revenue", "orders"]):
        ax.plot(labels, table[f"leavers {metric}"], "o-", color="crimson", label="leavers")
        ax.plot(labels, table[f"stayers {metric}"], "o-", color="steelblue", label="stayers")
        ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{metric.capitalize()} before leaving (Q-4 = 100)")
        ax.legend()
    save_chart(fig, "08_fading_pattern.png")
    return {"index": table, "freq": freq_table}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def key_findings(trend, season, segments, conc, activity, leavers, fading):
    section("KEY FINDINGS")
    region_tbl = segments["region"]
    tier_tbl = segments["tier"]
    lt = leavers["table"]
    q2q3 = season.loc[[i for i in season.index if "Q2->Q3" in i], "ALL"]
    last_year = trend["without"].loc[LAST_4_QUARTERS].sum().sum()
    bullets = [
        f"Last 4 quarters revenue (outlier excluded): {millions(last_year)}; "
        f"{region_tbl.index[0]} is the biggest region ({region_tbl['share_%'].iloc[0]}%).",
        f"One {trend['region']} order line in {trend['quarter']} inflates that region-quarter by "
        f"{trend['impact']:.0%} - it is excluded from all trends and forecasts.",
        f"Strong Q3 seasonality: total revenue moved {q2q3.iloc[0]:+.1f}% (2024) and {q2q3.iloc[1]:+.1f}% (2025) "
        f"from Q2 to Q3, so the 2026Q3 forecast must not simply copy 2026Q2.",
        f"Target attainment by tier ranges from {tier_tbl['attainment_%'].min():.0f}% "
        f"({tier_tbl['attainment_%'].idxmin()}) to {tier_tbl['attainment_%'].max():.0f}% "
        f"({tier_tbl['attainment_%'].idxmax()}).",
        f"Revenue concentration: top 10 partners make {conc['top10_share_%'].min()}-"
        f"{conc['top10_share_%'].max()}% of each region's revenue.",
        f"The partner base grew from {activity['onboarded_partners'].iloc[0]:,} to "
        f"{activity['onboarded_partners'].iloc[-1]:,} onboarded partners; new partners ramp up "
        "and must not be flagged just for low early revenue.",
        f"Historical leaver rate: {lt['ALL_rate_%'].min():.1f}-{lt['ALL_rate_%'].max():.1f}% of active partners "
        f"per quarter (latest {LAST_QUARTER}: {lt.loc[LAST_QUARTER, 'ALL_rate_%']:.1f}%).",
        f"{leavers['return_rate']:.0%} of past leavers were active again in a later quarter: in the history, "
        "'leaving' is almost always a temporary gap, not permanent churn.",
        f"Past leavers did NOT fade first - at Q-1 their revenue index was "
        f"{fading['index'].loc['Q-1', 'leavers revenue']:.0f} (Q-4 = 100) vs "
        f"{fading['index'].loc['Q-1', 'stayers revenue']:.0f} for stayers. What sets them apart is order frequency: "
        f"median {fading['freq'].loc['leavers', 'median orders/quarter']} order lines per quarter vs "
        f"{fading['freq'].loc['stayers', 'median orders/quarter']} for stayers - rare, lumpy buyers.",
        "Implication for the model: order frequency/regularity and recency are likely stronger risk signals "
        "than revenue trend; the 2026Q3 leavers may behave differently from these historical gaps, so validate "
        "with time-travel labels rather than assuming a fade.",
    ]
    for bullet in bullets:
        log(f"- {bullet}")


def main():
    print("Loading data...")
    sales_all = load_sales(exclude_outliers=False)
    sales = sales_all[~sales_all["is_outlier"]].copy()
    master = load_master()
    targets = load_targets()
    print(f"  {len(sales_all):,} order lines, {len(master):,} partners, {len(targets):,} target rows")

    check_joins(sales, master, targets)
    trend = revenue_trend(sales_all)
    season = seasonality(trend["without"])
    segments = segment_performance(sales, master, targets)
    conc = concentration(sales, master)

    quarters, revenue, orders, token, existed = partner_quarter_tables(sales, master)
    activity = partner_activity(master, quarters, revenue, token, existed)
    leavers = historical_leavers(master, quarters, revenue, token)
    fading = fading_pattern(quarters, revenue, orders, existed, leavers)
    key_findings(trend, season, segments, conc, activity, leavers, fading)

    (EDA_DIR / "findings.txt").write_text("\n".join(_LOG_LINES) + "\n")
    print(f"\nSaved {len(SAVED_CHARTS)} charts and findings.txt to {EDA_DIR}")


if __name__ == "__main__":
    main()
