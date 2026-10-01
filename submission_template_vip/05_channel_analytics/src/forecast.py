"""2026Q3 revenue forecast per region (APJ, EMEA, AMS) and in total, with an 80% interval.

WHY three simple methods + a backtest: with only 10 quarters of history, fancy models
over-fit. Simple seasonal methods are robust, and a backtest on past quarters
(using only data available at the time) tells us honestly which one to trust.

Run (after health.py, for the revenue-at-risk numbers):
    .venv/bin/python submission_template_vip/05_channel_analytics/src/forecast.py
Writes reports/forecast.csv, reports/backtest.csv, reports/forecast_chart.png
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FORECAST_QUARTER, LAST_QUARTER, REGIONS, REPORTS_DIR, load_sales

ALL = "ALL"
SERIES = REGIONS + [ALL]
BACKTEST = [("2025Q2", "2025Q3"), ("2025Q3", "2025Q4"), ("2025Q4", "2026Q1"), ("2026Q1", "2026Q2")]
Z_80 = 1.2816       # 80% two-sided interval: 10% in each tail of a normal distribution
MIN_SIGMA = 0.045   # never claim to be more than ~4.5% certain (in log terms); 4 backtest points is little evidence
BIAS_SHRINK = 0.5   # correct only half of the past average bias - 4 errors are too few to trust fully
TIE_TOLERANCE = 0.1  # MAPE within 0.1 percentage points counts as a tie -> prefer ENS


# ---------------------------------------------------------------------------
# 1. Quarterly revenue
# ---------------------------------------------------------------------------
def quarterly_revenue():
    """quarter x series table (APJ, EMEA, AMS, ALL), outlier line excluded."""
    sales = load_sales(exclude_outliers=True)
    table = sales.pivot_table(index="quarter", columns="region", values="revenue_usd", aggfunc="sum")[REGIONS]
    table[ALL] = table[REGIONS].sum(axis=1)
    return table.sort_index()


def shift(quarter, n):
    return str(pd.Period(quarter, freq="Q") + n)


def quarter_of_year(quarter):
    return pd.Period(quarter, freq="Q").quarter


# ---------------------------------------------------------------------------
# 2. Methods - each sees ONLY `history` (quarters up to the forecast origin)
# ---------------------------------------------------------------------------
def method_seasonal_growth(history, target):
    """A: same quarter last year x recent year-on-year growth (last 2 quarters vs a year earlier).
    WHY: keeps the seasonal shape of last year, scaled by how fast we are growing now."""
    origin = history.index[-1]
    recent = history[[shift(origin, -1), origin]].sum()
    year_before = history[[shift(origin, -5), shift(origin, -4)]].sum()
    return history[shift(target, -4)] * recent / year_before


def method_seasonal_ratio(history, target):
    """B: last quarter x the average historical step from the previous quarter-of-year to the target one
    (e.g. for Q3: average of Q3/Q2 in past years). WHY: captures the typical Q2->Q3 dip directly."""
    ratios = [history[q] / history[shift(q, -1)] for q in history.index
              if quarter_of_year(q) == quarter_of_year(target) and shift(q, -1) in history.index]
    return history.iloc[-1] * np.mean(ratios)


def method_regression(history, target):
    """C: least squares on log(revenue) = a + b*time + quarter-of-year effects.
    WHY log: growth is multiplicative, so a straight line in log space = steady % growth."""
    def design(quarters):
        t = np.array([pd.Period(q, freq="Q").ordinal for q in quarters], dtype=float)
        qoy = np.array([quarter_of_year(q) for q in quarters])
        dummies = [(qoy == k).astype(float) for k in (2, 3, 4)]  # Q1 is the baseline
        return np.column_stack([np.ones_like(t), t] + dummies)

    coef, *_ = np.linalg.lstsq(design(history.index), np.log(history.to_numpy()), rcond=None)
    return float(np.exp(design([target]) @ coef)[0])


def method_ensemble(history, target):
    """ENS: average of A, B and C - errors of different methods partly cancel out."""
    return np.mean([m(history, target) for m in (method_seasonal_growth, method_seasonal_ratio, method_regression)])


METHODS = {"A_seasonal_growth": method_seasonal_growth, "B_seasonal_ratio": method_seasonal_ratio,
           "C_regression": method_regression, "ENS": method_ensemble}


# ---------------------------------------------------------------------------
# 3. Backtest
# ---------------------------------------------------------------------------
def backtest(revenue):
    """Forecast each past quarter using only the data before it, and record the errors."""
    rows = []
    for origin, target in BACKTEST:
        for series in SERIES:
            history = revenue.loc[:origin, series]  # no peeking past the origin
            actual = revenue.loc[target, series]
            for name, method in METHODS.items():
                f = method(history, target)
                rows.append({"origin": origin, "target": target, "series": series, "method": name,
                             "forecast": f, "actual": actual, "pct_error": (f / actual - 1) * 100,
                             "log_error": np.log(f / actual)})
    return pd.DataFrame(rows)


def choose_method(bt):
    """Lowest average MAPE across the three regions; ENS wins ties (within 0.1 pp)."""
    regions = bt[bt["series"].isin(REGIONS)]
    mape = regions.assign(ape=regions["pct_error"].abs()).groupby(["method", "series"])["ape"].mean().unstack()
    mape["avg_regions"] = mape[REGIONS].mean(axis=1)
    mape[ALL] = bt[bt["series"] == ALL].assign(ape=lambda d: d["pct_error"].abs()).groupby("method")["ape"].mean()
    best = mape["avg_regions"].idxmin()
    if mape.loc["ENS", "avg_regions"] - mape.loc[best, "avg_regions"] <= TIE_TOLERANCE:
        best = "ENS"
    return best, mape.round(2)


# ---------------------------------------------------------------------------
# 4-6. Final forecast, interval, revenue at risk
# ---------------------------------------------------------------------------
def interval_sigma(log_errors):
    """Spread of past log errors; the largest of std and RMSE (RMSE also covers bias), floored at 3%."""
    std = np.std(log_errors, ddof=1) if len(log_errors) > 1 else 0.0
    rmse = np.sqrt(np.mean(np.square(log_errors)))
    return max(std, rmse, MIN_SIGMA)


def revenue_at_risk():
    """2026Q2 revenue of the partners flagged at_risk, per region (from health.py output)."""
    path = REPORTS_DIR / "partner_health.csv"
    if not path.exists():
        print("  (partner_health.csv not found - run health.py first; skipping revenue at risk)")
        return None
    health = pd.read_csv(path)
    flagged = health[health["at_risk"] == 1]
    per_region = flagged.groupby("region")["rev_q"].sum().reindex(REGIONS, fill_value=0)
    per_region[ALL] = per_region.sum()
    counts = flagged["region"].value_counts().reindex(REGIONS, fill_value=0)
    counts[ALL] = counts.sum()
    return pd.DataFrame({"flagged_partners": counts, "revenue_at_risk_q2": per_region})


def final_forecast(revenue, bt, method):
    """Chosen method per region, bias-corrected with shrinkage; ALL = sum of the adjusted regions."""
    chosen = bt[bt["method"] == method]
    rows = []
    for series in REGIONS:
        raw = METHODS[method](revenue[series], FORECAST_QUARTER)
        # Past forecasts were on average too high/low by mean_bias (in log terms).
        # WHY only half: with 4 backtest points the bias estimate itself is noisy.
        bias = chosen.loc[chosen["series"] == series, "log_error"].mean()
        factor = np.exp(-BIAS_SHRINK * bias)
        rows.append({"region": series, "raw_forecast": raw, "mean_bias": bias,
                     "adjust_%": (factor - 1) * 100, "forecast": raw * factor})
    rows.append({"region": ALL,  # ALL = sum of regions, both before and after adjustment
                 "raw_forecast": sum(r["raw_forecast"] for r in rows),
                 "mean_bias": chosen.loc[chosen["series"] == ALL, "log_error"].mean(),
                 "forecast": sum(r["forecast"] for r in rows)})
    table = pd.DataFrame(rows).set_index("region")
    table.loc[ALL, "adjust_%"] = (table.loc[ALL, "forecast"] / table.loc[ALL, "raw_forecast"] - 1) * 100

    print("Bias correction (mean log error of past forecasts; >0 means we forecast too high):")
    print(table[["mean_bias", "adjust_%"]].round(4).to_string())

    table["sigma"] = [interval_sigma(chosen.loc[chosen["series"] == s, "log_error"].to_numpy()) for s in table.index]
    table["lo80"] = table["forecast"] * np.exp(-Z_80 * table["sigma"])
    table["hi80"] = table["forecast"] * np.exp(Z_80 * table["sigma"])
    table["method"] = method
    table["q2_2026_actual"] = revenue.loc[LAST_QUARTER, table.index]
    table["q3_2025_actual"] = revenue.loc[shift(FORECAST_QUARTER, -4), table.index]
    table["yoy_%"] = (table["forecast"] / table["q3_2025_actual"] - 1) * 100
    table["qoq_%"] = (table["forecast"] / table["q2_2026_actual"] - 1) * 100
    assert (table["lo80"] <= table["forecast"]).all() and (table["forecast"] <= table["hi80"]).all()
    table["backtest_coverage"] = [coverage(chosen, s, table.loc[s, "sigma"]) for s in table.index]
    return table


def coverage(chosen, series, sigma):
    """How many backtest actuals fall inside the 80% range we would have drawn then.

    Uses the same rules as the final forecast: regions get the half bias correction;
    ALL is the sum of regions, so its correction is skipped here (it is small).
    Note: the bias was estimated on these same 4 points, so this check is a bit optimistic.
    """
    errors = chosen.loc[chosen["series"] == series, "log_error"].to_numpy()
    if series != ALL:
        errors = errors - BIAS_SHRINK * errors.mean()
    inside = int((np.abs(errors) <= Z_80 * sigma).sum())
    return f"{inside}/{len(errors)}"


def save_chart(revenue, table):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, series in zip(axes.flat, SERIES):
        ax.plot(revenue.index, revenue[series] / 1e6, marker="o", label="actual")
        row = table.loc[series]
        ax.errorbar([FORECAST_QUARTER], [row["forecast"] / 1e6],
                    yerr=[[(row["forecast"] - row["lo80"]) / 1e6], [(row["hi80"] - row["forecast"]) / 1e6]],
                    fmt="s", color="crimson", capsize=6, label="2026Q3 forecast (80% range)")
        ax.set_title(series)
        ax.set_ylabel("$M")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle(f"Quarterly revenue and {FORECAST_QUARTER} forecast (method: {table['method'].iloc[0]})")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "forecast_chart.png", dpi=120)
    plt.close(fig)


def main():
    print("1. Quarterly revenue (outlier excluded)...")
    revenue = quarterly_revenue()
    print((revenue / 1e6).round(1).to_string())

    print("\n3. Backtest - % error (forecast vs actual), each forecast uses only data up to its origin")
    print("=" * 90)
    bt = backtest(revenue)
    bt.to_csv(REPORTS_DIR / "backtest.csv", index=False)
    errors = bt.pivot_table(index=["target", "series"], columns="method", values="pct_error").round(1)
    print(errors.to_string())
    print("\nForecasting 2025Q3 from 2025Q2 (the only past Q3 we can test - the seasonal dip):")
    print(errors.loc["2025Q3"].to_string())

    method, mape = choose_method(bt)
    print("\nMAPE % per method (4 backtest quarters):")
    print(mape.to_string())
    best_raw = mape["avg_regions"].idxmin()
    why = ("lowest average MAPE across APJ/EMEA/AMS" if method == best_raw
           else f"within {TIE_TOLERANCE} pp of the best ({best_raw}), and ties go to the ensemble")
    print(f"\nChosen method: {method} ({mape.loc[method, 'avg_regions']:.2f}% average regional MAPE) - {why}.")

    print(f"\n4-5. Final {FORECAST_QUARTER} forecast with 80% interval")
    print("=" * 90)
    table = final_forecast(revenue, bt, method)
    shown = table.drop(columns=["mean_bias", "method"]).copy()
    for column in ["raw_forecast", "forecast", "lo80", "hi80", "q2_2026_actual", "q3_2025_actual"]:
        shown[column] = (shown[column] / 1e6).round(1)
    shown = shown.rename(columns=lambda c: f"{c}_$M" if c in ("raw_forecast", "forecast", "lo80", "hi80") else c)
    print(f"(method {method}; raw = before bias correction; coverage = backtest actuals inside the 80% range)")
    print(shown.round({"adjust_%": 2, "sigma": 4, "yoy_%": 1, "qoq_%": 1}).to_string())

    print("\n6. Revenue at risk (2026Q2 revenue of at_risk partners)")
    print("=" * 90)
    at_risk = revenue_at_risk()
    if at_risk is not None:
        at_risk["share_of_q2_%"] = (at_risk["revenue_at_risk_q2"] / revenue.loc[LAST_QUARTER, at_risk.index] * 100)
        printable = at_risk.assign(revenue_at_risk_q2=(at_risk["revenue_at_risk_q2"] / 1e6).round(2))
        print(printable.rename(columns={"revenue_at_risk_q2": "revenue_at_risk_q2_$M"}).round(2).to_string())
        table = table.join(at_risk)

    table.reset_index().to_csv(REPORTS_DIR / "forecast.csv", index=False)
    save_chart(revenue, table)
    print("\nSaved forecast.csv, backtest.csv and forecast_chart.png")


if __name__ == "__main__":
    main()
