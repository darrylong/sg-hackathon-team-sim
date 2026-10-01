"""FastAPI backend for the Partner Channel Health dashboard (Problem 05).

Start (from the project root):
    .venv/bin/python -m uvicorn api:app --app-dir submission_template_vip/05_channel_analytics/src --port 8001
Docs: http://127.0.0.1:8001/docs

Everything is loaded ONCE at startup from the files our pipeline already wrote
(data_prep.py, health.py, forecast.py). Requests only slice and summarise - no retraining.
"""

import json
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import FORECAST_QUARTER, LAST_QUARTER, REGIONS, REPORTS_DIR, load_master, load_sales, load_targets
from data_prep import build_monthly
from health import PartnerData, features_at, shift_quarter

ALL = "ALL"

# ---------------------------------------------------------------------------
# Load once at startup
# ---------------------------------------------------------------------------
print("Loading data and reports...")
SALES_ALL = load_sales(exclude_outliers=False, verbose=False)   # only for the outlier insight
SALES = SALES_ALL[~SALES_ALL["is_outlier"]].copy()             # everything else uses clean sales
MASTER = load_master()
TARGETS = load_targets()
PANEL = pd.read_csv(REPORTS_DIR / "panel.csv")
HEALTH = pd.read_csv(REPORTS_DIR / "partner_health.csv")
FORECAST = pd.read_csv(REPORTS_DIR / "forecast.csv")
BACKTEST = pd.read_csv(REPORTS_DIR / "backtest.csv")
QUARTERS = sorted(PANEL["quarter"].unique())
SALES = SALES.merge(MASTER[["partner_id", "partner_type"]], on="partner_id")

# Decline group a year ago - same feature code as health.py, so numbers match the model.
_data = PartnerData(PANEL, build_monthly(SALES, MASTER), MASTER)
YEAR_AGO = shift_quarter(LAST_QUARTER, -4)
FEATURES_YEAR_AGO = features_at(_data, YEAR_AGO)
print("Ready.")

app = FastAPI(title="HPE Partner Channel Health API",
              description="Performance, insights, partner health and the 2026Q3 forecast.")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def records(df):
    """DataFrame -> JSON-safe list of dicts (NaN -> null, numpy numbers -> plain numbers,
    dates -> "YYYY-MM-DD..." strings instead of epoch milliseconds)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def last_quarters(n):
    return QUARTERS[-n:]


def weighted_margin(df):
    """Revenue-weighted margin %: big deals count more than small ones."""
    return float((df["revenue_usd"] * df["margin_pct"]).sum() / df["revenue_usd"].sum())


def region_filter(df, region):
    return df if region == ALL else df[df["region"] == region]


def quarterly_by_region(sales):
    table = sales.pivot_table(index="quarter", columns="region", values="revenue_usd", aggfunc="sum")[REGIONS]
    table[ALL] = table.sum(axis=1)
    return table


def chart(kind, x, series, y_title="", x_title=""):
    """Small, front-end-agnostic chart spec: {type, x, series: [{name, y}], y_title, x_title}."""
    return {"type": kind, "x": [str(v) for v in x], "y_title": y_title, "x_title": x_title,
            "series": [{"name": name, "y": [None if pd.isna(v) else float(v) for v in values]}
                       for name, values in series.items()]}


def pct(x):
    return round(float(x) * 100, 1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "partners": len(MASTER), "last_quarter": LAST_QUARTER,
            "forecast_quarter": FORECAST_QUARTER, "flagged": int(HEALTH["at_risk"].sum())}


@app.get("/overview")
def overview():
    """Headline KPIs for the last 4 quarters + revenue history + the forecast."""
    q4 = last_quarters(4)
    recent_sales = SALES[SALES["quarter"].isin(q4)]
    recent_panel = PANEL[PANEL["quarter"].isin(q4)]
    flagged = HEALTH[HEALTH["at_risk"] == 1]
    q_last_revenue = PANEL.loc[PANEL["quarter"] == LAST_QUARTER, "revenue"].sum()
    kpis = {
        "period": f"{q4[0]}..{q4[-1]}",
        "revenue_usd": float(recent_sales["revenue_usd"].sum()),
        "margin_pct": round(weighted_margin(recent_sales), 2),
        "attainment_pct": pct(recent_panel["revenue"].sum() / recent_panel["target_usd"].sum()),
        "active_partners": int((PANEL.loc[PANEL["quarter"] == LAST_QUARTER, "revenue"] > 0).sum()),
        "total_partners": len(MASTER),
        "at_risk_partners": len(flagged),
        "revenue_at_risk_usd": float(flagged["rev_q"].sum()),
        "revenue_at_risk_share_pct": pct(flagged["rev_q"].sum() / q_last_revenue),
    }
    history = quarterly_by_region(SALES)
    return {"kpis": kpis,
            "quarterly_revenue": {"quarters": list(history.index),
                                  **{s: [float(v) for v in history[s]] for s in REGIONS + [ALL]}},
            "forecast": records(FORECAST[["region", "forecast", "lo80", "hi80", "yoy_%", "qoq_%"]])}


@app.get("/performance")
def performance(by: Literal["region", "tier", "partner_type", "product_family"] = "region",
                region: Literal["ALL", "APJ", "EMEA", "AMS"] = "ALL",
                quarters: int = Query(4, ge=1, le=10)):
    """Revenue, share, margin and attainment per group, plus each group's quarterly revenue."""
    period = last_quarters(quarters)
    sales = region_filter(SALES, region)
    recent = sales[sales["quarter"].isin(period)]
    grouped = recent.groupby(by)
    table = pd.DataFrame({
        "revenue_usd": grouped["revenue_usd"].sum(),
        "share_pct": grouped["revenue_usd"].sum() / recent["revenue_usd"].sum() * 100,
        "margin_pct": grouped.apply(weighted_margin, include_groups=False),
    })
    if by != "product_family":  # targets are set per partner, not per product
        panel = region_filter(PANEL, region)
        panel = panel[panel["quarter"].isin(period)]
        sums = panel.groupby(by)[["revenue", "target_usd"]].sum()
        table["attainment_pct"] = sums["revenue"] / sums["target_usd"] * 100
    table = table.sort_values("revenue_usd", ascending=False).round(2)
    series = sales.pivot_table(index="quarter", columns=by, values="revenue_usd", aggfunc="sum").fillna(0)
    return {"by": by, "region": region, "period": f"{period[0]}..{period[-1]}",
            "rows": records(table.reset_index().rename(columns={by: "group"})),
            "quarterly": {"quarters": list(series.index), **{str(g): [float(v) for v in series[g]] for g in series}}}


def build_insights():
    """Seven data-driven insights (real numbers, no LLM) + recommendations."""
    insights = []
    q4 = last_quarters(4)

    # 1. Outlier impact
    out = SALES_ALL[SALES_ALL["is_outlier"]].iloc[0]
    with_o = SALES_ALL[(SALES_ALL["region"] == out["region"]) & (SALES_ALL["quarter"] == out["quarter"])]["revenue_usd"].sum()
    without_o = with_o - out["revenue_usd"]
    insights.append({
        "id": "outlier", "title": "One order line would distort the trend",
        "text": f"A single ${out['revenue_usd'] / 1e6:,.1f}M {out['product_family']} line ({out['partner_id']}, "
                f"{out['order_date']:%Y-%m-%d}) inflates {out['region']} {out['quarter']} revenue by "
                f"{(with_o / without_o - 1) * 100:.0f}%. It is excluded from all trends, health scores and forecasts.",
        "chart": chart("bar", ["with outlier", "without outlier"], {f"{out['region']} {out['quarter']}": [with_o, without_o]},
                       "revenue (USD)")})

    # 2. Q3 seasonality
    qr = quarterly_by_region(SALES)
    dips = {f"{y} Q2->Q3": [(qr.loc[f"{y}Q3", s] / qr.loc[f"{y}Q2", s] - 1) * 100 for s in REGIONS + [ALL]]
            for y in ("2024", "2025")}
    insights.append({
        "id": "seasonality", "title": "Q3 dips after Q2 every year",
        "text": f"Total revenue moved {dips['2024 Q2->Q3'][-1]:+.1f}% (2024) and {dips['2025 Q2->Q3'][-1]:+.1f}% (2025) "
                f"from Q2 to Q3; EMEA dips most ({dips['2024 Q2->Q3'][1]:+.1f}%, {dips['2025 Q2->Q3'][1]:+.1f}%). "
                "The 2026Q3 forecast builds this dip in instead of copying Q2.",
        "chart": chart("bar", REGIONS + [ALL], dips, "% change Q2->Q3")})

    # 3. AMS growth + concentration
    prev4 = QUARTERS[-8:-4]
    growth = {r: (qr.loc[q4, r].sum() / qr.loc[prev4, r].sum() - 1) * 100 for r in REGIONS}
    recent = SALES[SALES["quarter"].isin(q4)]
    by_partner = recent.groupby(["region", "partner_id"])["revenue_usd"].sum()
    top10 = {r: by_partner.loc[r].nlargest(10).sum() / by_partner.loc[r].sum() * 100 for r in REGIONS}
    ams_share = qr.loc[q4, "AMS"].sum() / qr.loc[q4, ALL].sum() * 100
    insights.append({
        "id": "ams_growth", "title": "AMS drives growth - and is the most concentrated",
        "text": f"Last 4 quarters vs the 4 before: AMS {growth['AMS']:+.0f}%, EMEA {growth['EMEA']:+.0f}%, "
                f"APJ {growth['APJ']:+.0f}%. AMS is {ams_share:.0f}% of revenue, and its top 10 partners bring "
                f"{top10['AMS']:.0f}% of it (APJ {top10['APJ']:.0f}%, EMEA {top10['EMEA']:.0f}%).",
        "chart": chart("bar", REGIONS, {"growth %": [growth[r] for r in REGIONS],
                                        "top-10 share %": [top10[r] for r in REGIONS]}, "%")})

    # 4. Attainment by tier
    panel4 = PANEL[PANEL["quarter"].isin(q4)]
    att = panel4.groupby("tier")[["revenue", "target_usd"]].sum()
    att = (att["revenue"] / att["target_usd"] * 100).sort_values()
    insights.append({
        "id": "attainment", "title": f"{att.index[0]} partners miss target most",
        "text": f"Target attainment over the last 4 quarters: " + ", ".join(f"{t} {v:.0f}%" for t, v in att.items())
                + f". {att.index[0]} is {att.iloc[-1] - att.iloc[0]:.0f} points behind {att.index[-1]}.",
        "chart": chart("bar", list(att.index), {"attainment %": list(att.values)}, "attainment %")})

    # 5. Margin by partner type + GreenLake
    m_type = recent.groupby("partner_type").apply(weighted_margin, include_groups=False).sort_values()
    fam = recent.groupby("product_family")
    fam_margin = fam.apply(weighted_margin, include_groups=False)
    fam_share = fam["revenue_usd"].sum() / recent["revenue_usd"].sum() * 100
    insights.append({
        "id": "margin", "title": "GreenLake has the best margin but the smallest share",
        "text": f"Partner margin: " + ", ".join(f"{t} {v:.1f}%" for t, v in m_type.items())
                + f". GreenLake earns {fam_margin['GreenLake']:.1f}% margin vs {fam_margin['ProLiant']:.1f}% for ProLiant, "
                f"yet is only {fam_share['GreenLake']:.0f}% of revenue.",
        "chart": chart("bar", list(fam_margin.index), {"margin %": list(fam_margin.values),
                                                       "revenue share %": list(fam_share[fam_margin.index].values)}, "%")})

    # 6. New decline group
    # Counted over all partners, exactly like health.py's choose_n printout.
    now_n = int((HEALTH["n_consecutive_declines"] >= 4).sum())
    then_n = int((FEATURES_YEAR_AGO["n_consecutive_declines"] >= 4).sum())
    fading = HEALTH[HEALTH["strong_fade"] == 1]
    insights.append({
        "id": "decliners", "title": "A new group of steadily declining partners",
        "text": f"{now_n} partners have shrunk 4+ quarters in a row in {LAST_QUARTER}, vs {then_n} a year ago. "
                f"{len(fading)} of them are also down by half or more year-on-year - flagged as 'Fading' "
                f"(${fading['rev_q'].sum() / 1e6:,.1f}M revenue last quarter).",
        "chart": chart("bar", [YEAR_AGO, LAST_QUARTER], {"partners with 4+ declines": [then_n, now_n]}, "partners")})

    # 7. At-risk count vs revenue share
    flagged = HEALTH[HEALTH["at_risk"] == 1]
    q_rev = PANEL.loc[PANEL["quarter"] == LAST_QUARTER, "revenue"].sum()
    by_type = flagged.groupby("risk_type").agg(partners=("partner_id", "size"), revenue=("rev_q", "sum"))
    insights.append({
        "id": "at_risk", "title": "Many partners at risk, little revenue",
        "text": f"{len(flagged)} partners ({len(flagged) / len(HEALTH) * 100:.1f}%) are flagged, but they made only "
                f"{flagged['rev_q'].sum() / q_rev * 100:.1f}% of {LAST_QUARTER} revenue - mostly small "
                f"{flagged['tier'].mode()[0]}-tier partners. Retention effort can be cheap and targeted.",
        "chart": chart("bar", list(by_type.index), {"partners": list(by_type["partners"]),
                                                    "revenue ($k)": list(by_type["revenue"] / 1e3)})})

    recommendations = [
        f"Plan 2026Q3 for a seasonal dip: forecast ${FORECAST.set_index('region').loc[ALL, 'forecast'] / 1e6:,.0f}M "
        f"total ({FORECAST.set_index('region').loc[ALL, 'qoq_%']:+.1f}% vs Q2), with the 80% range as the planning band.",
        f"Call the {len(fading)} 'Fading' partners first - they are still ordering, so there is time to win them back.",
        f"Run a low-cost re-activation campaign for the {int(flagged['gone_quiet'].sum())} partners that have gone "
        "quiet (no order for 90+ days).",
        f"Review {att.index[0]}-tier targets and enablement: attainment {att.iloc[0]:.0f}% vs {att.iloc[-1]:.0f}% "
        f"for {att.index[-1]}.",
        f"Grow GreenLake through partners: highest margin ({fam_margin['GreenLake']:.1f}%) but only "
        f"{fam_share['GreenLake']:.0f}% of revenue; reduce reliance on AMS top-10 concentration.",
    ]
    return {"insights": insights, "recommendations": recommendations}


INSIGHTS = build_insights()  # computed once: the data doesn't change while the server runs


@app.get("/insights")
def insights():
    return INSIGHTS


AT_RISK_FIELDS = ["partner_id", "partner_name", "region", "tier", "partner_type", "health_score", "risk_type",
                  "reason", "rev_q", "rev_avg4", "days_since_last_order"]


@app.get("/at-risk")
def at_risk(region: str | None = None, risk_type: str | None = None, tier: str | None = None):
    """Flagged partners, riskiest first. Every filter is optional."""
    flagged = HEALTH[HEALTH["at_risk"] == 1].sort_values("health_score")
    for column, value in (("region", region), ("risk_type", risk_type), ("tier", tier)):
        if value and value != ALL:
            flagged = flagged[flagged[column] == value]
    return {"count": len(flagged), "partners": records(flagged[AT_RISK_FIELDS])}


@app.get("/partners/{partner_id}")
def partner(partner_id: str):
    """Everything about one partner: master data, health and its quarterly history."""
    if partner_id not in set(MASTER["partner_id"]):
        raise HTTPException(status_code=404, detail=f"Unknown partner_id {partner_id}")
    master = records(MASTER[MASTER["partner_id"] == partner_id].drop(columns="onboarded_quarter"))[0]
    health_fields = ["health_score", "at_risk", "risk_type", "reason", "gap_prob", "fade_score", "strong_fade",
                     "gone_quiet", "rev_q", "rev_avg4", "rev_ratio_yoy", "n_consecutive_declines", "orders_avg4",
                     "days_since_last_order", "attainment_avg4", "is_new"]
    health_row = records(HEALTH.loc[HEALTH["partner_id"] == partner_id, health_fields])[0]
    series = PANEL.loc[PANEL["partner_id"] == partner_id, ["quarter", "revenue", "n_orders", "target_usd", "attainment"]]
    return {"master": master, "health": health_row, "quarterly": records(series)}


@app.get("/forecast")
def forecast():
    """Forecast table, history, backtest accuracy and a plain-English method description."""
    history = quarterly_by_region(SALES)
    ape = BACKTEST.assign(ape=BACKTEST["pct_error"].abs())
    mape = ape.pivot_table(index="method", columns="series", values="ape", aggfunc="mean")[REGIONS + [ALL]]
    mape["avg_regions"] = mape[REGIONS].mean(axis=1)
    method = FORECAST["method"].iloc[0]
    text = (f"We average three simple seasonal methods ({method}): (A) same quarter last year x recent growth, "
            "(B) last quarter x the typical Q2->Q3 step, (C) a log-linear trend with quarter effects. "
            "Each was backtested on the last 4 quarters using only data available at the time; the ensemble "
            f"had the lowest error ({mape.loc[method, 'avg_regions']:.1f}% average regional MAPE). "
            "Because past forecasts ran slightly high, we correct half of that bias. The 80% range comes from the "
            "backtest errors, with a minimum width of +/-4.5% (log). The single ~$37.8M outlier order is excluded.")
    return {"forecast_quarter": FORECAST_QUARTER, "method": method, "method_text": text,
            "table": records(FORECAST),
            "history": {"quarters": list(history.index), **{s: [float(v) for v in history[s]] for s in REGIONS + [ALL]}},
            "backtest_mape": records(mape.round(2).reset_index())}


@app.get("/whatif")
def whatif(target_change_pct: float = Query(10, ge=-50, le=100)):
    """If last quarter's targets had been X% higher/lower: who would miss, and the new attainment, by tier."""
    last = PANEL[(PANEL["quarter"] == LAST_QUARTER) & PANEL["target_usd"].notna()].copy()
    last["new_target"] = last["target_usd"] * (1 + target_change_pct / 100)
    rows = []
    for tier, group in list(last.groupby("tier")) + [("ALL", last)]:
        rows.append({
            "tier": tier, "partners": len(group),
            "missing_now": int((group["revenue"] < group["target_usd"]).sum()),
            "missing_after": int((group["revenue"] < group["new_target"]).sum()),
            "attainment_now_pct": pct(group["revenue"].sum() / group["target_usd"].sum()),
            "attainment_after_pct": pct(group["revenue"].sum() / group["new_target"].sum()),
        })
    return {"quarter": LAST_QUARTER, "target_change_pct": target_change_pct, "by_tier": rows}
