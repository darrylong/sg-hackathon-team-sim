"""HPE Partner Channel Health - Streamlit dashboard (Problem 05).

Start with:  streamlit run dashboard.py   (NOT python dashboard.py)
Easiest: ./run_channel.sh from the project root starts the API and this dashboard together.

The dashboard never reads the data or models itself: everything comes from the FastAPI
backend (api.py) over HTTP, so judges, tests and the dashboard all see the same numbers.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit.runtime import exists as running_in_streamlit

if not running_in_streamlit():
    # `python dashboard.py` would render nothing - tell the user the right command instead.
    print("Start with:  streamlit run dashboard.py   (NOT python dashboard.py)")
    sys.exit(1)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8001")
START_COMMAND = ("./run_channel.sh\n\n# or just the API:\n.venv/bin/python -m uvicorn api:app "
                 "--app-dir submission_template_vip/05_channel_analytics/src --port 8001")
REGIONS = ["ALL", "APJ", "EMEA", "AMS"]
RISK_COLORS = {"Gone quiet": "#6b7280", "Fading": "#dc2626", "Mixed": "#d97706",
               "Irregular buyer (gap risk)": "#2563eb"}

st.set_page_config(page_title="HPE Partner Channel Health", layout="wide")


# ---------------------------------------------------------------------------
# API calls (cached: the backend data does not change while the app runs)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def api(path, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def money(x):
    return f"${x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}k"


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def spec_to_figure(spec):
    """Turn the API's {type, x, series} chart spec into a plotly figure."""
    fig = go.Figure()
    for s in spec["series"]:
        if spec["type"] == "line":
            fig.add_trace(go.Scatter(x=spec["x"], y=s["y"], name=s["name"], mode="lines+markers"))
        else:
            fig.add_trace(go.Bar(x=spec["x"], y=s["y"], name=s["name"]))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), barmode="group",
                      yaxis_title=spec.get("y_title", ""), legend=dict(orientation="h"))
    return fig


def history_figure(history, series_names, forecast_rows=None, height=380):
    """Quarterly revenue lines, optionally with the 2026Q3 forecast point and its 80% error bar."""
    fig = go.Figure()
    for name in series_names:
        fig.add_trace(go.Scatter(x=history["quarters"], y=history[name], name=name, mode="lines+markers"))
    if forecast_rows:
        for row in forecast_rows:
            if row["region"] not in series_names:
                continue
            fig.add_trace(go.Scatter(
                x=["2026Q3"], y=[row["forecast"]], mode="markers", marker=dict(size=11, symbol="diamond"),
                name=f"{row['region']} 2026Q3 forecast",
                error_y=dict(type="data", symmetric=False, array=[row["hi80"] - row["forecast"]],
                             arrayminus=[row["forecast"] - row["lo80"]])))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="revenue (USD)",
                      legend=dict(orientation="h"))
    return fig


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_overview(region):
    data = api("/overview")
    k = data["kpis"]
    st.caption(f"Last 4 quarters: {k['period']}")
    c = st.columns(6)
    c[0].metric("Revenue", money(k["revenue_usd"]))
    c[1].metric("Margin (rev-weighted)", f"{k['margin_pct']:.1f}%")
    c[2].metric("Target attainment", f"{k['attainment_pct']:.1f}%")
    c[3].metric("Active partners (2026Q2)", f"{k['active_partners']:,}", f"of {k['total_partners']:,}",
                delta_color="off")
    c[4].metric("At-risk partners", k["at_risk_partners"])
    c[5].metric("Revenue at risk (Q2)", money(k["revenue_at_risk_usd"]), f"{k['revenue_at_risk_share_pct']}% of Q2",
                delta_color="off")

    left, right = st.columns([3, 1])
    with left:
        st.subheader("Quarterly revenue by region")
        names = ["APJ", "EMEA", "AMS"] if region == "ALL" else [region]
        st.plotly_chart(history_figure(data["quarterly_revenue"], names), width="stretch")
        st.caption("The single ~$37.8M outlier order (EMEA, 2026Q1) is excluded.")
    with right:
        st.subheader("2026Q3 forecast")
        row = next(r for r in data["forecast"] if r["region"] == region)
        st.metric(f"{region} forecast", money(row["forecast"]), f"{row['qoq_%']:+.1f}% vs Q2", delta_color="off")
        st.write(f"80% range: **{money(row['lo80'])} - {money(row['hi80'])}**")
        st.write(f"Year-on-year: **{row['yoy_%']:+.1f}%**")


def tab_performance(region):
    c1, c2 = st.columns(2)
    by = c1.selectbox("Group by", ["region", "tier", "partner_type", "product_family"], index=1)
    quarters = c2.slider("Quarters", 1, 10, 4)
    data = api("/performance", by=by, region=region, quarters=quarters)
    rows = pd.DataFrame(data["rows"])
    st.caption(f"{region} - {data['period']}")

    cols = st.columns(3)
    cols[0].plotly_chart(px.bar(rows, x="group", y="revenue_usd", title="Revenue", height=320),
                         width="stretch")
    cols[1].plotly_chart(px.bar(rows, x="group", y="margin_pct", title="Margin %", height=320),
                         width="stretch")
    if "attainment_pct" in rows:
        fig = px.bar(rows, x="group", y="attainment_pct", title="Target attainment %", height=320)
        fig.add_hline(y=100, line_dash="dash", line_color="gray")
        cols[2].plotly_chart(fig, width="stretch")
    else:
        cols[2].info("No attainment by product: targets are set per partner, not per product.")

    quarterly = data["quarterly"]
    fig = go.Figure([go.Scatter(x=quarterly["quarters"], y=quarterly[g], name=g, mode="lines+markers")
                     for g in quarterly if g != "quarters"])
    fig.update_layout(height=320, title="Quarterly revenue per group", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(rows, hide_index=True, width="stretch")


def tab_insights():
    data = api("/insights")
    for i in range(0, len(data["insights"]), 2):
        cols = st.columns(2)
        for col, insight in zip(cols, data["insights"][i:i + 2]):
            with col.container(border=True):
                st.markdown(f"#### {insight['title']}")
                st.write(insight["text"])
                st.plotly_chart(spec_to_figure(insight["chart"]), width="stretch",
                                key=f"insight_{insight['id']}")
    st.subheader("Recommendations")
    for rec in data["recommendations"]:
        st.markdown(f"- {rec}")


def tab_at_risk(region):
    c1, c2 = st.columns(2)
    risk_type = c1.selectbox("Risk type", ["ALL"] + list(RISK_COLORS))
    tier = c2.selectbox("Tier", ["ALL", "Platinum", "Gold", "Silver", "Business"])
    data = api("/at-risk", region=region, risk_type=risk_type, tier=tier)
    df = pd.DataFrame(data["partners"])
    st.write(f"**{data['count']} flagged partners** (riskiest first)")
    if df.empty:
        return

    counts = df["risk_type"].value_counts().reset_index()
    st.plotly_chart(px.bar(counts, x="risk_type", y="count", color="risk_type", color_discrete_map=RISK_COLORS,
                           height=250), width="stretch")

    def color_row(row):
        return [f"color: {RISK_COLORS.get(row['risk_type'], 'black')}" if c == "risk_type" else "" for c in row.index]

    shown = df[["partner_id", "partner_name", "region", "tier", "risk_type", "reason", "health_score",
                "rev_q", "rev_avg4", "days_since_last_order"]]
    st.dataframe(shown.style.apply(color_row, axis=1).format({"health_score": "{:.3f}", "rev_q": "${:,.0f}",
                                                               "rev_avg4": "${:,.0f}"}),
                 hide_index=True, width="stretch")

    choice = st.selectbox("Look at one partner", df["partner_id"] + " - " + df["partner_name"])
    detail = api(f"/partners/{choice.split(' - ')[0]}")
    q = pd.DataFrame(detail["quarterly"])
    fig = go.Figure([go.Bar(x=q["quarter"], y=q["revenue"], name="revenue"),
                     go.Scatter(x=q["quarter"], y=q["target_usd"], name="target", mode="lines+markers")])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    h = detail["health"]
    st.markdown(f"**{detail['master']['partner_name']}** - {detail['master']['tier']} "
                f"{detail['master']['partner_type']}, {detail['master']['country']} "
                f"(onboarded {detail['master']['onboarded_date'][:10]})")
    st.info(f"[{h['risk_type']}] {h['reason']}")
    st.plotly_chart(fig, width="stretch")


def tab_forecast():
    data = api("/forecast")
    table = pd.DataFrame(data["table"])
    rows = data["table"]
    cols = st.columns(2)
    for i, region in enumerate(["APJ", "EMEA", "AMS", "ALL"]):
        with cols[i % 2]:
            st.markdown(f"**{region}**")
            st.plotly_chart(history_figure(data["history"], [region], rows, height=280), width="stretch",
                            key=f"fc_{region}")
    st.subheader("2026Q3 forecast")
    st.dataframe(table[["region", "forecast", "lo80", "hi80", "yoy_%", "qoq_%", "sigma", "adjust_%"]]
                 .style.format({"forecast": "${:,.0f}", "lo80": "${:,.0f}", "hi80": "${:,.0f}", "yoy_%": "{:+.1f}%",
                                "qoq_%": "{:+.1f}%", "sigma": "{:.3f}", "adjust_%": "{:+.2f}%"}),
                 hide_index=True, width="stretch")
    st.subheader("Backtest accuracy (MAPE %, last 4 quarters)")
    st.dataframe(pd.DataFrame(data["backtest_mape"]), hide_index=True, width="stretch")
    st.subheader("How the forecast works")
    st.write(data["method_text"])


def tab_whatif():
    change = st.slider("Change all 2026Q2 targets by (%)", -10, 30, 10)
    data = api("/whatif", target_change_pct=change)
    df = pd.DataFrame(data["by_tier"])
    total = df[df["tier"] == "ALL"].iloc[0]
    c = st.columns(3)
    c[0].metric("Partners missing target", int(total["missing_after"]),
                int(total["missing_after"] - total["missing_now"]), delta_color="inverse")
    c[1].metric("Attainment", f"{total['attainment_after_pct']:.1f}%",
                f"{total['attainment_after_pct'] - total['attainment_now_pct']:+.1f} pts", delta_color="off")
    c[2].metric("Partners", int(total["partners"]))
    tiers = df[df["tier"] != "ALL"]
    fig = go.Figure([go.Bar(x=tiers["tier"], y=tiers["missing_now"], name="missing now"),
                     go.Bar(x=tiers["tier"], y=tiers["missing_after"], name=f"missing with {change:+d}% targets")])
    fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df, hide_index=True, width="stretch")


def tab_how():
    st.markdown("""
**Data cleaning** - 273k order lines, 2,600 partners, 2024Q1-2026Q2. One order line of ~$37.8M (EMEA, 2026Q1) is
more than 10x the 99.99th percentile of all lines; it is excluded everywhere so it can't distort trends or forecasts.

**Health score (machine learning)** - no churn labels exist, so we build them by *time travel*: at each past quarter
we compute features using only data up to then (revenue trend, order frequency, days since last order, past empty
quarters, target attainment, tenure...) and label whether the partner went quiet the next quarter. A gradient-boosting
model trained on 2024-2025 and tested on later quarters reached **ROC-AUC 0.87** (vs 0.70 for "days since last order"
alone). The health score is 1 minus the partner's risk rank.

**Targeted fade boost** - partners whose revenue fell 4+ quarters in a row *and* halved year-on-year are lifted to the
top 1% of risk. A broad "revenue down" score hurt accuracy, so only steep, sustained declines get the boost.

**Who is flagged** - (1) partners that have *gone quiet* (no orders in 2026Q2 and 90+ days since the last one),
(2) *fading* partners, and (3) the top-N by model risk, where N = active partners x (historical leaver rate + the new
excess of decliners). Partners onboarded in the last two quarters are still ramping and are not flagged for low revenue.

**Forecast** - average of three seasonal methods (same-quarter-last-year x growth, typical Q2->Q3 step, log-linear
trend with quarter effects), chosen by backtest on 4 past quarters. Past forecasts ran slightly high, so half of that
bias is corrected. The 80% range comes from backtest errors, with a minimum width of +/-4.5% (log).

**Limitations** - only 10 quarters of history and 4 backtest points; historical "leavers" mostly came back later, so
labels are noisy; the decline group is new, so its leaver rate is estimated from few cases; no data on pipeline,
pricing or partner programmes.
""")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main():
    st.title("HPE Partner Channel Health")
    st.caption("Start with: `streamlit run dashboard.py` (NOT `python dashboard.py`) - or `./run_channel.sh` "
               "from the project root.")

    health = None
    try:
        health = api("/health")
    except requests.exceptions.RequestException:
        st.error(f"Can't reach the channel API at {API_URL}. Start it from the project root with:")
        st.code(START_COMMAND, language="bash")
        st.stop()

    region = st.sidebar.selectbox("Region", REGIONS)
    st.sidebar.caption(f"Data to {health['last_quarter']} - forecasting {health['forecast_quarter']}")
    st.sidebar.caption(f"{health['partners']:,} partners, {health['flagged']} flagged at risk")

    tabs = st.tabs(["Overview", "Performance", "Insights", "At-risk partners", "Forecast", "What-if", "How it works"])
    with tabs[0]:
        tab_overview(region)
    with tabs[1]:
        tab_performance(region)
    with tabs[2]:
        tab_insights()
    with tabs[3]:
        tab_at_risk(region)
    with tabs[4]:
        tab_forecast()
    with tabs[5]:
        tab_whatif()
    with tabs[6]:
        tab_how()


main()
