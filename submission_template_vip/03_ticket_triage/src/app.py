"""Streamlit front end for the ticket-triage MVP.

The app NEVER loads the model itself: every prediction goes to the FastAPI backend
over HTTP. That keeps one source of truth (api.py) for the judges, the app and tests.

Start both servers with ./run_triage.sh from the project root, or by hand:
    .venv/bin/python -m uvicorn api:app --app-dir submission_template_vip/03_ticket_triage/src --port 8000
    .venv/bin/python -m streamlit run submission_template_vip/03_ticket_triage/src/app.py --server.port 8501
"""

import io
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from config import CURVEBALL_CSV, EVAL_CSV, SUBMISSION_COLUMNS, load_cost_matrix

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
START_COMMAND = ("./run_triage.sh\n\n# or just the API:\n.venv/bin/python -m uvicorn api:app "
                 "--app-dir submission_template_vip/03_ticket_triage/src --port 8000")

# One sentence per abstain reason, so the agent understands WHY a human is needed.
REASON_EXPLAINED = {
    "Too short to understand": "The ticket has too few words for the model to tell what it is about.",
    "Unfamiliar topic": "The ticket looks unlike any category we have seen before, so no known team clearly owns it.",
    "Urgent and unsure": "This looks urgent, and the model is not confident enough in the team - "
                         "a wrong route on an urgent ticket costs more than a human check.",
    "Low confidence": "The model is not confident enough in the team; a hand-off is cheaper than a likely misroute.",
}
EXAMPLE_NONE = "(write your own)"

st.set_page_config(page_title="HPE Smart Ticket Triage", layout="wide")


# ---------------------------------------------------------------------------
# Talking to the API
# ---------------------------------------------------------------------------
def api_get(path):
    response = requests.get(f"{API_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def get_meta():
    """Label lists and thresholds - they never change while the app runs, so cache them."""
    return api_get("/meta")


def post_triage(ticket):
    response = requests.post(f"{API_URL}/triage", json=ticket, timeout=30)
    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.json().get('detail')}")
        return None
    return response.json()


def post_batch(filename, csv_bytes):
    response = requests.post(f"{API_URL}/triage/batch",
                             files={"file": (filename, csv_bytes, "text/csv")}, timeout=300)
    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.json().get('detail')}")
        return None
    return response.json()


@st.cache_data(show_spinner=False)
def load_curveballs():
    return pd.read_csv(CURVEBALL_CSV)


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------
def banner(text, color):
    """A small coloured box - the only custom HTML in the app."""
    st.markdown(
        f"<div style='background:{color};color:white;padding:14px 18px;border-radius:8px;"
        f"font-size:1.3rem;font-weight:600;margin:6px 0'>{text}</div>",
        unsafe_allow_html=True,
    )


def metric_card(column, label, value, confidence):
    with column:
        st.metric(label, value)
        st.progress(min(max(confidence, 0.0), 1.0), text=f"confidence {confidence:.0%}")


# ---------------------------------------------------------------------------
# Tab 1: one ticket
# ---------------------------------------------------------------------------
def fill_from_example():
    """Callback: copy the chosen curveball ticket into the input widgets."""
    choice = st.session_state["example"]
    if choice == EXAMPLE_NONE:
        return
    ticket = load_curveballs().set_index("ticket_id").loc[choice.split(" - ")[0]]
    st.session_state["subject"] = ticket["subject"]
    st.session_state["body"] = ticket["body"]
    st.session_state["product"] = ticket["product"]
    st.session_state["channel"] = ticket["channel"]


def show_triage_result(r):
    if r["abstain"]:
        banner(f"SEND TO HUMAN - {r['abstain_reason']}", "#d97706")
        st.caption(REASON_EXPLAINED.get(r["abstain_reason"], ""))
    else:
        banner(f"ROUTE -> {r['assigned_team']}", "#15803d")
    if r["escalation_flag"]:
        banner("Escalation risk - angry customer, or frustrated on an urgent ticket", "#b91c1c")
    for warning in r["warnings"]:
        st.warning(warning)

    st.subheader("Suggestion for the human" if r["abstain"] else "Prediction")
    cols = st.columns(4)
    metric_card(cols[0], "Category", r["category"], r["category_conf"])
    metric_card(cols[1], "Priority", r["priority"], r["priority_conf"])
    metric_card(cols[2], "Team", r["assigned_team"], r["team_conf"])
    metric_card(cols[3], "Sentiment", r["sentiment"], r["sentiment_conf"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Top 3 teams**")
        st.bar_chart(pd.DataFrame(r["top3_teams"]).set_index("team")["prob"], height=220)
    with right:
        st.markdown("**Priority probabilities**")
        st.bar_chart(pd.Series(r["prob_priority"], name="prob"), height=220)

    with st.expander("Why this team?"):
        st.write("Words and phrases that pushed the team model towards "
                 f"**{r['assigned_team']}** (higher weight = stronger push):")
        st.dataframe(pd.DataFrame(r["explanation"])[["term", "weight"]], hide_index=True)
        st.caption(f"Similarity to the nearest known category: {r['similarity']:.3f} "
                   "(low = unfamiliar topic).")


def tab_single(meta):
    curveballs = load_curveballs()
    options = [EXAMPLE_NONE] + [f"{row.ticket_id} - {row.subject}" for row in curveballs.itertuples()]
    st.selectbox("Load an example", options, key="example", on_change=fill_from_example)

    # Defaults the first time the page loads.
    st.session_state.setdefault("product", "GreenLake")
    st.session_state.setdefault("channel", "email")

    st.text_input("Subject", key="subject")
    st.text_area("Body", key="body", height=140)
    c1, c2 = st.columns(2)
    c1.selectbox("Product", meta["products"], key="product")
    c2.selectbox("Channel", meta["channels"], key="channel")

    if st.button("Triage", type="primary"):
        if not st.session_state.get("body", "").strip():
            st.error("Please enter a ticket body.")
            return
        ticket = {k: st.session_state[k] for k in ("subject", "body", "product", "channel")}
        with st.spinner("Asking the model..."):
            result = post_triage(ticket)
        if result:
            show_triage_result(result)


# ---------------------------------------------------------------------------
# Tab 2: batch CSV
# ---------------------------------------------------------------------------
def results_table(results):
    """Flatten the API results into a table (nested fields become text)."""
    df = pd.DataFrame(results)
    df["warnings"] = df["warnings"].apply("; ".join)
    df["why"] = df["explanation"].apply(lambda terms: ", ".join(t["term"] for t in terms))
    return df.drop(columns=["prob_priority", "top3_teams", "explanation"])


def submission_format(df):
    """Same shape as submission.csv: blank category/team when the ticket goes to a human."""
    sub = df.copy()
    sub.loc[sub["abstain"] == 1, ["category", "assigned_team"]] = ""
    return sub[SUBMISSION_COLUMNS]


def tab_batch():
    st.write("Upload a CSV with columns **subject, body, product, channel** (ticket_id optional).")
    uploaded = st.file_uploader("CSV file", type="csv")
    c1, c2 = st.columns(2)
    if c1.button("Triage uploaded CSV", disabled=uploaded is None):
        with st.spinner("Triaging..."):
            st.session_state["batch"] = post_batch(uploaded.name, uploaded.getvalue())
    if c2.button("Use eval sample (first 300 eval tickets)"):
        sample = pd.read_csv(EVAL_CSV).head(300)
        with st.spinner("Triaging 300 eval tickets..."):
            st.session_state["batch"] = post_batch("eval_sample.csv", sample.to_csv(index=False).encode())

    # Results live in session_state so changing a filter doesn't lose them.
    batch = st.session_state.get("batch")
    if not batch:
        return
    summary = batch["summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tickets", summary["n"])
    m2.metric("Sent to humans", f"{summary['abstain_rate']:.1%}")
    m3.metric("Escalation risks", summary["escalations"])
    m4.metric("15% cap applied", "Yes" if summary["cap_applied"] else "No")
    st.caption(summary["cap_note"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Routed tickets per team**")
        st.bar_chart(pd.Series(summary["per_team"], name="tickets"), height=250)
    with right:
        st.markdown("**Hand-offs per reason**")
        if summary["per_reason"]:
            st.bar_chart(pd.Series(summary["per_reason"], name="tickets"), height=250)
        else:
            st.write("No hand-offs.")

    df = results_table(batch["results"])
    f1, f2 = st.columns(2)
    teams = f1.multiselect("Filter by team", sorted(df["assigned_team"].unique()))
    decision = f2.selectbox("Filter by decision", ["All", "Routed", "Sent to human"])
    shown = df
    if teams:
        shown = shown[shown["assigned_team"].isin(teams)]
    if decision != "All":
        shown = shown[shown["abstain"] == (1 if decision == "Sent to human" else 0)]
    st.dataframe(shown, hide_index=True, width="stretch")

    d1, d2 = st.columns(2)
    d1.download_button("Download full results CSV", df.to_csv(index=False), "triage_results.csv", "text/csv")
    d2.download_button("Download submission-format CSV", submission_format(df).to_csv(index=False),
                       "submission_format.csv", "text/csv")


# ---------------------------------------------------------------------------
# Tab 3: how it works
# ---------------------------------------------------------------------------
def tab_how(meta, health):
    st.subheader("The pipeline")
    st.markdown(
        "1. **Clean the text** - remove the customer's own severity tag (it is only their opinion), "
        "lowercase, and add product / channel / severity as extra words.\n"
        "2. **Turn text into numbers** - TF-IDF on words and word pairs, plus character n-grams "
        "that still match when words are misspelled.\n"
        "3. **Four logistic regressions** predict category, priority, team and sentiment.\n"
        "4. **Route or hand off** - safety rules (too short, unfamiliar topic) first, then cost-aware "
        "confidence thresholds, within the human desk's 15% capacity."
    )

    st.subheader("Routing cost (penalty per ticket)")
    costs = load_cost_matrix()
    cost_table = pd.Series(costs).unstack()[["correct", "misroute", "abstain"]]
    cost_table.index.name = "true priority"
    st.dataframe(cost_table, width="content")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Confidence needed to route**")
        st.dataframe(pd.Series(meta["confidence_thresholds"], name="min team confidence"))
        st.caption("Break-even: 1 - abstain cost / misroute cost.")
    c2.metric("Unfamiliar-topic similarity threshold", f"{meta['unfamiliar_sim_threshold']:.3f}")
    c2.caption("Below this (and team confidence < 0.90) a ticket counts as unfamiliar.")
    c3.metric("Active policy", health["policy"])
    c3.caption(f"Max hand-off rate: {health['max_abstain_rate']:.0%}")

    st.subheader("Strategy comparison (20% validation set)")
    metrics_path = REPORTS_DIR / "metrics.csv"
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path), hide_index=True, width="stretch")
    chart_path = REPORTS_DIR / "routing_cost_by_strategy.png"
    if chart_path.exists():
        st.image(str(chart_path), width=650)

    st.subheader("Known limitations")
    st.markdown(
        "- **Sarcasm** - \"great job, it rebooted four times\" uses happy words in an angry ticket.\n"
        "- **Non-English text** - the model was trained on English; German or Malay tickets are often "
        "sent to a human as unfamiliar.\n"
        "- **Team exceptions** - e.g. GreenLake SSO / MFA issues belong to Account-Admin, not Security-Ops; "
        "rare exceptions are easy to miss.\n"
        "- **Noisy training labels** - some historical tickets were labelled wrongly, which caps accuracy."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main():
    st.title("HPE Smart Ticket Triage")
    st.write("Predicts category, priority, team and sentiment - and hands a ticket to a human "
             "when a mistake would cost more than a hand-off.")

    try:
        health = api_get("/health")
        meta = get_meta()
    except requests.exceptions.RequestException:
        st.error(f"Can't reach the triage API at {API_URL}. Start it from the project root with:")
        st.code(START_COMMAND, language="bash")
        st.stop()

    single, batch, how = st.tabs(["Triage a ticket", "Batch CSV", "How it works"])
    with single:
        tab_single(meta)
    with batch:
        tab_batch()
    with how:
        tab_how(meta, health)


main()
