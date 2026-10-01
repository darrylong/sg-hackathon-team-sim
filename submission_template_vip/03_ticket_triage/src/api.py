"""FastAPI backend for the ticket-triage MVP.

Start the server (from the project root):
    .venv/bin/python -m uvicorn api:app --app-dir submission_template_vip/03_ticket_triage/src --port 8000

Then open http://127.0.0.1:8000/docs to try every endpoint in the browser.

Endpoints:
    GET  /health        is the server up, which model/policy is loaded
    GET  /meta          label/product/channel lists + confidence thresholds (for dropdowns)
    POST /triage        triage ONE ticket (JSON in, JSON out)
    POST /triage/batch  triage a CSV upload, with the 15% abstain cap across the batch
"""

import io
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import policy
from config import (
    CATEGORIES,
    CHANNELS,
    PRIORITIES,
    PRODUCTS,
    SENTIMENTS,
    TEAMS,
    confidence_thresholds,
    load_max_abstain_rate,
)
from predict import load_bundle, predict_df
from text_utils import build_text

MODEL_FILE = "models_full.joblib"
REQUIRED_CSV_COLUMNS = ["subject", "body", "product", "channel"]
N_EXPLAIN_TERMS = 6
# Batches smaller than this skip the 15% abstain cap (see triage_batch).
MIN_BATCH_FOR_CAP = 100

# ---------------------------------------------------------------------------
# Load everything ONCE when the server starts (loading per request would be slow).
# ---------------------------------------------------------------------------
BUNDLE = load_bundle()
SIM_THRESHOLD = BUNDLE["unfamiliar_sim_threshold"]
MAX_ABSTAIN_RATE = load_max_abstain_rate()
# Names of all ~97k features, e.g. "word__array down" or "char__ rai"; used for explanations.
FEATURE_NAMES = BUNDLE["vectorizer"].get_feature_names_out()

app = FastAPI(
    title="HPE Ticket Triage API",
    description="Predicts category, priority, team and sentiment, and decides route vs. hand-off to a human.",
)

# CORS lets a front end running on another localhost port (e.g. Streamlit on 8501) call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------
class Ticket(BaseModel):
    """One ticket as an agent would paste it. Pydantic rejects bad input with a clear 422 error."""

    subject: str = ""
    body: str
    product: str = "GreenLake"
    channel: Literal["email", "portal", "chat"] = "email"

    @field_validator("body")
    @classmethod
    def body_not_empty(cls, value):
        # WHY: an empty body gives the model nothing to read - reject it up front.
        if not value.strip():
            raise ValueError("body must not be empty")
        return value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def explain_team(x_row, team):
    """Top terms that pushed the team model TOWARD the predicted team.

    Logistic regression scores a class as sum(feature_value * coefficient), so each
    feature's contribution is simply value * coefficient for that team.
    """
    model = BUNDLE["models"]["assigned_team"]
    class_index = list(model.classes_).index(team)
    row = x_row.tocsr()
    contributions = row.data * model.coef_[class_index, row.indices]

    terms, seen = [], set()
    for position in np.argsort(contributions)[::-1]:
        if contributions[position] <= 0:
            break
        kind, _, text = FEATURE_NAMES[row.indices[position]].partition("__")
        text = text.strip()
        # Only whole words/phrases are readable to an agent; character chunks like "nla" are not.
        if kind != "word" or not text or text in seen:
            continue
        seen.add(text)
        terms.append({"term": text, "type": kind, "weight": round(float(contributions[position]), 4)})
        if len(terms) == N_EXPLAIN_TERMS:
            break
    return terms


def input_warnings(product, channel, word_count):
    """Non-fatal problems the agent should know about."""
    warnings = []
    if product not in PRODUCTS:
        warnings.append(f"Unknown product '{product}' - prediction may be less reliable. Known: {', '.join(PRODUCTS)}")
    if channel not in CHANNELS:
        warnings.append(f"Unknown channel '{channel}' - expected one of {', '.join(CHANNELS)}")
    if word_count < policy.MIN_WORDS:
        warnings.append(f"Only {word_count} words - too short to triage reliably")
    return warnings


def build_result(row, x_row, abstain, reason, warnings):
    """Turn one prediction row into the JSON response (plain Python types, not numpy)."""
    top3 = sorted(((t, float(row[f"prob_team_{t}"])) for t in TEAMS), key=lambda item: item[1], reverse=True)[:3]
    return {
        "ticket_id": row["ticket_id"],
        # When abstain=1 we STILL return the best guesses - the human sees them as a suggestion.
        "category": row["category"],
        "category_conf": round(float(row["category_conf"]), 4),
        "priority": row["priority"],
        "priority_conf": round(float(row["priority_conf"]), 4),
        "prob_priority": {p: round(float(row[f"prob_priority_{p}"]), 4) for p in PRIORITIES},
        "assigned_team": row["assigned_team"],
        "team_conf": round(float(row["team_conf"]), 4),
        "top3_teams": [{"team": t, "prob": round(p, 4)} for t, p in top3],
        "sentiment": row["sentiment"],
        "sentiment_conf": round(float(row["sentiment_conf"]), 4),
        "similarity": round(float(row["similarity"]), 4),
        "escalation_flag": int(row["escalation_flag"]),
        "abstain": int(abstain),
        "abstain_reason": reason or "",
        "warnings": warnings,
        "explanation": explain_team(x_row, row["assigned_team"]),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_FILE, "policy": policy.POLICY, "max_abstain_rate": MAX_ABSTAIN_RATE}


@app.get("/meta")
def meta():
    """Everything the front end needs to build its dropdowns and explain thresholds."""
    return {
        "categories": CATEGORIES,
        "priorities": PRIORITIES,
        "teams": TEAMS,
        "sentiments": SENTIMENTS,
        "products": PRODUCTS,
        "channels": CHANNELS,
        "confidence_thresholds": confidence_thresholds(),
        "unfamiliar_sim_threshold": SIM_THRESHOLD,
        "max_abstain_rate": MAX_ABSTAIN_RATE,
        "policy": policy.POLICY,
    }


@app.post("/triage")
def triage(ticket: Ticket):
    """Triage one ticket. No capacity cap here - with one ticket there is no batch to rank."""
    df = pd.DataFrame([{"ticket_id": "TK-API-0001", **ticket.model_dump()}])
    row = predict_df(df).iloc[0]
    abstain, reason, _ = policy.decide_single(
        row["team_conf"], row["similarity"],
        {p: row[f"prob_priority_{p}"] for p in PRIORITIES}, row["word_count"],
        sim_threshold=SIM_THRESHOLD, predicted_priority=row["priority"],
    )
    x_row = BUNDLE["vectorizer"].transform(build_text(df))
    warnings = input_warnings(ticket.product, ticket.channel, int(row["word_count"]))
    return build_result(row, x_row, abstain, reason, warnings)


@app.post("/triage/batch")
async def triage_batch(file: UploadFile = File(...)):
    """Triage a CSV of tickets, applying the 15% abstain cap across the whole batch."""
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as error:  # not a CSV, wrong encoding, empty file...
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {error}")

    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing column(s): {', '.join(missing)}. "
                   f"Required: {', '.join(REQUIRED_CSV_COLUMNS)} (ticket_id optional).",
        )
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    if "ticket_id" not in df.columns:
        df.insert(0, "ticket_id", [f"TK-UPLOAD-{i:04d}" for i in range(1, len(df) + 1)])
    df["product"] = df["product"].fillna("").astype(str)
    df["channel"] = df["channel"].fillna("").astype(str)

    predictions = predict_df(df)
    # The 15% cap models the human desk's capacity over a large stream of tickets.
    # On a handful of tickets it would allow only 1-2 hand-offs, so small batches are uncapped.
    cap_applied = len(df) >= MIN_BATCH_FOR_CAP
    if cap_applied:
        cap_note = f"{MAX_ABSTAIN_RATE:.0%} hand-off cap applied across {len(df)} tickets."
    else:
        cap_note = (f"No cap: batch has {len(df)} tickets (< {MIN_BATCH_FOR_CAP}); "
                    "each ticket is decided on its own, like /triage.")
    decided = policy.decide_with_policy(predictions, MAX_ABSTAIN_RATE if cap_applied else 1.0,
                                        sim_threshold=SIM_THRESHOLD)
    X = BUNDLE["vectorizer"].transform(build_text(df))

    results = []
    for i, (_, row) in enumerate(decided.iterrows()):
        warnings = input_warnings(df["product"].iloc[i], df["channel"].iloc[i], int(row["word_count"]))
        results.append(build_result(row, X[i], row["abstain"], row["abstain_reason"], warnings))

    routed = decided[decided["abstain"] == 0]
    summary = {
        "n": len(decided),
        "abstain_rate": round(float(decided["abstain"].mean()), 4),
        "per_team": {k: int(v) for k, v in routed["assigned_team"].value_counts().items()},
        "per_reason": {k: int(v) for k, v in decided.loc[decided["abstain"] == 1, "abstain_reason"]
                       .value_counts().items()},
        "escalations": int(decided["escalation_flag"].sum()),
        "cap_applied": cap_applied,
        "cap_note": cap_note,
    }
    return {"summary": summary, "results": results}
