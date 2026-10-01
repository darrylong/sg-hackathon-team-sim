"""Make the two submission files (eval + curveball) and check them against every README rule.

Uses models_full.joblib (trained on 100% of tickets.csv by train.py).

Run:
    .venv/bin/python submission_template_vip/03_ticket_triage/src/predict.py

Writes:
    submission.csv, submission_curveball.csv           -> the hand-in files (OUTPUT_DIR)
    src/reports/eval_predictions_full.csv,
    src/reports/curveball_predictions_full.csv         -> extra detail for the app and slides
"""

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

import policy
from centroids import max_similarity
from config import (
    CATEGORIES,
    CURVEBALL_CSV,
    EVAL_CSV,
    MODELS_DIR,
    OUTPUT_DIR,
    PRIORITIES,
    SAMPLE_SUB_CSV,
    SAMPLE_SUB_CURVEBALL_CSV,
    SENTIMENTS,
    SUBMISSION_COLUMNS,
    TEAMS,
    TICKETS_CSV,
    load_max_abstain_rate,
)
from text_utils import build_text, word_count

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Model target -> short name used for confidence columns (team_conf, not assigned_team_conf).
TARGETS = {"category": "category", "priority": "priority", "assigned_team": "team", "sentiment": "sentiment"}


# ---------------------------------------------------------------------------
# Prediction (reused by the API later)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_bundle():
    """Load the 100% model bundle once and keep it in memory (the API calls this repeatedly)."""
    path = MODELS_DIR / "models_full.joblib"
    print(f"Loading {path.name}...")
    return joblib.load(path)


def predict_df(df):
    """Predict all 4 labels for any DataFrame with ticket_id, channel, subject, body, product.

    Returns one row per ticket with the predicted labels, their confidences,
    priority probabilities, word_count and escalation_flag - everything
    policy.decide needs. No abstain decision is made here.
    """
    bundle = load_bundle()
    # Exactly the same text cleaning as training, or the model sees unfamiliar input.
    X = bundle["vectorizer"].transform(build_text(df))

    out = pd.DataFrame({"ticket_id": df["ticket_id"].values})
    out["word_count"] = word_count(df).values
    for target, short in TARGETS.items():
        model = bundle["models"][target]
        probs = model.predict_proba(X)
        out[target] = model.classes_[probs.argmax(axis=1)]
        out[f"{short}_conf"] = probs.max(axis=1)
        if target in ("priority", "assigned_team"):
            # Full probabilities: the policy weighs the chance of an urgent ticket,
            # and the app shows the top-3 candidate teams.
            for i, label in enumerate(model.classes_):
                out[f"prob_{short}_{label}"] = probs[:, i]
    # How close the ticket is to its nearest known category (low = unfamiliar topic).
    out["similarity"] = max_similarity(X, bundle["centroids"])
    out["escalation_flag"] = policy.escalation_flag(out["sentiment"], out["priority"])
    return out


def to_submission(decided):
    """Keep exactly the submission columns; blank category/team when we hand off to a human."""
    sub = decided.copy()
    handed_off = sub["abstain"] == 1
    sub.loc[handed_off, ["category", "assigned_team"]] = ""
    sub["abstain"] = sub["abstain"].astype(int)
    return sub[SUBMISSION_COLUMNS]


def run_file(input_csv, sub_path, full_path, max_abstain_rate):
    """Predict one input file, apply the abstain policy, and write both outputs."""
    tickets = pd.read_csv(input_csv)
    print(f"\nPredicting {input_csv.name} ({len(tickets):,} tickets)...")
    predictions = predict_df(tickets)
    # The cap applies to the whole file at once, so decide on all tickets together.
    decided = policy.decide_with_policy(predictions, max_abstain_rate,
                                        sim_threshold=load_bundle()["unfamiliar_sim_threshold"])

    to_submission(decided).to_csv(sub_path, index=False, encoding="utf-8")
    print(f"  Wrote {sub_path.name}")
    decided.to_csv(full_path, index=False, encoding="utf-8")
    print(f"  Wrote {full_path.name} (detail for app/slides - NOT the submission)")
    return tickets, decided


# ---------------------------------------------------------------------------
# Validation against the README rules
# ---------------------------------------------------------------------------
def validate_submission(sub_path, input_csv, sample_csv, check_abstain_rate):
    """Check one submission file against every README validity rule. Returns True if all pass."""
    print(f"\nValidating {sub_path.name}:")
    # Read everything as text and keep blanks as "" (not NaN) so empty cells are easy to check.
    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    input_ids = pd.read_csv(input_csv, dtype=str)["ticket_id"]
    sample_columns = list(pd.read_csv(sample_csv, nrows=0).columns)

    routed = sub["abstain"] == "0"
    checks = {
        "exact columns and order (same as sample file)": list(sub.columns) == sample_columns,
        "no duplicate ticket_ids": not sub["ticket_id"].duplicated().any(),
        "every input ticket_id present": set(input_ids) <= set(sub["ticket_id"]),
        "no unknown ticket_ids": set(sub["ticket_id"]) <= set(input_ids),
        "row count equals input count": len(sub) == len(input_ids),
        "abstain is 0 or 1": sub["abstain"].isin(["0", "1"]).all(),
        "priority filled and valid on every row": sub["priority"].isin(PRIORITIES).all(),
        "sentiment filled and valid on every row": sub["sentiment"].isin(SENTIMENTS).all(),
        "category filled and valid when abstain=0": sub.loc[routed, "category"].isin(CATEGORIES).all(),
        "assigned_team filled and valid when abstain=0": sub.loc[routed, "assigned_team"].isin(TEAMS).all(),
        "category/team blank or valid when abstain=1": (
            sub.loc[~routed, "category"].isin(CATEGORIES + [""]).all()
            and sub.loc[~routed, "assigned_team"].isin(TEAMS + [""]).all()
        ),
    }
    if check_abstain_rate:
        max_rate = load_max_abstain_rate()
        rate = (sub["abstain"] == "1").mean()
        checks[f"abstain rate {rate:.2%} <= {max_rate:.0%}"] = rate <= max_rate

    for name, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {name}")
    return all(checks.values())


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def sanity_report(decided):
    """Do eval predictions look like the training data? Big gaps would hint at a bug."""
    print("\n" + "=" * 70)
    print("SANITY REPORT - eval predictions vs training labels")
    print("=" * 70)
    train = pd.read_csv(TICKETS_CSV)
    for target in TARGETS:
        table = pd.DataFrame({
            "train %": train[target].value_counts(normalize=True) * 100,
            "eval predicted %": decided[target].value_counts(normalize=True) * 100,
        }).fillna(0).round(1)
        table["diff"] = (table["eval predicted %"] - table["train %"]).round(1)
        print(f"\n{target}:")
        print(table.to_string())

    print(f"\nAbstain rate: {decided['abstain'].mean():.2%} ({decided['abstain'].sum()} tickets)")
    print("Abstains per reason:")
    print(decided.loc[decided["abstain"] == 1, "abstain_reason"].value_counts().to_string())
    print(f"Escalation flags: {decided['escalation_flag'].sum()} ({decided['escalation_flag'].mean():.1%})")
    short = decided["word_count"] < policy.MIN_WORDS
    print(f"Tickets under {policy.MIN_WORDS} words: {short.sum()}, abstained: {decided.loc[short, 'abstain'].sum()}")


def curveball_table(tickets, decided):
    """Show all 15 hand-written edge cases in one readable table."""
    print("\n" + "=" * 70)
    print("CURVEBALL TICKETS")
    print("=" * 70)
    table = pd.DataFrame({
        "id": decided["ticket_id"],
        "subject": tickets["subject"].str.slice(0, 28).values,
        "category": decided["category"],
        "priority": decided["priority"],
        "team": decided["assigned_team"],
        "sentiment": decided["sentiment"],
        "abstain": decided["abstain"],
        "reason": decided["abstain_reason"],
        "team_conf": decided["team_conf"].round(2),
        "similarity": decided["similarity"].round(3),
    })
    print(f"(Unfamiliar if similarity < {load_bundle()['unfamiliar_sim_threshold']:.4f} "
          f"and team_conf < {policy.UNFAMILIAR_TEAM_CONF})")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.to_string(index=False))


def main():
    max_rate = load_max_abstain_rate()
    print(f"Max abstain rate: {max_rate:.0%}")
    names = {"C": "cost-aware + cap", "D": "hybrid: safety rules + thresholds"}
    costs = ", ".join(f"{k} = {v:.4f}" for k, v in policy.VALIDATION_COSTS.items())
    print(f"Abstain policy: {policy.POLICY} ({names[policy.POLICY]}) - "
          f"lowest validation routing cost ({costs})")

    eval_tickets, eval_decided = run_file(
        EVAL_CSV, OUTPUT_DIR / "submission.csv", REPORTS_DIR / "eval_predictions_full.csv", max_rate)
    # No cap for curveball: the README measures the 15% capacity on submission.csv only,
    # and the edge-case file is scored per ticket - so every ticket the rules flag may abstain.
    cb_tickets, cb_decided = run_file(
        CURVEBALL_CSV, OUTPUT_DIR / "submission_curveball.csv", REPORTS_DIR / "curveball_predictions_full.csv",
        max_abstain_rate=1.0)

    eval_ok = validate_submission(OUTPUT_DIR / "submission.csv", EVAL_CSV, SAMPLE_SUB_CSV, check_abstain_rate=True)
    # The README measures the 15% cap on submission.csv only - the edge-case file is scored separately.
    cb_ok = validate_submission(OUTPUT_DIR / "submission_curveball.csv", CURVEBALL_CSV, SAMPLE_SUB_CURVEBALL_CSV,
                                check_abstain_rate=False)

    sanity_report(eval_decided)
    curveball_table(cb_tickets, cb_decided)

    if not (eval_ok and cb_ok):
        sys.exit("\n❌ Submission validation FAILED - fix the ❌ items above before handing in.")
    print("\n✅ Both submission files pass every validity check.")


if __name__ == "__main__":
    main()
