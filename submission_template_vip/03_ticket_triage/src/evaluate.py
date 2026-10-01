"""Compare abstain strategies on the 20% validation set, scored like the hackathon.

Strategies:
  A  Never abstain               - route every ticket
  B  Simple thresholds           - abstain if team confidence < README rule of thumb
                                   for the PREDICTED priority, capped at 15%
  C  Ours: cost-aware + cap      - policy.decide (expected cost using priority
                                   probabilities, short/unfamiliar rules, 15% cap)

Run after train.py:
    .venv/bin/python submission_template_vip/03_ticket_triage/src/evaluate.py
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save charts to files, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import policy
from centroids import max_similarity
from config import (
    CATEGORIES,
    MODELS_DIR,
    PRIORITIES,
    SENTIMENTS,
    TEAMS,
    TICKETS_CSV,
    confidence_thresholds,
    load_cost_matrix,
    load_max_abstain_rate,
)
from text_utils import build_text, word_count

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
ABSTAINED = "<abstain>"  # placeholder label so an abstained ticket counts as a miss


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_validation():
    """Validation predictions from train.py plus word_count from the original tickets."""
    print("Loading validation predictions...")
    val = pd.read_csv(REPORTS_DIR / "val_predictions.csv")
    tickets = pd.read_csv(TICKETS_CSV)
    tickets["word_count"] = word_count(tickets)
    val = val.merge(tickets[["ticket_id", "word_count"]], on="ticket_id", how="left")

    # Older val_predictions.csv files may lack these columns: recompute them with the
    # saved 80% bundle (no retraining - the validation tickets were never trained on).
    bundle = joblib.load(MODELS_DIR / "models.joblib")
    missing = [c for c in ("category_conf", "similarity") if c not in val.columns]
    if missing:
        print(f"  {', '.join(missing)} missing - computing with models.joblib...")
        rows = tickets.set_index("ticket_id").loc[val["ticket_id"]].reset_index()
        X = bundle["vectorizer"].transform(build_text(rows))
        if "category_conf" in missing:
            val["category_conf"] = bundle["models"]["category"].predict_proba(X).max(axis=1)
        if "similarity" in missing:
            val["similarity"] = max_similarity(X, bundle["centroids"])
        # Save the new columns back so the next run (and the app) can reuse them.
        val.drop(columns="word_count").to_csv(REPORTS_DIR / "val_predictions.csv", index=False)
        print("  Added them to val_predictions.csv")

    # Threshold learned by centroids.py / train.py from validation similarity.
    val.attrs["sim_threshold"] = bundle["unfamiliar_sim_threshold"]
    print(f"  {len(val):,} tickets; UNFAMILIAR_SIM_THRESHOLD = {val.attrs['sim_threshold']:.4f}")
    return val


# ---------------------------------------------------------------------------
# Strategies - each returns an abstain column (0/1) for the validation set
# ---------------------------------------------------------------------------
def strategy_never(val, _max_rate):
    return pd.Series(0, index=val.index)


def strategy_simple(val, max_rate):
    """README rule of thumb applied to the predicted priority, then capped by lowest confidence."""
    thresholds = confidence_thresholds()
    needed = val["pred_priority"].map(thresholds)
    want = val[val["team_conf"] < needed]
    allowed = policy.abstain_budget(len(val), max_rate)
    chosen = want["team_conf"].nsmallest(allowed).index
    abstain = pd.Series(0, index=val.index)
    abstain[chosen] = 1
    return abstain


def strategy_ours(val, max_rate):
    decided = policy.decide(val, max_rate, sim_threshold=val.attrs["sim_threshold"])
    val["abstain_reason"] = decided["abstain_reason"]  # kept for the per-reason breakdown
    return decided["abstain"]


def strategy_hybrid(val, max_rate):
    decided = policy.decide_hybrid(val, max_rate, sim_threshold=val.attrs["sim_threshold"],
                                   priority_col="pred_priority")
    val["abstain_reason_D"] = decided["abstain_reason"]
    return decided["abstain"]


STRATEGY_C = "C: Ours (cost-aware + cap)"
STRATEGY_D = "D: Hybrid (safety rules + thresholds)"
STRATEGIES = {
    "A: Never abstain": strategy_never,
    "B: Simple thresholds": strategy_simple,
    STRATEGY_C: strategy_ours,
    STRATEGY_D: strategy_hybrid,
}


# ---------------------------------------------------------------------------
# Scoring (mirrors "How you are scored" in the pack README)
# ---------------------------------------------------------------------------
def per_ticket_cost(val, abstain, costs):
    """Penalty for each ticket, using its TRUE priority."""
    misroute = val["true_priority"].map(lambda p: costs[(p, "misroute")])
    abstain_cost = val["true_priority"].map(lambda p: costs[(p, "abstain")])
    wrong_team = val["pred_team"] != val["true_team"]
    return np.where(abstain == 1, abstain_cost, np.where(wrong_team, misroute, 0.0))


def score(val, abstain, costs):
    """All headline metrics for one strategy."""
    routed = abstain == 0
    # An abstained ticket gets no category, so it can never be a hit.
    category_pred = val["pred_category"].where(routed, ABSTAINED)
    true_esc = policy.escalation_flag(val["true_sentiment"], val["true_priority"])
    pred_esc = policy.escalation_flag(val["pred_sentiment"], val["pred_priority"])
    return {
        "routing_cost": per_ticket_cost(val, abstain, costs).mean(),
        "abstain_rate": abstain.mean(),
        "category_macro_f1": f1_score(val["true_category"], category_pred, labels=CATEGORIES,
                                      average="macro", zero_division=0),
        "team_macro_f1_routed": f1_score(val.loc[routed, "true_team"], val.loc[routed, "pred_team"],
                                         labels=TEAMS, average="macro", zero_division=0),
        "priority_macro_f1": f1_score(val["true_priority"], val["pred_priority"], labels=PRIORITIES,
                                      average="macro", zero_division=0),
        "sentiment_macro_f1": f1_score(val["true_sentiment"], val["pred_sentiment"], labels=SENTIMENTS,
                                       average="macro", zero_division=0),
        "P1_recall": recall_score(val["true_priority"] == "P1", val["pred_priority"] == "P1"),
        "P1_precision": precision_score(val["true_priority"] == "P1", val["pred_priority"] == "P1"),
        "escalation_f1": f1_score(true_esc, pred_esc),
    }


def explain_ours(val, abstain, costs, label="C", reason_col="abstain_reason"):
    """Breakdown for one of our strategies: why we abstained, and where the remaining cost comes from."""
    print(f"\nStrategy {label} - abstains per reason:")
    reasons = val.loc[abstain == 1, reason_col].value_counts()
    print(reasons.to_string())

    cost = pd.Series(per_ticket_cost(val, abstain, costs), index=val.index)
    breakdown = pd.DataFrame({
        "tickets": val.groupby("true_priority").size(),
        "abstained": abstain.groupby(val["true_priority"]).sum(),
        "misrouted": ((abstain == 0) & (val["pred_team"] != val["true_team"])).groupby(val["true_priority"]).sum(),
        "total_cost": cost.groupby(val["true_priority"]).sum().round(1),
    })
    breakdown["share_of_cost_%"] = (breakdown["total_cost"] / breakdown["total_cost"].sum() * 100).round(1)
    print(f"\nStrategy {label} - routing cost split by TRUE priority:")
    print(breakdown.to_string())


def save_chart(table):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["lightgray", "steelblue", "seagreen", "darkorange"]
    bars = ax.bar(table.index, table["routing_cost"], color=colors)
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel("routing cost (avg penalty per ticket, lower is better)")
    ax.set_title("Routing cost on validation by abstain strategy")
    fig.tight_layout()
    path = REPORTS_DIR / "routing_cost_by_strategy.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path.name}")


def main():
    val = load_validation()
    costs = load_cost_matrix()
    max_rate = load_max_abstain_rate()
    print(f"Max abstain rate {max_rate:.0%} -> budget {policy.abstain_budget(len(val), max_rate)} tickets")

    results, abstains = {}, {}
    for name, strategy in STRATEGIES.items():
        print(f"Running {name}...")
        abstains[name] = strategy(val, max_rate)
        results[name] = score(val, abstains[name], costs)

    table = pd.DataFrame(results).T
    table.index.name = "strategy"
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON (20% validation set)")
    print("=" * 100)
    print(table.round(4).to_string())

    explain_ours(val, abstains[STRATEGY_C], costs, "C", "abstain_reason")
    explain_ours(val, abstains[STRATEGY_D], costs, "D", "abstain_reason_D")

    print()
    table.round(4).to_csv(REPORTS_DIR / "metrics.csv")
    print("Saved metrics.csv")
    save_chart(table)

    # Pick C or D for predict.py: the one with the lower validation routing cost.
    cost_c, cost_d = table.loc[STRATEGY_C, "routing_cost"], table.loc[STRATEGY_D, "routing_cost"]
    best = "D" if cost_d < cost_c else "C"
    print("\n" + "=" * 100)
    print(f"POLICY CHOICE: {best}  (validation routing cost C = {cost_c:.4f}, D = {cost_d:.4f})")
    print(f"  -> set POLICY = \"{best}\" and VALIDATION_COSTS = {{\"C\": {cost_c:.4f}, \"D\": {cost_d:.4f}}} "
          f"in policy.py (currently POLICY = \"{policy.POLICY}\")")
    print("=" * 100)

    a, b = table.iloc[0], table.iloc[1]
    ours = table.loc[STRATEGY_D if best == "D" else STRATEGY_C]
    print("\nSUMMARY FOR THE SLIDE")
    print("=" * 100)
    print(f"1. Routing every ticket costs {a['routing_cost']:.3f} per ticket; our policy cuts it to "
          f"{ours['routing_cost']:.3f} ({1 - ours['routing_cost'] / a['routing_cost']:.0%} lower).")
    # Word line 2 from the actual numbers - never claim a win the data doesn't show.
    gap = b["routing_cost"] - ours["routing_cost"]
    if gap > 0.002:
        print(f"2. It beats simple confidence thresholds ({b['routing_cost']:.3f}).")
    elif gap >= -0.002:
        print(f"2. It matches simple confidence thresholds ({b['routing_cost']:.3f}) on cost, and adds what they lack: "
              f"it abstains on too-short and unfamiliar-topic tickets (the eval traps validation doesn't contain).")
    else:
        print(f"2. It costs slightly more than simple thresholds ({b['routing_cost']:.3f}) on validation, because it "
              f"reserves budget for too-short and unfamiliar-topic tickets (eval traps validation doesn't contain).")
    print(f"3. It hands {ours['abstain_rate']:.1%} of tickets to humans - inside the {max_rate:.0%} capacity limit.")
    print(f"4. Routed tickets reach the right team with macro-F1 {ours['team_macro_f1_routed']:.3f}; "
          f"category macro-F1 is {ours['category_macro_f1']:.3f} even counting every hand-off as a miss.")


if __name__ == "__main__":
    main()
