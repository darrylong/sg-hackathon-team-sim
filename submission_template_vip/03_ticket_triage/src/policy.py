"""Route-or-abstain policy: decide which tickets go to the human triage desk.

WHY: a wrong team costs far more on urgent tickets (P1 misroute = 8, P1 abstain = 1).
So for each ticket we compare the EXPECTED cost of routing with the cost of
handing it to a human, then spend our limited abstain budget (15%) where it
saves the most.

All functions are pure: they take numbers/DataFrames and return results.
The only thing read from disk is the cost matrix (via config) when none is passed in.
"""

import math

import numpy as np
import pandas as pd

from config import PRIORITIES, confidence_thresholds, load_cost_matrix

# Which strategy predict.py and the API use: "C" (cost-aware) or "D" (hybrid).
# Set from the validation routing cost printed by evaluate.py - see VALIDATION_COSTS.
# D chosen: lower validation routing cost than C (evaluate.py, 6,000 validation tickets).
POLICY = "D"
VALIDATION_COSTS = {"C": 0.1585, "D": 0.1568}

# "Unfamiliar topic" = the ticket is far from every known category centroid
# (similarity below the threshold) AND the team model isn't very sure either.
# The eval set contains a category never seen in training (README).
# The similarity threshold is learned from data (2nd percentile on validation,
# see centroids.py) and stored in the model bundle as "unfamiliar_sim_threshold";
# callers pass it in. This value is only a fallback for tests.
UNFAMILIAR_SIM_THRESHOLD = 0.20
# Even if a ticket looks unusual, a very confident team prediction means we know who owns it.
UNFAMILIAR_TEAM_CONF = 0.90
# Tickets shorter than this carry too little information to route ("it broke").
MIN_WORDS = 8
# Use slightly less than the full capacity so we never go over the limit by rounding.
CAPACITY_SAFETY = 0.98
URGENT_PRIORITIES = ["P1", "P2"]

REASON_SHORT = "Too short to understand"
REASON_UNFAMILIAR = "Unfamiliar topic"
REASON_URGENT = "Urgent and unsure"
REASON_LOW_CONF = "Low confidence"

# Rank unfamiliar tickets above every cost-based candidate when the budget is tight.
UNFAMILIAR_BONUS = 1_000.0


def expected_costs(team_conf, priority_probs, cost_matrix):
    """Expected cost of routing vs abstaining, using the model's own probabilities.

    team_conf:      probability our predicted team is right (number or array)
    priority_probs: P(P1..P4) - a DataFrame/2-D array with columns in PRIORITIES
                    order, or a dict {"P1": .., ...} for a single ticket
    cost_matrix:    {(priority, outcome): cost} from load_cost_matrix()

    WHY use priority PROBABILITIES instead of the predicted priority: if the
    model says "P3, but 30% chance it's P1", that 30% of an 8-point misroute
    matters. Averaging over all priorities captures that risk.

    Returns (route_cost, abstain_cost, saving). saving > 0 means abstaining is cheaper.
    """
    if isinstance(priority_probs, dict):
        probs = np.array([priority_probs[p] for p in PRIORITIES], dtype=float)
    elif isinstance(priority_probs, pd.DataFrame):
        probs = priority_probs[[f"prob_priority_{p}" for p in PRIORITIES]].to_numpy(dtype=float)
    else:
        probs = np.asarray(priority_probs, dtype=float)

    misroute = np.array([cost_matrix[(p, "misroute")] for p in PRIORITIES])
    abstain = np.array([cost_matrix[(p, "abstain")] for p in PRIORITIES])
    team_wrong = 1 - np.asarray(team_conf, dtype=float)

    # probs @ misroute = expected misroute penalty IF the team is wrong.
    route_cost = team_wrong * (probs @ misroute)
    abstain_cost = probs @ abstain
    return route_cost, abstain_cost, route_cost - abstain_cost


def abstain_budget(n_tickets, max_abstain_rate):
    """How many tickets we may abstain on, with a small safety margin."""
    return math.floor(max_abstain_rate * n_tickets * CAPACITY_SAFETY)


def _cost_reason(p_urgent):
    """Explain a cost-based abstain in words an agent understands."""
    return REASON_URGENT if p_urgent >= 0.5 else REASON_LOW_CONF


def is_unfamiliar(similarity, team_conf, sim_threshold):
    """True when a ticket looks unlike every known category AND the team is uncertain."""
    return (similarity < sim_threshold) & (team_conf < UNFAMILIAR_TEAM_CONF)


def decide(df, max_abstain_rate, cost_matrix=None, sim_threshold=UNFAMILIAR_SIM_THRESHOLD):
    """Decide abstain (1) or route (0) for every ticket in df, respecting capacity.

    df needs columns: team_conf, similarity, prob_priority_P1..P4, word_count.
    sim_threshold: pass bundle["unfamiliar_sim_threshold"] (learned from validation).
    Returns a copy of df with new columns: abstain, abstain_reason, saving.
    """
    cost_matrix = cost_matrix or load_cost_matrix()
    out = df.copy()
    _, _, saving = expected_costs(out["team_conf"], out, cost_matrix)
    out["saving"] = saving
    p_urgent = out["prob_priority_P1"] + out["prob_priority_P2"]

    # Rules 1-3: who WANTS to abstain, and why.
    forced = out["word_count"] < MIN_WORDS
    unfamiliar = ~forced & is_unfamiliar(out["similarity"], out["team_conf"], sim_threshold)
    costly = ~forced & ~unfamiliar & (out["saving"] > 0)

    reason = pd.Series("", index=out.index)
    reason[forced] = REASON_SHORT
    reason[unfamiliar] = REASON_UNFAMILIAR
    reason[costly] = p_urgent[costly].apply(_cost_reason)

    # Rule 4: capacity. Score = how much we want to abstain; forced first, then
    # unfamiliar, then by expected saving. Keep only the top `allowed`.
    score = out["saving"].copy()
    score[unfamiliar] += UNFAMILIAR_BONUS
    score[forced] += 2 * UNFAMILIAR_BONUS
    candidates = score[forced | unfamiliar | costly].sort_values(ascending=False)
    chosen = candidates.index[: abstain_budget(len(out), max_abstain_rate)]

    out["abstain"] = 0
    out.loc[chosen, "abstain"] = 1
    out["abstain_reason"] = np.where(out["abstain"] == 1, reason, "")
    return out


def _threshold_reason(predicted_priority):
    """Reason text for a threshold-based abstain, from the PREDICTED priority."""
    return REASON_URGENT if predicted_priority in URGENT_PRIORITIES else REASON_LOW_CONF


def decide_hybrid(df, max_abstain_rate, cost_matrix=None, sim_threshold=UNFAMILIAR_SIM_THRESHOLD,
                  priority_col="priority"):
    """Strategy D: safety rules first, then the README rule of thumb for the rest.

      1. word_count < 8                         -> forced abstain "Too short to understand"
      2. centroid unfamiliar rule              -> abstain "Unfamiliar topic"
      3. remaining budget: team_conf below the cost-break-even threshold for the
         PREDICTED priority (e.g. 0.875 for P1), biggest shortfall first.

    WHY: on validation, simple per-priority thresholds (strategy B) were the cheapest
    way to use the budget, but B has no protection against short or brand-new-topic
    tickets. D keeps B's ranking and adds those two safety nets.

    priority_col: column holding the predicted priority ("priority" from predict_df,
    "pred_priority" in val_predictions.csv).
    Returns a copy of df with abstain, abstain_reason, saving (saving kept for reporting).
    """
    cost_matrix = cost_matrix or load_cost_matrix()
    out = df.copy()
    _, _, out["saving"] = expected_costs(out["team_conf"], out, cost_matrix)

    # Break-even confidence per priority, e.g. {"P1": 0.875, ...}, from the cost matrix.
    needed = out[priority_col].map(confidence_thresholds())
    gap = needed - out["team_conf"]  # > 0 means less confident than routing requires

    forced = out["word_count"] < MIN_WORDS
    unfamiliar = ~forced & is_unfamiliar(out["similarity"], out["team_conf"], sim_threshold)
    below = ~forced & ~unfamiliar & (gap > 0)

    reason = pd.Series("", index=out.index)
    reason[forced] = REASON_SHORT
    reason[unfamiliar] = REASON_UNFAMILIAR
    reason[below] = out.loc[below, priority_col].apply(_threshold_reason)

    # Capacity: forced first, then unfamiliar, then the largest confidence shortfall.
    score = gap.copy()
    score[unfamiliar] = UNFAMILIAR_BONUS + (sim_threshold - out["similarity"])
    score[forced] = 2 * UNFAMILIAR_BONUS
    candidates = score[forced | unfamiliar | below].sort_values(ascending=False)
    chosen = candidates.index[: abstain_budget(len(out), max_abstain_rate)]

    out["abstain"] = 0
    out.loc[chosen, "abstain"] = 1
    out["abstain_reason"] = np.where(out["abstain"] == 1, reason, "")
    return out


def decide_with_policy(df, max_abstain_rate, sim_threshold, priority_col="priority", strategy=None):
    """Run whichever strategy POLICY selects ("C" = cost-aware, "D" = hybrid)."""
    strategy = strategy or POLICY
    if strategy == "D":
        return decide_hybrid(df, max_abstain_rate, sim_threshold=sim_threshold, priority_col=priority_col)
    return decide(df, max_abstain_rate, sim_threshold=sim_threshold)


def decide_single(team_conf, similarity, priority_probs, word_count, cost_matrix=None,
                  sim_threshold=UNFAMILIAR_SIM_THRESHOLD, predicted_priority=None, strategy=None):
    """Route-or-abstain for ONE ticket (used by the API, where there is no batch to rank).

    Applies rules 1-3 only - capacity can't be enforced one ticket at a time.
    priority_probs is a dict {"P1": .., "P2": .., "P3": .., "P4": ..}.
    predicted_priority defaults to the most likely priority in priority_probs.
    strategy: "C" or "D" (defaults to POLICY).
    Returns (abstain 0/1, reason, saving).
    """
    strategy = strategy or POLICY
    cost_matrix = cost_matrix or load_cost_matrix()
    _, _, saving = expected_costs(team_conf, priority_probs, cost_matrix)
    saving = float(saving)

    if word_count < MIN_WORDS:
        return 1, REASON_SHORT, saving
    if is_unfamiliar(similarity, team_conf, sim_threshold):
        return 1, REASON_UNFAMILIAR, saving
    if strategy == "D":
        predicted_priority = predicted_priority or max(priority_probs, key=priority_probs.get)
        if team_conf < confidence_thresholds()[predicted_priority]:
            return 1, _threshold_reason(predicted_priority), saving
    elif saving > 0:
        return 1, _cost_reason(priority_probs["P1"] + priority_probs["P2"]), saving
    return 0, "", saving


def escalation_flag(sentiment, priority):
    """1 if the ticket is an escalation risk: Angry, or Frustrated on a P1/P2.

    Works on single values ("Angry", "P3") or on whole columns (pandas Series / arrays).
    """
    sentiment = np.asarray(sentiment)
    flag = (sentiment == "Angry") | ((sentiment == "Frustrated") & np.isin(priority, URGENT_PRIORITIES))
    flag = flag.astype(int)
    return int(flag) if flag.ndim == 0 else flag
