"""The project's "address book": every other script imports paths, labels and costs from here.

Usage in another script (in the same src/ folder):
    from config import TICKETS_CSV, load_cost_matrix, RANDOM_STATE

Run this file directly to check that everything it points to exists:
    .venv/bin/python submission_template_vip/03_ticket_triage/src/config.py
"""

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Project root
# ---------------------------------------------------------------------------
# This file is <root>/submission_template_vip/03_ticket_triage/src/config.py.
# Going 3 folders up from it gives the hpe-hackathon folder, wherever you run from.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
# Input data (read-only - never write into SG_Hackathon_Pack/).
PACK_DIR = PROJECT_ROOT / "SG_Hackathon_Pack" / "03_ticket_triage"
TICKETS_CSV = PACK_DIR / "data" / "tickets.csv"
EVAL_CSV = PACK_DIR / "eval" / "eval_tickets.csv"
CURVEBALL_CSV = PACK_DIR / "eval" / "curveball_tickets.csv"
COST_MATRIX_CSV = PACK_DIR / "artifacts" / "routing_cost_matrix.csv"
CAPACITY_JSON = PACK_DIR / "artifacts" / "triage_capacity.json"
SAMPLE_SUB_CSV = PACK_DIR / "artifacts" / "sample_submission.csv"
SAMPLE_SUB_CURVEBALL_CSV = PACK_DIR / "artifacts" / "sample_submission_curveball.csv"

# Our outputs: submission files go in OUTPUT_DIR, trained models in MODELS_DIR.
# OUTPUT_DIR is found from this file's own location (src/ -> 03_ticket_triage/), so it keeps
# working if the team folder is renamed from "submission_template_vip".
OUTPUT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = OUTPUT_DIR / "src" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)  # create it on first use so saving never fails

# ---------------------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------------------
# Pass this to every model / split so results are the same on every run.
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 3-5. Costs and abstain rules (always read from the files, never hard-coded)
# ---------------------------------------------------------------------------
def load_cost_matrix():
    """Return the routing costs as {(priority, outcome): cost}.

    Example: {("P1", "misroute"): 8.0, ("P1", "abstain"): 1.0, ...}
    We read the CSV every time so that if the organisers change the numbers,
    our code automatically uses the new ones.
    """
    df = pd.read_csv(COST_MATRIX_CSV)
    return {
        (row.true_priority, row.outcome): float(row.cost)
        for row in df.itertuples()
    }


def load_max_abstain_rate():
    """Return the largest share of eval tickets we may abstain on (e.g. 0.15)."""
    capacity = json.loads(CAPACITY_JSON.read_text())
    return float(capacity["max_abstain_rate"])


def confidence_thresholds():
    """Return the minimum team-confidence needed to route (not abstain), per priority.

    If the model is p confident in its team, routing costs on average
    (1 - p) * misroute_cost, while abstaining always costs abstain_cost.
    Routing is the cheaper choice when (1 - p) * misroute_cost < abstain_cost,
    which rearranges to  p > 1 - abstain_cost / misroute_cost.
    """
    costs = load_cost_matrix()
    thresholds = {}
    for priority in PRIORITIES:
        misroute = costs[(priority, "misroute")]
        abstain = costs[(priority, "abstain")]
        thresholds[priority] = 1 - abstain / misroute
    return thresholds


# ---------------------------------------------------------------------------
# 6. Label lists and submission format
# ---------------------------------------------------------------------------
def _sorted_labels(column):
    """Read one column of tickets.csv and return its unique values, sorted."""
    values = pd.read_csv(TICKETS_CSV, usecols=[column])[column].dropna().unique()
    return sorted(values)


CATEGORIES = _sorted_labels("category")
PRIORITIES = _sorted_labels("priority")
TEAMS = _sorted_labels("assigned_team")
SENTIMENTS = _sorted_labels("sentiment")
# Known products and channels (inputs, not labels) - used by the API and front end dropdowns.
PRODUCTS = _sorted_labels("product")
CHANNELS = _sorted_labels("channel")

# The exact column order the graders expect, taken from their sample file.
SUBMISSION_COLUMNS = list(pd.read_csv(SAMPLE_SUB_CSV, nrows=0).columns)


# ---------------------------------------------------------------------------
# 8. Self-check when run directly
# ---------------------------------------------------------------------------
def _print_paths():
    print("Paths:")
    paths = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "TICKETS_CSV": TICKETS_CSV,
        "EVAL_CSV": EVAL_CSV,
        "CURVEBALL_CSV": CURVEBALL_CSV,
        "COST_MATRIX_CSV": COST_MATRIX_CSV,
        "CAPACITY_JSON": CAPACITY_JSON,
        "SAMPLE_SUB_CSV": SAMPLE_SUB_CSV,
        "SAMPLE_SUB_CURVEBALL_CSV": SAMPLE_SUB_CURVEBALL_CSV,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MODELS_DIR": MODELS_DIR,
    }
    for name, path in paths.items():
        mark = "✅" if path.exists() else "❌"
        # Show paths relative to the root so they're short and readable.
        shown = path if path == PROJECT_ROOT else path.relative_to(PROJECT_ROOT)
        print(f"  {mark} {name:<25} {shown}")
    print()


def _print_example_tickets():
    """Print one full ticket per channel so we can see what the raw text looks like."""
    tickets = pd.read_csv(TICKETS_CSV)
    for channel in ["email", "portal", "chat"]:
        ticket = tickets[tickets["channel"] == channel].iloc[0]
        print(f"--- Example {channel} ticket ---")
        for column, value in ticket.items():
            print(f"  {column:<14}: {value}")
        print()


if __name__ == "__main__":
    _print_paths()

    print("Cost matrix (priority, outcome) -> cost:")
    for key, cost in load_cost_matrix().items():
        print(f"  {key}: {cost}")
    print()

    print(f"Max abstain rate: {load_max_abstain_rate()}\n")

    print("Confidence needed to route instead of abstain:")
    for priority, threshold in confidence_thresholds().items():
        print(f"  {priority}: {threshold:.3f}")
    print()

    print(f"CATEGORIES ({len(CATEGORIES)}): {CATEGORIES}")
    print(f"PRIORITIES ({len(PRIORITIES)}): {PRIORITIES}")
    print(f"TEAMS      ({len(TEAMS)}): {TEAMS}")
    print(f"SENTIMENTS ({len(SENTIMENTS)}): {SENTIMENTS}")
    print(f"SUBMISSION_COLUMNS: {SUBMISSION_COLUMNS}")
    print(f"RANDOM_STATE: {RANDOM_STATE}\n")

    _print_example_tickets()
