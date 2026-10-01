"""Train the 4 ticket-triage models (category, priority, team, sentiment).

What it does:
  1. Builds model text from tickets.csv (see text_utils.build_text).
  2. Holds out 20% as a validation set and scores the models on it.
  3. Saves validation predictions + probabilities (needed for the abstain policy).
  4. Saves models.joblib      -> trained on 80%, for honest validation / tuning.
  5. Saves models_full.joblib -> trained on 100%, used to make the submission.

Run:
    .venv/bin/python submission_template_vip/03_ticket_triage/src/train.py
"""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

from centroids import compute_centroids, max_similarity, similarity_threshold
from config import MODELS_DIR, RANDOM_STATE, TICKETS_CSV, load_cost_matrix
from text_utils import build_text

TARGETS = ["category", "priority", "assigned_team", "sentiment"]
# Short names for output columns, e.g. "team_conf" instead of "assigned_team_conf".
SHORT_NAME = {"category": "category", "priority": "priority", "assigned_team": "team", "sentiment": "sentiment"}

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def make_vectorizer():
    """Turn text into numbers with two TF-IDF views joined side by side.

    - word 1-2 grams: catch meaningful words and phrases ("array down", "invoice").
    - char 2-5 grams (within words): robust to typos and chat-speak ("rebuidling",
      "pls chek") because misspelled words still share most character chunks.
    sublinear_tf dampens words repeated many times in one ticket.
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=200_000)),
    ])


def make_classifier():
    """Logistic regression: fast, strong on TF-IDF text, and gives probabilities.

    class_weight="balanced" makes rare classes (like P1, 11% of tickets) count
    more, so the model doesn't just ignore them to get easy accuracy.
    """
    return LogisticRegression(C=5, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)


def train_models(X, df):
    """Train one classifier per target on the already-vectorised text X."""
    models = {}
    for target in TARGETS:
        print(f"  Training {target} model...", end=" ", flush=True)
        start = time.time()
        model = make_classifier()
        model.fit(X, df[target])
        models[target] = model
        print(f"done in {time.time() - start:.1f}s")
    return models


def fit_everything(df):
    """Fit the vectorizer once, then all 4 models and the category centroids.

    Returns (vectorizer, models, centroids).
    """
    print(f"  Fitting vectorizer on {len(df):,} tickets...", end=" ", flush=True)
    start = time.time()
    vectorizer = make_vectorizer()
    X = vectorizer.fit_transform(df["text"])
    print(f"done in {time.time() - start:.1f}s ({X.shape[1]:,} features)")
    # Centroids power the "Unfamiliar topic" check (see centroids.py).
    centroids = compute_centroids(X, df["category"])
    return vectorizer, train_models(X, df), centroids


def save_bundle(vectorizer, models, centroids, sim_threshold, filename, description):
    """Save everything needed for prediction in one file."""
    bundle = {"vectorizer": vectorizer, "models": models, "targets": TARGETS, "description": description,
              "centroids": centroids, "unfamiliar_sim_threshold": sim_threshold}
    path = MODELS_DIR / filename
    joblib.dump(bundle, path)
    print(f"  Saved {path.name} ({path.stat().st_size / 1e6:.0f} MB) -> {description}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def evaluate(models, X_val, val):
    """Print per-target scores and return (summary table, predictions table)."""
    rows = []
    predictions = pd.DataFrame({"ticket_id": val["ticket_id"].values})
    for target in TARGETS:
        model = models[target]
        short = SHORT_NAME[target]
        probs = model.predict_proba(X_val)
        pred = model.classes_[probs.argmax(axis=1)]
        truth = val[target].values

        predictions[f"true_{short}"] = truth
        predictions[f"pred_{short}"] = pred
        predictions[f"{short}_conf"] = probs.max(axis=1)
        # Full probability columns for team and priority - the abstain policy needs them.
        if target in ("assigned_team", "priority"):
            for i, label in enumerate(model.classes_):
                predictions[f"prob_{short}_{label}"] = probs[:, i]

        accuracy = accuracy_score(truth, pred)
        macro_f1 = f1_score(truth, pred, average="macro")
        rows.append({"target": target, "accuracy": accuracy, "macro_f1": macro_f1})

        print(f"\n--- {target} ---  accuracy {accuracy:.3f}   macro-F1 {macro_f1:.3f}")
        print(classification_report(truth, pred, digits=3, zero_division=0))
        if target == "priority":
            p1_recall = recall_score(truth, pred, labels=["P1"], average="macro", zero_division=0)
            p1_precision = precision_score(truth, pred, labels=["P1"], average="macro", zero_division=0)
            print(f"P1 recall {p1_recall:.3f}  (share of real P1s we caught)")
            print(f"P1 precision {p1_precision:.3f}  (share of predicted P1s that really are P1)")
            rows[-1].update({"P1_recall": p1_recall, "P1_precision": p1_precision})
    return pd.DataFrame(rows), predictions


def routing_cost_preview(predictions, most_common_team):
    """Compare routing cost with no abstaining: our model vs. 'send everything to one team'.

    Cost per ticket (from routing_cost_matrix.csv): 0 if the team is right,
    otherwise the misroute cost for the ticket's TRUE priority.
    """
    costs = load_cost_matrix()
    misroute_cost = predictions["true_priority"].map(lambda p: costs[(p, "misroute")])
    abstain_cost = predictions["true_priority"].map(lambda p: costs[(p, "abstain")])

    model_wrong = predictions["pred_team"] != predictions["true_team"]
    baseline_wrong = predictions["true_team"] != most_common_team

    table = pd.DataFrame([
        {"strategy": "our model, route all", "misroutes": model_wrong.sum(),
         "total_cost": (misroute_cost * model_wrong).sum()},
        {"strategy": f"baseline: all -> {most_common_team}", "misroutes": baseline_wrong.sum(),
         "total_cost": (misroute_cost * baseline_wrong).sum()},
        {"strategy": "abstain on everything (reference)", "misroutes": 0,
         "total_cost": abstain_cost.sum()},
    ])
    table["cost_per_ticket"] = (table["total_cost"] / len(predictions)).round(4)
    print(table.to_string(index=False))

    # Where does our model's cost come from? Misroutes on urgent tickets hurt most.
    by_priority = (misroute_cost * model_wrong).groupby(predictions["true_priority"]).agg(["sum", "count"])
    by_priority["misroutes"] = model_wrong.groupby(predictions["true_priority"]).sum()
    by_priority.columns = ["cost", "tickets", "misroutes"]
    print("\nOur model's routing cost split by true priority:")
    print(by_priority.to_string())
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    total_start = time.time()

    print("Step 1: loading tickets and building text...")
    df = pd.read_csv(TICKETS_CSV)
    df["text"] = build_text(df)
    print(f"  {len(df):,} tickets")

    print("\nStep 2: stratified 80/20 split on priority...")
    # Stratify so the rare P1 class is equally represented in train and validation.
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["priority"], random_state=RANDOM_STATE)
    print(f"  train {len(train_df):,} / validation {len(val_df):,}")
    val_ids_path = REPORTS_DIR / "val_ticket_ids.csv"
    val_df[["ticket_id"]].to_csv(val_ids_path, index=False)
    print(f"  Saved validation ticket_ids to {val_ids_path.name}")

    print("\nStep 3-4: fitting features and models on the 80% training split...")
    vectorizer, models, centroids = fit_everything(train_df)

    print("\nStep 5: scoring on the 20% validation split...")
    X_val = vectorizer.transform(val_df["text"])
    summary, predictions = evaluate(models, X_val, val_df)
    predictions["similarity"] = max_similarity(X_val, centroids)
    sim_threshold = similarity_threshold(predictions["similarity"])
    print(f"\nUNFAMILIAR_SIM_THRESHOLD (2nd percentile of validation similarity) = {sim_threshold:.4f}")

    print("\nStep 6: saving validation predictions...")
    predictions_path = REPORTS_DIR / "val_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    print(f"  Saved {predictions_path.name} ({len(predictions):,} rows, {predictions.shape[1]} columns)")

    print("\nStep 7: routing-cost preview on validation (no abstaining yet)...")
    most_common_team = train_df["assigned_team"].mode()[0]
    routing_cost_preview(predictions, most_common_team)

    print("\nStep 8: saving the 80% model bundle...")
    save_bundle(vectorizer, models, centroids, sim_threshold, "models.joblib",
                "trained on 80% of tickets.csv - use for validation and tuning")

    print("\nStep 9: retraining on 100% of tickets.csv for the submission...")
    full_vectorizer, full_models, full_centroids = fit_everything(df)
    # Reuse the threshold measured on held-out validation tickets - the honest estimate.
    save_bundle(full_vectorizer, full_models, full_centroids, sim_threshold, "models_full.joblib",
                "trained on 100% of tickets.csv - use for submission.csv")

    print("\n" + "=" * 60)
    print("SCORES SUMMARY (20% validation, models.joblib)")
    print("=" * 60)
    print(summary.round(3).to_string(index=False))
    print(f"\nTotal time: {(time.time() - total_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
