"""Category centroids: detect tickets that look unlike ANY known category.

WHY: the eval set contains a category never seen in training (README). A classifier
always picks one of the 8 known categories, sometimes confidently. Instead we ask
a different question: "how similar is this ticket to the typical ticket of each
category?" A centroid is the average TF-IDF vector of a category's tickets; a ticket
far from every centroid is probably about something new.

Run this to add centroids to the EXISTING model bundles without retraining:
    .venv/bin/python submission_template_vip/03_ticket_triage/src/centroids.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from config import CATEGORIES, MODELS_DIR, TICKETS_CSV
from text_utils import build_text

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
# 2% of normal validation tickets fall below the threshold - the rest count as "familiar".
SIM_PERCENTILE = 2


def compute_centroids(X, categories):
    """One L2-normalised mean vector per category (rows in CATEGORIES order).

    We normalise X first so every ticket counts equally, whatever its length.
    """
    X = normalize(X)
    categories = np.asarray(categories)
    rows = [np.asarray(X[categories == c].mean(axis=0)).ravel() for c in CATEGORIES]
    return normalize(np.vstack(rows))


def max_similarity(X, centroids):
    """Cosine similarity of each ticket to its NEAREST category centroid (0..1)."""
    return (normalize(X) @ centroids.T).max(axis=1)


def similarity_threshold(val_similarity):
    """Threshold = the 2nd percentile of similarity on validation (known-category) tickets."""
    return float(np.percentile(val_similarity, SIM_PERCENTILE))


def bundle_similarity(bundle, texts):
    """Similarity for raw model texts using a saved bundle's vectorizer + centroids."""
    return max_similarity(bundle["vectorizer"].transform(texts), bundle["centroids"])


def main():
    print("Loading tickets and building text...")
    tickets = pd.read_csv(TICKETS_CSV)
    tickets["text"] = build_text(tickets)
    val_ids = set(pd.read_csv(REPORTS_DIR / "val_ticket_ids.csv")["ticket_id"])
    is_val = tickets["ticket_id"].isin(val_ids)
    train_part, val_part = tickets[~is_val], tickets[is_val]
    print(f"  train split {len(train_part):,} / validation {len(val_part):,}")

    # 80% bundle: centroids from the training split only, so validation stays unseen.
    print("\nmodels.joblib: computing centroids from the 80% training split...")
    bundle = joblib.load(MODELS_DIR / "models.joblib")
    X_train = bundle["vectorizer"].transform(train_part["text"])
    bundle["centroids"] = compute_centroids(X_train, train_part["category"])
    val_sim = bundle_similarity(bundle, val_part["text"])
    threshold = similarity_threshold(val_sim)
    print(f"  Validation similarity: median {np.median(val_sim):.3f}, min {val_sim.min():.3f}")
    print(f"  UNFAMILIAR_SIM_THRESHOLD ({SIM_PERCENTILE}nd percentile on validation) = {threshold:.4f}")
    bundle["unfamiliar_sim_threshold"] = threshold
    joblib.dump(bundle, MODELS_DIR / "models.joblib")
    print("  Re-saved models.joblib (classifiers unchanged)")

    # 100% bundle: centroids from all tickets; reuse the honest threshold from validation.
    print("\nmodels_full.joblib: computing centroids from 100% of tickets...")
    full = joblib.load(MODELS_DIR / "models_full.joblib")
    X_all = full["vectorizer"].transform(tickets["text"])
    full["centroids"] = compute_centroids(X_all, tickets["category"])
    full["unfamiliar_sim_threshold"] = threshold
    joblib.dump(full, MODELS_DIR / "models_full.joblib")
    print("  Re-saved models_full.joblib (classifiers unchanged)")


if __name__ == "__main__":
    main()
