import joblib
import numpy as np
import pandas as pd

# Load the 4 saved models
_m = joblib.load("models.joblib")
cat_model, prio_model, team_model, senti_model = _m["cat"], _m["prio"], _m["team"], _m["senti"]

misroute_cost = {"P1": 8, "P2": 2, "P3": 1, "P4": 1}
abstain_cost  = {"P1": 1.0, "P2": 0.5, "P3": 0.3, "P4": 0.3}
mis_vec = np.array([misroute_cost[p] for p in prio_model.classes_])
abs_vec = np.array([abstain_cost[p] for p in prio_model.classes_])


def triage(tickets, use_cap=True):
    t = tickets.fillna("").copy()
    text = t["product"] + " | " + t["channel"] + " | " + t["subject"] + " | " + t["body"]

    cat  = cat_model.predict(text)
    sent = senti_model.predict(text)
    prio_p = prio_model.predict_proba(text)
    prio   = prio_model.classes_[prio_p.argmax(axis=1)]
    team_p = team_model.predict_proba(text)
    team   = team_model.classes_[team_p.argmax(axis=1)]
    p_right = team_p.max(axis=1)

    low = (t["subject"] + " " + t["body"]).str.lower()
    acct = (t["product"] == "GreenLake").values & \
           low.str.contains(r"sso|saml|mfa|okta|api key|api-key|role|permission").values
    cat     = np.where(acct, "Access", cat)
    team    = np.where(acct, "Account-Admin", team)
    p_right = np.where(acct, 0.95, p_right)

    saving = (1 - p_right) * (prio_p @ mis_vec) - prio_p @ abs_vec
    n_words = t["body"].str.split().str.len().values
    saving[n_words < 10] = 99

    if use_cap:
        limit = int(0.145 * len(t))
        order = np.argsort(-saving)
        abstain = np.zeros(len(t), dtype=bool)
        abstain[order[:limit]] = saving[order[:limit]] > 0
    else:
        abstain = saving > 0

    out = pd.DataFrame({
        "ticket_id": t["ticket_id"], "category": cat, "priority": prio,
        "assigned_team": team, "sentiment": sent, "abstain": abstain.astype(int),
    })
    out.loc[out["abstain"] == 1, ["category", "assigned_team"]] = ""
    return out