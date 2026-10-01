"""Partner health score and at-risk flags for 2026Q3.

Two signals, combined:
  1. GAP MODEL (machine learning): "time travel" - at each past quarter Q we compute
     features using only data up to Q, and label whether the partner went quiet in Q+1.
     A gradient-boosting model learns which partners tend to have empty quarters.
  2. FADE SIGNAL (explainable rule): is the partner's revenue shrinking right now?
     (consecutive declines, trend, year-on-year, last 3 months, missed targets)

Run after data_prep.py:
    .venv/bin/python submission_template_vip/05_channel_analytics/src/health.py
Writes src/reports/partner_health.csv and src/models/health_models.joblib
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from config import LAST_QUARTER, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, load_master, load_sales
from data_prep import build_monthly, build_panel

TOKEN_SHARE = 0.05             # same "token order" definition as explore.py
SNAPSHOTS = ["2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"]
TRAIN_UNTIL = "2025Q3"         # time-based validation: train <= 2025Q3, test after
TOP_K = 50
GAP_WEIGHT, FADE_WEIGHT = 0.6, 0.4  # only used to report the old blend in validation
NEW_PARTNER_DISCOUNT = 0.3     # new partners are still ramping - don't flag them for low revenue
GONE_QUIET_DAYS = 90

NUMERIC_FEATURES = [
    "rev_q", "rev_avg4", "rev_trend4", "rev_ratio_yoy", "n_consecutive_declines", "orders_q", "orders_avg4",
    "active_months_6m", "days_since_last_order", "past_gap_quarters", "rev_cv4", "last3m_vs_prev3m",
    "attainment_q", "attainment_avg4", "quarters_below_target_4", "margin_pct", "tenure_quarters", "is_new",
]
CATEGORICAL = ["tier", "partner_type", "region"]


# ---------------------------------------------------------------------------
# Helpers on wide (partner x quarter) tables
# ---------------------------------------------------------------------------
def shift_quarter(quarter, n):
    return str(pd.Period(quarter, freq="Q") + n)


def wide(panel, column):
    """partner x quarter table; NaN where the partner was not yet onboarded."""
    return panel.pivot(index="partner_id", columns="quarter", values=column)


def row_slope(values):
    """Least-squares slope of each row over its non-missing points (NaN if < 2 points)."""
    x = np.arange(values.shape[1], dtype=float)
    mask = ~np.isnan(values)
    n = mask.sum(axis=1)
    x_mean = np.where(n > 0, (mask * x).sum(axis=1) / np.maximum(n, 1), np.nan)
    y_mean = np.nansum(np.where(mask, values, 0.0), axis=1) / np.maximum(n, 1)  # rows with no data -> NaN below
    dx = np.where(mask, x - x_mean[:, None], 0.0)
    dy = np.where(mask, values - y_mean[:, None], 0.0)
    denom = (dx ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((n >= 2) & (denom > 0), (dx * dy).sum(axis=1) / denom, np.nan)


def safe_ratio(numerator, denominator):
    """numerator / denominator, NaN when the denominator is 0 or missing."""
    return numerator / denominator.where(denominator > 0)


# ---------------------------------------------------------------------------
# A. Features at a snapshot quarter (no peeking past its end)
# ---------------------------------------------------------------------------
class PartnerData:
    """Wide tables built once and reused for every snapshot."""

    def __init__(self, panel, monthly, master):
        self.quarters = sorted(panel["quarter"].unique())
        self.revenue = wide(panel, "revenue")
        self.orders = wide(panel, "n_orders")
        self.attainment = wide(panel, "attainment")
        self.margin = wide(panel, "margin_pct")
        self.days = wide(panel, "days_since_last_order")
        self.tenure = wide(panel, "tenure_quarters")
        self.monthly = monthly.reindex(self.revenue.index)
        self.master = master.set_index("partner_id").reindex(self.revenue.index)

    def history(self, quarter):
        return [q for q in self.quarters if q <= quarter]

    def token_level(self, quarter):
        """5% of the partner's average quarterly revenue, using only quarters up to `quarter`."""
        return TOKEN_SHARE * self.revenue[self.history(quarter)].mean(axis=1)

    def months_of(self, quarter):
        start = pd.Period(quarter, freq="Q").asfreq("M", "start")
        return [str(start + i) for i in range(3)]


def consecutive_declines(revenue, history):
    """Quarters in a row (ending at the snapshot) where revenue was lower than the quarter before."""
    count = pd.Series(0, index=revenue.index)
    still_declining = pd.Series(True, index=revenue.index)
    for later, earlier in zip(history[::-1], history[::-1][1:]):
        declined = (revenue[later] < revenue[earlier]).fillna(False)
        still_declining &= declined
        count += still_declining.astype(int)
    return count


def features_at(data, quarter):
    """One row per partner onboarded by `quarter`, using data up to the end of `quarter` only."""
    hist = data.history(quarter)
    last4 = hist[-4:]
    rev = data.revenue
    token = data.token_level(quarter)
    exists = rev[quarter].notna()

    f = pd.DataFrame(index=rev.index)
    f["rev_q"] = rev[quarter]
    f["rev_avg4"] = rev[last4].mean(axis=1)
    f["rev_trend4"] = row_slope(rev[last4].to_numpy(dtype=float)) / f["rev_avg4"].where(f["rev_avg4"] > 0)
    year_ago = shift_quarter(quarter, -4)
    f["rev_ratio_yoy"] = safe_ratio(rev[quarter], rev[year_ago]) if year_ago in rev else np.nan
    f["n_consecutive_declines"] = consecutive_declines(rev, hist)
    f["orders_q"] = data.orders[quarter]
    f["orders_avg4"] = data.orders[last4].mean(axis=1)

    this_months = data.months_of(quarter)
    prev_months = data.months_of(shift_quarter(quarter, -1))
    prev_months_available = [m for m in prev_months if m in data.monthly.columns]
    f["active_months_6m"] = (data.monthly[prev_months_available + this_months] > 0).sum(axis=1)
    f["last3m_vs_prev3m"] = (safe_ratio(data.monthly[this_months].sum(axis=1),
                                        data.monthly[prev_months_available].sum(axis=1))
                             if prev_months_available else np.nan)
    f["days_since_last_order"] = data.days[quarter]

    # Earlier empty/token quarters, skipping the onboarding quarter (a partial first quarter is not a gap).
    earlier = hist[:-1]
    gaps = rev[earlier].le(token, axis=0) & data.tenure[earlier].gt(0) if earlier else None
    f["past_gap_quarters"] = gaps.sum(axis=1) if earlier else 0
    f["rev_cv4"] = rev[last4].std(axis=1, ddof=0) / f["rev_avg4"].where(f["rev_avg4"] > 0)

    f["attainment_q"] = data.attainment[quarter]
    f["attainment_avg4"] = data.attainment[last4].mean(axis=1)
    f["quarters_below_target_4"] = (data.attainment[last4] < 1).sum(axis=1)
    f["margin_pct"] = data.margin[quarter]
    f["tenure_quarters"] = data.tenure[quarter]
    f["is_new"] = (f["tenure_quarters"] <= 1).astype(int)  # onboarded this quarter or last
    f["total_orders"] = data.orders[hist].sum(axis=1)
    f["token_level"] = token
    f["active"] = (rev[quarter] > token).astype(int)
    for column in CATEGORICAL:
        f[column] = data.master[column]
    f["snapshot"] = quarter
    return f[exists]


def one_hot(features):
    """Model matrix: numeric features + one-hot tier/partner_type/region."""
    dummies = pd.get_dummies(features[CATEGORICAL], dtype=int)
    return pd.concat([features[NUMERIC_FEATURES], dummies], axis=1)


# ---------------------------------------------------------------------------
# B. Time-travel labels
# ---------------------------------------------------------------------------
def labelled_snapshots(data):
    """Features at every snapshot Q for partners active in Q, labelled 1 if they went quiet in Q+1."""
    frames = []
    for quarter in SNAPSHOTS:
        f = features_at(data, quarter)
        f = f[f["active"] == 1].copy()
        next_rev = data.revenue.loc[f.index, shift_quarter(quarter, 1)]
        f["label"] = (next_rev <= f["token_level"]).astype(int)
        frames.append(f)
    return pd.concat(frames)


# ---------------------------------------------------------------------------
# C. Gap model with time-based validation
# ---------------------------------------------------------------------------
def make_model():
    # Small, regularised trees: few positives (~1%), so we avoid memorising noise.
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )


def ranking_metrics(y, score, k=TOP_K):
    order = np.argsort(-score)
    top_hits = y[order[:k]].sum()
    return {"ROC-AUC": roc_auc_score(y, score), "PR-AUC": average_precision_score(y, score),
            f"precision@{k}": top_hits / k, f"recall@{k}": top_hits / max(y.sum(), 1)}


def validate(labelled):
    """Train on early snapshots, test on later ones - the honest way to judge a forecast."""
    columns = one_hot(labelled).columns
    train = labelled[labelled["snapshot"] <= TRAIN_UNTIL]
    # A partner appears once per snapshot, so partner_id repeats: give rows a unique index.
    test = labelled[labelled["snapshot"] > TRAIN_UNTIL].reset_index()
    model = make_model().fit(one_hot(train)[columns], train["label"])
    y = test["label"].to_numpy()

    days = test["days_since_last_order"].fillna(test["days_since_last_order"].median())
    trend = test["rev_trend4"].fillna(0)
    gap_prob = pd.Series(model.predict_proba(one_hot(test)[columns])[:, 1], index=test.index)
    # The blend we use at 2026Q2, rebuilt per test snapshot so ranks are within one quarter.
    by_snapshot = test.groupby("snapshot")
    fade = by_snapshot.apply(fade_score, include_groups=False).droplevel(0).reindex(test.index)
    gap_rank = gap_prob.groupby(test["snapshot"]).rank(pct=True)
    fade_rank = fade.groupby(test["snapshot"]).rank(pct=True)
    combined = GAP_WEIGHT * gap_rank + FADE_WEIGHT * fade_rank
    table = pd.DataFrame({
        "gap model (HGB)": ranking_metrics(y, gap_prob.to_numpy()),
        "fade_score alone": ranking_metrics(y, fade.to_numpy()),
        f"combined {GAP_WEIGHT}*gap + {FADE_WEIGHT}*fade": ranking_metrics(y, combined.to_numpy()),
        "baseline: days_since_last_order": ranking_metrics(y, days.to_numpy()),
        "baseline: -rev_trend4": ranking_metrics(y, -trend.to_numpy()),
    }).T.round(3)
    print(f"Train snapshots {SNAPSHOTS[0]}..{TRAIN_UNTIL}: {len(train):,} rows, {train['label'].sum()} positives")
    print(f"Test snapshots  {test['snapshot'].min()}..{test['snapshot'].max()}: "
          f"{len(test):,} rows, {int(y.sum())} positives")
    print(table.to_string())

    # Same targeted boost as at 2026Q2, applied within each test snapshot.
    is_fade = strong_fade(test)
    boosted = pd.concat([apply_fade_boost(gap_rank[idx], is_fade[idx])
                         for idx in test.groupby("snapshot").groups.values()]).reindex(test.index)
    boost_table = pd.DataFrame({
        "gap model alone (rank)": ranking_metrics(y, gap_rank.to_numpy()),
        "gap + targeted fade boost": ranking_metrics(y, boosted.to_numpy()),
    }).T.round(3)
    fade_rate = test.loc[is_fade, "label"].mean() if is_fade.any() else float("nan")
    print(f"\nStrong-fade rows in the test snapshots: {int(is_fade.sum())}; "
          f"{int(test.loc[is_fade, 'label'].sum())} left next quarter -> leaver rate {fade_rate:.1%} "
          f"vs base rate {test['label'].mean():.1%}")
    print(f"Targeted fade boost (strong-fade lifted to >= {BOOST_PERCENTILE:.0%} percentile of risk):")
    print(boost_table.to_string())
    return pd.concat([table, boost_table]), columns


# ---------------------------------------------------------------------------
# D. Fade signal
# ---------------------------------------------------------------------------
# (feature, direction): +1 = higher value means more fading, -1 = lower value means more fading.
FADE_COMPONENTS = [("n_consecutive_declines", 1), ("rev_trend4", -1), ("rev_ratio_yoy", -1),
                   ("last3m_vs_prev3m", -1), ("quarters_below_target_4", 1)]


def fade_score(features):
    """Average percentile rank of the fade components (0 = growing, 1 = fading hardest).
    Missing values (e.g. no history yet) count as neutral 0.5.
    Kept for the dashboard; it is NOT part of the health score (it hurt validation AUC)."""
    ranks = [(direction * features[col]).rank(pct=True).fillna(0.5) for col, direction in FADE_COMPONENTS]
    return pd.concat(ranks, axis=1).mean(axis=1)


# A "strong fade" is a steep, sustained decline - not just a soft or lumpy quarter.
BOOST_PERCENTILE = 0.99


def is_gone_quiet(features):
    """Established partner with nothing this quarter and no order for 90+ days."""
    return ((features["rev_q"] == 0) & (features["days_since_last_order"] >= GONE_QUIET_DAYS)
            & (features["is_new"] == 0))


def strong_fade(features):
    """True for established, still-ordering partners in a sustained decline:
    4+ quarters of decline in a row AND at most half of last year's revenue.

    WHY only this test: a looser "big drop vs last year" test mostly caught lumpy
    buyers (one big order a year ago), not partners that are really fading.
    Gone-quiet partners are excluded - they have their own, stronger flag."""
    sustained = (features["n_consecutive_declines"] >= 4) & (features["rev_ratio_yoy"] <= 0.5)
    return sustained & (features["is_new"] == 0) & ~is_gone_quiet(features)


def apply_fade_boost(risk, is_strong_fade):
    """Lift strong-fade partners to at least the 95th percentile of risk.

    WHY targeted instead of blending: blending a fade score into everyone hurt AUC,
    because mild declines are normal. Only steep collapses deserve attention.
    """
    floor = risk.quantile(BOOST_PERCENTILE)
    return risk.where(~is_strong_fade, np.maximum(risk, floor))


# ---------------------------------------------------------------------------
# E-G. Combine, flag and explain
# ---------------------------------------------------------------------------
def combine(now):
    """risk = percentile rank of gap_prob (the validated ML signal), then three adjustments.
    health_score = 1 - risk."""
    now["gap_rank"] = now["gap_prob"].rank(pct=True)
    now["fade_rank"] = now["fade_score"].rank(pct=True)
    risk = now["gap_rank"].copy()

    # New partners are still ramping: low revenue is expected, so damp their risk.
    ramping = (now["is_new"] == 1) & (now["total_orders"] > 0)
    risk[ramping] *= NEW_PARTNER_DISCOUNT

    # Steep, sustained collapses get lifted to at least the 95th percentile.
    now["strong_fade"] = strong_fade(now).astype(int)
    risk = apply_fade_boost(risk, now["strong_fade"] == 1)
    print(f"  Strong-fade partners boosted to >= {BOOST_PERCENTILE:.0%} percentile of risk: "
          f"{int(now['strong_fade'].sum())}")

    # Established partners with nothing in 2026Q2 and no order for 90+ days have gone quiet.
    now["gone_quiet"] = is_gone_quiet(now).astype(int)
    risk[now["gone_quiet"] == 1] = 1 + now["days_since_last_order"] / 10_000  # above everyone else
    now["risk"] = risk
    now["health_score"] = 1 - risk
    return now


def choose_n(labelled, data):
    """How many partners to flag: expected leavers = historical rate + any NEW excess of decliners."""
    leaver_rate = labelled.groupby("snapshot")["label"].mean().mean()
    now = features_at(data, LAST_QUARTER)
    year_ago = features_at(data, shift_quarter(LAST_QUARTER, -4))
    share_now = (now.loc[now["active"] == 1, "n_consecutive_declines"] >= 4).mean()
    share_then = (year_ago.loc[year_ago["active"] == 1, "n_consecutive_declines"] >= 4).mean()
    excess = max(0.0, share_now - share_then)
    active_now = int(now["active"].sum())
    n = round(active_now * (leaver_rate + excess))
    print(f"Partners with 4+ consecutive quarterly declines: {LAST_QUARTER} "
          f"{int((now['n_consecutive_declines'] >= 4).sum())} ({share_now:.2%} of active) vs "
          f"{shift_quarter(LAST_QUARTER, -4)} {int((year_ago['n_consecutive_declines'] >= 4).sum())} "
          f"({share_then:.2%})")
    print(f"Historical average leaver rate: {leaver_rate:.2%}; excess decliner share now: {excess:.2%}")
    print(f"N = round({active_now:,} active partners x ({leaver_rate:.2%} + {excess:.2%})) = {n} "
          f"({n / active_now:.2%})")
    return n


def pct(x):
    return f"{x:+.0%}"


def explain(row):
    """risk_type and a plain-English reason with the partner's real numbers."""
    days = int(row["days_since_last_order"])
    if row["gone_quiet"]:
        avg_before = row["rev_avg4"] * 4 / 3  # avg4 includes the empty current quarter
        return "Gone quiet", (f"No orders in {LAST_QUARTER} and none for {days} days "
                              f"(it averaged ~${avg_before / 1e3:,.0f}k per quarter before).")

    fading = []
    if row["n_consecutive_declines"] >= 2:
        yoy = f" ({pct(row['rev_ratio_yoy'] - 1)} vs a year ago)" if pd.notna(row["rev_ratio_yoy"]) else ""
        fading.append(f"Revenue fell {int(row['n_consecutive_declines'])} quarters in a row{yoy}")
    elif pd.notna(row["rev_ratio_yoy"]) and row["rev_ratio_yoy"] < 0.7:
        fading.append(f"Revenue is {pct(row['rev_ratio_yoy'] - 1)} vs the same quarter last year")
    if pd.notna(row["last3m_vs_prev3m"]) and row["last3m_vs_prev3m"] < 0.6:
        fading.append(f"last 3 months {pct(row['last3m_vs_prev3m'] - 1)} vs the 3 before")
    if row["quarters_below_target_4"] >= 3 and not fading:
        fading.append(f"missed target in {int(row['quarters_below_target_4'])} of the last 4 quarters")

    irregular = []
    if row["orders_avg4"] < 1:
        irregular.append(f"Orders less than once per quarter on average ({row['orders_avg4']:.2f})")
    elif row["orders_avg4"] <= 5:
        times = round(row["orders_avg4"])
        irregular.append(f"Buys only ~{times} time{'s' if times != 1 else ''} per quarter")
    if row["past_gap_quarters"] >= 1:
        irregular.append(f"has had {int(row['past_gap_quarters'])} empty quarter(s) before")

    if row["strong_fade"]:
        # Always show the decline numbers that triggered the strong-fade boost.
        parts = []
        if row["n_consecutive_declines"] >= 2:
            parts.append(f"Revenue fell {int(row['n_consecutive_declines'])} quarters in a row")
        if pd.notna(row["rev_ratio_yoy"]):
            subject = "" if parts else "Revenue is "  # start the sentence with a subject
            parts.append(f"{subject}{pct(row['rev_ratio_yoy'] - 1)} vs the same quarter last year")
        if pd.notna(row["last3m_vs_prev3m"]):
            parts.append(f"last 3 months {pct(row['last3m_vs_prev3m'] - 1)} vs the 3 before")
        risk_type = "Fading"
    elif fading and irregular and row["gap_rank"] >= 0.9:
        risk_type, parts = "Mixed", [fading[0], irregular[0]]
    else:
        risk_type, parts = "Irregular buyer (gap risk)", irregular[:2] or fading[:1] or [
            f"Order pattern resembles past partners who went quiet ({row['orders_q']:.0f} orders this quarter)"]
    # Capitalise only the start of the sentence: "Revenue fell ...; buys only ...".
    text = "; ".join([parts[0]] + [p[0].lower() + p[1:] for p in parts[1:]])
    return risk_type, f"{text[0].upper()}{text[1:]}; last order {days} day{'s' if days != 1 else ''} ago."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data and building the panel...")
    sales = load_sales(exclude_outliers=True)
    master = load_master()
    panel = build_panel(sales, master)
    data = PartnerData(panel, build_monthly(sales, master), master)

    print("\nB. Building time-travel labels...")
    labelled = labelled_snapshots(data)
    rates = labelled.groupby("snapshot")["label"].agg(["sum", "count", "mean"])
    rates.columns = ["leavers", "active", "rate"]
    print(rates.assign(rate=(rates["rate"] * 100).round(2)).to_string())

    print("\nC. Gap model - time-based validation")
    print("=" * 78)
    validation, columns = validate(labelled)
    validation.to_csv(REPORTS_DIR / "health_validation.csv")

    print("\nRetraining the gap model on all snapshots...")
    model = make_model().fit(one_hot(labelled)[columns], labelled["label"])

    print(f"\nD/E. Scoring all partners at {LAST_QUARTER}...")
    now = features_at(data, LAST_QUARTER)
    now["gap_prob"] = model.predict_proba(one_hot(now).reindex(columns=columns, fill_value=0))[:, 1]
    now["fade_score"] = fade_score(now)
    now = combine(now)

    print("\nF. How many to flag")
    print("=" * 78)
    n_flag = choose_n(labelled, data)
    # Gone-quiet and strong-fade partners are flagged on top of N, so they don't crowd out
    # the partners the gap model ranks highest. The two groups never overlap (see strong_fade).
    gone_quiet = now["gone_quiet"] == 1
    fading = now["strong_fade"] == 1
    top_n = now.loc[~gone_quiet & ~fading, "risk"].nlargest(n_flag).index
    now["at_risk"] = 0
    now.loc[gone_quiet | fading, "at_risk"] = 1
    now.loc[top_n, "at_risk"] = 1
    print(f"at_risk = {int(gone_quiet.sum())} gone-quiet + {int(fading.sum())} strong-fade + top {n_flag} "
          f"by risk among the rest = {int(now['at_risk'].sum())} partners ({now['at_risk'].mean():.2%} of all)")

    print("\nG. Reasons for flagged partners...")
    now["risk_type"], now["reason"] = "", ""
    flagged = now["at_risk"] == 1
    explained = now[flagged].apply(explain, axis=1, result_type="expand")
    now.loc[flagged, "risk_type"], now.loc[flagged, "reason"] = explained[0], explained[1]

    out = now.reset_index().merge(master[["partner_id", "partner_name", "country", "onboarded_date"]],
                                  on="partner_id")
    out.to_csv(REPORTS_DIR / "partner_health.csv", index=False)
    joblib.dump({"gap_model": model, "columns": columns, "n_flag": n_flag,
                 "boost_percentile": BOOST_PERCENTILE, "new_partner_discount": NEW_PARTNER_DISCOUNT},
                MODELS_DIR / "health_models.joblib")
    print(f"  Saved partner_health.csv ({len(out):,} partners) and health_models.joblib")

    print("\nH. Results")
    print("=" * 78)
    fl = out[out["at_risk"] == 1]
    print(f"Flagged: {len(fl)} of {len(out):,} partners")
    for column in ["risk_type", "region", "tier"]:
        print(f"\nFlagged by {column}:")
        print(fl[column].value_counts().to_string())
    print(f"\nNew (ramping) partners among flagged: {int(fl['is_new'].sum())} "
          f"(of {int(out['is_new'].sum())} new partners)")
    print(f"Strong-fade partners flagged: {int(fl['strong_fade'].sum())} of {int(out['strong_fade'].sum())}")

    print("\n5 example reasons per risk_type:")
    for risk_type, group in fl.sort_values("risk", ascending=False).groupby("risk_type", sort=False):
        print(f"\n  [{risk_type}]")
        for row in group.head(5).itertuples():
            print(f"    {row.partner_id} {row.partner_name[:26]:<26} {row.region:<4} {row.tier:<8} {row.reason}")

    print("\n10 healthiest partners:")
    best = out.nlargest(10, "health_score")
    print(best[["partner_id", "partner_name", "region", "tier", "rev_q", "rev_trend4", "orders_avg4",
                "days_since_last_order", "health_score"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
