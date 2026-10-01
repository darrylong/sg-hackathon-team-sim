"""Exploratory data analysis (EDA) for Problem 03 - ticket triage.

WHY: before training any model we want to know what the data looks like:
which labels are rare, which clues predict the team and priority, which
"traps" exist (customer severity tag, sarcasm, typos), and how the eval and
curveball tickets differ from training. Every finding here shapes a model choice.

Run:
    .venv/bin/python submission_template_vip/03_ticket_triage/src/explore.py

Outputs (in src/reports/eda/):
    *.png         one chart per question
    findings.txt  everything printed below, ready to copy into slides
"""

import difflib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw charts to files only - no pop-up windows needed
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    CATEGORIES,
    CURVEBALL_CSV,
    EVAL_CSV,
    PRIORITIES,
    SENTIMENTS,
    TICKETS_CSV,
    load_cost_matrix,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Show full tables when printing (pandas hides columns by default).
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# Every line we print is also kept here so we can save it to findings.txt.
_LOG_LINES = []
SAVED_CHARTS = []


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(text=""):
    """Print a line AND remember it for findings.txt."""
    print(text)
    _LOG_LINES.append(str(text))


def section(title):
    """Print a clear header so the long output is easy to scan."""
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


def save_chart(fig, filename):
    """Save a matplotlib figure into the report folder and close it to free memory."""
    path = REPORT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    SAVED_CHARTS.append(path)
    log(f"[chart saved] {path.name}")


def heatmap(table, title, filename, fmt="{:.0f}"):
    """Draw a table as a coloured grid with the number written in every cell."""
    fig, ax = plt.subplots(figsize=(1.1 * len(table.columns) + 3, 0.5 * len(table.index) + 2))
    ax.imshow(table.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    biggest = table.values.max()
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            # White text on dark cells, black on light cells, so numbers stay readable.
            color = "white" if value > biggest * 0.6 else "black"
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8, color=color)
    ax.set_title(title)
    save_chart(fig, filename)


def word_count(df):
    """Number of words in subject + body - the text our model will actually read."""
    text = df["subject"].fillna("") + " " + df["body"].fillna("")
    return text.str.split().str.len()


def percent(series):
    """Value counts as percentages, rounded for printing."""
    return (series.value_counts(normalize=True) * 100).round(1)


# ---------------------------------------------------------------------------
# 1. Label balance
# ---------------------------------------------------------------------------
def label_balance(train):
    """WHY: rare classes (e.g. P1) are easy for a model to ignore. If a class is
    rare we may need class weights, and plain accuracy will look misleadingly good."""
    section("1. LABEL BALANCE - how common is each value?")
    columns = ["category", "priority", "assigned_team", "sentiment", "channel", "product"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    results = {}
    for ax, column in zip(axes.flat, columns):
        counts = train[column].value_counts()
        table = pd.DataFrame({"count": counts, "%": percent(train[column])})
        log(f"\n{column}:")
        log(table.to_string())
        results[column] = table
        counts.plot.bar(ax=ax, color="steelblue")
        ax.set_title(column)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    save_chart(fig, "01_label_balance.png")
    return results


# ---------------------------------------------------------------------------
# 2. Category -> team
# ---------------------------------------------------------------------------
def category_vs_team(train):
    """WHY: if every category always went to one team, we could just map
    category -> team. CLAUDE.md says that is NOT true - let's measure how far off it is."""
    section("2. CATEGORY -> TEAM - does category alone decide the team?")
    table = pd.crosstab(train["category"], train["assigned_team"])
    log(table.to_string())
    heatmap(table, "Category vs assigned team (ticket counts)", "02_category_vs_team.png")

    log("\nShare of each category that goes to its most common team:")
    top_share = (table.max(axis=1) / table.sum(axis=1) * 100).round(1)
    summary = pd.DataFrame({"top_team": table.idxmax(axis=1), "% to top team": top_share})
    log(summary.sort_values("% to top team").to_string())
    return summary


# ---------------------------------------------------------------------------
# 3. Product changes the team
# ---------------------------------------------------------------------------
def product_changes_team(train, category_summary):
    """WHY: e.g. a 'Firmware' ticket about a switch may go to Network-Support,
    while one about a server goes to Server-HW. If so, the model MUST see the product."""
    section("3. PRODUCT CHANGES THE TEAM - same category, different product")
    exceptions = []
    for category in ["Firmware", "Hardware", "Storage", "Security", "Access"]:
        subset = train[train["category"] == category]
        table = pd.crosstab(subset["product"], subset["assigned_team"])
        log(f"\n{category}  (overall top team: {category_summary.loc[category, 'top_team']})")
        log(table.to_string())
        heatmap(table, f"{category}: product vs team", f"03_product_team_{category.lower()}.png")

        # An "exception" is a product whose usual team differs from the category's usual team.
        default_team = category_summary.loc[category, "top_team"]
        for product, row in table.iterrows():
            team = row.idxmax()
            if team != default_team:
                exceptions.append({
                    "category": category,
                    "product": product,
                    "team": team,
                    "instead_of": default_team,
                    "tickets": int(row.max()),
                    "% of this product": round(row.max() / row.sum() * 100, 1),
                })

    log("\nTop exceptions: (category, product) -> team that differs from the category default")
    exceptions = pd.DataFrame(exceptions).sort_values("tickets", ascending=False)
    log(exceptions.to_string(index=False) if len(exceptions) else "  none found")
    return exceptions


# ---------------------------------------------------------------------------
# 4. Customer severity tag
# ---------------------------------------------------------------------------
# A bracket like "[Some label: Value]". We check the label part with difflib below,
# so typos such as "Cusotmer-selceted sevverity" still count.
BRACKET_PATTERN = re.compile(r"\[([^\]:]{3,50}):\s*([^\]]{1,20})\]")
TAG_KEYWORDS = ["customer", "selected", "severity"]
SEVERITY_LEVELS = ["Critical", "High", "Medium", "Low"]
# What priority each severity "should" mean if the customer were right.
SEVERITY_TO_PRIORITY = {"Critical": "P1", "High": "P2", "Medium": "P3", "Low": "P4"}


def _looks_like_severity_label(label):
    """True if any word in the label is close to 'customer', 'selected' or 'severity'."""
    words = re.split(r"[\s\-_]+", label.lower())
    return any(difflib.get_close_matches(word, TAG_KEYWORDS, n=1, cutoff=0.7) for word in words)


def normalise_severity(raw_value):
    """Map a possibly-misspelled value ('Criitcal', 'gigh', 'Lo') to Critical/High/Medium/Low."""
    value = raw_value.strip().lower()
    lowered = [level.lower() for level in SEVERITY_LEVELS]
    match = difflib.get_close_matches(value, lowered, n=1, cutoff=0.5)
    if not match:
        return None
    return SEVERITY_LEVELS[lowered.index(match[0])]


def extract_severity(body):
    """Return (raw_value, normalised_value) for the first severity tag in a body, or (None, None)."""
    for label, value in BRACKET_PATTERN.findall(str(body)):
        if _looks_like_severity_label(label):
            return value, normalise_severity(value)
    return None, None


def customer_severity(train):
    """WHY: CLAUDE.md warns the portal severity is the customer's opinion, not the label.
    We measure how often it matches the real priority to decide how much to trust it."""
    section("4. CUSTOMER SEVERITY TAG (portal) vs TRUE PRIORITY")
    portal = train[train["channel"] == "portal"].copy()
    extracted = portal["body"].apply(extract_severity)
    portal["raw_severity"] = extracted.str[0]
    portal["severity"] = extracted.str[1]

    found = portal["raw_severity"].notna().sum()
    clean = portal["raw_severity"].isin(SEVERITY_LEVELS).sum()
    unresolved = portal["raw_severity"].notna() & portal["severity"].isna()
    log(f"Portal tickets: {len(portal):,}; tag found in {found:,} ({found / len(portal):.1%})")
    log(f"  spelled correctly: {clean:,}; typo fixed by fuzzy matching: {found - clean - unresolved.sum():,}")
    log(f"  could not be normalised: {unresolved.sum()}  {sorted(portal.loc[unresolved, 'raw_severity'].unique())}")

    typo_examples = portal.loc[portal["raw_severity"].notna() & ~portal["raw_severity"].isin(SEVERITY_LEVELS),
                               ["raw_severity", "severity"]].drop_duplicates().head(10)
    log("\nExample typo fixes (raw -> normalised):")
    log(typo_examples.to_string(index=False))

    # Does the tag also appear outside the portal? (A model should treat it the same way.)
    other = train[train["channel"] != "portal"]["body"].apply(lambda b: extract_severity(b)[0] is not None)
    log(f"\nSeverity tags in non-portal tickets: {other.sum()}")

    tagged = portal.dropna(subset=["severity"])
    table = pd.crosstab(tagged["severity"], tagged["priority"], normalize="index") * 100
    table = table.reindex(index=SEVERITY_LEVELS, columns=PRIORITIES).round(1)
    log("\nRow % - for each customer severity, what the true priority was:")
    log(table.to_string())
    heatmap(table, "Customer severity vs true priority (row %)", "04_severity_vs_priority.png", fmt="{:.0f}%")

    agree = (tagged["severity"].map(SEVERITY_TO_PRIORITY) == tagged["priority"]).mean()
    log(f"\nTag 'agrees' with priority (Critical=P1, High=P2, Medium=P3, Low=P4): {agree:.1%}")
    critical_is_p1 = (tagged.loc[tagged["severity"] == "Critical", "priority"] == "P1").mean()
    log(f"When the customer says Critical, the ticket is really P1 only {critical_is_p1:.1%} of the time")
    return {"agree": agree, "critical_is_p1": critical_is_p1, "typos": found - clean}


# ---------------------------------------------------------------------------
# 5. Sentiment vs priority
# ---------------------------------------------------------------------------
def sentiment_vs_priority(train):
    """WHY: angry customers are not always urgent (and calm ones can be P1).
    If sentiment strongly predicted priority, the model might over-rely on tone."""
    section("5. SENTIMENT vs PRIORITY")
    table = pd.crosstab(train["sentiment"], train["priority"], normalize="index") * 100
    table = table.reindex(index=SENTIMENTS, columns=PRIORITIES).round(1)
    log("Row % - for each sentiment, the mix of priorities:")
    log(table.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    table.plot.bar(stacked=True, ax=ax, colormap="RdYlGn")
    ax.set_ylabel("% of tickets")
    ax.set_title("Priority mix within each sentiment")
    ax.legend(title="priority", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    save_chart(fig, "05_sentiment_vs_priority.png")
    return table


# ---------------------------------------------------------------------------
# 6. Channel style
# ---------------------------------------------------------------------------
def channel_style(train):
    """WHY: chat is short and sloppy, email is long and polite. Text cleaning and
    features (e.g. lowercasing, '!' counts) should work for every writing style."""
    section("6. CHANNEL STYLE - how does each channel write?")
    df = train.copy()
    df["words"] = word_count(df)
    letters = df["body"].str.count(r"[A-Za-z]")
    df["% lowercase letters"] = df["body"].str.count(r"[a-z]") / letters * 100
    df["all lowercase"] = df["body"] == df["body"].str.lower()
    df["has !"] = df["body"].str.contains("!", regex=False)

    table = df.groupby("channel").agg(
        tickets=("ticket_id", "count"),
        avg_words=("words", "mean"),
        pct_lowercase_letters=("% lowercase letters", "mean"),
        pct_all_lowercase=("all lowercase", lambda s: s.mean() * 100),
        pct_with_exclamation=("has !", lambda s: s.mean() * 100),
    ).round(1)
    log(table.to_string())

    for channel in table.index:
        log(f"\nExample {channel} bodies:")
        for body in df.loc[df["channel"] == channel, "body"].sample(2, random_state=42):
            log(f"  - {body}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column, title in zip(
        axes,
        ["avg_words", "pct_all_lowercase", "pct_with_exclamation"],
        ["Average words", "% tickets all lowercase", "% tickets with '!'"],
    ):
        table[column].plot.bar(ax=ax, color="darkorange")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=0)
    save_chart(fig, "06_channel_style.png")
    return table


# ---------------------------------------------------------------------------
# 7. Length
# ---------------------------------------------------------------------------
def ticket_length(datasets):
    """WHY: very short tickets ('it broke') carry almost no signal. Those are the
    ones we should abstain on, and we need to know how many exist in eval."""
    section("7. TICKET LENGTH (words in subject + body)")
    rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, df in datasets.items():
        words = word_count(df)
        short = (words < 8).sum()
        rows.append({
            "file": name, "tickets": len(df), "median_words": words.median(),
            "mean_words": round(words.mean(), 1), "min": words.min(), "max": words.max(),
            "< 8 words": short, "% < 8 words": round(short / len(df) * 100, 1),
        })
        # density=True so files of very different sizes are comparable on one chart.
        ax.hist(words, bins=range(0, 90, 3), alpha=0.5, density=True, label=f"{name} (n={len(df)})")
    table = pd.DataFrame(rows).set_index("file")
    log(table.to_string())
    ax.set_xlabel("words in subject + body")
    ax.set_ylabel("share of tickets")
    ax.set_title("Ticket length distribution")
    ax.legend()
    save_chart(fig, "07_length_histogram.png")
    return table


# ---------------------------------------------------------------------------
# 8. Train vs eval vs curveball
# ---------------------------------------------------------------------------
def compare_splits(datasets):
    """WHY: a model only works on data that looks like its training data.
    If eval/curveball contain new products, languages or channels, we must plan for it."""
    section("8. TRAIN vs EVAL vs CURVEBALL")
    unseen = {}
    for column in ["channel", "product"]:
        table = pd.DataFrame({name: percent(df[column]) for name, df in datasets.items()}).fillna(0)
        log(f"\n{column} % in each file:")
        log(table.to_string())

        fig, ax = plt.subplots(figsize=(10, 5))
        table.plot.bar(ax=ax)
        ax.set_ylabel("% of tickets")
        ax.set_title(f"{column}: train vs eval vs curveball")
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        save_chart(fig, f"08_split_{column}.png")

        train_values = set(datasets["train"][column])
        for name in ["eval", "curveball"]:
            new_values = sorted(set(datasets[name][column]) - train_values)
            unseen[(name, column)] = new_values
            log(f"  {column} values in {name} never seen in training: {new_values or 'none'}")

    log("\nAll curveball tickets:")
    for _, ticket in datasets["curveball"].iterrows():
        log(f"\n  {ticket['ticket_id']}  [{ticket['channel']}]  product: {ticket['product']}")
        log(f"    subject: {ticket['subject']}")
        log(f"    body:    {ticket['body']}")
    return unseen


# ---------------------------------------------------------------------------
# 9. Storage deep dive
# ---------------------------------------------------------------------------
# Story for the presentation: "Storage is the riskiest category, and customer
# emotion is an early-warning signal for urgency". Each step below collects the
# evidence - and the numbers decide whether the story holds.
SENTIMENT_ORDER = ["Angry", "Frustrated", "Neutral", "Positive"]


def why_storage(train):
    """9a. WHY: compare all categories on the things that make a ticket costly -
    how many P1s it contains, how angry customers are, how often the team is
    ambiguous, and what a misroute would cost on average."""
    section("9a. WHY STORAGE? - risk profile of every category")
    costs = load_cost_matrix()
    df = train.copy()
    # Cost we would pay if THIS ticket were misrouted, read from routing_cost_matrix.csv.
    df["misroute_cost"] = df["priority"].map(lambda p: costs[(p, "misroute")])
    df["is_p1"] = df["priority"] == "P1"
    df["is_angry"] = df["sentiment"] == "Angry"

    grouped = df.groupby("category")
    team_counts = pd.crosstab(df["category"], df["assigned_team"])
    table = pd.DataFrame({
        "% of all tickets": grouped.size() / len(df) * 100,
        "% of all P1 tickets": grouped["is_p1"].sum() / df["is_p1"].sum() * 100,
        "P1 rate in category %": grouped["is_p1"].mean() * 100,
        "Angry %": grouped["is_angry"].mean() * 100,
        "% to top team": team_counts.max(axis=1) / team_counts.sum(axis=1) * 100,
        "avg misroute cost": grouped["misroute_cost"].mean(),
    }).round(2).sort_values("avg misroute cost", ascending=False)
    log(table.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    table[["% of all tickets", "% of all P1 tickets"]].plot.bar(ax=ax, color=["lightgray", "crimson"])
    ax.set_ylabel("%")
    ax.set_title("Share of all tickets vs share of all P1 tickets, by category")
    ax.tick_params(axis="x", rotation=0)
    save_chart(fig, "09a_tickets_vs_p1_share.png")
    return table


def storage_sentiment_vs_priority(storage):
    """9b. WHY: if emotion predicts urgency inside Storage, it can act as an early
    warning - an angry storage ticket deserves a closer look before routing."""
    section("9b. STORAGE: SENTIMENT vs PRIORITY")
    table = pd.crosstab(storage["sentiment"], storage["priority"], normalize="index") * 100
    table = table.reindex(index=SENTIMENT_ORDER, columns=PRIORITIES).fillna(0).round(1)
    log("Row % - within Storage, the priority mix for each sentiment:")
    log(table.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    table.plot.bar(stacked=True, ax=ax, colormap="RdYlGn")
    ax.set_ylabel("% of Storage tickets")
    ax.set_title("Storage: priority mix within each sentiment")
    ax.legend(title="priority", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    save_chart(fig, "09b_storage_sentiment_vs_priority.png")

    angry, positive = table.loc["Angry", "P1"], table.loc["Positive", "P1"]
    log(f"\nAn Angry Storage ticket is P1 {angry}% of the time vs {positive}% for Positive.")
    return table


def storage_product_vs_team(storage):
    """9c. WHY: Storage is split between two teams. If the product explains the
    split, the model can learn it - and we can explain it to judges in one sentence."""
    section("9c. STORAGE: PRODUCT vs TEAM")
    table = pd.crosstab(storage["product"], storage["assigned_team"])
    log(table.to_string())
    heatmap(table, "Storage: product vs assigned team", "09c_storage_product_vs_team.png")

    log("\nThe rule in plain English (each product's most common team):")
    rules = {}
    for product, row in table.iterrows():
        team, share = row.idxmax(), row.max() / row.sum() * 100
        rules[product] = (team, share, int(row.sum()))
        log(f"  {product:<22} -> {team:<16} ({share:.0f}% of {row.sum()} tickets)")
    arrays = [p for p, (t, _, n) in rules.items() if t == "Storage-Support" and n >= 50]
    servers = [p for p, (t, _, n) in rules.items() if t == "Server-HW" and n >= 50]
    log(f"\n  Storage arrays/cloud ({', '.join(arrays)}) -> Storage-Support")
    log(f"  Servers ({', '.join(servers)}) -> Server-HW (disks/RAID inside a server stay with the server team)")
    small = [p for p, (_, _, n) in rules.items() if n < 50]
    log(f"  Too few tickets to call a rule (<50): {', '.join(small)}")
    return rules


def escalation_risk(train):
    """9d. WHY: an 'escalation risk' flag (Angry, or Frustrated AND urgent) is something
    a support desk can act on. We check which categories produce the most of them."""
    section("9d. ESCALATION RISK = Angry, or Frustrated with P1/P2")
    df = train.copy()
    df["escalation_risk"] = (df["sentiment"] == "Angry") | (
        (df["sentiment"] == "Frustrated") & df["priority"].isin(["P1", "P2"])
    )
    risky = df[df["escalation_risk"]]
    table = pd.DataFrame({
        "tickets": df.groupby("category").size(),
        "escalation risk %": df.groupby("category")["escalation_risk"].mean() * 100,
        "% of risky that are P1": risky.groupby("category")["priority"].apply(lambda s: (s == "P1").mean() * 100),
    }).round(1).sort_values("escalation risk %", ascending=False)
    log(table.to_string())

    overall_rate = df["escalation_risk"].mean() * 100
    overall_p1 = (risky["priority"] == "P1").mean() * 100
    base_p1 = (df["priority"] == "P1").mean() * 100
    log(f"\nOverall: {overall_rate:.1f}% of tickets are escalation risks; "
        f"{overall_p1:.1f}% of those are P1 (vs {base_p1:.1f}% P1 across all tickets).")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["crimson" if c == "Storage" else "steelblue" for c in table.index]
    table["escalation risk %"].plot.bar(ax=ax, color=colors)
    ax.axhline(overall_rate, color="gray", linestyle="--", label=f"all tickets ({overall_rate:.1f}%)")
    ax.set_ylabel("% of tickets flagged")
    ax.set_title("Escalation-risk rate by category")
    ax.legend()
    ax.tick_params(axis="x", rotation=0)
    save_chart(fig, "09d_escalation_risk_by_category.png")
    return {"table": table, "overall_rate": overall_rate, "overall_p1": overall_p1, "base_p1": base_p1}


# Positive-sounding phrases that, inside an urgent ticket, are usually sarcasm.
# \b = whole words only; (?<!not ) skips honest phrases like "not great".
SARCASM_PATTERN = re.compile(
    r"\b(?<!not )(?:love it|love how|wonderful|great|fantastic|brilliant|amazing|thanks for nothing|"
    r"just perfect|well done|inspiring|what an achievement)\b",
    re.IGNORECASE,
)


def storage_sarcasm_examples(storage):
    """9e. WHY: a bag-of-words model sees 'great' and 'love it' as happy words.
    In a P1 ticket they are usually sarcasm - these examples show why tone is tricky."""
    section("9e. STORAGE P1 TICKETS WITH SARCASM-LIKE POSITIVE WORDS")
    p1 = storage[storage["priority"] == "P1"]
    matches = p1[p1["body"].str.contains(SARCASM_PATTERN)]
    log(f"{len(matches)} of {len(p1)} Storage P1 tickets ({len(matches) / len(p1):.1%}) contain a positive-sounding phrase.")
    log("Labelled sentiment of those tickets:")
    log(matches["sentiment"].value_counts().to_string())
    for _, ticket in matches.sample(min(5, len(matches)), random_state=42).iterrows():
        found = ", ".join(sorted({m.lower() for m in SARCASM_PATTERN.findall(ticket["body"])}))
        log(f"\n  {ticket['ticket_id']}  [{ticket['channel']}]  {ticket['product']}  "
            f"sentiment={ticket['sentiment']}  matched: {found}")
        log(f"    subject: {ticket['subject']}")
        log(f"    body:    {ticket['body']}")
    return {"count": len(matches), "p1_total": len(p1)}


def storage_deep_dive(train):
    """Run 9a-9e and return the numbers the KEY FINDINGS need."""
    storage = train[train["category"] == "Storage"]
    return {
        "why": why_storage(train),
        "sentiment": storage_sentiment_vs_priority(storage),
        "rules": storage_product_vs_team(storage),
        "escalation": escalation_risk(train),
        "sarcasm": storage_sarcasm_examples(storage),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def storage_findings(deep):
    """Plain-English Storage bullets, worded from the numbers (not assumed)."""
    why = deep["why"]
    rank = list(why.index).index("Storage") + 1
    s = why.loc["Storage"]
    sent = deep["sentiment"]
    esc = deep["escalation"]
    esc_rank = list(esc["table"].index).index("Storage") + 1
    sar = deep["sarcasm"]
    server_share = sum(n for _, (t, _, n) in deep["rules"].items() if t == "Server-HW")
    storage_total = sum(n for _, (_, _, n) in deep["rules"].items())
    return [
        f"Storage ranks #{rank} of 8 categories on average misroute cost ({s['avg misroute cost']:.2f}): "
        f"it is {s['% of all tickets']:.1f}% of tickets but {s['% of all P1 tickets']:.1f}% of all P1s "
        f"(P1 rate {s['P1 rate in category %']:.1f}%).",
        f"Storage is the most team-ambiguous category ({s['% to top team']:.0f}% to its top team): "
        f"about {server_share / storage_total:.0%} of Storage tickets go to Server-HW because they are about "
        "disks inside servers - the product decides it.",
        f"Emotion is an early warning in Storage: an Angry Storage ticket is P1 {sent.loc['Angry', 'P1']}% of the time "
        f"vs {sent.loc['Positive', 'P1']}% for Positive; Storage ranks #{esc_rank} of 8 on escalation risk "
        f"({esc['table'].loc['Storage', 'escalation risk %']}%).",
        f"Tone is tricky: {sar['count']} of {sar['p1_total']} Storage P1 tickets use positive words "
        "('great', 'love it', 'thanks for nothing') that are often sarcasm - a word-count model can be fooled.",
    ]


def key_findings(balance, cat_summary, exceptions, severity, sentiment, style, length, unseen, deep):
    """Turn the numbers above into plain-English bullets for the slides."""
    section("KEY FINDINGS")
    priority = balance["priority"]["%"]
    teams = balance["assigned_team"]["%"]
    worst = cat_summary["% to top team"].idxmin()
    angry_p1 = sentiment.loc["Angry", "P1"]
    positive_p1 = sentiment.loc["Positive", "P1"]
    short_eval = length.loc["eval", "< 8 words"]
    bullets = [
        f"Training set has {balance['category']['count'].sum():,} tickets; P1 is the rarest priority "
        f"({priority['P1']}%) yet the costliest to misroute - use class weights and check P1 recall.",
        f"Teams are imbalanced: {teams.index[0]} {teams.iloc[0]}% vs {teams.index[-1]} {teams.iloc[-1]}%.",
        f"Category alone does not decide the team: only {cat_summary.loc[worst, '% to top team']}% of "
        f"{worst} tickets go to {cat_summary.loc[worst, 'top_team']}.",
        f"Product overrides the category default in {len(exceptions)} (category, product) pairs, "
        + (f"e.g. {exceptions.iloc[0]['category']} + {exceptions.iloc[0]['product']} -> {exceptions.iloc[0]['team']}"
           if len(exceptions) else "none found")
        + " - the product must be a model feature.",
        f"The portal severity tag agrees with the true priority only {severity['agree']:.0%} of the time "
        f"('Critical' is really P1 just {severity['critical_is_p1']:.0%}) - strip it from the text or treat it as a weak feature.",
        f"{severity['typos']} severity tags contain typos, so any rule touching them must be fuzzy.",
        f"Sentiment is a strong hint for priority: {angry_p1}% of Angry tickets are P1 vs "
        f"{positive_p1}% of Positive ones - but most Angry tickets are still not P1, so tone alone is not enough.",
        f"Chat is written differently: {style.loc['chat', 'pct_all_lowercase']}% of chat tickets are all "
        f"lowercase vs {style.loc['email', 'pct_all_lowercase']}% of emails - lowercase all text before vectorising.",
        f"The shortest training ticket has {length.loc['train', 'min']} words, but {short_eval} eval tickets have "
        "fewer than 8 - the model has never seen text that short, so these are prime candidates for abstaining.",
        "Curveball tickets test sarcasm, non-English text (Malay, German), all-caps shouting, "
        "vague 'it broke' messages, false urgency and out-of-scope requests (e.g. aircon); "
        f"new products/channels vs training: {sum(len(v) for v in unseen.values()) or 'none'}.",
    ] + storage_findings(deep)
    for bullet in bullets:
        log(f"- {bullet}")


def main():
    print("Loading data...")
    datasets = {
        "train": pd.read_csv(TICKETS_CSV),
        "eval": pd.read_csv(EVAL_CSV),
        "curveball": pd.read_csv(CURVEBALL_CSV),
    }
    for name, df in datasets.items():
        print(f"  {name}: {len(df):,} rows")
    train = datasets["train"]

    # Sanity check: the labels in config.py come from the same file.
    assert sorted(train["category"].unique()) == CATEGORIES

    balance = label_balance(train)
    cat_summary = category_vs_team(train)
    exceptions = product_changes_team(train, cat_summary)
    severity = customer_severity(train)
    sentiment = sentiment_vs_priority(train)
    style = channel_style(train)
    length = ticket_length(datasets)
    unseen = compare_splits(datasets)
    deep = storage_deep_dive(train)
    key_findings(balance, cat_summary, exceptions, severity, sentiment, style, length, unseen, deep)

    findings_path = REPORT_DIR / "findings.txt"
    findings_path.write_text("\n".join(_LOG_LINES) + "\n")
    print(f"\nSaved {len(SAVED_CHARTS)} charts and {findings_path.name} to {REPORT_DIR}")


if __name__ == "__main__":
    main()
