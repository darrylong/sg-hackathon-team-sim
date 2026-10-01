"""Text helpers shared by training, prediction and the app.

WHY a separate file: the model must see EXACTLY the same text at training time
and at prediction time. Keeping the cleaning in one place guarantees that.

Run directly to see what the model's input text looks like:
    .venv/bin/python submission_template_vip/03_ticket_triage/src/text_utils.py
"""

import difflib
import re

import pandas as pd

# A bracket like "[Some label: Value]". The label part is checked with difflib
# (see _is_severity_label), so typos such as "Cusotmer-selceted sevverity" still match.
BRACKET_PATTERN = re.compile(r"\[([^\]:]{3,50}):\s*([^\]]{1,20})\]")
TAG_KEYWORDS = ["customer", "selected", "severity"]
SEVERITY_LEVELS = ["Critical", "High", "Medium", "Low"]
NO_SEVERITY = "None"


def _is_severity_label(label):
    """True if any word in the bracket label is close to 'customer', 'selected' or 'severity'."""
    words = re.split(r"[\s\-_]+", label.lower())
    return any(difflib.get_close_matches(word, TAG_KEYWORDS, n=1, cutoff=0.7) for word in words)


def _normalise_severity(raw_value):
    """Map a misspelled value ('Criitcal', 'gigh', 'Lo') to Critical/High/Medium/Low, or None."""
    lowered = [level.lower() for level in SEVERITY_LEVELS]
    match = difflib.get_close_matches(raw_value.strip().lower(), lowered, n=1, cutoff=0.5)
    return SEVERITY_LEVELS[lowered.index(match[0])] if match else None


def _find_severity_tag(body):
    """Return the regex match of the severity tag in body, or None if there isn't one."""
    if not isinstance(body, str):  # NaN / missing body
        return None
    for match in BRACKET_PATTERN.finditer(body):
        if _is_severity_label(match.group(1)):
            return match
    return None


def extract_severity(body):
    """Return the customer-selected severity ("Critical"/"High"/"Medium"/"Low"), or "None".

    WHY: the customer's severity is only their opinion (it matches the true
    priority ~44% of the time), but it is still a useful hint - e.g. "Low" is
    almost never P1. We give it to the model as one clean token instead of
    leaving it buried, typo-ridden, in the text.
    """
    match = _find_severity_tag(body)
    if match is None:
        return NO_SEVERITY
    return _normalise_severity(match.group(2)) or NO_SEVERITY


def strip_severity_tag(body):
    """Return the body with the severity bracket removed (missing body -> "")."""
    if not isinstance(body, str):
        return ""
    match = _find_severity_tag(body)
    if match is None:
        return body
    return (body[:match.start()] + body[match.end():]).strip()


def _clean_column(series):
    """Turn NaN into "" so string operations never crash on missing values."""
    return series.fillna("").astype(str)


def build_text(df):
    """Build the single text string the model reads for each ticket.

    Example output:
        "array down the alletra is down ... product_alletra_6010 channel_email severity_none"

    WHY the extra "product_..." / "channel_..." / "severity_..." words: TF-IDF only
    understands words. Turning these columns into unique made-up words lets the
    same vectorizer use them - and EDA showed the product decides the team.
    """
    subject = _clean_column(df["subject"])
    body = _clean_column(df["body"])
    body_without_tag = body.apply(strip_severity_tag)
    severity = body.apply(extract_severity)
    product = _clean_column(df["product"]).str.replace(" ", "_")
    channel = _clean_column(df["channel"])

    text = (
        subject + " " + body_without_tag
        + " product_" + product
        + " channel_" + channel
        + " severity_" + severity
    )
    # Lowercase because chat is often all lowercase while emails are not (EDA section 6).
    return text.str.lower()


def word_count(df):
    """Number of words in subject + body - short tickets carry little signal."""
    text = _clean_column(df["subject"]) + " " + _clean_column(df["body"])
    return text.str.split().str.len()


if __name__ == "__main__":
    from config import EVAL_CSV, TICKETS_CSV

    tickets = pd.read_csv(TICKETS_CSV)
    print("build_text for 3 training tickets (one per channel):\n")
    examples = tickets.groupby("channel").head(1)
    for ticket_id, text in zip(examples["ticket_id"], build_text(examples)):
        print(f"{ticket_id}: {text}\n")

    eval_df = pd.read_csv(EVAL_CSV)
    eval_df["words"] = word_count(eval_df)
    shortest = eval_df.nsmallest(3, "words")
    print("3 shortest eval tickets:\n")
    for (_, row), text in zip(shortest.iterrows(), build_text(shortest)):
        print(f"{row['ticket_id']} ({row['words']} words): {text}\n")

    # Quick self-test of the fuzzy matching on typo examples seen in the data.
    tests = ["[Cusotmer-selceted sevverity: Criitcal] x", "[Customer-selected severity: gigh] x", "no tag", None]
    for body in tests:
        print(f"extract_severity({body!r}) -> {extract_severity(body)}")
