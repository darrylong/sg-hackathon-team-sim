# Team <your team name>

Members: <name 1>, <name 2>, <name 3>, <name 4>

This folder is your team's hand-in. Keep this layout:

```
<your team>/
  README.md                 this file
  requirements.txt          Python packages (same file as in the project root)
  run_triage.sh             starts the Problem 03 MVP
  run_channel.sh            starts the Problem 05 MVP
  slides/                   ONE slide deck (PDF or PPTX, about 6-10 slides) covering both problems
  03_ticket_triage/         submission files + src/ (code, trained models, reports, full-stack MVP)
  05_channel_analytics/     submission files + src/ (code, model, reports, full-stack MVP)
```

## How to run

| Problem | Start the MVP (one command) | Regenerate the submission files (in this order) |
|---|---|---|
| 03 Ticket Triage | `./run_triage.sh` then open http://localhost:8501 (API docs: http://127.0.0.1:8000/docs) | `python 03_ticket_triage/src/train.py && python 03_ticket_triage/src/evaluate.py && python 03_ticket_triage/src/predict.py` |
| 05 Channel Analytics | `./run_channel.sh` then open http://localhost:8502 (API docs: http://127.0.0.1:8001/docs) | `python 05_channel_analytics/src/data_prep.py && python 05_channel_analytics/src/health.py && python 05_channel_analytics/src/forecast.py && python 05_channel_analytics/src/make_submission.py` |

Both MVPs use different ports, so they can run at the same time. If a port is busy, choose others, e.g.
`API_PORT=8100 APP_PORT=8601 ./run_triage.sh`. Stop an MVP with Ctrl+C (it stops its API and front end together).

**Setup**

1. Put the pack next to the team folder - the code finds the data at `../SG_Hackathon_Pack/` relative to this folder:
   ```
   <project root>/
     SG_Hackathon_Pack/        the organisers' data pack (read-only, unchanged)
     <your team>/              this folder
     .venv/                    Python 3.12 virtual environment (created below)
   ```
2. Create the environment in the project root with **Python 3.12** and install the packages:
   ```
   cd <project root>
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r <your team>/requirements.txt
   ```
3. Start an MVP. `run_triage.sh` and `run_channel.sh` work from either place - inside this folder, or the identical
   copies in the project root. They find the team folder and `.venv` by themselves.
4. To regenerate the submissions, activate the environment and run the commands from the table **from inside this
   folder**: `source ../.venv/bin/activate`. Every script uses paths relative to its own file, so it also works from any
   other folder. `train.py` (Problem 03) takes about 4 minutes; everything else takes seconds.
   The trained models and reports are already included, so the MVPs start without retraining.

Optional exploration scripts (not needed for the submission): `03_ticket_triage/src/explore.py`,
`05_channel_analytics/src/explore.py` (charts and findings in each `src/reports/eda/`), and API smoke tests
`src/test_api.py` in both problems.

## Approach

### 03 Ticket Triage

**Text cleaning** (`text_utils.py`). The portal's "[Customer-selected severity: X]" tag is only the customer's opinion: it
matched the true priority just 44% of the time. We remove it from the text with a fuzzy matcher (212 tags had typos such as
"Cusotmer-selceted sevverity: Criitcal") and keep the normalised value as a single token. Product, channel and severity
are added as extra words (e.g. `product_alletra_6010`). Product is essential because the team is not a pure function of the
category: Storage tickets about servers go to Server-HW, while array tickets go to Storage-Support. All text is lowercased
because 59% of chat tickets are written in lowercase.

**Models** (`train.py`). TF-IDF on word 1-2-grams plus character 2-5-grams. The character n-grams make the model robust to
typos and chat-speak. Four logistic regressions (category, priority, team, sentiment) use `class_weight="balanced"` so that
rare P1 tickets are not ignored. We validate on a stratified 80/20 split, then retrain on 100% of the data for the
submission.

**Abstain policy** (`policy.py`). The 15% hand-off capacity is filled in this order (strategy "D: Hybrid"):
1. **Too short to understand**: fewer than 8 words. Every training ticket has at least 9 words, but 101 eval tickets do
   not.
2. **Unfamiliar topic**: the ticket's cosine similarity to its nearest category centroid is below the 2nd percentile of
   validation tickets, *and* team confidence is below 0.90. This targets the eval category that never appears in
   training.
3. **Below break-even confidence**: team confidence is below `1 - abstain_cost / misroute_cost` for the predicted
   priority (0.875 for P1 ... 0.70 for P3/P4), read from `routing_cost_matrix.csv`. The largest shortfall goes first.

On validation, D beat our earlier cost-aware ranking (C), so `policy.POLICY = "D"`. Single tickets in the API use rules
1-3 without the cap; batches of 100+ tickets get the 15% cap. The curveball file is scored per ticket, so it is uncapped.

**Noisy labels, sarcasm, mixed language.** Regularised linear models plus class balancing cope well with label noise.
Sarcasm ("Truly inspiring stuff" after four reboots) is partly caught by the sentiment model. Non-English tickets usually
look unfamiliar to the model and are handed to a human rather than guessed (e.g. the German curveball CB-011).

### 05 Channel Analytics

**Key insights** (`explore.py`, dashboard "Insights" tab - every insight has a chart):
- **Outlier:** a single $37.8M order line inflates EMEA 2026Q1 revenue by 15%. It is excluded everywhere (threshold =
  10x the 99.99th percentile, computed, not hard-coded).
- **Q3 seasonality:** revenue moves Q2->Q3 by -2.8% (2024) and -5.2% (2025); EMEA dips 9-12%.
- **AMS growth and concentration:** AMS grew +45% (last 4 quarters vs the 4 before) and is 58% of revenue. Its top 10
  partners bring 18% of it.
- **Attainment and margin:** Silver partners have the lowest target attainment (89% vs 98% for Business). GreenLake has the
  best margin (21.7%) but only a 10% revenue share.
- **New decline group:** 22 partners have shrunk 4+ quarters in a row, vs 6 a year ago.

**Health score** (`data_prep.py`, `health.py`). There are no churn labels, so we build them by **time travel**. At each past
quarter Q we compute features from data up to Q only: revenue trend and year-on-year change, order frequency, days since
the last order, past empty quarters, lumpiness, target attainment, margin, tenure, tier, type and region. The label is
whether the partner placed no orders, or only a token order (< 5% of its own average), in Q+1.
- **Model:** a HistGradientBoosting classifier learns these "gaps". The health score is 1 - percentile rank of its
  probability.
- **Fade boost:** partners with 4+ quarters of decline that are also down by at least half year-on-year are lifted to the
  top 1% of risk.
- **New partners:** partners onboarded in the last 2 quarters are still ramping, so their risk is damped.

**Flags** (`at_risk = 1`):
- **Gone quiet:** no orders in 2026Q2 and none for 90+ days.
- **Fading:** the steep, sustained decliners described above.
- **Top N by model risk:** N = active partners x (historical leaver rate 1.11% + excess decliners 0.55%) = 43.

Each flagged partner gets a plain-language reason with its own numbers (shown in the dashboard, not in the CSV).

**Forecast** (`forecast.py`).
- **Methods:** we average three seasonal methods: same quarter last year x recent growth, last quarter x the typical
  Q2->Q3 step, and a log-linear trend with quarter-of-year effects.
- **Selection:** the methods were backtested on 4 past quarters, each forecast using only the data available at the time.
  The ensemble had the lowest error.
- **Bias correction:** past forecasts ran slightly high, so we correct half of that bias.
- **ALL:** the sum of the three regional forecasts.
- **80% interval:** `forecast x exp(+/-1.2816 x sigma)`, with sigma taken from the backtest log errors (minimum 0.045).

## Results

All numbers are on our own hold-out data. The organisers' eval labels are not available to us.

**03 Ticket Triage** - 20% stratified validation split (6,000 tickets), models trained on the other 80%:

| Target | Accuracy | Macro-F1 |
|---|---|---|
| Category | 0.861 | 0.860 |
| Priority | 0.760 | 0.753 (P1 recall 0.749, P1 precision 0.672) |
| Assigned team | 0.900 | 0.895 |
| Sentiment | 0.862 | 0.867 |

| Abstain strategy (validation) | Routing cost | Abstain rate | Category macro-F1* | Team macro-F1 (routed) |
|---|---|---|---|---|
| A: Never abstain | 0.195 | 0% | 0.860 | 0.895 |
| B: Simple confidence thresholds | 0.156 | 14.7% | 0.824 | 0.939 |
| **D: Hybrid (submitted)** | **0.157** | **14.7%** | 0.823 | 0.938 |

\*Abstained tickets count as category misses, as in the official scoring. Escalation-risk F1 is 0.855.

- **Strategy D vs B:** D costs about the same as B on validation but reserves budget for the eval traps (too-short and
  unseen-category tickets) that the validation set does not contain.
- **Eval submission:** abstain rate 14.70% (cap 15%): 335 low confidence, 317 urgent and unsure, 129 unfamiliar topic,
  101 too short. All 101 tickets under 8 words are handed off.
- **Curveball:** 5 of 15 are handed to a human:
  - CB-002 "it broke"
  - CB-004 server-room aircon (out of scope)
  - CB-011 German
  - CB-013 all caps
  - CB-015 "same thing as before"

**05 Channel Analytics**

| Health model - time-based validation (train <= 2025Q3, test 2025Q4 + 2026Q1) | ROC-AUC | PR-AUC | Precision@50 |
|---|---|---|---|
| Gap model (HistGradientBoosting) | 0.872 | 0.064 | 0.12 |
| **Gap model + targeted fade boost (submitted)** | **0.870** | 0.065 | 0.12 |
| Baseline: days since last order | 0.695 | 0.053 | 0.10 |
| Baseline: revenue trend | 0.451 | 0.021 | 0.08 |

- **Flags:** 91 partners (3.5%) are flagged: 30 gone quiet, 18 fading, 33 mixed, 10 irregular buyers. Together they made
  only $12.7M (1.1%) of 2026Q2 revenue - mostly small Business-tier partners.
- **New partners:** none of the 120 partners still ramping are flagged.

| 2026Q3 forecast | Forecast | 80% range | vs 2025Q3 | vs 2026Q2 |
|---|---|---|---|---|
| APJ | $192.4M | $181.6M - $203.8M | +8.4% | -5.8% |
| EMEA | $227.3M | $214.5M - $240.8M | +8.2% | -10.9% |
| AMS | $715.3M | $675.2M - $757.8M | +43.4% | -1.2% |
| ALL | $1,135.0M | $1,071.4M - $1,202.4M | +28.0% | -4.1% |

- **Backtest error (MAPE over 4 quarters):** ensemble 3.02% average across regions, 1.92% for ALL. All 4 backtest actuals
  fell inside the 80% ranges.

## Limitations and next steps

- **03 - Non-English and sarcastic tickets:** they are handled mostly by handing them off, not by understanding them. Next
  step: a multilingual sentence-embedding model, or translation, as an extra feature.
- **03 - Unfamiliar-topic threshold:** it is set from validation tickets of *known* categories, so we could not measure
  how well it catches the eval's unseen category.
- **03 - Cost-aware ranking:** ranking by expected cost (strategy C) did not beat simple thresholds, probably because
  class-balanced probabilities are not calibrated. Next step: probability calibration (`CalibratedClassifierCV`).
- **05 - Few leavers, temporary gaps:** about 1% of active partners per quarter, and 99% of past leavers ordered again
  later. The time-travel labels are noisy, and the fade signal on its own does not predict leaving.
- **05 - Short history:** only 10 quarters and 4 backtest points, so the forecast intervals rest on little evidence. AMS
  growth (+43% year-on-year) dominates the total; if it slows, the ALL forecast will be too high.
- **05 - Missing drivers:** no data on pipeline, pricing or partner programmes. Next step: add them, and re-check flags
  after each quarter closes.

## Tools and models used

- **Libraries:** Python 3.12, pandas, numpy, scikit-learn (TF-IDF, LogisticRegression, HistGradientBoostingClassifier),
  matplotlib, plotly, FastAPI + uvicorn + pydantic (APIs), Streamlit (front ends), joblib, requests, httpx (API tests).
  Full list in `requirements.txt`.
- **Models:** no pretrained models. All models are trained from scratch on the pack's data and run fully offline.
- **No LLM at prediction time:** neither the shared LLM gateway nor any other LLM is used for predictions, health scores,
  forecasts, insights or reasons. All are computed from the data, so the MVPs work without network access. No API keys
  are needed or stored.
- **AI coding assistant:** Claude Code (Anthropic) was used to write and review the code. The team made the modelling
  decisions and checked the results.
