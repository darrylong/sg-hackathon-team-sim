# Problem 03 – Intelligent Support Ticket Triage

Predicts a support ticket's **category, priority (P1–P4), owning team and customer sentiment**, and decides per ticket whether to **route it** or **send it to the human triage desk** (abstain).

## Files in this folder

| File | What it is |
|---|---|
| `model.ipynb` | Notebook that trains the 4 models and makes the submission files |
| `models.joblib` | The 4 trained models, saved |
| `triage_core.py` | The `triage()` function: loads the models, predicts, applies the abstain rule |
| `api.py` | Backend API (FastAPI): `POST /triage` and `POST /triage_csv` |
| `ui.py` | Web front end (Streamlit) |

## How to run the app

**1. Install (once):**
```
py -m pip install pandas scikit-learn fastapi uvicorn python-multipart streamlit requests
```

**2. Start the backend.** Open a terminal **in this folder** and run:
```
py -m uvicorn api:app
```

**3. Start the web page.** Open a **second** terminal in this folder and run:
```
py -m streamlit run ui.py
```

**4. Open http://localhost:8501 in your browser.**
- **Single ticket** tab: paste a subject and body, pick the product, click **Triage**.
- **Upload CSV** tab: upload a CSV with columns `subject, body, product, channel` (e.g. `eval/curveball_tickets.csv`).

API docs: http://127.0.0.1:8000/docs

## Approach

- Text = product + channel + subject + body.
- Features: word TF-IDF (1–2 words) + character TF-IDF (2–5 letters, which handles typos and mixed languages).
- One logistic regression per task (category, priority, team, sentiment).
- **Abstain rule:** for each ticket, compare the expected cost of routing (chance the team is wrong × penalty, weighted by predicted priority) with the cost of a hand-off. The tickets where abstaining saves the most go to humans, capped at 14.5% (limit 15%).
- Extra rules: tickets under 10 words always go to a human; GreenLake SSO / MFA / API-key / role issues go to Account-Admin (README rule, confirmed in the training data).

## Results (validation: 6,000 held-out training tickets)

| Metric | Value |
|---|---|
| Category macro-F1 | 0.865 |
| Priority macro-F1 | 0.759 |
| Team macro-F1 | 0.906 |
| Sentiment macro-F1 | 0.869 |
| Routing cost, never abstaining | 0.174 |
| Routing cost, with abstain rule | 0.149 |
| Abstain rate | 14.5% |
