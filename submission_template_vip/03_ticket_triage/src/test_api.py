"""Smoke-test the API without starting a server (FastAPI's TestClient calls it in-process).

Run:
    .venv/bin/python submission_template_vip/03_ticket_triage/src/test_api.py
"""

import io

from fastapi.testclient import TestClient

from api import app
from config import CURVEBALL_CSV

client = TestClient(app)

# Four curveball tickets, copied from curveball_tickets.csv.
CURVEBALLS = {
    "CB-002": {"channel": "chat", "subject": "it broke", "body": "it broke", "product": "GreenLake"},
    "CB-004": {
        "channel": "portal", "subject": "server room aircon dead", "product": "Synergy 480 Gen11",
        "body": "[Customer-selected severity: Critical] The aircon in our server room died about an hour ago, "
                "room is at 38C and the Synergy frame is screaming. Who do we call?",
    },
    "CB-005": {
        "channel": "email", "subject": "volume read-only", "product": "Alletra 6010",
        "body": "At 09:14 the payments volume on the Alletra went read-only. Card transactions are failing "
                "in all 140 stores. We have not touched anything.",
    },
    "CB-010": {
        "channel": "email", "subject": "logical drive failed", "product": "ProLiant DL380 Gen11",
        "body": "Smart Array on DL380 host db07 reports logical drive 1 FAILED after a power blip; "
                "Windows won't boot. It's our staging SQL box.",
    },
}


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_result(name, r):
    decision = f"ABSTAIN ({r['abstain_reason']})" if r["abstain"] else "ROUTE"
    print(f"\n{name}: {decision}")
    print(f"  category  {r['category']:<16} conf {r['category_conf']:.2f}")
    print(f"  priority  {r['priority']:<16} conf {r['priority_conf']:.2f}   probs {r['prob_priority']}")
    print(f"  team      {r['assigned_team']:<16} conf {r['team_conf']:.2f}")
    print(f"  top3      " + ", ".join(f"{t['team']} {t['prob']:.2f}" for t in r["top3_teams"]))
    print(f"  sentiment {r['sentiment']:<16} conf {r['sentiment_conf']:.2f}")
    print(f"  similarity {r['similarity']:.3f}   escalation_flag {r['escalation_flag']}")
    print(f"  why this team: " + ", ".join(f"{e['term']!r}" for e in r["explanation"]))
    if r["warnings"]:
        print(f"  warnings: {r['warnings']}")


def main():
    header("GET /health")
    response = client.get("/health")
    print(response.status_code, response.json())

    header("GET /meta")
    meta = client.get("/meta").json()
    for key, value in meta.items():
        print(f"  {key}: {value}")

    header("POST /triage - 4 curveball tickets")
    for ticket_id, ticket in CURVEBALLS.items():
        response = client.post("/triage", json=ticket)
        assert response.status_code == 200, response.text
        show_result(ticket_id, response.json())

    header("POST /triage - error handling")
    response = client.post("/triage", json={"body": "   "})
    print(f"  empty body       -> {response.status_code} {response.json()['detail'][0]['msg']}")
    response = client.post("/triage", json={"body": "Our new widget router keeps rebooting every hour",
                                            "product": "Widget 9000"})
    print(f"  unknown product  -> {response.status_code} warnings: {response.json()['warnings']}")
    bad_csv = io.BytesIO(b"subject,body\nhello,world\n")
    response = client.post("/triage/batch", files={"file": ("bad.csv", bad_csv, "text/csv")})
    print(f"  CSV missing cols -> {response.status_code} {response.json()['detail']}")

    header("POST /triage/batch - curveball_tickets.csv (15 tickets: below MIN_BATCH_FOR_CAP, so no cap)")
    with open(CURVEBALL_CSV, "rb") as f:
        response = client.post("/triage/batch", files={"file": ("curveball_tickets.csv", f, "text/csv")})
    assert response.status_code == 200, response.text
    data = response.json()
    for key, value in data["summary"].items():
        print(f"  {key}: {value}")
    print()
    print(f"  {'id':<8}{'category':<12}{'priority':<9}{'team':<18}{'sentiment':<11}{'abstain':<8}"
          f"{'reason':<26}{'team_conf':>9}")
    for r in data["results"]:
        print(f"  {r['ticket_id']:<8}{r['category']:<12}{r['priority']:<9}{r['assigned_team']:<18}"
              f"{r['sentiment']:<11}{r['abstain']:<8}{r['abstain_reason']:<26}{r['team_conf']:>9.2f}")

    print("\n✅ All API calls succeeded.")


if __name__ == "__main__":
    main()
