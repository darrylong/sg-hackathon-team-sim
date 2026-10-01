"""Smoke-test every API endpoint in-process (no server needed).

Run:
    .venv/bin/python submission_template_vip/05_channel_analytics/src/test_api.py
"""

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def get(path, **params):
    response = client.get(path, params=params)
    assert response.status_code == 200, f"{path} -> {response.status_code}: {response.text[:300]}"
    return response.json()


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    header("GET /health")
    print(get("/health"))

    header("GET /overview")
    data = get("/overview")
    for key, value in data["kpis"].items():
        print(f"  {key}: {value:,.2f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"  quarterly_revenue: {len(data['quarterly_revenue']['quarters'])} quarters")
    for row in data["forecast"]:
        print(f"  forecast {row['region']:<5} ${row['forecast'] / 1e6:,.1f}M "
              f"[{row['lo80'] / 1e6:,.1f} .. {row['hi80'] / 1e6:,.1f}]")

    for by in ["region", "tier", "partner_type", "product_family"]:
        header(f"GET /performance?by={by}")
        data = get("/performance", by=by)
        for row in data["rows"]:
            att = f"{row['attainment_pct']:.1f}%" if row.get("attainment_pct") is not None else "n/a"
            print(f"  {row['group']:<17} ${row['revenue_usd'] / 1e6:>8,.1f}M  share {row['share_pct']:5.1f}%  "
                  f"margin {row['margin_pct']:5.1f}%  attainment {att}")
    data = get("/performance", by="tier", region="EMEA", quarters=2)
    print(f"  (EMEA, last 2 quarters, by tier: {len(data['rows'])} rows, period {data['period']})")

    header("GET /insights")
    data = get("/insights")
    for insight in data["insights"]:
        print(f"  [{insight['id']}] {insight['title']}\n      {insight['text']}")
    print("  Recommendations:")
    for rec in data["recommendations"]:
        print(f"   - {rec}")

    header("GET /at-risk")
    data = get("/at-risk")
    print(f"  all flagged: {data['count']}")
    for filters in ({"region": "EMEA"}, {"risk_type": "Fading"}, {"tier": "Platinum"}):
        print(f"  {filters}: {get('/at-risk', **filters)['count']}")
    first = data["partners"][0]
    print(f"  riskiest: {first['partner_id']} {first['partner_name']} [{first['risk_type']}] {first['reason']}")

    header(f"GET /partners/{first['partner_id']}")
    data = get(f"/partners/{first['partner_id']}")
    print(f"  {data['master']['partner_name']} | {data['master']['tier']} {data['master']['partner_type']} | "
          f"health {data['health']['health_score']:.3f} | {len(data['quarterly'])} quarters of history")
    missing = client.get("/partners/PT-99999")
    print(f"  unknown partner -> {missing.status_code} {missing.json()['detail']}")

    header("GET /forecast")
    data = get("/forecast")
    print(f"  method: {data['method']}")
    for row in data["backtest_mape"]:
        print(f"  MAPE {row['method']:<18} avg regions {row['avg_regions']:.2f}%  ALL {row['ALL']:.2f}%")
    print(f"  {data['method_text'][:160]}...")

    header("GET /whatif?target_change_pct=10")
    for row in get("/whatif", target_change_pct=10)["by_tier"]:
        print(f"  {row['tier']:<9} partners {row['partners']:>5}  missing {row['missing_now']:>5} -> "
              f"{row['missing_after']:>5}   attainment {row['attainment_now_pct']:.1f}% -> {row['attainment_after_pct']:.1f}%")
    bad = client.get("/whatif", params={"target_change_pct": 500})
    print(f"  out-of-range change -> {bad.status_code}")

    print("\n✅ All endpoints responded.")


if __name__ == "__main__":
    main()
