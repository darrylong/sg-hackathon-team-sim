import requests
import pandas as pd
import streamlit as st

API = "http://127.0.0.1:8000"
PRODUCTS = ["GreenLake", "ProLiant DL380 Gen11", "ProLiant DL360 Gen10", "Synergy 480 Gen11",
            "Alletra 6010", "Nimble HF40", "Aruba CX 6300", "Aruba CX 8360", "iLO 6"]

st.set_page_config(page_title="HPE Ticket Triage", page_icon="🎫", layout="wide")
st.title("🎫 HPE Support Ticket Triage")
st.caption("Paste a ticket or upload a CSV. Tickets the model is unsure about go to the human triage desk.")

tab1, tab2 = st.tabs(["Single ticket", "Upload CSV"])

with tab1:
    subject = st.text_input("Subject")
    body = st.text_area("Body", height=150)
    c1, c2 = st.columns(2)
    product = c1.selectbox("Product", PRODUCTS)
    channel = c2.selectbox("Channel", ["email", "portal", "chat"])

    if st.button("Triage", type="primary"):
        if not body.strip():
            st.warning("Please paste the ticket body.")
        else:
            r = requests.post(f"{API}/triage", timeout=30, json={
                "subject": subject, "body": body, "product": product, "channel": channel}).json()

            if r["abstain"] == 1:
                st.error("⚠️ SEND TO HUMAN: the model is not confident enough to route this ticket.")
            else:
                st.success(f"✅ ROUTE to {r['assigned_team']}")

            if r["sentiment"] == "Angry" or (r["sentiment"] == "Frustrated" and r["priority"] in ["P1", "P2"]):
                st.warning("🔥 Escalation risk: unhappy customer on an urgent ticket")

            a, b, c, d = st.columns(4)
            a.metric("Category", r["category"] or "-")
            b.metric("Priority", r["priority"])
            c.metric("Team", r["assigned_team"] or "-")
            d.metric("Sentiment", r["sentiment"])

with tab2:
    f = st.file_uploader("CSV with columns: subject, body, product, channel", type="csv")
    if f is not None and st.button("Triage CSV", type="primary"):
        r = requests.post(f"{API}/triage_csv", timeout=300,
                          files={"file": (f.name, f.getvalue(), "text/csv")}).json()
        res = pd.DataFrame(r)
        st.write(f"**{len(res)} tickets**, {res['abstain'].mean():.1%} sent to humans")
        st.dataframe(res)
        st.download_button("Download results", res.to_csv(index=False), "triage_results.csv")