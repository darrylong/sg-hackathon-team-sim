import io
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from triage_core import triage

app = FastAPI(title="HPE Ticket Triage")


class Ticket(BaseModel):
    subject: str = ""
    body: str
    product: str = ""
    channel: str = "email"


def to_json(df):
    return json.loads(df.to_json(orient="records"))


@app.post("/triage")
def triage_one(ticket: Ticket):
    df = pd.DataFrame([{"ticket_id": "single", **ticket.model_dump()}])
    result = to_json(triage(df, use_cap=False))[0]
    result["decision"] = "Send to human" if result["abstain"] == 1 else "Route"
    return result


@app.post("/triage_csv")
async def triage_csv(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"row-{i+1}" for i in range(len(df))]
    for col in ["subject", "body", "product", "channel"]:
        if col not in df.columns:
            df[col] = ""
    return to_json(triage(df, use_cap=len(df) >= 100))