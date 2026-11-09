import io
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
import pandas as pd

from nonstop_search import run_search

BUCKET = os.environ["RESULTS_BUCKET"]
CONFIG_KEY = os.environ.get("CONFIG_KEY", "config/searches.xlsx")
s3 = boto3.client("s3")


def is_blank(value) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def parse_row(row) -> dict:
    """Turn one spreadsheet row into a search dict. return_days blank means one-way."""
    return {
        "origin": str(row["origin"]).strip(),
        "destination": str(row["destination"]).strip(),
        "days_ahead": int(row["days_ahead"]),
        "return_days": None if is_blank(row.get("return_days")) else int(row["return_days"]),
    }


def load_searches(event) -> list[dict]:
    """Use searches sent in the event if present, otherwise read searches.xlsx from S3."""
    if "searches" in event:
        return event["searches"]
    obj = s3.get_object(Bucket=BUCKET, Key=CONFIG_KEY)
    df = pd.read_excel(io.BytesIO(obj["Body"].read()), engine="openpyxl")
    df = df.dropna(how="all")  # skip fully blank rows
    return [parse_row(row) for _, row in df.iterrows()]


def lambda_handler(event, context):
    results = []
    for s in load_searches(event):
        origin = s["origin"]
        destination = s["destination"]
        days_ahead = s["days_ahead"]
        return_days = s["return_days"]

        try:
            df = run_search(origin, destination, days_ahead, return_days)
        except (SystemExit, Exception) as e:  # run_search uses sys.exit for bad input
            results.append({"search": s, "error": str(e) or repr(e)})
            continue

        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%dT%H%M%S")
        route = f"{origin}_{destination}".replace(",", "-").upper()
        suffix = f"d{days_ahead}" + (f"_r{return_days}" if return_days is not None else "")
        key = f"flights/{route}/{stamp}_{suffix}.parquet"

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)  # requires pyarrow
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/vnd.apache.parquet",
        )
        results.append({"search": s, "rows": len(df), "s3_key": key})

    return {"results": results}