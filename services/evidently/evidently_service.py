"""
Evidently Microservice — Standalone FastAPI wrapper for drift detection.
Stores results in MinIO (drift-reports bucket).

Endpoints:
  POST /run          — run all drift reports
  GET  /reports      — list available reports in MinIO
  GET  /reports/{n}  — download a specific report HTML/JSON
  GET  /health       — health check
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config from env ─────────────────────────────────────────────────────────────
REPORT_DIR = Path(os.getenv("REPORT_DIR", "/app/reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "/app/data/processed"))

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "drift-reports")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]


# ── S3 client ──────────────────────────────────────────────────────────────────
def _get_s3():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=MINIO_REGION,
        config=BotoConfig(signature_version="s3v4"),
    )


def _upload(local_path: Path, s3_key: str):
    """Upload file to MinIO. s3_key should NOT include bucket name (already set in _get_s3)."""
    try:
        _get_s3().upload_file(str(local_path), MINIO_BUCKET, s3_key)
        logger.info(f"  Uploaded → s3://{MINIO_BUCKET}/{s3_key}")
    except ClientError as e:
        logger.warning(f"  MinIO upload failed for {s3_key}: {e}")


# ── Data loaders ────────────────────────────────────────────────────────────────
def _load_reference():
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    available = [c for c in FEATURE_COLS if c in X_train.columns]
    logger.info(f"Reference loaded: {X_train.shape[0]} rows × {len(available)} features")
    return X_train[available]


def _load_current():
    snapshot = PROCESSED_DIR / "current.parquet"
    if snapshot.exists():
        df = pd.read_parquet(snapshot)
        v_cols = [f"V{i}" for i in range(1, 29)]
        valid_cols = [c for c in v_cols + ["Amount_scaled", "Time_scaled"] if c in df.columns]
        if len(df) >= 10 and len(valid_cols) >= 20:
            logger.info(f"Current data loaded from snapshot: {df.shape[0]} rows, {len(valid_cols)} features")
            return df[valid_cols]
        logger.info(f"current.parquet too small ({len(df)} rows) — using X_test as proxy")

    X_test = PROCESSED_DIR / "X_test.parquet"
    if X_test.exists():
        df = pd.read_parquet(X_test)
        available = [c for c in FEATURE_COLS if c in df.columns]
        logger.info(f"Using X_test.parquet as current data proxy: {df.shape[0]} rows")
        return df[available]

    raise FileNotFoundError(
        f"No current data found. Ensure X_test.parquet exists at {PROCESSED_DIR}."
    )


# ── Evidently report runners ───────────────────────────────────────────────────
def run_data_drift_report() -> dict:
    """Run DataDriftPreset, save HTML/JSON, upload to MinIO."""
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset

    reference_df = _load_reference()
    current_df = _load_current()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=None)

    html_path = REPORT_DIR / "data_drift_report.html"
    json_path = REPORT_DIR / "data_drift_report.json"

    report.save_html(str(html_path))
    report.save_json(str(json_path))

    # Upload to MinIO
    _upload(html_path, "data_drift_report.html")
    _upload(json_path, "data_drift_report.json")

    # Extract drift score from legacy report
    result = report.as_dict()
    try:
        drift_score = result["metrics"][0]["result"]["share_of_drifted_columns"]
    except Exception:
        drift_score = None

    is_drift = drift_score is not None and drift_score >= 0.5

    # Write + upload alert JSON
    alert = {
        "drift_detected": is_drift,
        "drift_score": float(drift_score) if drift_score is not None else None,
        "retrain_recommended": is_drift,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "data_drift",
    }
    alert_path = REPORT_DIR / "drift_alert.json"
    with open(alert_path, "w") as f:
        json.dump(alert, f, indent=2)
    _upload(alert_path, "drift_alert.json")

    logger.info(f"Data drift report done — score={drift_score}, drift={is_drift}")
    return alert


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Evidently Drift Service",
    description="Standalone Evidently microservice for fraud detection drift monitoring",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {
        "service": "evidently",
        "status": "running",
        "message": "Use /docs for API docs, /health for health check, and /run to generate drift reports.",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "evidently"}


class RunResponse(BaseModel):
    drift_detected: bool
    drift_score: float | None
    retrain_recommended: bool
    generated_at: str
    report_type: str


@app.post("/run", response_model=RunResponse)
async def run_reports():
    """Run all Evidently drift reports. Returns alert JSON."""
    try:
        alert = run_data_drift_report()
        return JSONResponse(content=alert)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception:
        logger.exception("Drift detection failed")
        raise HTTPException(500, "Drift detection failed. Check logs.")


@app.get("/reports")
async def list_reports():
    """List available reports in MinIO drift-reports bucket."""
    try:
        s3 = _get_s3()
        resp = s3.list_objects_v2(Bucket=MINIO_BUCKET)

        files = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            # Support both formats:
            # 1) New keys: data_drift_report.html
            # 2) Legacy keys: drift-reports/data_drift_report.html
            if key.endswith("/"):
                continue
            if key.startswith("drift-reports/"):
                key = key.replace("drift-reports/", "", 1)
            files.append(key)

        return {"reports": files}
    except ClientError as e:
        raise HTTPException(502, f"MinIO error: {e}")


@app.get("/reports/{name}")
async def get_report(name: str):
    """Download a specific report file (served from local cache / MinIO)."""
    local_path = REPORT_DIR / name

    # Try local first
    if local_path.exists():
        media = "application/json" if name.endswith(".json") else "text/html"
        return FileResponse(str(local_path), media_type=media)

    # Fallback to MinIO. Try current key format first, then legacy prefixed key.
    candidate_keys = [name, f"drift-reports/{name}"]
    try:
        s3 = _get_s3()
        downloaded = False
        for s3_key in candidate_keys:
            try:
                s3.download_file(MINIO_BUCKET, s3_key, str(local_path))
                downloaded = True
                break
            except ClientError as inner:
                inner_code = inner.response.get("Error", {}).get("Code", "")
                if inner_code in {"404", "NoSuchKey", "NotFound"}:
                    continue
                raise

        if not downloaded:
            raise HTTPException(404, f"Report '{name}' not found in MinIO")
    except ClientError as e:
        raise HTTPException(502, f"MinIO error: {e}")

    media = "application/json" if name.endswith(".json") else "text/html"
    return FileResponse(str(local_path), media_type=media)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
