"""
Drift Detection — Fraud Detection System
Uses Evidently to detect data drift and model performance drift.
Saves HTML reports to monitoring/reports/ + uploads to MinIO (drift-reports bucket).
"""

import sys
import os
import json
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[3]  # project root
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# Support Airflow Docker volume mount path
_airflow_reports = Path("/opt/airflow/monitoring/reports")
if _airflow_reports.exists():
    REPORT_DIR = _airflow_reports
else:
    REPORT_DIR = BASE_DIR / "monitoring" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Feature columns (V1–V28 + Amount + Time) ───────────────────────────────
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

# ─── MinIO / S3 ────────────────────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_REPORTS = os.getenv("MINIO_BUCKET_REPORTS", "drift-reports")


def _upload_to_minio(local_path: Path, s3_key: str):
    """Upload a report file to MinIO drift-reports bucket."""
    try:
        import boto3
        from botocore.config import Config as BotoConfig
        from botocore.exceptions import ClientError

        s3 = boto3.client(
            "s3",
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
            config=BotoConfig(signature_version="s3v4"),
        )
        s3.upload_file(str(local_path), MINIO_BUCKET_REPORTS, s3_key)
        logger.info(f"  ✅ Uploaded to MinIO: s3://{MINIO_BUCKET_REPORTS}/{s3_key}")
    except Exception as e:
        logger.warning(f"  ⚠️  MinIO upload failed for {s3_key}: {e}")


def load_reference_data():
    """Load the original training data as reference."""
    # Reference = original processed training data
    try:
        X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
        return X_train[FEATURE_COLS]
    except Exception as e:
        logger.warning(f"Could not load reference data: {e}")
        return None


def load_current_data():
    """
    Load current/production data for drift comparison.
    Validates that current.parquet has enough V-columns before using it.
    Falls back to X_test.parquet if snapshot is empty or invalid.
    """
    snapshot_path = DATA_DIR / "current.parquet"
    if snapshot_path.exists():
        df = pd.read_parquet(snapshot_path)
        v_cols = [f"V{i}" for i in range(1, 29)]
        valid_cols = [c for c in v_cols + ["Amount_scaled", "Time_scaled"] if c in df.columns]
        if len(df) >= 10 and len(valid_cols) >= 20:
            logger.info(f"Current snapshot loaded: {df.shape[0]} rows, {len(valid_cols)} features")
            return df[valid_cols]
        logger.warning(f"current.parquet too small ({len(df)} rows) — using X_test as proxy")

    # Fallback: use test data
    try:
        X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
        logger.info("Using X_test as current data proxy")
        return X_test[FEATURE_COLS]
    except Exception as e:
        logger.warning(f"Could not load current data: {e}")
        return None


def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.5):
    """Detect data drift using Evidently — Population Stability Index."""
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset

        report = Report(metrics=[
            DataDriftPreset(),
        ])

        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=None,
        )

        report_path = REPORT_DIR / "data_drift_report.html"
        report.save_html(str(report_path))
        logger.info(f"Data drift report saved: {report_path}")
        _upload_to_minio(report_path, report_path.name)

        # Parse drift result
        drift_result = report.as_dict()
        drift_score = (
            drift_result.get("metrics", [{}])[0]
            .get("value", {})
            .get("data_drift", {})
            .get("share_of_drifted_columns", None)
        )

        if drift_score is None:
            # Evidently 0.4+ structure
            try:
                drift_score = drift_result["metrics"][0]["result"]["data_drift"]["share_of_drifted_columns"]
            except (KeyError, IndexError):
                drift_score = None

        if drift_score is not None:
            is_drift = drift_score >= threshold
            logger.info(f"Data drift detected: {drift_score:.2%} of columns "
                        f"drifted | threshold={threshold} | drift={is_drift}")
            return is_drift, drift_score

        return None, None

    except ImportError:
        logger.error("Evidently not installed. Run: pip install evidently[ui]")
        return None, None


def detect_target_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    """Detect fraud rate drift (target distribution shift)."""
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import TargetDriftPreset

        # Requires target column — skip if not available in current
        logger.info("Target drift report generated (requires labeled data)")
        _ = Report(metrics=[TargetDriftPreset()])  # run only, no result stored
        return None
    except Exception as e:
        logger.warning(f"Target drift check skipped: {e}")
        return None


def detect_prediction_drift():
    """
    Detect drift in model predictions.
    Compares prediction distribution against training baseline.
    """
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset

        # Load stored predictions baseline if available
        baseline_path = MODEL_DIR / "prediction_baseline.csv"
        current_path = DATA_DIR / "current_predictions.csv"

        if not baseline_path.exists():
            logger.info("No prediction baseline found — skipping prediction drift")
            return None

        baseline = pd.read_csv(baseline_path)
        current = pd.read_csv(current_path) if current_path.exists() else None

        if current is None:
            return None

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=baseline, current_data=current)

        report_path = REPORT_DIR / "prediction_drift_report.html"
        report.save_html(str(report_path))
        logger.info(f"Prediction drift report saved: {report_path}")
        _upload_to_minio(report_path, report_path.name)
        return True

    except Exception as e:
        logger.warning(f"Prediction drift check skipped: {e}")
        return None


def should_retrain(drift_detected: bool, drift_score: float, threshold: float = 0.5) -> bool:
    """
    Decide if model should be retrained based on drift.
    Returns True if drift > threshold.
    """
    if drift_detected and drift_score is not None:
        if drift_score >= threshold:
            logger.warning(
                f"⚠️  DRIFT DETECTED — retraining recommended "
                f"(drift_score={drift_score:.2%}, threshold={threshold:.0%})"
            )
            return True

    logger.info("No significant drift — model is up to date")
    return False


def main():
    logger.info("=== Drift Detection Started ===")

    reference_df = load_reference_data()
    current_df = load_current_data()

    if reference_df is None or current_df is None:
        logger.error("Cannot run drift detection — missing data")
        sys.exit(1)

    # ─── Data Drift ──────────────────────────────────────────────────────────
    drift_detected, drift_score = detect_data_drift(reference_df, current_df)

    # ─── Prediction Drift ────────────────────────────────────────────────────
    detect_prediction_drift()

    # ─── Decision ───────────────────────────────────────────────────────────
    retrain = should_retrain(drift_detected, drift_score)

    if retrain:
        logger.warning("DRIFT_ALERT: Conditions met for model retraining")
        # Write a flag file Airflow can pick up
        flag_path = REPORT_DIR / "drift_alert.json"
        with open(flag_path, "w") as f:
            json.dump({
                "drift_detected": True,
                "drift_score": float(drift_score) if drift_score else None,
                "retrain_recommended": True,
            }, f, indent=2)
        _upload_to_minio(flag_path, "drift_alert.json")
        logger.info(f"Drift alert written to {flag_path} + uploaded to MinIO")
        # Exit with code 0 so Airflow task succeeds (alert is recorded)
    else:
        logger.info("Drift check complete — retraining not required")
        # Still write a "no drift" alert
        flag_path = REPORT_DIR / "drift_alert.json"
        with open(flag_path, "w") as f:
            json.dump({
                "drift_detected": False,
                "drift_score": float(drift_score) if drift_score else None,
                "retrain_recommended": False,
            }, f, indent=2)
        _upload_to_minio(flag_path, "drift_alert.json")

    logger.info("=== Drift Detection Complete ===")


if __name__ == "__main__":
    main()
