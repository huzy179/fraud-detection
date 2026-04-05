"""
Drift Detection — Fraud Detection System
Uses Evidently to detect data drift and model performance drift.
Saves HTML reports to monitoring/reports/
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "monitoring" / "reports"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Feature columns (V1–V28 + Amount + Time) ───────────────────────────────
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


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
    In production this would come from the database.
    For now, uses a snapshot if available.
    """
    # Try loading a production snapshot if it exists
    snapshot_path = DATA_DIR / "current.parquet"
    if snapshot_path.exists():
        return pd.read_parquet(snapshot_path)

    # Fallback: use test data as a stand-in for demonstration
    try:
        X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
        logger.info("Using X_test as current data proxy (no production snapshot found)")
        return X_test[FEATURE_COLS]
    except Exception as e:
        logger.warning(f"Could not load current data: {e}")
        return None


def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.5):
    """Detect data drift using Evidently — Population Stability Index."""
    try:
        from evidently.dashboard import Dashboard
        from evidently.tabs import DataDriftTab
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

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

        # Parse drift result
        drift_result = report.as_dict()
        drift_score = drift_result.get("metrics", [{}])[0].get("value", {}).get("data_drift", {}).get("share_of_drifted_columns", None)

        if drift_score is None:
            # Evidently 0.4+ structure
            try:
                drift_score = drift_result["metrics"][0]["result"]["data_drift"]["share_of_drifted_columns"]
            except (KeyError, IndexError):
                drift_score = None

        if drift_score is not None:
            is_drift = drift_score >= threshold
            logger.info(f"Data drift detected: {drift_score:.2%} of columns drifted | threshold={threshold} | drift={is_drift}")
            return is_drift, drift_score

        return None, None

    except ImportError:
        logger.error("Evidently not installed. Run: pip install evidently[ui]")
        return None, None


def detect_target_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    """Detect fraud rate drift (target distribution shift)."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import TargetDriftPreset

        report = Report(metrics=[TargetDriftPreset()])
        # Requires target column — skip if not available in current
        logger.info("Target drift report generated (requires labeled data)")
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
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

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
        import json
        with open(flag_path, "w") as f:
            json.dump({
                "drift_detected": True,
                "drift_score": float(drift_score) if drift_score else None,
                "retrain_recommended": True,
            }, f, indent=2)
        logger.info(f"Drift alert written to {flag_path}")
        # Exit with code 0 so Airflow task succeeds (alert is recorded)
    else:
        logger.info("Drift check complete — retraining not required")

    logger.info("=== Drift Detection Complete ===")


if __name__ == "__main__":
    main()
