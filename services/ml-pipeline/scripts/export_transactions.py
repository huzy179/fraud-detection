"""
Export Transactions — Fraud Detection System
Reads transaction data from PostgreSQL, scales features, and exports to parquet
for use as "current data" in Evidently drift detection.
"""

import os
import sys
import json
import logging
from pathlib import Path

import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[3]  # project root
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "monitoring" / "reports"

# ─── Feature columns ─────────────────────────────────────────────────────────
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
# PostgreSQL stores mixed-case columns (V1-V28, amount). read_sql returns them as-is.
# Scalers trained on capitalized "Amount" / "Time".
RAW_COLS_PG = [f"V{i}" for i in range(1, 29)] + ["amount"]


def _resolve_path(path, env_var, docker_path, fallback_path):
    """Resolve path with priority: env var > docker path > fallback."""
    if env_var:
        return Path(env_var)
    if os.path.exists(docker_path):
        return Path(docker_path)
    return Path(fallback_path)


def get_data_dir():
    # Airflow: /opt/airflow/data/processed
    # Docker volume: /app/data/processed
    # Local dev: project root relative
    env = os.getenv("DATA_DIR", "")
    if env and os.path.exists(env):
        return Path(env)
    for candidate in [
        "/opt/airflow/data/processed",
        "/app/data/processed",
        BASE_DIR / "data" / "processed",
    ]:
        if os.path.exists(str(candidate)):
            return Path(candidate)
    return BASE_DIR / "data" / "processed"


def get_database_url():
    """Get PostgreSQL connection URL."""
    url = os.getenv("DATABASE_URL", "")
    if url:
        return url
    # Airflow runtime
    return "postgresql+psycopg2://postgres:postgres@postgres:5432/fraud_detection"


def load_scalers():
    """Load Time and Amount scalers from processed data directory."""
    data_dir = get_data_dir()
    time_scaler_path = data_dir / "time_scaler.joblib"
    amount_scaler_path = data_dir / "amount_scaler.joblib"

    if not time_scaler_path.exists() or not amount_scaler_path.exists():
        logger.warning(f"Scalers not found at {data_dir}, scaling will be skipped")
        return None, None

    time_scaler = joblib.load(time_scaler_path)
    amount_scaler = joblib.load(amount_scaler_path)
    logger.info(f"Scalers loaded from {data_dir}")
    return time_scaler, amount_scaler


def fetch_transactions_from_db(db_url: str) -> pd.DataFrame:
    """
    Fetch all scored transactions from PostgreSQL.
    Falls back to SQLite for local development.
    """
    try:
        from sqlalchemy import create_engine
        engine = create_engine(db_url)

        query = """
            SELECT id, amount,
                   "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                   "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                   "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
                   fraud_probability, created_at
            FROM public.transactions
            ORDER BY created_at DESC
        """
        df = pd.read_sql(query, engine)
        engine.dispose()
        logger.info(f"Fetched {len(df)} transactions from PostgreSQL")
        return df
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL: {e}")
        logger.info("Checking for SQLite fallback...")

        sqlite_path = BASE_DIR / "fraud_detection.db"
        if sqlite_path.exists():
            try:
                engine = create_engine(f"sqlite:///{sqlite_path}")
                df = pd.read_sql("SELECT * FROM transactions ORDER BY created_at DESC", engine)
                engine.dispose()
                logger.info(f"Fetched {len(df)} transactions from SQLite")
                return df
            except Exception as sqlite_err:
                logger.warning(f"SQLite fallback also failed: {sqlite_err}")

        logger.error("No database available. Cannot export transactions.")
        return None


def build_current_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale Amount and Time, select only the 30 feature columns,
    and prepare current.parquet for Evidently.
    """
    if df is None or len(df) == 0:
        logger.error("No data to process")
        return None

    # Check for required V columns — match exact names returned by read_sql
    # PostgreSQL: "V1".."V28" (mixed-case), "amount" (lowercase)
    raw_v = [f"V{i}" for i in range(1, 29)] + ["amount"]
    missing_v = [c for c in raw_v if c not in df.columns]
    if missing_v:
        logger.error(f"Missing required columns: {missing_v}")
        return None

    # Work on a copy of the feature columns
    result = pd.DataFrame()

    for i in range(1, 29):
        # Column names from read_sql match exactly: "V1".."V28"
        src_col = f"V{i}"
        if src_col in df.columns:
            result[src_col] = pd.to_numeric(df[src_col], errors="coerce")

    # Scale Amount and Time
    time_scaler, amount_scaler = load_scalers()

    if time_scaler is not None and amount_scaler is not None:
        # DB has "amount" (lowercase); scaler expects capitalized "Amount"
        amount_col = "amount" if "amount" in df.columns else "Amount"
        time_col = "time" if "time" in df.columns else None
        result["Amount_scaled"] = amount_scaler.transform(df[[amount_col]].values)
        if time_col:
            result["Time_scaled"] = time_scaler.transform(df[[time_col]].values)
        else:
            # Use zeros if time not available (time column absent from DB)
            result["Time_scaled"] = 0.0
        logger.info("Amount and Time scaled successfully")
    else:
        # Fallback: no scaler available — use unscaled values
        amount_col = "amount" if "amount" in df.columns else "Amount"
        time_col = "time" if "time" in df.columns else None
        result["Amount_scaled"] = pd.to_numeric(df[amount_col], errors="coerce")
        result["Time_scaled"] = pd.to_numeric(df[time_col], errors="coerce") if time_col else 0.0
        logger.warning("Using unscaled Amount/Time as fallback")

    # Drop rows with NaN in all V columns
    v_cols = [f"V{i}" for i in range(1, 29)]
    result = result.dropna(subset=v_cols, how="all")
    logger.info(f"Prepared {len(result)} rows with {len(result.columns)} features")

    return result


def build_predictions_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build current_predictions.csv containing fraud_probability per transaction.
    Used for prediction drift detection.
    """
    if df is None or len(df) == 0:
        return None

    if "fraud_probability" not in df.columns:
        logger.warning("fraud_probability column not found, skipping predictions CSV")
        return None

    out = pd.DataFrame({
        "transaction_id": df["id"] if "id" in df.columns else range(len(df)),
        "fraud_probability": pd.to_numeric(df["fraud_probability"], errors="coerce").fillna(0),
    })
    return out


def save_outputs(current_df: pd.DataFrame, predictions_df: pd.DataFrame):
    """Save parquet and CSV outputs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    current_path = DATA_DIR / "current.parquet"
    predictions_path = DATA_DIR / "current_predictions.csv"
    metadata_path = DATA_DIR / "export_metadata.json"

    if current_df is not None and len(current_df) > 0:
        current_df.to_parquet(current_path, index=False)
        logger.info(f"Saved current.parquet ({len(current_df)} rows) → {current_path}")
    else:
        logger.warning("No current data to save")

    if predictions_df is not None and len(predictions_df) > 0:
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved current_predictions.csv ({len(predictions_df)} rows) → {predictions_path}")

    # Write export metadata
    metadata = {
        "exported_at": pd.Timestamp.now().isoformat(),
        "rows": int(len(current_df)) if current_df is not None else 0,
        "current_parquet": str(current_path),
        "predictions_csv": str(predictions_path),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved export metadata → {metadata_path}")


def main():
    logger.info("=== Transaction Export Started ===")

    db_url = get_database_url()
    logger.info(f"Database URL: {db_url.split('@')[-1] if '@' in db_url else db_url}")

    # Step 1: Fetch from DB
    raw_df = fetch_transactions_from_db(db_url)

    if raw_df is None:
        logger.error("Export failed: no database connection available")
        sys.exit(1)

    if len(raw_df) == 0:
        logger.warning("No transactions found in database. Skipping export (detect_drift will use X_test as proxy).")
        # Write empty metadata so downstream knows export ran but found nothing
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        metadata_path = DATA_DIR / "export_metadata.json"
        metadata = {
            "exported_at": pd.Timestamp.now().isoformat(),
            "rows": 0,
            "note": "No transactions in database yet",
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("=== Transaction Export Complete (empty — detect_drift will use X_test) ===")
        return

    # Step 2: Build feature parquet
    current_df = build_current_parquet(raw_df)

    # Step 3: Build predictions CSV
    predictions_df = build_predictions_csv(raw_df)

    # Step 4: Save
    save_outputs(current_df, predictions_df)

    logger.info(f"=== Transaction Export Complete ({len(current_df) if current_df is not None else 0} rows) ===")


if __name__ == "__main__":
    main()
