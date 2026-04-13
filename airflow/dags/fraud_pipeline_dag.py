"""
Airflow DAG — Fraud Detection ML Pipeline
Orchestrates: preprocess → train → evaluate → detect drift (via Evidently microservice)
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.models.baseoperator import TriggerRule
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

# ─── DAG Defaults ───────────────────────────────────────────────────────────
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_ml_pipeline",
    default_args=default_args,
    description="Fraud detection end-to-end ML pipeline: preprocess → train → drift check",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["fraud", "ml", "pipeline"],
) as dag:

    # ─── Step 1: Download raw data ───────────────────────────────────────────
    download_data = BashOperator(
        task_id="download_data",
        bash_command="cd /opt/airflow && PYTHONPATH=/opt/airflow python data/scripts/download_data.py",
    )

    # ─── Step 2: Preprocess data ─────────────────────────────────────────────
    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="cd /opt/airflow/services/ml-pipeline && PYTHONPATH=/opt/airflow/services/ml-pipeline python scripts/preprocess.py",
    )

    # ─── Step 3: Train models ─────────────────────────────────────────────────
    train = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow/services/ml-pipeline && PYTHONPATH=/opt/airflow/services/ml-pipeline python scripts/train.py",
    )

    # ─── Step 4: Export production transactions ──────────────────────────────────
    export_transactions = BashOperator(
        task_id="export_transactions",
        bash_command=(
            "cd /opt/airflow/services/ml-pipeline && "
            "PYTHONPATH=/opt/airflow/services/ml-pipeline "
            "python scripts/export_transactions.py"
        ),
        trigger_rule="all_done",
    )

    # ─── Step 5: Detect drift via Evidently microservice ──────────────────────
    # First ensure current.parquet exists (export step ran), then call service
    detect_drift = SimpleHttpOperator(
        task_id="detect_drift",
        method="POST",
        http_conn_id="evidently_service",
        endpoint="/run",
        response_check=lambda response: response.json().get("drift_score") is not None,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ─── Pipeline flow ────────────────────────────────────────────────────────
    download_data >> preprocess >> train >> export_transactions >> detect_drift
