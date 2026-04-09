# 06 — Airflow: ML Pipeline Orchestration

## Tổng quan

Apache Airflow điều phối toàn bộ ML pipeline theo schedule, tự động chạy retraining khi cần.

- **Port:** 8080 (Webserver)
- **Scheduler:** LocalExecutor (single-node)
- **Metadata DB:** PostgreSQL `airflow_db`
- **DAG file:** [airflow/dags/fraud_pipeline_dag.py](airflow/dags/fraud_pipeline_dag.py)

---

## DAG Structure

```
fraud_ml_pipeline (runs @daily)
│
├── download_data
│   └── BashOperator: download_data.py
│
├── preprocess_data
│   └── BashOperator: preprocess.py
│
├── train_model
│   └── BashOperator: train.py
│
├── export_transactions
│   └── BashOperator: export_transactions.py (trigger_rule: all_done)
│
└── detect_drift
    └── BashOperator: detect_drift.py (trigger_rule: all_done)
```

---

## Cách sử dụng

### Truy cập Airflow UI

```bash
# Mở browser
open http://localhost:8080

# Login: airflow / airflow (default)
```

### Trigger DAG manually

```bash
# Qua CLI (trong container)
docker compose exec airflow-scheduler airflow dags trigger fraud_ml_pipeline

# Hoặc qua UI: DAGs → fraud_ml_pipeline → Trigger
```

### Check DAG status

```bash
docker compose exec airflow-scheduler airflow tasks list fraud_ml_pipeline
docker compose exec airflow-scheduler airflow dags list-runs fraud_ml_pipeline
```

---

## DAG Code

```python
# airflow/dags/fraud_pipeline_dag.py

with DAG(
    dag_id="fraud_ml_pipeline",
    schedule_interval="@daily",      # Chạy mỗi ngày lúc midnight
    start_date=days_ago(1),
    catchup=False,                   # Không chạy lại những ngày miss
    tags=["fraud", "ml", "pipeline"],
) as dag:

    download_data = BashOperator(
        task_id="download_data",
        bash_command="cd /opt/airflow && PYTHONPATH=/opt/airflow python data/scripts/download_data.py",
    )

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="cd /opt/airflow/services/ml-pipeline && "
                     "PYTHONPATH=/opt/airflow/services/ml-pipeline "
                     "python scripts/preprocess.py",
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow/services/ml-pipeline && "
                     "PYTHONPATH=/opt/airflow/services/ml-pipeline "
                     "python scripts/train.py",
    )

    export_transactions = BashOperator(
        task_id="export_transactions",
        bash_command="cd /opt/airflow/services/ml-pipeline && "
                     "PYTHONPATH=/opt/airflow/services/ml-pipeline "
                     "python scripts/export_transactions.py",
        trigger_rule="all_done",  # Chạy kể cả khi train thất bại
    )

    detect_drift = BashOperator(
        task_id="detect_drift",
        bash_command="cd /opt/airflow/services/ml-pipeline && "
                     "PYTHONPATH=/opt/airflow/services/ml-pipeline "
                     "python scripts/detect_drift.py",
        trigger_rule="all_done",
    )

    # Dependency graph
    download_data >> preprocess >> train >> export_transactions >> detect_drift
```

---

## Task Dependencies

```
download_data  ──→  preprocess_data  ──→  train_model
                                                  │
                                                  ▼
                          export_transactions  ←  ┘
                                  │
                                  ▼
                              detect_drift
```

### Dependency Operators

```python
# Sequential: A >> B >> C
A >> B >> C

# Equivalently:
B.set_upstream(A)
C.set_upstream(B)
```

### Trigger Rules

| Rule | Khi nào chạy? |
|------|--------------|
| `all_success` (default) | Tất cả upstream tasks thành công |
| `all_done` | Tất cả upstream tasks đã finish (success hoặc fail) |
| `one_success` | Ít nhất 1 upstream task thành công |
| `all_failed` | Tất cả upstream tasks fail |

**Tại sao `trigger_rule="all_done"` cho export và detect_drift?**
- Ngay cả khi train fail, vẫn muốn export current transactions và check drift
- Drift detection không cần model mới — so sánh reference data với current production data
- Alert vẫn được ghi nếu có drift, независимо от train result

---

## Vì sao dùng Airflow?

### So sánh với alternatives

| Feature | Airflow | Prefect | Dagster | Cron + Scripts |
|---------|---------|---------|---------|-------------------|
| Python-native | ✅ | ✅ | ✅ | ❌ |
| DAG UI | ✅ | ✅ | ✅ | ❌ |
| Scheduling | ✅ | ✅ | ✅ | ⚠️ (basic) |
| Retries | ✅ | ✅ | ✅ | ❌ |
| SLA monitoring | ✅ | ✅ | ✅ | ❌ |
| Complex dependencies | ✅ | ✅ | ✅ | ❌ |
| Learning curve | Medium | Easy | Medium | Easy |

### Tại sao LocalExecutor thay vì CeleryExecutor/KubernetesExecutor?
- **Single-node deployment**: không cần distributed workers
- **Simplicity**: không cần Redis/Celery broker
- **Sufficient for demo**: dataset nhỏ, training không tốn nhiều resources
- **Production scale**: khi cần scale, đổi sang CeleryExecutor + Redis

### Tại sao `catchup=False`?
```python
catchup=False  # Không chạy lại những DAG runs miss khi Airflow down
```
- Trong fraud detection, daily retraining là nice-to-have, không phải hard requirement
- Không muốn backfill nhiều ngày khi restart sau maintenance
- Production có thể đổi thành `catchup=True` nếu cần historical runs

### Tại sao @daily schedule?
- **Retraining quá thường xuyên** (hourly): tốn compute, có thể overfit
- **Retraining quá ít** (monthly): có thể miss drift patterns
- **Daily**: reasonable balance — catches drift weekly/monthly patterns
- Có thể điều chỉnh: `@hourly` cho production cao traffic, `@weekly` cho stable patterns

### Tại sao separate `download_data` task?
- Raw data có thể được update định kỳ (streaming new transactions, new Kaggle dataset)
- Tách biệt download logic → có thể add source validation
- Dễ debug: nếu download fail, preprocess không chạy

### Tại sao `PYTHONPATH` set trong mỗi task?
- Airflow scheduler chạy task trong subprocess
- `PYTHONPATH=/opt/airflow/services/ml-pipeline` để scripts có thể import modules từ thư mục đó
- Prevents `ModuleNotFoundError` khi chạy `python scripts/preprocess.py`
