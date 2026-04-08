# PLAN — Fraud Detection End-to-End ML Ops

## Cấu trúc project

```
fraud-detection/
├── docker-compose.yml              # 8 services: postgres, mlflow, api,
│                                    #   frontend, prometheus, grafana,
│                                    #   ml-pipeline, airflow
├── .github/workflows/ci.yml        # CI/CD pipeline
├── Dockerfile.airflow               # Airflow custom Docker image
├── data/
│   ├── raw/creditcard.csv           # Raw dataset (98MB, Kaggle)
│   ├── processed/                  # Parquet files + scalers
│   └── scripts/download_data.py
├── models/
│   ├── lgbm_model.txt               # Active model (LightGBM, ~1MB)
│   ├── xgboost_model.json          # XGBoost (~1.5MB)
│   ├── rf_model.joblib              # RandomForest (~6.8MB)
│   └── best_config.json             # Best model metadata
├── services/
│   ├── ml-pipeline/                 # Preprocess + Train scripts
│   ├── ml-serving/                  # FastAPI inference + PostgreSQL
│   └── frontend/                    # Next.js 14 dashboard
├── airflow/                         # DAGs, logs, plugins, config
├── mlflow_artifacts/               # MLflow artifact storage
├── mlflow.db                        # SQLite backend (local dev)
├── monitoring/
│   ├── prometheus.yml               # Scrape config (2 targets)
│   └── grafana/provisioning/       # Auto-provisioned dashboards
└── postgres-init/                  # PostgreSQL init scripts
```

---

## Luồng ML Ops

```
creditcard.csv (raw, 98MB)
        │
        ▼ download_data.py
StandardScaler + StratifiedSplit + SMOTE
        │
        ▼ preprocess.py
X_train/test.parquet + scalers
        │
        ▼ train.py (5-fold CV, 3 models)
lgbm_model.txt + xgboost_model.json + rf_model.joblib
        │
        ├─► MLflow (PostgreSQL backend + artifact root)
        └─► models/best_config.json

ml-pipeline Dockerfile ──► ml-serving loads models ──► FastAPI
                                                           │
                                                  PostgreSQL (transactions)
                                                           │
                                                     Prometheus (scrape)
                                                           │
                                                     Grafana (dashboard)
                                                           │
                                                   GitHub Actions (CI/CD)
```

---

## Trạng thái hiện tại

### ✅ Hoàn thành

- [x] Download dataset + tiền xử lý (StandardScaler, SMOTE)
- [x] Huấn luyện 3 mô hình (LightGBM ⭐, XGBoost, RandomForest)
- [x] MLflow tracking (PostgreSQL backend, artifact root)
- [x] FastAPI inference server với KNN serving + PostgreSQL
- [x] Next.js frontend dashboard
- [x] Prometheus metrics scraping (2 targets: api + prometheus self)
- [x] Grafana dashboard (10 panels, auto-provisioned)
- [x] Apache Airflow orchestration (webserver + scheduler)
- [x] Docker Compose multi-service orchestration
- [x] GitHub Actions CI/CD (lint + test + docker build)
- [x] README + PRESENTATION documentation

### 🚧 Còn cần làm (TODO)

- [ ] Thiết lập Airflow Fernet key (`AIRFLOW_FERNET_KEY`)
- [ ] API authentication (hiện tại open)
- [ ] Model versioning với MLflow Model Registry
- [ ] Alerting rules cho Prometheus/Grafana
- [ ] CI: chạy trên Ubuntu thay vì self-hosted

---

## Debug checklist

```bash
# 1. Containers status
docker-compose ps
# → postgres: healthy
# → mlflow:    running
# → api:        healthy
# → prometheus: running
# → grafana:    running

# 2. API health
curl http://localhost:8000/health
# → {"status":"healthy","model_loaded":true,"model_type":"lightgbm"}

# 3. MLflow
open http://localhost:5001

# 4. Prometheus targets
curl "http://localhost:9090/api/v1/query?query=up"
# → up{job="api"} = 1, up{job="prometheus"} = 1

# 5. Grafana
open http://localhost:3002  # admin / admin
```

---

*Cập nhật: 2026-04-08*
