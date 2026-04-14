# 01 — Overview: Kiến trúc tổng quan

## Tổng quan

Hệ thống **fraud-detection** là một ML Ops platform end-to-end cho bài toán credit card fraud detection, được xây dựng với 12 Docker services.

- **Dataset:** Kaggle Credit Card Fraud (284,807 giao dịch, ~0.17% fraud rate)
- **Output:** Fraud probability + SHAP explanation cho mỗi transaction
- **Kiến trúc:** Real-time inference (FastAPI) + Batch ML pipeline (Airflow) + Drift Detection (Evidently)
- **Object Storage:** MinIO (S3-compatible) — lưu MLflow artifacts + drift reports

---

## Sơ đồ kiến trúc

```
┌──────────────────────────────────────────────────────────────────┐
│  Client / Demo                                                   │
│  http://localhost:3000 (Frontend — Next.js 14)                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP POST /predict
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  API Service (FastAPI)  ── port 8000                            │
│  • ML Inference (KNN lookup + LightGBM fallback)                  │
│  • SHAP Explainability                                           │
│  • Transaction CRUD (PostgreSQL)                                 │
│  • Prometheus /metrics endpoint                                  │
│  • Drift status + report endpoints                              │
└──────┬───────────────┬────────────────────┬──────────────────────┘
       │               │                    │
       ▼               ▼                    ▼
┌────────────┐   ┌──────────┐      ┌─────────────┐
│ PostgreSQL │   │  MinIO   │      │ Prometheus  │
│ port 5432  │   │:9000/9001│      │  port 9090  │
│• Txn store │   │• MLflow  │      │• Scrapes    │
│• Airflow   │   │  artifacts│      │  /metrics   │
│• MLflow DB │   │• drift   │      └──────┬──────┘
└────────────┘   │  reports │             │
                 └─────┬────┘             ▼
                       │           ┌─────────────┐
                       ▼           │  Grafana   │
              ┌──────────────┐    │ port 3002  │
              │   MLflow     │    │ Dashboard  │
              │ port 5001    │    └─────────────┘
              └──────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Airflow (port 8080) — DAG: preprocess → train → export → drift  │
│  Evidently Service (port 8002) — Drift detection API             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
1. Kaggle CSV (raw/creditcard.csv)
       │
       ▼
2. preprocess.py ── StandardScaler(Time, Amount)
       │          ── Stratified train/test split (80/20)
       │          ── SMOTE oversampling (sampling_strategy=0.5)
       ▼
3. train.py ──────── 5-fold Stratified CV
       │          ── 3 models: LightGBM, XGBoost, RandomForest
       │          ── Threshold optimization (max F1)
       │          ── Logs to MLflow (S3 artifact root)
       ▼
4. models/lgbm_model.txt (active model)
       │
       ▼
5. FastAPI (/predict endpoint)
       │  KNN nearest-neighbor lookup trong preprocessed feature space
       │  LightGBM fallback cho direct inference
       │  SHAP explainability
       ▼
6. PostgreSQL (transactions table)
       │
       ▼
7. Next.js Dashboard (real-time view)
       │
       ▼
8. Prometheus ── Grafana (metrics visualization)
       │
       ▼
9. Evidently Service (detect_drift.py → HTTP call port 8002)
       │  Compare reference vs. current data distribution
       │  Report saved to MinIO drift-reports bucket
       ▼
10. Airflow DAG ── triggers retraining nếu drift > threshold
```

---

## Tech Stack

| Layer | Technology | Port | Purpose |
|-------|-----------|------|---------|
| **Database** | PostgreSQL 15 | 5432 | Transaction storage + MLflow/Airflow backend |
| **Object Storage** | MinIO (S3) | 9000/9001 | MLflow artifacts + drift reports |
| **API** | FastAPI (Python 3.9+) | 8000 | ML inference + CRUD |
| **Frontend** | Next.js 14 (TypeScript) | 3000 | Real-time dashboard |
| **ML Tracking** | MLflow | 5001 | Experiment tracking, model registry |
| **Drift Detection** | Evidently AI Service | 8002 | Drift detection HTTP API |
| **Orchestration** | Apache Airflow | 8080 | DAG scheduling |
| **Metrics** | Prometheus v2.47 | 9090 | Metrics collection |
| **Visualization** | Grafana 10.1 | 3002 | Dashboard |
| **Container** | Docker Compose | — | Service orchestration |

---

## Vì sao chọn kiến trúc này?

### FastAPI thay vì Flask
- **Async native:** FastAPI hỗ trợ `async/await` tự nhiên, phù hợp với I/O-bound tasks (DB calls, external API)
- **Auto-generated docs:** Swagger UI tự động từ type hints
- **Pydantic validation:** Request/response được validate tự động, giảm boilerplate
- **Performance:** Nhanh hơn Flask nhờ Starlette + Pydantic tối ưu

### KNN + Model hybrid thay vì chỉ dùng model
- **KNN lookup** cho serving: không cần chạy model inference, chỉ tìm nearest neighbor trong preprocessed data
- **LightGBM/XGBoost fallback**: khi serving index không khả dụng
- **Tradeoff**: KNN cho speed, gradient boosting cho accuracy. Hệ thống tự fallback linh hoạt.

### MinIO thay vì AWS S3
- **Self-hosted S3-compatible storage**: không phụ thuộc cloud provider
- **MinIO chạy trong Docker**: cùng infra với toàn bộ stack
- **MLflow artifact root**: artifacts được lưu vào MinIO thay vì local filesystem
- **Evidently reports**: drift reports được upload lên MinIO thay vì local disk

### Evidently microservice riêng biệt
- Chạy như service độc lập (port 8002), không phụ thuộc ml-pipeline
- API Server gọi được HTTP endpoint `/drift-status` và `/drift-report`
- Evidently reports (JSON + HTML) được lưu trong MinIO
- Tách biệt: batch drift detection script và real-time API

### 12 services riêng biệt
- **Separation of concerns**: mỗi service có 1 responsibility rõ ràng
- **Independent scaling**: Prometheus/Grafana không cần scale, ML Serving có thể scale horizontally
- **Fault isolation**: 1 service down không ảnh hưởng toàn bộ hệ thống
- **minio-init**: khởi tạo buckets tự động khi MinIO ready

---

## Bảng Services đầy đủ

| # | Service | Công nghệ | Port | Restart Policy | Mục đích |
|---|---------|-----------|------|---------------|---------|
| 1 | `postgres` | PostgreSQL 15 | 5432 | always | Transaction storage + MLflow/Airflow metadata |
| 2 | `minio` | MinIO S3 | 9000/9001 | always | Artifact storage + drift reports |
| 3 | `minio-init` | mc client | — | no | Bootstrap buckets (one-shot) |
| 4 | `mlflow` | MLflow | 5001 | always | Experiment tracking server |
| 5 | `evidently-service` | FastAPI + Evidently | 8002 | unless-stopped | Drift detection microservice |
| 6 | `api` | FastAPI + Uvicorn | 8000 | unless-stopped | ML inference + transaction CRUD |
| 7 | `frontend` | Next.js 14 | 3000 | always | Real-time dashboard |
| 8 | `ml-pipeline` | Python/Sklearn | — | no | One-shot batch job (preprocess → train → drift) |
| 9 | `prometheus` | Prometheus v2.47 | 9090 | unless-stopped | Metrics collection |
| 10 | `grafana` | Grafana 10.1 | 3002 | unless-stopped | Visualization dashboard |
| 11 | `airflow-webserver` | Apache Airflow | 8080 | unless-stopped | DAG UI + manual trigger |
| 12 | `airflow-scheduler` | Apache Airflow | — | unless-stopped | DAG scheduler (LocalExecutor) |

---

## Deployment Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                        docker-compose.yml                             │
│                                                                      │
│  Infrastructure Layer     │   Application Layer     │  Orchestration │
│  ─────────────────────    │   ─────────────────    │  ──────────────│
│  postgres                 │   api                   │  airflow-*      │
│  minio (+ minio-init)     │   frontend              │  ml-pipeline    │
│  mlflow                   │   evidently-service     │                 │
│  prometheus               │                         │                 │
│  grafana                  │                         │                 │
└──────────────────────────────────────────────────────────────────────┘
```
