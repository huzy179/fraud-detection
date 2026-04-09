# 01 — Overview: Kiến trúc tổng quan

## Tổng quan

Hệ thống **fraud-detection** là một ML Ops platform end-to-end cho bài toán credit card fraud detection, được xây dựng với 8 Docker services.

- **Dataset:** Kaggle Credit Card Fraud (284,807 giao dịch, ~0.17% fraud rate)
- **Output:** Fraud probability + SHAP explanation cho mỗi transaction
- **Kiến trúc:** Real-time inference (FastAPI) + Batch ML pipeline (Airflow)

---

## Sơ đồ kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client / Demo                               │
│              http://localhost:3000 (Frontend)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /predict
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│   API Service (FastAPI)  ── port 8000                           │
│   • ML Inference (KNN lookup + LightGBM fallback)               │
│   • SHAP Explainability                                          │
│   • Transaction CRUD (PostgreSQL)                                │
│   • Prometheus /metrics endpoint                                 │
└──────┬───────────────┬────────────────────┬─────────────────────┘
       │               │                    │
       ▼               ▼                    ▼
┌─────────────┐  ┌──────────┐       ┌─────────────┐
│ PostgreSQL  │  │ MLflow   │       │ Prometheus  │
│ port 5432   │  │ port 5001│       │ port 9090   │
│ • Txn store│  │ • Metrics│       │ • Scrapes   │
│ • Airflow  │  │ • Model  │       │   /metrics  │
└─────────────┘  └──────────┘       └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │  Grafana   │
                                    │ port 3002  │
                                    │ Dashboard  │
                                    └─────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Airflow (port 8080) — DAG: preprocess → train → drift check   │
└─────────────────────────────────────────────────────────────────┘
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
       │          ── Logs to MLflow
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
9. Evidently drift detection (detect_drift.py)
       │  Compare reference vs. current data distribution
       ▼
10. Airflow DAG ── triggers retraining nếu drift > threshold
```

---

## Tech Stack

| Layer | Technology | Port | Purpose |
|-------|-----------|------|---------|
| **Database** | PostgreSQL 15 | 5432 | Transaction storage + MLflow backend |
| **API** | FastAPI (Python 3.9+) | 8000 | ML inference + CRUD |
| **Frontend** | Next.js 13 (TypeScript) | 3000 | Real-time dashboard |
| **ML Tracking** | MLflow | 5001 | Experiment tracking, model registry |
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

### Evidently cho drift detection
- Chuyên biệt cho ML monitoring, hỗ trợ Population Stability Index (PSI) và feature-level drift
- Tích hợp sẵn với Grafana qua Prometheus metrics
- HTML report tức thì cho việc phân tích

### 8 services riêng biệt
- **Separation of concerns**: mỗi service có 1 responsibility rõ ràng
- **Independent scaling**: Prometheus/Grafana không cần scale, ML Serving có thể scale horizontally
- **Fault isolation**: 1 service down không ảnh hưởng toàn bộ hệ thống
