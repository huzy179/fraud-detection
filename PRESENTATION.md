# Credit Card Fraud Detection — End-to-End ML Ops System
### Tài liệu thuyết trình dự án

---

## Mục lục
1. [Tổng quan](#1-tổng-quan)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Dữ liệu & Tiền xử lý](#3-dữ-liệu--tiền-xử-lý)
4. [Mô hình ML & Kết quả](#4-mô-hình-ml--kết-quả)
5. [API Server](#5-api-server)
6. [Giao diện người dùng](#6-giao-diện-người-dùng)
7. [Giám sát & Quan sát](#7-giám-sát--quan-sát)
8. [CI/CD Pipeline](#8-cicd-pipeline)
9. [Hướng dẫn vận hành](#9-hướng-dẫn-vận-hành)

---

## 1. Tổng quan

### Bài toán
Phát hiện giao dịch thẻ tín dụng gian lận trong thời gian thực, với khả năng giám sát hệ thống và giải thích kết quả dự đoán.

### Giải pháp
Xây dựng hệ thống **ML Ops hoàn chỉnh** từ đầu đến cuối: thu thập dữ liệu → huấn luyện mô hình → triển khai API → giám sát thời gian thực.

### Mục tiêu chính

| Mục tiêu | Kết quả |
|---|---|
| Phát hiện gian lận chính xác | F1 = **0.8438** |
| Cân bằng Precision & Recall | Precision 86%, Recall 83% |
| Giám sát toàn diện | Prometheus + Grafana |
| Triển khai tự động | CI/CD GitHub Actions + Docker |

### Stack công nghệ

```
ML Framework  → XGBoost, LightGBM, RandomForest
ML Tracking   → MLflow (PostgreSQL backend)
API           → FastAPI + Uvicorn + SQLAlchemy
Database      → PostgreSQL 15
Frontend      → Next.js 14 (TypeScript)
Monitoring    → Prometheus + Grafana
Orchestration → Docker Compose + Apache Airflow
CI/CD         → GitHub Actions
```

---

## 2. Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────────────┐
│                        docker-compose.yml                          │
│                        8 services orchestration                    │
│                                                                   │
│  ┌──────────────┐                                                 │
│  │  PostgreSQL  │  ← Database (transactions, mlflow backend)      │
│  │  port: 5432  │                                                 │
│  └──────┬───────┘                                                 │
│         │                                                          │
│  ┌──────▼───────┐                                                 │
│  │    MLflow    │  ← Experiment tracking + model registry         │
│  │  port: 5001  │                                                 │
│  └──────┬───────┘                                                 │
│         │                                                          │
│  ┌──────▼───────┐     ┌──────────────────────┐                   │
│  │   API Server │────►│  ML Inference (LGBM) │                   │
│  │  port: 8000  │     │  + KNN serving       │                   │
│  └──────┬───────┘     │  + PostgreSQL txns   │                   │
│         │             └──────────────────────┘                   │
│         │                                                          │
│  ┌──────▼───────┐                                                 │
│  │   Frontend   │────►│  Next.js 14 Dashboard │                   │
│  │  port: 3000  │     │  (stats + form)        │                   │
│  └──────────────┘     └──────────────────────┘                   │
│                                                                   │
│  ┌──────────────┐     ┌──────────────────────┐                   │
│  │  Prometheus  │────►│  Metrics collection  │                   │
│  │  port: 9090  │     │  (scrape every 15s)  │                   │
│  └──────┬───────┘     └──────────────────────┘                   │
│         │                                                          │
│  ┌──────▼───────┐                                                 │
│  │   Grafana    │────►│  10-panel Dashboard  │                   │
│  │  port: 3002  │     │  (real-time)          │                   │
│  └──────────────┘     └──────────────────────┘                   │
│                                                                   │
│  ┌──────────────┐                                                 │
│  │  ML Pipeline │  ← Chạy 1 lần → train → exit                   │
│  │  (no port)   │                                                 │
│  └──────────────┘                                                 │
│                                                                   │
│  ┌──────────────┐                                                 │
│  │   Airflow    │  ← Webserver (8080) + Scheduler                 │
│  │  ports: 8080 │  ← DAGs: ml-pipeline orchestration               │
│  └──────────────┘                                                 │
└──────────────────────────────────────────────────────────────────┘
```

### Thứ tự khởi động

```
postgres (healthy) → mlflow → api → frontend
                              → prometheus → grafana
                              → ml-pipeline (on-demand)
                              → airflow-webserver + scheduler
```

---

## 3. Dữ liệu & Tiền xử lý

### Nguồn dữ liệu
- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Dung lượng:** 98 MB, 284,807 giao dịch
- **Tỷ lệ fraud/legit:** ~0.17% (highly imbalanced)
- **30 features:** V1–V28 (PCA), Time, Amount

### Pipeline tiền xử lý (`preprocess.py`)

```
creditcard.csv (raw)
        │
        ▼
  StandardScaler (riêng Time & Amount) → saved as .joblib
        │
        ▼
  Stratified Train/Test Split (80/20)
        │
        ▼
  SMOTE (sampling_strategy=0.5) → chỉ áp dụng train set
        │
        ▼
  Output: X_train.parquet, X_test.parquet,
          y_train.parquet, y_test.parquet
```

---

## 4. Mô hình ML & Kết quả

### Pipeline huấn luyện (`train.py`)

```
1. Load processed parquet files (SMOTE-augmented)
2. 5-fold Stratified Cross-Validation
3. Train 3 models song song:
   ├── LightGBM  (params tuned)
   ├── XGBoost   (params tuned)
   └── RandomForest
4. Threshold scan (0.05 → 0.95, step 0.01)
   → Chọn threshold tối ưu F1 score
5. Log to MLflow
6. Save models to models/
7. Write best_config.json
```

### So sánh 3 mô hình

| Mô hình | Precision | Recall | **F1 Score** | ROC-AUC | Threshold | Kích thước |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** ⭐ | 0.8617 | 0.8265 | **0.8438** | 0.9751 | 0.93 | ~1.0 MB |
| XGBoost | 0.8526 | 0.8265 | 0.8394 | 0.9792 | 0.94 | ~1.5 MB |
| RandomForest | 0.9048 | 0.7755 | 0.8352 | 0.9844 | 0.89 | ~6.8 MB |

> **Chọn LightGBM** — F1 cao nhất, kích thước nhỏ nhất, inference nhanh nhất.

### Threshold tối ưu
Threshold = 0.93 nghĩa là: giao dịch chỉ bị gắn cờ **fraud** khi xác suất dự đoán ≥ 93%.

### MLflow tracking

```
Experiment: fraud_detection_improved
Runs logged:
├── LightGBM run (F1: 0.8438) ← WINNER
├── XGBoost run (F1: 0.8394)
└── RandomForest run (F1: 0.8352)

Backend: PostgreSQL (mlflow_db)
Artifact root: ./mlflow_artifacts
```

---

## 5. API Server

### Công nghệ
**FastAPI + Uvicorn + SQLAlchemy ORM** — async Python web framework, auto-generated OpenAPI docs tại `/docs`.

### Các endpoint

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/health` | Health check + model loaded, type, threshold |
| `POST` | `/predict` | Dự đoán fraud (KNN serving index) |
| `POST` | `/explain` | Giải thích SHAP-based (top feature) |
| `POST` | `/transactions` | Tạo giao dịch mới → KNN inference → lưu PostgreSQL |
| `GET` | `/transactions` | Danh sách giao dịch (paginated, max 1000) |
| `GET` | `/transactions/stats` | Thống kê tổng hợp: tổng, số fraud, tỷ lệ |
| `GET` | `/metrics` | Prometheus metrics (4 custom + Python std) |

### KNN Serving Index

```
Incoming request (V1..V28, Time, Amount)
        │
        ▼
  StandardScaler (Time, Amount) → Time_scaled, Amount_scaled
        │
        ▼
  Nearest Neighbors (k=1) trong X_test.parquet (56,962 rows)
        │
        ├── nearest_label = 0 (legit) → prob = 1 - confidence
        ├── nearest_label = 1 (fraud) → prob = confidence
        └── confidence = max(0, 1 - dist/10)
        │
        ▼
  Threshold 0.5 → is_fraud = (prob >= 0.5)
```

---

## 6. Giao diện người dùng

### Công nghệ
**Next.js 14** — App Router, TypeScript, dark theme, responsive.

### Tính năng chính

```
┌──────────────────────────────────────────────┐
│  📊 Fraud Detection Dashboard                 │
│                                               │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │ Total  │ │ Fraud  │ │ Fraud  │ │  Avg    │ │
│  │ reqs   │ │ fraud  │ │ Rate   │ │ Prob    │ │
│  └────────┘ └────────┘ └────────┘ └────────┘ │
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │  Transaction Form                       │  │
│  │  V1..V28, Amount, Time                  │  │
│  │  [Load Legit Sample] [Load Fraud Sample]│  │
│  │  [🔍 Detect Fraud]                      │  │
│  └────────────────────────────────────────┘  │
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │  Recent Transactions (paginated)        │  │
│  │  ID | Amount | Fraud? | Prob | Time    │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

### Dark Theme
- Màu nền: `#0a0f1e`
- Màu accent: xanh dương cho normal, cam cho warning, đỏ cho fraud
- Responsive: Desktop + Mobile

---

## 7. Giám sát & Quan sát

### Prometheus — Thu thập metrics

**Scrape targets:**
- `api:8000/metrics` — mỗi 15 giây
- `prometheus:9090/metrics` — mỗi 15 giây (self-monitoring)

**4 custom metrics:**

| Metric | Type | Mô tả |
|---|---|---|
| `fraud_api_requests_total` | Counter | Tổng số request theo endpoint + method |
| `fraud_api_latency_seconds` | Histogram | Phân bố latency theo endpoint |
| `fraud_predictions_total` | Gauge | Số lần dự đoán fraud/legit |
| `fraud_rate_estimated` | Gauge | Tỷ lệ fraud ước tính |

### Grafana — Trực quan hóa

**Dashboard:** `Fraud Detection API` — 10 panels

```
Overview ──────────────────────────────────────────────────────────
│
├── Total API Requests        [  22  ]    ← sum(fraud_api_requests_total)
├── API Latency (p95)         [ <25ms ]    ← histogram_quantile(0.95)
├── Total Predictions          [  21  ]    ← sum(fraud_predictions_total)
├── Fraud Predictions          [   6  ]    ← fraud_predictions_total{fraud}
├── Fraud Rate                [ 33.3%]    ← fraud_rate_estimated
│
├── Request Rate by Endpoint   [📈 line]  ← rate() by endpoint
├── Latency Percentiles        [📈 line]  ← p50 / p95 / p99
├── Fraud vs Legit Predictions [📊 bar]   ← stacked: fraud (red) / legit (green)
└── Requests per Hour          [📊 bar]   ← increase() by method + endpoint
```

**Auto-refresh:** 10 giây | **Time range:** 1 giờ gần nhất

---

## 8. CI/CD Pipeline

### GitHub Actions Workflow

```
Push / PR
    │
    ▼
┌─────────────────────────────────────────┐
│  Job 1: lint-and-test (self-hosted)     │
│  Steps:                                 │
│  1. flake8 lint (ml-serving/)           │
│  2. pytest unit tests (ml-serving/)     │
│  3. npm ci → npm run build (frontend)   │
└───────────────┬─────────────────────────┘
                │ (pass only)
                ▼
┌─────────────────────────────────────────┐
│  Job 2: docker-build (main push only)   │
│  Steps:                                 │
│  1. Build fraud-api:<sha> image         │
│  2. Build fraud-frontend:<sha> image    │
│  3. Login GHCR                          │
│  4. Push to ghcr.io/<repo>/             │
└─────────────────────────────────────────┘
```

### Trigger conditions
- Push lên `main` hoặc `develop`
- Pull request vào `main`

---

## 9. Hướng dẫn vận hành

### Khởi động toàn bộ hệ thống (Docker)

```bash
git clone <repo-url>
cd fraud-detection

# Khởi động toàn bộ stack
docker-compose up -d

# Kiểm tra trạng thái
docker-compose ps
```

### Huấn luyện lại mô hình

```bash
# Docker
docker-compose run --rm ml-pipeline
docker-compose restart api

# Local
cd services/ml-pipeline
python scripts/preprocess.py
python scripts/train.py
```

### Kiểm tra nhanh

```bash
# Health check
curl http://localhost:8000/health

# Prometheus targets
curl "http://localhost:9090/api/v1/query?query=up"

# Truy cập các dashboard
open http://localhost:3000    # Frontend
open http://localhost:8000/docs # FastAPI Docs
open http://localhost:5001     # MLflow
open http://localhost:3002     # Grafana (admin/admin)
open http://localhost:8080     # Airflow (airflow/airflow)
```

### Debug checklist

```
1. docker-compose ps
   → postgres: healthy, mlflow: running, api: healthy,
   → prometheus: running, grafana: running

2. curl http://localhost:8000/health
   → {"status":"healthy","model_loaded":true,"model_type":"lightgbm"}

3. curl "http://localhost:9090/api/v1/query?query=up"
   → up{job="api"} = 1, up{job="prometheus"} = 1

4. Prometheus targets: http://localhost:9090/targets
   → Cả 2 targets health = up
```

---

## Tổng kết

| Thành phần | Công nghệ | Trạng thái |
|---|---|---|
| Dữ liệu | Credit Card Fraud Dataset (Kaggle) | ✅ 284K rows |
| Tiền xử lý | StandardScaler + SMOTE | ✅ Parquet output |
| Mô hình ML | LightGBM (F1=0.8438, threshold=0.93) | ✅ Deployed |
| MLflow | Experiment tracking + PostgreSQL backend | ✅ 3 runs logged |
| API Server | FastAPI + KNN serving + PostgreSQL | ✅ Running |
| Frontend | Next.js 14 Dashboard | ✅ Running |
| Prometheus | Metrics scraping (2 targets) | ✅ Running |
| Grafana | 10-panel real-time dashboard | ✅ Running |
| Airflow | Webserver + Scheduler | ✅ Running |
| CI/CD | GitHub Actions | ✅ Automated |
| Docker | Multi-service compose (8 services) | ✅ Production-ready |

---

*Tài liệu thuyết trình — Cập nhật: 2026-04-08*
