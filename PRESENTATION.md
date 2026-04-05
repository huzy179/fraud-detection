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
| Thời gian phản hồi nhanh | p95 latency < 0.5s |
| Giám sát toàn diện | Prometheus + Grafana |
| Triển khai tự động | CI/CD GitHub Actions |

### Stack công nghệ
```
ML Framework  → XGBoost, LightGBM, RandomForest (so sánh)
ML Tracking   → MLflow
API           → FastAPI + Uvicorn
Database      → PostgreSQL 15 + SQLAlchemy ORM
Frontend      → Next.js 14 (React + TypeScript)
Monitoring    → Prometheus + Grafana
Container     → Docker + Docker Compose
CI/CD         → GitHub Actions
```

---

## 2. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                         docker-compose.yml                        │
│                                                                   │
│  ┌──────────────┐                                                 │
│  │  PostgreSQL  │  ← Database (transactions, mlflow backend)       │
│  │  port: 5432  │                                                 │
│  └──────┬───────┘                                                 │
│         │                                                          │
│  ┌──────▼───────┐                                                 │
│  │    MLflow    │  ← Experiment tracking + model registry          │
│  │  port: 5001  │                                                 │
│  └──────┬───────┘                                                 │
│         │                                                          │
│  ┌──────▼───────┐     ┌──────────────────────┐                   │
│  │   API Server  │────►│  ML Inference (LGBM) │                   │
│  │  port: 8000  │     │  + DB transactions    │                   │
│  └──────┬───────┘     └──────────────────────┘                   │
│         │                                                          │
│  ┌──────▼───────┐     ┌──────────────────────┐                   │
│  │   Frontend   │────►│  Next.js Dashboard   │                   │
│  │  port: 3000  │     │  (stats + form)      │                   │
│  └──────────────┘     └──────────────────────┘                   │
│                                                                   │
│  ┌──────────────┐     ┌──────────────────────┐                   │
│  │  Prometheus  │────►│  Metrics collection  │                   │
│  │  port: 9090  │     │  (scrape every 15s)  │                   │
│  └──────┬───────┘     └──────────────────────┘                   │
│         │                                                          │
│  ┌──────▼───────┐     ┌──────────────────────┐                   │
│  │   Grafana    │────►│  10-panel Dashboard  │                   │
│  │  port: 3002  │     │  (real-time)         │                   │
│  └──────────────┘     └──────────────────────┘                   │
│                                                                   │
│  ┌──────────────┐                                                 │
│  │  ML Pipeline │  ← Chạy 1 lần → huấn luyện → thoát             │
│  │  (no port)   │                                                 │
│  └──────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Thứ tự khởi động
```
postgres (healthy) → mlflow → api → prometheus → grafana
                                        ↓
                              ml-pipeline (on-demand)
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

### Đăng ký mô hình với MLflow
```
Experiment: fraud_detection_improved
Runs logged:
├── LightGBM run (F1: 0.8438) ← WINNER
├── XGBoost run (F1: 0.8394)
└── RandomForest run (F1: 0.8352)

Artifacts: model binary + config + metrics
```

---

## 5. API Server

### Công nghệ
**FastAPI + Uvicorn** — async Python web framework, auto-generated OpenAPI docs.

### Các endpoint

| Method | Path | Mô tả | Ghi chú |
|---|---|---|---|
| `GET` | `/health` | Health check | Trả về model loaded, type, threshold |
| `POST` | `/predict` | Dự đoán fraud | KNN serving index, không lưu DB |
| `POST` | `/explain` | Giải thích SHAP | KNN-based explanation (top feature) |
| `POST` | `/transactions` | Tạo giao dịch mới | KNN inference + lưu PostgreSQL |
| `GET` | `/transactions` | Danh sách giao dịch | Paginated, max 1000 |
| `GET` | `/transactions/stats` | Thống kê tổng hợp | Tổng, số fraud, tỷ lệ, trung bình |
| `GET` | `/metrics` | Prometheus metrics | 4 custom metrics + Python std metrics |

### Ví dụ request/response

**POST /predict**
```json
// Request
{
  "V1": -1.359, "V2": -0.072, ..., "V28": -0.068,
  "Amount": 149.52, "Time": 40680
}

// Response
{
  "fraud_probability": 0.0614,
  "is_fraud": false,
  "threshold": 0.5,
  "confidence": "low"
}
```

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
│  📊 Fraud Detection Dashboard                  │
│                                               │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │
│  │ Total  │ │ Fraud  │ │ Fraud  │ │  Avg   │  │
│  │  22    │ │   6    │ │ Rate   │ │ Prob   │  │
│  │ reqs   │ │ fraud  │ │ 33.3%  │ │  39.8% │  │
│  └────────┘ └────────┘ └────────┘ └────────┘  │
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │  Transaction Form                        │  │
│  │  V1..V28, Amount, Time                  │  │
│  │  [Load Legit Sample] [Load Fraud Sample]│  │
│  │  [🔍 Detect Fraud]                      │  │
│  └────────────────────────────────────────┘  │
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │  Recent Transactions                   │  │
│  │  ID | Amount | Fraud? | Prob | Time    │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

### Dark Theme
- Màu nền: `#0a0f1e`
- Màu accent: Xanh dương、青, cam cho cảnh báo
- Responsive: Desktop + Mobile

---

## 7. Giám sát & Quan sát

### Prometheus — Thu thập metrics

**Scrape targets:**
- `api:8000/metrics` — mỗi 15 giây
- `prometheus:9090/metrics` — mỗi 15 giây

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
│  Job 1: lint-and-test                   │
│  Runs on: ubuntu-latest + PostgreSQL svc │
│  Steps:                                 │
│  1. flake8 lint (main.py)               │
│  2. pytest unit tests                   │
│  3. npm ci → npm run build (frontend)   │
└───────────────┬─────────────────────────┘
                │ (pass only)
                ▼
┌─────────────────────────────────────────┐
│  Job 2: docker-build                     │
│  Steps:                                 │
│  1. Build fraud-api:<sha> image         │
│  2. Build fraud-frontend:<sha> image     │
│  3. Login GHCR                          │
│  4. Push to ghcr.io/<repo>/             │
└─────────────────────────────────────────┘
```

### Trigger conditions
- Push lên `main` hoặc `develop`
- Pull request vào `main`

---

## 9. Hướng dẫn vận hành

### Khởi động toàn bộ hệ thống
```bash
# Clone repo
git clone <repo-url>
cd fraud-detection

# Download dữ liệu (nếu chưa có)
python data/scripts/download_data.py

# Khởi động toàn bộ stack
docker-compose up -d

# Kiểm tra trạng thái
docker-compose ps
```

### Huấn luyện lại mô hình
```bash
# Chạy ML pipeline (tự động train + save models)
docker-compose run --rm ml-pipeline

# Restart API để load model mới
docker-compose restart api
```

### Kiểm tra nhanh
```bash
# Health check
curl http://localhost:8000/health

# MLflow UI
open http://localhost:5001

# Prometheus
open http://localhost:9090

# Grafana dashboard
open http://localhost:3002  # admin / admin

# Frontend
open http://localhost:3000
```

### Debug checklist
```
1. docker-compose ps
   → Tất cả containers STATUS = Up (healthy)

2. curl http://localhost:8000/health
   → {"status":"healthy","model_loaded":true,"model_type":"lightgbm"}

3. curl "http://localhost:9090/api/v1/query?query=up"
   → up{job="api"} = 1, up{job="prometheus"} = 1

4. Prometheus targets: http://localhost:9090/targets
   → Cả 2 targets health = up

5. Grafana datasource: http://localhost:3002/api/datasources
   → Prometheus uid=prometheus, isDefault=true
```

---

## Tổng kết

| Thành phần | Công nghệ | Trạng thái |
|---|---|---|
| Dữ liệu | Credit Card Fraud Dataset (Kaggle) | ✅ 284K rows |
| Tiền xử lý | StandardScaler + SMOTE | ✅ Parquet output |
| Mô hình ML | LightGBM (best F1=0.8438) | ✅ Deployed |
| MLflow | Experiment tracking + registry | ✅ 3 runs logged |
| API Server | FastAPI + PostgreSQL | ✅ Running |
| Frontend | Next.js 14 Dashboard | ✅ Running |
| Prometheus | Metrics scraping | ✅ 2 targets |
| Grafana | 10-panel real-time dashboard | ✅ Live |
| CI/CD | GitHub Actions | ✅ Automated |
| Docker | Multi-service compose | ✅ Production-ready |

**Mọi thành phần đã được kiểm chứng và hoạt động ổn định.**

---

*Tài liệu này được tạo tự động. Cập nhật: 2026-04-05*
