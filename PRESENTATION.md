# Credit Card Fraud Detection — End-to-End ML Ops System
### Tài liệu thuyết trình dự án

---

## Mục lục
1. [Tổng quan](#1-tổng-quan)
2. [Giao diện người dùng](#2-giao-diện-người-dùng)
3. [Dữ liệu & Tiền xử lý](#3-dữ-liệu--tiền-xử-lý)
4. [Mô hình ML & Kết quả](#4-mô-hình-ml--kết-quả)
5. [API Server](#5-api-server)
6. [Giám sát & Quan sát](#6-giám-sát--quan-sát)
7. [CI/CD Pipeline](#7-cicd-pipeline)
8. [Hướng dẫn vận hành](#8-hướng-dẫn-vận-hành)

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
ML Framework    → XGBoost, LightGBM, RandomForest
ML Tracking      → MLflow (PostgreSQL backend, S3 artifact root)
Object Storage   → MinIO (S3-compatible, 2 buckets: mlflow-artifacts, drift-reports)
Drift Detection  → Evidently AI microservice (port 8002)
API              → FastAPI + Uvicorn + SQLAlchemy
Database         → PostgreSQL 15
Frontend         → Next.js 14 (TypeScript)
Monitoring       → Prometheus + Grafana
Orchestration    → Docker Compose + Apache Airflow
CI/CD            → GitHub Actions
```

---

## 2. Giao diện người dùng

### Công nghệ
**Next.js 14** — App Router, TypeScript, dark theme, responsive. Truy cập tại `http://localhost:3000`.

### Source chính
- **Component:** `services/frontend/pages/index.tsx`
- **Styles:** `services/frontend/styles/globals.css`
- **Dockerfile:** `services/frontend/Dockerfile`

### Giao diện chính

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
│  │  V1..V28, Amount, Time                │  │
│  │  [Load Legit Sample] [Load Fraud Sample]│  │
│  │  [🔍 Detect Fraud]                      │  │
│  └────────────────────────────────────────┘  │
│                                               │
│  ┌────────────────────────────────────────┐  │
│  │  Recent Transactions (paginated)        │  │
│  │  ID | Amount | Fraud? | Prob | Time   │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

### API Endpoints được sử dụng

| Endpoint | Method | Purpose |
|---|---|---|
| `/transactions?limit=20` | GET | Danh sách giao dịch gần nhất |
| `/transactions/stats` | GET | KPI: total, fraud count, fraud rate, avg prob |
| `/transactions` | POST | Tạo giao dịch mới + chạy dự đoán fraud |

### Sample data được pre-extract

```typescript
// 2 samples đại diện được extract từ Kaggle test set
const SAMPLE_LEGIT = {
  V1: -0.67, V2: 1.41, ..., V28: 0.29,
  Amount: 23.00, Time: 160760.00
};
// Fraud sample: Amount = 0.01 (fraudsters dùng số tiền nhỏ để tránh detection)
const SAMPLE_FRAUD = {
  V1: -1.27, ..., V28: -0.08,
  Amount: 0.01, Time: 57007.00
};
```

### Các hàm chính trong `index.tsx`

| Hàm | Mô tả |
|---|---|
| `fetchData()` | Gọi song song `GET /transactions` + `GET /transactions/stats` bằng `Promise.all`. Fail graceful — nếu stats fail thì transactions vẫn hiển thị. |
| `handleSubmit()` | POST dữ liệu form → `POST /transactions` → cập nhật result + gọi `fetchData()` để refresh bảng |
| `loadSample(type)` | Gán `SAMPLE_LEGIT` hoặc `SAMPLE_FRAUD` vào form state |

---

## 3. Dữ liệu & Tiền xử lý

### Nguồn dữ liệu
- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Dung lượng:** 98 MB, 284,807 giao dịch
- **Tỷ lệ fraud/legit:** ~0.17% (highly imbalanced)
- **30 features:** V1–V28 (PCA), Time, Amount

### Pipeline tiền xử lý (`services/ml-pipeline/scripts/preprocess.py`)

```
creditcard.csv (raw)
        │
        ▼ load_data()
  Đọc CSV → 284,807 rows × 31 cols
        │
        ▼ clean_data()
  Drop NaN + lọc Amount ≥ 0
        │
        ▼ scale_features()
  StandardScaler(Time)  → time_scaler.joblib
  StandardScaler(Amount) → amount_scaler.joblib
        │
        ▼ split_data()
  Stratified 80/20 train/test (stratify=y)
        │
        ▼ handle_imbalance() — SMOTE
  sampling_strategy=0.5 → 50% fraud ratio
        │
        ▼ save_processed_data()
  X_train.parquet | X_test.parquet | y_train.parquet | y_test.parquet
```

### Chi tiết từng hàm trong `preprocess.py`

| Hàm | Mô tả |
|---|---|
| `load_data()` | `pd.read_csv(raw/creditcard.csv)` — đọc toàn bộ dataset, log số rows/columns |
| `clean_data()` | `dropna()` nếu có NaN; lọc `df = df[df["Amount"] >= 0]` để loại outliers |
| `scale_features()` | Tạo 2 `StandardScaler` riêng cho Time và Amount. `fit_transform()` trên toàn bộ data (trước split) rồi `joblib.dump()` ra file. Lý do: scaler cần cả train+test distribution để inference đúng. |
| `split_data()` | `StratifiedKFold`-style split: giữ nguyên tỷ lệ fraud trong cả train và test. `y_train.sum()` và `y_test.sum()` được log. |
| `handle_imbalance(strategy="smote")` | `SMOTE(sampling_strategy=0.5)` — tăng minority class (fraud) lên 50% của majority. **Chỉ áp dụng train set** để tránh data leakage. |
| `save_processed_data()` | Ghi 4 file parquet (Parquet ≈ 10× nhỏ hơn CSV, đọc nhanh hơn). `to_parquet()` giữ nguyên dtype. |

---

## 4. Mô hình ML & Kết quả

### Pipeline huấn luyện (`services/ml-pipeline/scripts/train.py`)

```
1. load_data()       → X_train, X_test, y_train, y_test (parquet)
2. wait_for_mlflow() → retry polling MLflow server 60s (graceful fallback)
3. _configure_mlflow_s3() → set AWS_* env vars → MinIO as artifact root
4. train_with_cv() cho 3 models:
   ├── LightGBM  (scale_pos_weight = class_ratio)
   ├── XGBoost   (scale_pos_weight = class_ratio)
   └── RandomForest (class_weight="balanced")
   Mỗi model:
   ├── 5-fold Stratified CV → mean ± std metrics
   ├── Final model trên full X_train
   ├── Threshold scan (0.05→0.95, step 0.01) → maximize F1
   ├── Log to MLflow (nếu available)
   └── Save: lgbm_model.txt | xgboost_model.json | rf_model.joblib
5. Chọn model F1 cao nhất → lưu best_config.json
```

### Chi tiết từng hàm trong `train.py`

| Hàm | Mô tả |
|---|---|
| `wait_for_mlflow(uri, timeout=60, interval=5)` | Retry loop: gọi `urllib.request.urlopen(uri + "/health")` mỗi 5s trong 60s. Nếu timeout → fallback sang local-only training. Không crash nếu MLflow không có. |
| `_configure_mlflow_s3()` | Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT_URL` → MLflow biết ghi artifacts vào MinIO thay vì local disk |
| `find_optimal_threshold(y_true, y_proba)` | Scan 90 giá trị từ 0.05 đến 0.94 (step 0.01). Với mỗi threshold, tính F1. Trả về threshold cho F1 cao nhất. Đây là **tuning step quan trọng** vì default 0.5 không tối ưu cho imbalanced data. |
| `evaluate(y_true, y_pred, y_proba, name)` | Tính 5 metrics: precision, recall, F1, ROC-AUC, average_precision. Log bảng classification_report + confusion_matrix. |
| `train_with_cv(model_cfg, model_name, ...)` | 1) 5-fold CV → thu metrics per fold; 2) fit final model trên **toàn bộ** X_train (không hold-out); 3) scan threshold trên X_test; 4) save model. XGBoost/LightGBM dùng `eval_set=[(X_test, y_test)]` cho early stopping. |
| `_train_with_mlflow(model_cfg, model_name, ...)` | Wrapper quanh `train_with_cv()`. Bên trong `mlflow.start_run(run_name=model_name)` → log params, metrics, model artifact (`.log_model()`). Nếu MLflow fail → trả về `train_with_cv()` thường. |
| `main()` | Chạy 3 models → `max(all_models, key=lambda x: x[1]["f1"])` → best model → `json.dump(best_config.json)`. |

### So sánh 3 mô hình

| Mô hình | Precision | Recall | **F1 Score** | ROC-AUC | Threshold | Kích thước |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** ⭐ | 0.8617 | 0.8265 | **0.8438** | 0.9751 | 0.93 | ~1.0 MB |
| XGBoost | 0.8526 | 0.8265 | 0.8394 | 0.9792 | 0.94 | ~1.5 MB |
| RandomForest | 0.9048 | 0.7755 | 0.8352 | 0.9844 | 0.89 | ~6.8 MB |

> **Chọn LightGBM** — F1 cao nhất, kích thước nhỏ nhất, inference nhanh nhất.

### Threshold tối ưu
Threshold = 0.93 nghĩa là: giao dịch chỉ bị gắn cờ **fraud** khi xác suất dự đoán ≥ 93%. Threshold cao như vậy là do dữ liệu imbalanced nặng — cần đủ "chắc chắn" mới gọi là fraud.

---

## 5. API Server

### Công nghệ
**FastAPI + Uvicorn + SQLAlchemy ORM** — async Python web framework, auto-generated OpenAPI docs tại `/docs`. Kết nối MinIO qua `boto3` và Evidently qua HTTP.

### Các endpoint

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/health` | Health check + model loaded, type, threshold |
| `POST` | `/predict` | Dự đoán fraud (KNN serving index) |
| `POST` | `/explain` | Giải thích SHAP-based (top feature) |
| `POST` | `/transactions` | Tạo giao dịch mới → KNN inference → lưu PostgreSQL |
| `GET` | `/transactions` | Danh sách giao dịch (paginated, max 1000) |
| `GET` | `/transactions/stats` | Thống kê tổng hợp: tổng, số fraud, tỷ lệ |
| `GET` | `/drift-status` | JSON drift status (từ MinIO hoặc local) |
| `GET` | `/drift-report` | HTML drift report (từ MinIO hoặc local) |
| `POST` | `/run-drift` | Trigger Evidently microservice |
| `GET` | `/metrics` | Prometheus metrics (4 custom + Python std) |

### Chi tiết từng hàm trong `main.py`

#### Khởi tạo (module-level, khi container start)

| Hàm | Mô tả |
|---|---|
| `load_model()` | Thử load LightGBM Booster trước (`lgb.Booster(model_file=...)`). Nếu fail → load XGBoost Classifier. Nếu cả 2 fail → `RuntimeError`. Priority: LightGBM vì inference nhanh hơn. |
| `load_scalers()` | `joblib.load(time_scaler.joblib)` + `joblib.load(amount_scaler.joblib)`. Dùng cho inference: scale Time/Amount của request trước khi KNN lookup. |
| `_build_serving_index()` | Gọi khi container start hoặc first request: đọc `X_test.parquet` (56,962 rows × 30 features) → fit `NearestNeighbors(n_neighbors=1, algorithm="ball_tree")`. Ball_tree tối ưu cho high-dimensional data (30 dims). |
| `_get_s3()` | Lazy-init boto3 S3 client với MinIO endpoint + `signature_version="s3v4"`. Dùng để đọc/ghi drift reports. |

#### Core inference

| Hàm | Mô tả |
|---|---|
| `_knn_predict_from_request(tx)` | 1) Tách Time/Amount khỏi request body; 2) Scale bằng scaler để match feature space của training data; 3) Build query vector với 30 features; 4) `knn.kneighbors(query)` → distance + index; 5) `confidence = max(0, 1 - dist/10)` — distance càng nhỏ → confidence càng cao; 6) Nếu nearest label=1 (fraud) → prob=confidence; label=0 → prob=1-confidence |
| `predict_fraud(features)` | Direct model inference (LightGBM Booster hoặc XGBoost Classifier). LightGBM: `1/(1+exp(-raw_score))` vì Booster trả raw log-odds. |
| `_to_python(val)` | Convert numpy float/int → native Python. **Bắt buộc** vì psycopg2 không accept numpy dtypes → sẽ raise `NotImplementedError`. Priority: float → int → fallback. |

#### Database model

| Hàm | Mô tả |
|---|---|
| `TransactionDB` | SQLAlchemy ORM model cho bảng `transactions`. 30 cột V1–V28 + amount + fraud_probability + is_fraud + confidence + created_at. UUID primary key. |
| `get_db()` | FastAPI dependency — tạo `SessionLocal()`, yield cho request, `finally: db.close()`. FastAPI tự động quản lifecycle. |

#### Endpoints

| Endpoint | Chi tiết |
|---|---|
| `POST /transactions` | 1) KNN predict (lazy init on first call); 2) `db.add(TransactionDB(...))`; 3) `db.commit()`; 4) Prometheus: `PREDICTION_COUNT.labels("fraud"/"legit").inc()` |
| `GET /transactions/stats` | `SELECT COUNT(*)`, `SELECT COUNT(*) WHERE is_fraud=True`, avg fraud_probability. Update `FRAUD_RATE_GAUGE.set()`. |
| `GET /drift-status` | Thử download `drift_alert.json` từ MinIO trước → cache local → return JSON. Fallback: đọc local file. |
| `POST /run-drift` | `httpx.AsyncClient().post(f"{EVIDENTLY_SERVICE_URL}/run")` với 120s timeout. Return alert JSON. |

### KNN Serving Index

```
Incoming request (V1..V28, Time, Amount)
        │
        ▼ time_scaler + amount_scaler
  Time_scaled, Amount_scaled
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

## 6. Giám sát & Quan sát

### Prometheus — Thu thập metrics

**Scrape targets:**
- `api:8000/metrics` — mỗi 15 giây
- `prometheus:9090/metrics` — mỗi 15 giây (self-monitoring)

**4 custom metrics (định nghĩa trong `main.py`):**

```python
REQUEST_COUNT = Counter("fraud_api_requests_total", ["endpoint", "method"])
REQUEST_LATENCY = Histogram("fraud_api_latency_seconds", ["endpoint"])
PREDICTION_COUNT = Counter("fraud_predictions_total", ["prediction"])
FRAUD_RATE_GAUGE = Gauge("fraud_rate_estimated", "Estimated fraud rate")
DRIFT_SCORE_GAUGE = Gauge("fraud_drift_score", "Latest Evidently data drift score")
```

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

### Drift Detection Pipeline

| Script | Hàm chính | Mô tả |
|---|---|---|
| `export_transactions.py` | `fetch_transactions_from_db()` | SELECT tất cả transactions từ PostgreSQL |
| | `build_current_parquet()` | Scale Amount/Time → chuẩn bị 30 features cho Evidently |
| | `save_outputs()` | Ghi `current.parquet` + `current_predictions.csv` |
| `detect_drift.py` | `detect_data_drift()` | Evidently DataDriftPreset → PSI → HTML/JSON → upload MinIO |
| | `should_retrain()` | drift_score ≥ 0.5 → khuyến nghị retrain |
| | `main()` | Ghi `drift_alert.json` |
| `evidently_service.py` | `run_data_drift_report()` | Evidently Report → save HTML/JSON/JSON alert → upload MinIO |

---

## 7. CI/CD Pipeline

### GitHub Actions Workflow

```
Push / PR
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Job 1: lint-and-test                                    │
│                                                          │
│  Services: postgres:5432 (healthy) + minio:9000 (ready) │
│                                                          │
│  Steps:                                                  │
│  1. Setup MinIO buckets (mc alias → mb → anonymous set)  │
│  2. flake8 lint (ml-serving/)                            │
│  3. pytest tests (ml-serving/)                            │
│  4. flake8 lint (ml-pipeline/scripts/)                   │
│  5. Python import test (detect_drift.py)                 │
│  6. flake8 lint (evidently/)                             │
│  7. npm ci → npm run build (frontend)                    │
└───────────────┬──────────────────────────────────────────┘
                │ (pass only on push to main)
                ▼
┌──────────────────────────────────────────────────────────┐
│  Job 2: docker-build                                      │
│                                                          │
│  1. Build + push fraud-api:<sha>, :latest                │
│  2. Build + push fraud-ml-pipeline:<sha>, :latest         │
│  3. Build + push fraud-evidently:<sha>, :latest          │
│  4. Build + push fraud-airflow:<sha>, :latest            │
│  5. Build + push fraud-frontend:<sha>, :latest           │
└──────────────────────────────────────────────────────────┘
```

### Trigger conditions
- Push lên `main` hoặc `develop`
- Pull request vào `main`

---

## 8. Hướng dẫn vận hành

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
open http://localhost:8080     # Airflow (admin/admin)
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
| MLflow | Experiment tracking + PostgreSQL backend | ✅ Running |
| Object Storage | MinIO (S3-compatible, 2 buckets) | ✅ Running |
| Evidently Service | Drift detection API (port 8002) | ✅ Running |
| API Server | FastAPI + KNN serving + PostgreSQL | ✅ Running |
| Frontend | Next.js 14 Dashboard | ✅ Running |
| Prometheus | Metrics scraping (2 targets) | ✅ Running |
| Grafana | 10-panel real-time dashboard | ✅ Running |
| Airflow | Webserver (8080) + Scheduler | ✅ Running |
| CI/CD | GitHub Actions (5 Docker images) | ✅ Automated |
| Docker | Multi-service compose (12 services) | ✅ Production-ready |

---

*Tài liệu thuyết trình — Cập nhật: 2026-04-15*
