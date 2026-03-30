# Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng
## Credit Card Fraud Detection System

---

## 1. Giới Thiệu Dự Án

### 1.1 Bối cảnh

Gian lận thẻ tín dụng là một trong những vấn đề nghiêm trọng nhất trong ngành tài chính — gây thiệt hại hàng tỷ đô la mỗi năm trên toàn cầu. Với sự phát triển mạnh mẽ của thương mại điện tử và thanh toán số, nhu cầu phát hiện gian lận theo thời gian thực ngày càng cấp thiết.

### 1.2 Mục tiêu

- Xây dựng hệ thống phát hiện gian lận **theo thời gian thực** cho các giao dịch thẻ tín dụng
- Triển khai kiến trúc **microservices** đầy đủ, bao phủ toàn bộ vòng đời ML (ML lifecycle)
- Cung cấp **dashboard trực quan** để theo dõi và kiểm tra giao dịch
- Tích hợp **giám sát & cảnh báo** tự động

---

## 2. Kiến Trúc Hệ Thống

### 2.1 Tổng quan

Kiến trúc gồm **3 services chính** (API Server, Frontend, MLflow) + 2 services giám sát.

```
┌──────────────────────────────────────────────────────────────────┐
│                       CLIENT / BROWSER                            │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTP REST
                   ┌──────────▼──────────┐
                   │   Frontend          │
                   │   Next.js  :3000    │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   API Server        │
                   │   FastAPI  :8000    │
                   │  ┌──────────────┐  │
                   │  │ ML Inference │  │  ← XGBoost model
                   │  │ DB (SQLAlchemy)│ │  ← PostgreSQL
                   │  └──────────────┘  │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   MLflow             │
                   │   :5000              │
                   │   (Model Registry)   │
                   └──────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│   Prometheus :9090  ──▶  Grafana :3002                        │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 Các thành phần chi tiết

| Thành phần      | Công nghệ              | Vai trò                                      |
|----------------|-----------------------|----------------------------------------------|
| **API Server** | FastAPI + SQLAlchemy  | Dự đoán ML + quản lý giao dịch + PostgreSQL |
| **Frontend**   | Next.js               | Dashboard tương tác, hiển thị kết quả      |
| **MLflow**     | MLflow                | Tracking thí nghiệm, quản lý model          |
| **Prometheus** | Prometheus            | Thu thập metrics từ API server               |
| **Grafana**    | Grafana               | Trực quan hóa dashboard giám sát            |

---

## 3. Quy Trình Xử Lý Dữ Liệu & Huấn Luyện Model

### 3.1 Nguồn dữ liệu

- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features:** 284,807 giao dịch, 30 features (V1–V28 ẩn danh, Time, Amount), nhãn Class (0=legit, 1=fraud)
- **Tỷ lệ mất cân bằng:** ~0.17% fraud — một trong những dataset mất cân bằng nhất

### 3.2 Pipeline tiền xử lý (`preprocess.py`)

```
Raw CSV ──▶ Missing Values ──▶ Scaling (Time/Amount) ──▶ Train/Test Split
              Cleaning              StandardScaler         Stratified 80/20
                 │                                            │
                 ▼                                            ▼
           Handle Imbalance ──────────────────────────▶ Save Parquet
           (SMOTE oversampling to 50:50)
```

**Các bước chính:**
1. **Load & Clean:** Kiểm tra giá trị thiếu, loại bỏ outliers (Amount < 0)
2. **Scale:** StandardScaler cho `Time` và `Amount` → lưu scaler để reuse
3. **Split:** Train/Test 80/20, stratified theo Class
4. **Handle Imbalance:** Áp dụng **SMOTE** (Synthetic Minority Over-sampling Technique) để tăng tỷ lệ fraud từ 0.17% → 50%

### 3.3 Huấn luyện Model (`train.py`)

Hai model được huấn luyện và so sánh:

**XGBoost:**
- `max_depth=6`, `learning_rate=0.1`, `n_estimators=200`
- `scale_pos_weight` để cân bằng class
- `eval_metric=aucpr` (Area Under PR Curve)

**Random Forest:**
- `n_estimators=200`, `max_depth=10`
- `class_weight='balanced'`

**Metrics đánh giá:** Precision, Recall, F1, **ROC-AUC**, **Average Precision (PR-AUC)**

> **Average Precision (PR-AUC)** là metric quan trọng nhất vì phản ánh chính xác hiệu suất phân loại trên tập mất cân bằng nghiêm trọng

### 3.4 MLflow Tracking

```
Experiment: fraud_detection
├── Run: xgboost_fraud
│   ├── Params: max_depth, learning_rate, n_estimators, scale_pos_weight
│   └── Metrics: precision, recall, f1, roc_auc, average_precision
└── Run: random_forest_fraud
    ├── Params: n_estimators, max_depth, class_weight
    └── Metrics: precision, recall, f1, roc_auc, average_precision

Registered Model: FraudDetector (versioned)
```

---

## 4. API Server — FastAPI (Inference + DB + REST)

FastAPI đóng vai trò **hợp nhất**: vừa chạy ML inference, vừa quản lý database, vừa expose REST API cho Frontend.

### 4.1 Database Schema (PostgreSQL)

```sql
Table: transactions
├── id                  UUID (PK)
├── amount              FLOAT
├── V1..V28             FLOAT
├── fraud_probability   FLOAT
├── is_fraud            BOOLEAN
├── confidence          VARCHAR (high/medium/low)
└── created_at          TIMESTAMP
```

### 4.2 API Endpoints

| Endpoint              | Method | Mô tả                                      |
|----------------------|--------|---------------------------------------------|
| `/health`             | GET    | Health check, kiểm tra model loaded         |
| `/predict`            | POST   | Dự đoán xác suất gian lận cho 1 giao dịch  |
| `/explain`            | POST   | Giải thích SHAP cho kết quả dự đoán        |
| `/transactions`       | POST   | Tạo giao dịch → dự đoán ML → lưu vào DB    |
| `/transactions`       | GET    | Danh sách giao dịch gần nhất               |
| `/transactions/stats`  | GET    | Thống kê: tổng, fraud count, tỷ lệ        |
| `/metrics`            | GET    | Prometheus metrics                          |

### 4.3 Luồng xử lý giao dịch

```
POST /transactions
        │
        ▼
FastAPI TransactionsService
        │
        ├───▶ preprocess(features) ──▶ XGBoost.predict_proba
        │                                    │
        │     ┌───────────────────────────────┘
        │     ▼
        │  fraud_probability, is_fraud, confidence
        │     │
        ▼     ▼
Lưu vào PostgreSQL ──▶ Response về Frontend
```

### 4.4 SHAP Explainability

- Endpoint `/explain` trả về **SHAP values** cho 30 features
- Top 5 features có ảnh hưởng lớn nhất đến kết quả dự đoán
- Giúp analyst hiểu **tại sao** model đưa ra quyết định

---

## 5. Frontend — Next.js Dashboard

Dashboard tương tác để kiểm tra giao dịch và theo dõi thống kê fraud.

| Chức năng | Mô tả |
|-----------|--------|
| **Thống kê tổng quan** | Tổng giao dịch, số fraud, tỷ lệ fraud, xác suất trung bình |
| **Form test giao dịch** | Nhập 30 features V1–V28 + Amount + Time |
| **Quick load samples** | 2 mẫu Legit / Fraud có sẵn để demo nhanh |
| **Bảng lịch sử** | Danh sách giao dịch gần nhất, hiển thị nhãn fraud/legit |
| **Real-time refresh** | Tự động cập nhật sau mỗi lần submit |

**Truy cập:** `http://localhost:3000`

---

## 6. Giám Sát & Observability

### 6.1 Prometheus Metrics (từ FastAPI)

| Metric                       | Type     | Mô tả                              |
|------------------------------|----------|-------------------------------------|
| `fraud_api_requests_total`   | Counter  | Tổng số request theo endpoint      |
| `fraud_api_latency_seconds`  | Histogram| Latency phân bố của API            |
| `fraud_predictions_total`    | Gauge    | Số dự đoán theo loại (fraud/legit) |
| `fraud_rate_estimated`       | Gauge    | Tỷ lệ fraud ước tính               |

### 6.2 Grafana Dashboard

- **Request Rate:** Số request/giây đến API server
- **Latency Distribution:** P50, P95, P99 latency
- **Fraud Rate Over Time:** Biến động tỷ lệ fraud
- **Model Performance:** AUC, Precision, Recall (từ MLflow)

---

## 7. Triển Khai & CI/CD

### 7.1 CI/CD Pipeline (GitHub Actions)

Pipeline tự động chạy trên **mọi push và pull request**.

```yaml
Trigger: push (main, develop) / pull_request (main)

Jobs:
lint-and-test
  ├── Python (3.10): flake8 lint + pytest
  └── Node.js (18): npm ci + npm run build

docker-build (only on push)
  ├── docker build API Server  → fraud-api:SHA
  ├── docker build Frontend    → fraud-frontend:SHA
  └── docker push → GHCR
```

### 7.2 Docker Compose — Full Stack

```yaml
postgres  (:5432)    ← Lưu transactions + MLflow metadata
    │
    ├── mlflow       (:5000)  ← Model registry & tracking
    │
    ├── api          (:8000)  ← FastAPI (ML inference + DB + REST)
    │       │
    │       └───▶ Prometheus (:9090) ──▶ Grafana (:3002)
    │
    └── frontend     (:3000)  ← Next.js Dashboard
```

### 7.3 Environment Variables

| Variable               | Default                                              | Mô tả                   |
|------------------------|------------------------------------------------------|-------------------------|
| `FRAUD_THRESHOLD`      | `0.5`                                               | Ngưỡng phát hiện fraud |
| `DATABASE_URL`         | `postgresql://postgres:postgres@postgres:5432/...`  | PostgreSQL connection   |
| `MLFLOW_TRACKING_URI`  | `http://localhost:5000`                             | MLflow server URL       |
| `NEXT_PUBLIC_API_URL`  | `http://localhost:8000`                              | FastAPI URL (external)  |

---

## 8. Xử Lý Sự Cố

### 8.1 HTTP 502 Bad Gateway (Grafana/Prometheus)

Lỗi **502 Bad Gateway** xuất hiện khi Grafana không kết nối được đến Prometheus.

| Nguyên nhân | Cách khắc phục |
|------------|---------------|
| Prometheus chưa khởi động xong | Đợi 30–60s, kiểm tra `docker-compose logs prometheus` |
| Prometheus port bị trùng | `lsof -i :9090` để kiểm tra |
| `prometheus.yml` sai cấu hình | Kiểm tra `scrape_configs` trong `monitoring/prometheus.yml` |
| Prometheus URL trong Grafana sai | `http://prometheus:9090` (trong Docker) hoặc `http://localhost:9090` (trình duyệt) |
| Grafana chưa thấy data source | Thêm Data Source thủ công: Settings → Data Sources → Prometheus |

```bash
docker-compose logs prometheus
curl http://localhost:9090/-/healthy
curl http://localhost:8000/metrics
docker-compose restart grafana
```

### 8.2 Các lỗi thường gặp khác

| Lỗi | Nguyên nhân | Giải pháp |
|-----|------------|-----------|
| `Model not found` | Chưa chạy `train.py` | `cd services/ml-pipeline && python scripts/train.py` |
| `POST /predict 503` | ML model chưa load | `docker-compose logs api` |
| PostgreSQL connection failed | Container chưa healthy | `docker-compose ps postgres` |

---

## 9. Cấu Trúc Dự Án

```
fraud-detection/
├── docker-compose.yml              # Full stack orchestration
├── .github/workflows/ci.yml        # CI/CD pipeline
│
├── data/
│   ├── raw/                        # creditcard.csv (284,807 rows)
│   ├── processed/                  # X_train, X_test, y_train, y_test (.parquet)
│   └── scripts/
│       └── download_data.py        # Tải dataset từ Kaggle / direct link
│
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory Data Analysis
│
├── services/
│   ├── ml-pipeline/                # Python ML Pipeline
│   │   ├── scripts/
│   │   │   ├── preprocess.py       # Clean, scale, SMOTE, split
│   │   │   └── train.py            # XGBoost + RandomForest + MLflow
│   │   └── Dockerfile
│   │
│   ├── ml-serving/                 # FastAPI — Unified API Server
│   │   ├── main.py                 # ML inference + DB + REST API
│   │   ├── requirements.txt         # FastAPI + SQLAlchemy + XGBoost
│   │   └── Dockerfile
│   │
│   └── frontend/                  # Next.js Dashboard
│       ├── pages/
│       │   ├── index.tsx           # Main dashboard
│       │   └── _app.tsx
│       └── next.config.js
│
└── monitoring/
    └── prometheus.yml             # Scrape configs cho Prometheus
```

---

## 10. Cách Chạy Dự Án

### 10.1 Chuẩn bị & Huấn luyện

```bash
# 1. Tải dataset
cd services/ml-pipeline
python ../../data/scripts/download_data.py

# 2. Tiền xử lý
python scripts/preprocess.py

# 3. Huấn luyện model
python scripts/train.py
```

### 10.2 Chạy với Docker (khuyên dùng)

```bash
docker-compose up --build
docker-compose up -d    # chạy background
```

### 10.3 Chạy không có Docker

```bash
# API Server (FastAPI)
cd services/ml-serving
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd services/frontend
npm install && npm run dev
```

### 10.4 Truy cập

| Service    | URL                        |
|-----------|----------------------------|
| Frontend  | http://localhost:3000      |
| FastAPI   | http://localhost:8000/docs |
| MLflow    | http://localhost:5000      |
| Grafana   | http://localhost:3002      |
| Prometheus| http://localhost:9090      |

---

## 11. Điểm Nổi Bật

### ✅ Thành công của dự án

| Điểm mạnh                  | Chi tiết                                           |
|---------------------------|----------------------------------------------------|
| **Kiến trúc gọn nhẹ**     | Chỉ 3 services chính — FastAPI hợp nhất mọi thứ  |
| **Xử lý mất cân bằng**    | SMOTE oversampling → cân bằng 0.17% → 50%         |
| **Explainability**         | SHAP values cho mỗi dự đoán                        |
| **Full ML lifecycle**      | Data → Preprocess → Train → Serve → Monitor        |
| **Observability**          | Prometheus + Grafana, health checks                 |
| **Model registry**         | MLflow với versioning và comparison                |
| **CI/CD tự động**          | GitHub Actions end-to-end                          |
| **Real-time inference**    | FastAPI — sub-100ms latency                        |

### ⚠️ Hạn chế & Hướng phát triển

| Hạn chế hiện tại            | Hướng phát triển tương lai                       |
|----------------------------|---------------------------------------------------|
| Chưa có online learning    | Cập nhật model theo thời gian thực               |
| Threshold cố định (0.5)    | Dynamic threshold theo business rules             |
| Chưa có notification       | Slack/Email alert khi fraud rate tăng đột ngột   |
| Chưa có feature store      | Tích hợp Feast feature store                     |
| Chưa có A/B testing        | So sánh nhiều model versions trong production    |

---

## 12. Kết Luận

Hệ thống **Fraud Shield** là một giải pháp toàn diện cho việc phát hiện gian lận thẻ tín dụng, kết hợp sức mạnh của XGBoost với kiến trúc đơn giản — chỉ **3 services chính** nhờ FastAPI hợp nhất. Hệ thống có thể dễ dàng mở rộng, giám sát và triển khai vào production.
