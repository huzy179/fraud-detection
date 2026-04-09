# Credit Card Fraud Detection System

Hệ thống phát hiện giao dịch thẻ tín dụng gian lận trong thời gian thực, xây dựng theo mô hình ML Ops end-to-end.

> Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 giao dịch, ~0.17% fraud rate

## Kiến trúc

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client / Demo                              │
│                  http://localhost:3000 (Next.js)                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP POST /predict
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  API Service (FastAPI)  ── port 8000                             │
│  • ML Inference (KNN lookup + LightGBM fallback)                 │
│  • SHAP Explainability                                          │
│  • Transaction CRUD (PostgreSQL)                                 │
│  • Prometheus /metrics endpoint                                  │
└──────┬───────────────┬──────────────────────┬───────────────────┘
       │               │                      │
       ▼               ▼                      ▼
┌─────────────┐  ┌──────────┐       ┌─────────────┐
│ PostgreSQL  │  │ MLflow   │       │ Prometheus  │
│ port 5432   │  │ port 5001│       │ port 9090   │
│ • Txn store│  │ • Metrics│       └──────┬──────┘
│ • Airflow  │  │ • Model  │              │
└─────────────┘  └──────────┘              ▼
                                   ┌─────────────┐
                                   │  Grafana   │
                                   │ port 3002  │
                                   └─────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Airflow (port 8080) — DAG: preprocess → train → drift check    │
└──────────────────────────────────────────────────────────────────┘
```

**8 services:** PostgreSQL · MLflow · FastAPI · Next.js · Prometheus · Grafana · Airflow (webserver + scheduler) · ML Pipeline (one-shot)

---

## Stack công nghệ

| Thành phần     | Công nghệ                                | Port  |
|---------------|------------------------------------------|-------|
| ML Model      | LightGBM / XGBoost / RandomForest         | —     |
| ML Tracking   | MLflow + PostgreSQL backend              | 5001  |
| API Server    | FastAPI + Uvicorn + SQLAlchemy           | 8000  |
| Database      | PostgreSQL 15                             | 5432  |
| Frontend      | Next.js 14 (TypeScript)                  | 3000  |
| Metrics       | Prometheus v2.47                         | 9090  |
| Visualization | Grafana 10.1                             | 3002  |
| Orchestration | Apache Airflow (LocalExecutor)          | 8080  |
| CI/CD         | GitHub Actions                           | —     |
| Container     | Docker Compose                           | —     |

---

## Hướng dẫn chạy

### Cách 1: Docker Compose (Khuyến nghị)

```bash
# 1. Clone & vào thư mục
git clone <repo-url>
cd fraud-detection

# 2. Khởi động toàn bộ hệ thống
docker compose up -d

# 3. Kiểm tra trạng thái
docker compose ps
```

> Lần đầu chạy sẽ tự động khởi tạo PostgreSQL, MLflow, API, Frontend, Prometheus, Grafana và Airflow.

**Các service có sẵn:**

| Service      | URL                          | Tài khoản         |
|-------------|------------------------------|-------------------|
| Frontend    | http://localhost:3000        | —                 |
| FastAPI     | http://localhost:8000/docs   | —                 |
| MLflow UI   | http://localhost:5001        | —                 |
| Prometheus  | http://localhost:9090         | —                 |
| Grafana     | http://localhost:3002        | admin / admin     |
| Airflow     | http://localhost:8080        | airflow / airflow |

### Cách 2: Chạy từng service (Development)

**Yêu cầu:** Python 3.9+, Node.js 18+, PostgreSQL

```bash
# ── 1. Tiền xử lý dữ liệu ───────────────────────────────────
cd services/ml-pipeline
pip install -r requirements.txt
python scripts/preprocess.py

# ── 2. Huấn luyện mô hình ───────────────────────────────────
python scripts/train.py

# ── 3. API Server (terminal 1) ────────────────────────────────
cd ../ml-serving
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# ── 4. Frontend (terminal 2) ─────────────────────────────────
cd services/frontend
npm install
npm run dev
```

### Huấn luyện lại mô hình (Docker)

```bash
# Chạy pipeline train (tự động log MLflow + save model)
docker compose run --rm ml-pipeline

# Restart API để load model mới
docker compose restart api
```

---

## API Endpoints

| Method | Path                       | Mô tả                            |
|--------|----------------------------|----------------------------------|
| `GET`  | `/health`                  | Health check + model info         |
| `POST` | `/predict`                 | Dự đoán fraud (KNN serving)      |
| `POST` | `/explain`                 | Giải thích SHAP-based            |
| `POST` | `/transactions`            | Tạo giao dịch + dự đoán         |
| `GET`  | `/transactions`            | Danh sách giao dịch (paginated)  |
| `GET`  | `/transactions/stats`      | Thống kê fraud                  |
| `GET`  | `/drift-status`            | JSON drift status (Evidently)    |
| `GET`  | `/drift-report`            | HTML drift report (Evidently)     |
| `GET`  | `/metrics`                 | Prometheus metrics                |

### Ví dụ predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "V1": -0.67, "V2": 1.41, "V3": -1.11, "V4": -1.33, "V5": 1.39,
      "V6": -1.31, "V7": 1.89, "V8": -0.61, "V9": 0.31, "V10": 0.65,
      "V11": -0.86, "V12": -0.23, "V13": -0.20, "V14": 0.27, "V15": -0.05,
      "V16": -0.74, "V17": -0.61, "V18": -0.39, "V19": -0.16, "V20": 0.39,
      "V21": 0.08, "V22": 0.81, "V23": -0.22, "V24": 0.71, "V25": -0.14,
      "V26": 0.05, "V27": 0.53, "V28": 0.29,
      "Amount": 23.00, "Time": 160760.00
    }
  }'
```

---

## Kết quả mô hình

Sau khi chạy `train.py`, 3 models được so sánh qua 5-fold Stratified CV. Kết quả mẫu (sẽ khác nhau tùy random seed và dataset):

| Model           | Precision | Recall | F1 Score | ROC-AUC | Threshold |
|-----------------|:---------:|:------:|:--------:|:-------:|:---------:|
| **LightGBM** ⭐ | ~0.84     | ~0.84  | **~0.84** | ~0.98  | ~0.53     |
| XGBoost         | ~0.85     | ~0.84  | ~0.84    | ~0.98  | ~0.53     |
| RandomForest    | ~0.82     | ~0.81  | ~0.82    | ~0.98  | ~0.60     |

> Model tốt nhất (F1 cao nhất) được lưu tại `models/lgbm_model.txt`
> Metrics: F1 = 2×(Precision×Recall)/(Precision+Recall) — harmonic mean cân bằng precision/recall

---

## Biến môi trường

| Variable              | Default                                     | Mô tả                  |
|----------------------|--------------------------------------------|------------------------|
| `FRAUD_THRESHOLD`     | `0.5` (code), `0.93` (Docker)             | Ngưỡng phát hiện fraud |
| `MODEL_PATH`          | `/app/models` (Docker)                     | Đường dẫn model files  |
| `DATABASE_URL`        | `postgresql://postgres:postgres@postgres:5432/` | PostgreSQL connection |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000`                       | MLflow server URL      |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000`                   | FastAPI URL (frontend) |

---

## Debug nhanh

```bash
# 1. Kiểm tra containers
docker compose ps

# 2. Health check API
curl http://localhost:8000/health

# 3. Kiểm tra Prometheus targets
curl "http://localhost:9090/api/v1/query?query=up"

# 4. Xem logs
docker compose logs -f api
docker compose logs -f mlflow
```

---

## Cấu trúc project

```
fraud-detection/
├── docker-compose.yml           # Orchestration (8 services)
├── Dockerfile.airflow           # Airflow custom image
├── .github/workflows/ci.yml    # CI/CD pipeline
│
├── data/
│   ├── raw/creditcard.csv     # Raw dataset (98MB)
│   └── processed/             # Parquet train/test + scalers
│
├── models/
│   ├── lgbm_model.txt        # Active model (LightGBM)
│   ├── xgboost_model.json     # XGBoost
│   ├── rf_model.joblib        # RandomForest
│   └── best_config.json       # Best model config (sau khi train)
│
├── services/
│   ├── ml-pipeline/
│   │   ├── scripts/
│   │   │   ├── preprocess.py       # StandardScaler, SMOTE, stratified split
│   │   │   ├── train.py            # 5-fold CV, 3 models, threshold tuning
│   │   │   ├── detect_drift.py     # Evidently drift detection
│   │   │   └── export_transactions.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── ml-serving/
│   │   ├── main.py            # FastAPI: inference + CRUD + metrics
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── frontend/
│       ├── pages/index.tsx    # Next.js dashboard
│       ├── styles/globals.css
│       ├── package.json
│       └── Dockerfile
│
├── airflow/
│   ├── dags/fraud_pipeline_dag.py  # DAG: preprocess → train → drift
│   ├── logs/
│   └── config/
│
├── mlflow_artifacts/           # MLflow file artifact store
│
├── monitoring/
│   ├── prometheus.yml          # Prometheus scrape config
│   ├── reports/                # Evidently HTML drift reports
│   └── grafana/provisioning/   # Auto-provisioned dashboards
│
└── postgres-init/              # PostgreSQL init scripts
```

---

## Thuyết trình

Tài liệu thuyết trình chi tiết cho từng service nằm trong folder `presentation/`:

```
presentation/
├── README.md              # Mục lục + thứ tự trình bày
├── 01-overview.md         # Kiến trúc tổng quan
├── 02-ml-pipeline.md      # Batch ML workflow
├── 03-ml-serving.md       # FastAPI core service
├── 04-frontend.md         # Next.js dashboard
├── 05-mlflow.md          # Experiment tracking
├── 06-airflow.md          # DAG orchestration
├── 07-monitoring.md       # Prometheus + Grafana
└── 08-cicd.md           # GitHub Actions
```

---

MIT — Cập nhật: 2026-04-09
