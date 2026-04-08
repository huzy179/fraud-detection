# Credit Card Fraud Detection System

Hệ thống phát hiện giao dịch thẻ tín dụng gian lận trong thời gian thực, xây dựng theo mô hình ML Ops end-to-end.

## Kiến trúc

```
┌──────────────────────────────────────────────────────────────────┐
│                     docker-compose.yml                           │
│                                                                  │
│  PostgreSQL ──► MLflow ──► FastAPI ──► Next.js                  │
│      (5432)        (5001)      (8000)      (3000)                │
│                            │                                     │
│                     Prometheus ──► Grafana                      │
│                       (9090)       (3002)                       │
└──────────────────────────────────────────────────────────────────┘
```

## Stack công nghệ

| Thành phần    | Công nghệ                                |
|--------------|------------------------------------------|
| ML Model     | LightGBM (F1=0.8438, threshold=0.93)     |
| ML Tracking  | MLflow + PostgreSQL backend              |
| API Server   | FastAPI + Uvicorn + SQLAlchemy          |
| Database     | PostgreSQL 15                            |
| Frontend     | Next.js 14 (TypeScript)                  |
| Monitoring   | Prometheus + Grafana                    |
| Orchestration| Docker Compose + GitHub Actions          |
| Scheduler    | Apache Airflow                           |

---

## Hướng dẫn chạy

### Cách 1: Docker Compose (Khuyến nghị)

```bash
# 1. Clone & vào thư mục
git clone <repo-url>
cd fraud-detection

# 2. Khởi động toàn bộ hệ thống
docker-compose up -d

# 3. Kiểm tra trạng thái
docker-compose ps
```

> Lần đầu chạy sẽ tự động khởi tạo PostgreSQL, MLflow, API, Frontend, Prometheus, Grafana và Airflow.

**Các service có sẵn:**

| Service          | URL                   | Tài khoản        |
|-----------------|----------------------|-----------------|
| Frontend        | http://localhost:3000 | —               |
| FastAPI Docs    | http://localhost:8000/docs | —         |
| MLflow UI       | http://localhost:5001 | —               |
| Prometheus      | http://localhost:9090 | —               |
| Grafana         | http://localhost:3002 | admin / admin   |
| Airflow         | http://localhost:8080 | airflow / airflow |

### Cách 2: Chạy từng service (Development)

**Yêu cầu:** Python 3.9+, Node.js 18+, PostgreSQL

```bash
# ── 1. Download dữ liệu ──────────────────────────────────────
python data/scripts/download_data.py

# ── 2. Cài đặt ML Pipeline ────────────────────────────────────
cd services/ml-pipeline
pip install -r requirements.txt

# ── 3. Tiền xử lý dữ liệu ────────────────────────────────────
python scripts/preprocess.py

# ── 4. Huấn luyện mô hình ────────────────────────────────────
python scripts/train.py

# ── 5. Cài đặt API Server ─────────────────────────────────────
cd ../ml-serving
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# ── 6. Cài đặt Frontend (terminal khác) ─────────────────────
cd services/frontend
npm install
npm run dev
```

### Huấn luyện lại mô hình (Docker)

```bash
# Chạy pipeline train (tự động log MLflow + save model)
docker-compose run --rm ml-pipeline

# Restart API để load model mới
docker-compose restart api
```

### Huấn luyện lại mô hình (Local)

```bash
cd services/ml-pipeline
python scripts/preprocess.py
python scripts/train.py
# Restart API
cd ../ml-serving
uvicorn main:app --reload --port 8000
```

---

## API Endpoints

| Method | Path                    | Mô tả                              |
|--------|-------------------------|------------------------------------|
| `GET`  | `/health`               | Health check + model info          |
| `POST` | `/predict`              | Dự đoán fraud (KNN serving)       |
| `POST` | `/explain`              | Giải thích SHAP-based             |
| `POST` | `/transactions`         | Tạo giao dịch + dự đoán           |
| `GET`  | `/transactions`         | Danh sách giao dịch (paginated)    |
| `GET`  | `/transactions/stats`   | Thống kê fraud                    |
| `GET`  | `/metrics`              | Prometheus metrics                 |

### Ví dụ predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378,
    "V5": -0.338, "V6": 0.462, "V7": 0.239, "V8": 0.098,
    "V9": -0.664, "V10": 0.463, "V11": -0.931, "V12": -2.304,
    "V13": 0.772, "V14": -1.576, "V15": -0.230, "V16": -0.050,
    "V17": -0.844, "V18": -0.380, "V19": 0.597, "V20": -0.697,
    "V21": -0.055, "V22": -0.270, "V23": -0.233, "V24": 0.140,
    "V25": -0.052, "V26": 0.265, "V27": 0.825, "V28": -0.068,
    "Amount": 149.52, "Time": 40680
  }'
```

---

## Kết quả mô hình

| Model       | Precision | Recall | F1 Score | ROC-AUC | Threshold |
|-------------|:---------:|:------:|:--------:|:-------:|:---------:|
| **LightGBM** ⭐ | 0.8617  | 0.8265 | **0.8438** | 0.9751 | 0.93      |
| XGBoost     | 0.8526   | 0.8265 | 0.8394   | 0.9792 | 0.94      |
| RandomForest| 0.9048   | 0.7755 | 0.8352   | 0.9844 | 0.89      |

> **Active model:** LightGBM — threshold=0.93
> Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 giao dịch, ~0.17% fraud rate

---

## Biến môi trường

| Variable              | Default                                         | Mô tả                |
|----------------------|------------------------------------------------|----------------------|
| `FRAUD_THRESHOLD`     | `0.93`                                         | Ngưỡng phát hiện fraud |
| `DATABASE_URL`        | `postgresql://postgres:postgres@postgres:5432/` | PostgreSQL connection |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000`                           | MLflow server URL    |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000`                       | FastAPI URL (frontend) |

---

## Debug nhanh

```bash
# 1. Kiểm tra containers
docker-compose ps

# 2. Health check API
curl http://localhost:8000/health

# 3. Kiểm tra Prometheus targets
curl "http://localhost:9090/api/v1/query?query=up"

# 4. Xem logs
docker-compose logs -f api
docker-compose logs -f mlflow
```

---

## Cấu trúc project

```
fraud-detection/
├── docker-compose.yml           # Orchestration (8 services)
├── .github/workflows/ci.yml      # CI/CD pipeline
├── Dockerfile.airflow           # Airflow custom image
├── data/
│   ├── raw/creditcard.csv       # Raw dataset (98MB)
│   ├── processed/               # Parquet + scalers
│   └── scripts/download_data.py
├── models/
│   ├── lgbm_model.txt           # Active model (LightGBM)
│   ├── xgboost_model.json       # XGBoost
│   ├── rf_model.joblib          # RandomForest
│   └── best_config.json         # Best config metadata
├── services/
│   ├── ml-pipeline/             # Preprocess + Train
│   ├── ml-serving/              # FastAPI (inference + DB)
│   └── frontend/                # Next.js dashboard
├── airflow/                     # Airflow DAGs
├── mlflow_artifacts/            # MLflow run artifacts
├── mlflow.db                    # SQLite MLflow backend (local)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/provisioning/    # Dashboards + datasources
└── postgres-init/               # PostgreSQL init scripts
```

---

MIT — Cập nhật: 2026-04-08
