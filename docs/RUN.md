# Hướng Dẫn Chạy Dự Án
## Credit Card Fraud Detection — Quick Start

---

## Cách 1: Docker (Khuyên dùng)

```bash
# 1. Clone & chạy
git clone <repo-url>
cd fraud-detection
docker-compose up --build

# 2. Train model (trong container ml-pipeline)
docker-compose run --rm ml-pipeline
# Hoặc chạy trực tiếp bên ngoài (xem Cách 2)
```

**Truy cập:**
| Service   | URL                     |
|----------|-------------------------|
| Frontend | http://localhost:3000   |
| FastAPI  | http://localhost:8000   |
| API Docs | http://localhost:8000/docs |
| MLflow   | http://localhost:5000   |
| Grafana  | http://localhost:3002   |

---

## Cách 2: Chạy trên máy (Không Docker)

### Bước 1 — Cài đặt môi trường

```bash
# Python 3.10+ (dùng conda hoặc venv)
# Tạo conda env (khuyên)
conda create -n fraud python=3.10 -y
conda activate fraud

# Hoặc venv thuần
python3 -m venv .venv && source .venv/bin/activate
```

### Bước 2 — Cài dependencies

```bash
# ML Pipeline
cd services/ml-pipeline
pip install -r requirements.txt

# API Server (FastAPI)
cd ../ml-serving
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### Bước 3 — Tải dataset

```bash
# Cách nhanh (không cần Kaggle API key)
python data/scripts/download_data.py

# Kết quả: data/raw/creditcard.csv (284,807 rows)
```

### Bước 4 — Tiền xử lý dữ liệu

```bash
cd services/ml-pipeline
python scripts/preprocess.py
```

**Output:**
```
Loaded 284807 rows, 31 columns
Scaler saved.
Train: 227845 (394 fraud), Test: 56962 (98 fraud)
Applying SMOTE oversampling...
After resampling: 341176 samples (33.33% fraud)
Preprocessing complete!
```

### Bước 5 — Huấn luyện model

```bash
python scripts/train.py
```

**Output mẫu:**
```
XGBoost Metrics:
  precision: 0.5676
  recall:    0.8571
  f1:        0.6829
  roc_auc:   0.9761
  average_precision: 0.8484
Best model: XGBoost
```

> MLflow tracking bật nếu `MLFLOW_TRACKING_URI` được đặt và server reachable.
> Để tắt: `MLFLOW_TRACKING_URI="" python scripts/train.py`

### Bước 6 — Copy model sang API folder

```bash
cp models/xgboost_model.json services/ml-serving/models/
cp models/rf_model.joblib     services/ml-serving/models/
```

### Bước 7 — Khởi động API Server

```bash
cd services/ml-serving
python -m uvicorn main:app --port 8000 --reload
```

- API chạy tại: http://localhost:8000
- Docs tại: http://localhost:8000/docs

### Bước 8 — Khởi động Frontend

```bash
cd services/frontend
npm run dev
```

Dashboard tại: http://localhost:3000

---

## Test nhanh bằng curl

```bash
# Health check
curl http://localhost:8000/health

# Dự đoán giao dịch hợp lệ (Legit)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378,
      "V5": -0.338, "V6": 0.462, "V7": 0.239, "V8": 0.098,
      "V9": -0.664, "V10": 0.463, "V11": -0.931, "V12": -2.304,
      "V13": 0.772, "V14": -1.576, "V15": -0.230, "V16": -0.050,
      "V17": -0.844, "V18": -0.380, "V19": 0.597, "V20": -0.697,
      "V21": -0.055, "V22": -0.270, "V23": -0.233, "V24": 0.140,
      "V25": -0.052, "V26": 0.265, "V27": 0.825, "V28": -0.068,
      "Amount": 149.52, "Time": 40680
    }
  }'

# Kết quả mẫu:
# {"fraud_probability": 0.000347, "is_fraud": false, "confidence": "low"}
```

---

## Cấu trúc thư mục sau khi chạy

```
fraud-detection/
├── data/
│   ├── raw/
│   │   └── creditcard.csv         ← Bước 3 (284,807 rows)
│   └── processed/
│       ├── X_train.parquet        ← Bước 4
│       ├── X_test.parquet
│       ├── y_train.parquet
│       ├── y_test.parquet
│       ├── time_scaler.joblib      ← Bước 4
│       └── amount_scaler.joblib
│
├── models/
│   ├── xgboost_model.json         ← Bước 5
│   └── rf_model.joblib
│
└── services/ml-serving/
    └── models/                    ← Bước 6
        ├── xgboost_model.json
        └── rf_model.joblib
```

---

## Xử lý sự cố thường gặp

### `ModuleNotFoundError` khi chạy `uvicorn`
→ Dùng đúng Python đã cài dependencies:
```bash
which python3   # Kiểm tra Python path
# Nếu là /usr/bin/python3 → dùng conda env:
conda activate fraud
# Hoặc: /opt/anaconda3/bin/python3 -m uvicorn ...
```

### API trả `500 Internal Server Error`
```bash
# Kiểm tra model đã được copy chưa
ls services/ml-serving/models/
# Phải có: xgboost_model.json

# Kiểm tra scaler
ls data/processed/*_scaler.joblib
```

### `PostgreSQL connection failed` (Docker)
→ Chờ PostgreSQL khởi động xong:
```bash
docker-compose ps   # Kiểm tra trạng thái
docker-compose logs postgres  # Xem logs
```

### Cổng bị chiếm
```bash
lsof -i :8000   # Kiểm tra port 8000
lsof -i :3000   # Kiểm tra port 3000
kill <PID>      # Kill process chiếm port
```

---

## Kết quả model đã train

| Model         | Precision | Recall | F1    | ROC-AUC | Avg Precision |
|--------------|-----------|--------|-------|---------|--------------|
| **XGBoost**  | 0.5676    | 0.8571 | 0.6829| 0.9761  | **0.8484**   |
| Random Forest | 0.4330    | 0.8571 | 0.5753| 0.9815  | 0.7943       |

**Best model:** XGBoost (average_precision=0.8484)

- Recall cao (0.86) → phát hiện được ~86% giao dịch fraud
- ROC-AUC 0.97 → phân biệt tốt giữa fraud/legit
- Average Precision 0.85 → hiệu quả trên tập mất cân bằng
