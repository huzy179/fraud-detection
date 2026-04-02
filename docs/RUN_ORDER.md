# Thứ Tự Chạy Các File — Hệ Thống Phát Hiện Gian Lận

## Tổng Quan

Dự án gồm 3 nhóm chính: **ML Pipeline** (huấn luyện), **ML Serving** (API inference), **Frontend** (dashboard). Thứ tự chạy phải tuân theo dependency: dữ liệu → huấn luyện → phục vụ → giao diện.

---

## Sơ Đồ Phụ Thuộc

```
data/raw/creditcard.csv
        │
        ▼
services/ml-pipeline/scripts/preprocess.py
        │  (tạo X_train.parquet, X_test.parquet, ...)
        ▼
services/ml-pipeline/scripts/train.py
        │  (tạo lgbm_model.txt, xgboost_model.json, ...)
        │
        ├──▶ models/              (nguồn)
        │     └── lgbm_model.txt
        │     └── xgboost_model.json
        │
        └──▶ services/ml-serving/models/  (đích, copy vào)
              └── lgbm_model.txt
              └── xgboost_model.json
              └── rf_model.joblib

services/ml-serving/main.py
        │  (FastAPI — cần model + scaler từ data/processed)
        ▼
services/ml-serving/tests/test_main.py
        │
        ▼
services/frontend/
        └── pages/index.tsx
        └── (Next.js — cần API chạy ở :8000)
```

---

## Chi Tiết Từng Bước

### Bước 1 — Tải Dataset

```bash
python data/scripts/download_data.py
```

| Output | Đường dẫn |
|--------|-----------|
| Raw dataset | `data/raw/creditcard.csv` (284,807 rows × 31 cols) |

**File xử lý:** `data/scripts/download_data.py`
- Tải từ Kaggle (cần token) hoặc GitHub raw URL (fallback)
- Lưu vào `data/raw/creditcard.csv`

---

### Bước 2 — Tiền Xử Lý Dữ Liệu

```bash
cd services/ml-pipeline
python scripts/preprocess.py
```

| Input | Output |
|-------|--------|
| `data/raw/creditcard.csv` | `data/processed/X_train.parquet` |
|  | `data/processed/X_test.parquet` |
|  | `data/processed/y_train.parquet` |
|  | `data/processed/y_test.parquet` |
|  | `data/processed/time_scaler.joblib` |
|  | `data/processed/amount_scaler.joblib` |

**File xử lý:** `services/ml-pipeline/scripts/preprocess.py`

**Các bước bên trong:**
```
1. Load creditcard.csv
2. Drop missing values
3. Remove Amount < 0
4. StandardScaler on Time  → time_scaler.joblib
5. StandardScaler on Amount → amount_scaler.joblib
6. Train/Test split (80/20, stratified by Class)
7. SMOTE oversampling (0.17% → 50% fraud ratio)
8. Save all 4 parquet files
```

---

### Bước 3 — Huấn Luyện Model

```bash
python scripts/train.py
```

| Input | Output |
|-------|--------|
| `data/processed/X_train.parquet` | `models/lgbm_model.txt` |
| `data/processed/y_train.parquet` | `models/xgboost_model.json` |
|  | `models/rf_model.joblib` |
|  | `models/best_config.json` |

**File xử lý:** `services/ml-pipeline/scripts/train.py`

**Các bước bên trong:**
```
1. Load X_train, y_train (parquet)
2. 5-fold Stratified CV cho từng model:
   ├── XGBoost   (max_depth=6, lr=0.1, n_estimators=200)
   ├── LightGBM  (max_depth=6, lr=0.05, n_estimators=300)
   └── Random Forest (n_estimators=200, max_depth=12)
3. Tune threshold (0.05 → 0.95, step 0.01) để maximize F1
4. Đánh giá: Precision, Recall, F1, ROC-AUC, Avg Precision
5. Log to MLflow (nếu MLFLOW_TRACKING_URI set)
6. Register best model vào MLflow model registry
7. Save model files + best_config.json
```

---

### Bước 4 — Copy Model Sang Serving Folder

```bash
cp models/lgbm_model.txt     services/ml-serving/models/
cp models/xgboost_model.json services/ml-serving/models/
cp models/rf_model.joblib    services/ml-serving/models/
```

> ⚠️ **Bước bắt buộc** — API server đọc model từ `services/ml-serving/models/`, không phải thư mục gốc `models/`.

**Lưu ý:** Nếu chạy `docker-compose up --build`, Dockerfile tự động copy các model này vào image.

---

### Bước 5 — Chạy API Server (ML Serving)

```bash
cd services/ml-serving
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

| Dependency | Cần thiết |
|-----------|-----------|
| `models/lgbm_model.txt` | ✅ |
| `models/xgboost_model.json` | ✅ |
| `data/processed/time_scaler.joblib` | ✅ |
| `data/processed/amount_scaler.joblib` | ✅ |
| PostgreSQL / SQLite | ✅ (DB fallback SQLite nếu không có PG) |

**File xử lý:** `services/ml-serving/main.py`

**Load order bên trong `main.py`:**
```
1. SQLAlchemy engine (PostgreSQL hoặc SQLite)
2. Load time_scaler.joblib
3. Load amount_scaler.joblib
4. Try: LightGBM model (lgbm_model.txt)
   Else: XGBoost model (xgboost_model.json)
5. Initialize SHAP explainer
6. Register Prometheus metrics
7. Expose FastAPI app on :8000
```

**API ready at:** `http://localhost:8000/docs` (Swagger UI)

---

### Bước 6 — Seed Sample Data (Tùy chọn)

```bash
cd data/scripts
python seed_data.py
```

| Input | Tác động |
|-------|----------|
| `http://localhost:8000` | POST 10 sample transactions (5 legit, 5 fraud) |

**File xử lý:** `data/scripts/seed_data.py`
- Chạy sau khi API đã up
- Tạo 10 sample rows trong DB để dashboard không trống

---

### Bước 7 — Chạy Tests

```bash
cd services/ml-serving
pytest tests/test_main.py -v
```

| Test Class | Số test | Coverage |
|-----------|---------|----------|
| `TestHealth` | 3 | `/health` endpoint |
| `TestPredict` | 9 | `/predict` endpoint |
| `TestTransactions` | 9 | `/transactions` CRUD + stats |
| `TestMetrics` | 3 | `/metrics` Prometheus format |
| `TestFullFlow` | 2 | End-to-end flow |
| **Tổng** | **28** | **✅ All passed** |

**File xử lý:** `services/ml-serving/tests/test_main.py`
- Dùng `FastAPI.TestClient` (in-memory, không cần chạy server thật)
- Tự tạo test DB (SQLite tạm) — không ảnh hưởng production DB

---

### Bước 8 — Chạy Frontend

```bash
cd services/frontend
npm install
npm run dev
```

| Dependency | Cần thiết |
|-----------|-----------|
| API server chạy ở `:8000` | ✅ bắt buộc (frontend gọi API) |
| `NEXT_PUBLIC_API_URL=http://localhost:8000` | ✅ trong `.env` |

**File xử lý:** `services/frontend/pages/index.tsx`

**Frontend ready at:** `http://localhost:3000`

---

## Chạy Nhanh Với Docker (Khuyên Dùng)

```bash
# Chạy toàn bộ stack cùng lúc
docker-compose up --build

# Hoặc chạy background
docker-compose up -d
```

Docker tự động chạy đúng thứ tự:
```
1. postgres     (wait for healthy)
2. mlflow       (depends on postgres)
3. api          (depends on postgres, mlflow)
4. prometheus   (depends on api)
5. grafana      (depends on prometheus)
6. frontend     (depends on api)
```

---

## Tóm Tắt Nhanh

| # | Bước | File | Lệnh | Output |
|---|------|------|------|--------|
| 1 | Tải data | `data/scripts/download_data.py` | `python download_data.py` | `data/raw/creditcard.csv` |
| 2 | Preprocess | `services/ml-pipeline/scripts/preprocess.py` | `python preprocess.py` | `data/processed/*.parquet`, `*.joblib` |
| 3 | Train model | `services/ml-pipeline/scripts/train.py` | `python train.py` | `models/*.json`, `*.txt`, `*.joblib` |
| 4 | Copy model | — | `cp models/* services/ml-serving/models/` | Model trong serving folder |
| 5 | Chạy API | `services/ml-serving/main.py` | `uvicorn main:app --reload` | `http://localhost:8000` |
| 6 | Seed data | `data/scripts/seed_data.py` | `python seed_data.py` | 10 transactions in DB |
| 7 | Chạy tests | `services/ml-serving/tests/test_main.py` | `pytest tests/ -v` | 28/28 PASSED |
| 8 | Chạy frontend | `services/frontend/pages/index.tsx` | `npm run dev` | `http://localhost:3000` |

**Thứ tự dependency:** 1 → 2 → 3 → 4 → 5 → 6 (7 song song với 5) → 8
