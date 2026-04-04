# PLAN — Fraud Detection End-to-End ML Ops

## Cấu trúc hiện tại (sau khi dọn dẹp)

```
fraud-detection/
├── .env.example
├── .gitignore
├── PLAN.md
├── README.md
├── docker-compose.yml
├── mlflow_artifacts/           ← trống, docker-compose mount vào
├── models/
│   └── xgboost_model.json     ← model đã train
├── data/
│   ├── raw/creditcard.csv     ← dữ liệu gốc
│   ├── processed/             ← đã preprocess (parquet + scaler)
│   └── scripts/download_data.py
├── services/
│   ├── ml-pipeline/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── scripts/
│   │       ├── preprocess.py
│   │       └── train.py
│   ├── ml-serving/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── tests/test_main.py
│   └── frontend/
│       ├── Dockerfile
│       ├── package.json / package-lock.json
│       ├── next.config.js
│       ├── tsconfig.json
│       ├── .dockerignore
│       ├── pages/
│       │   ├── _app.tsx
│       │   └── index.tsx
│       └── styles/globals.css
└── monitoring/
    ├── prometheus.yml
    └── grafana/provisioning/
        ├── dashboards/
        │   ├── dashboard.yml
        │   └── fraud-api.json
        └── datasources/
            └── prometheus.yml     ← datasource cho Grafana ✅
```

---

## Luồng ML Ops (từ README)

```
Data (raw CSV)
    ↓ download_data.py
Preprocess (StandardScaler, split, SMOTE)
    ↓ preprocess.py
Train (XGBoost)
    ↓ train.py → xgboost_model.json + best_config.json
MLflow (log metrics, model registry)
    ↓
API Server (FastAPI) ← model + scalers + PostgreSQL
    ↓
Frontend (Next.js)
    ↓
Prometheus (scrape /metrics)
    ↓
Grafana (dashboard)
```

---

## Tổng hợp những gì CÓ rồi vs CHƯA CÓ

### ✅ Đã có (không cần sửa)

| Component | File | Trạng thái |
|---|---|---|
| Download data | `data/scripts/download_data.py` | ✅ OK |
| Preprocess | `services/ml-pipeline/scripts/preprocess.py` | ✅ OK |
| Train (XGBoost) | `services/ml-pipeline/scripts/train.py` | ✅ OK |
| Model | `models/xgboost_model.json` | ✅ OK (16MB) |
| Data processed | `data/processed/*.parquet, *_scaler.joblib` | ✅ OK |
| MLflow server | `docker-compose.yml` (mlflow service) | ✅ OK |
| API Server | `services/ml-serving/main.py` | ✅ OK |
| Prometheus config | `monitoring/prometheus.yml` | ✅ OK |
| Grafana datasource | `monitoring/grafana/provisioning/datasources/prometheus.yml` | ✅ OK |
| Grafana dashboard | `monitoring/grafana/provisioning/dashboards/fraud-api.json` | ✅ OK |
| Docker compose | `docker-compose.yml` | ✅ OK |
| MLflow datasource | `monitoring/grafana/provisioning/datasources/prometheus.yml` | ✅ OK |
| Frontend Dockerfile | `services/frontend/Dockerfile` | ✅ OK |
| ml-pipeline Dockerfile | `services/ml-pipeline/Dockerfile` | ✅ OK |
| ml-serving Dockerfile | `services/ml-serving/Dockerfile` | ✅ OK |

### ❌ Vấn đề cần fix

| # | Vấn đề | Ảnh hưởng | Ưu tiên |
|---|---|---|---|
| 1 | `docker-compose.yml`: api service **không có healthcheck** | Prometheus scrape fail/race condition | 🔴 Cao |
| 2 | `docker-compose.yml`: prometheus **không đợi** api healthy | Metrics miss early data 0-30s | 🔴 Cao |
| 3 | `train.py`: MLflow fallback **silent** — không log rõ khi kết nối thất bại | MLflow UI trắng, không hiểu tại sao | 🔴 Cao |
| 4 | `docker-compose.yml`: ml-pipeline **không có** `depends_on` mlflow healthy | Run train khi MLflow chưa up → silent fail | 🟡 Trung |
| 5 | `models/` **thiếu** `best_config.json` (file này chưa được tạo ở đâu) | API đọc threshold từ đâu? | 🟡 Trung |
| 6 | `docker-compose.yml`: thứ tự depends_on chưa rõ ràng | Khó debug khi có lỗi | 🟡 Trung |

---

## Các bước thực hiện (từng bước, xong bước này → xác nhận → bước sau)

### Bước 1 — Fix docker-compose.yml (healthcheck + depends_on)
**Mục tiêu:** Đảm bảo thứ tự khởi động đúng và Prometheus scrape được ngay từ đầu.

Cần sửa:
- Thêm `healthcheck` vào `api` service (endpoint `/health`)
- Thêm `depends_on` mlflow → **health condition** vào ml-pipeline
- Prometheus `depends_on` → đợi `api:service_healthy`
- ml-pipeline: đợi `mlflow:service_healthy` trước khi train

### Bước 2 — Fix train.py (MLflow logging rõ ràng)
**Mục tiêu:** Khi MLflow kết nối thất bại, log rõ ràng thay vì silent fallback.

Cần sửa:
- Thêm `wait_for_mlflow()` function — retry kết nối với timeout
- Log: `"MLflow connected ✅"` hoặc `"MLflow unavailable, running local-only"`
- Đảm bảo runs được ghi đúng vào MLflow server (không fallback local)
- Tạo `models/best_config.json` sau khi train xong

### Bước 3 — Kiểm tra API đọc model đúng chỗ
**Mục tiêu:** API phải load được model + scaler từ Docker volume.

Cần kiểm:
- `MODEL_PATH` mount trong docker-compose đúng chưa?
- `FRAUD_THRESHOLD` đọc từ env hay hardcoded?
- Scaler paths có đúng không?

### Bước 4 — Test toàn bộ luồng với docker-compose
**Mục tiêu:** Toàn bộ stack chạy, MLflow + Prometheus + Grafana hiển thị data.

Commands:
```bash
docker-compose down -v
docker-compose up --build
docker-compose ps                    # tất cả healthy
docker-compose logs mlflow            # MLflow up
docker-compose logs api               # model loaded
docker-compose logs ml-pipeline       # train xong
curl http://localhost:5001            # MLflow UI
curl http://localhost:9090/api/v1/query?query=fraud_api_requests_total  # Prometheus
curl http://localhost:8000/health    # API
http://localhost:3000                # Frontend
http://localhost:3002                # Grafana
```

### Bước 5 — Verify Prometheus scrape + Grafana dashboard
**Mục tiêu:** Prometheus scrape được metrics, Grafana hiển thị.

Check:
- Prometheus targets: `http://localhost:9090/targets`
- Grafana → Dashboards → Fraud Detection API → có data hiển thị

---

## Debug checklist (sau khi fix xong)

```
1. docker-compose ps
   → postgres: healthy ✅
   → mlflow:    healthy ✅
   → api:        healthy ✅
   → prometheus: running ✅
   → grafana:    running ✅

2. docker-compose logs mlflow | grep "Uvicorn running"
   → MLflow UI accessible at :5001

3. docker-compose logs ml-pipeline | grep "MLflow"
   → "MLflow connected ✅" HOẶC "MLflow unavailable, local-only"

4. http://localhost:5001
   → Có experiments trong MLflow UI

5. curl http://localhost:8000/health
   → {"status": "healthy", "model_loaded": true, ...}

6. curl http://localhost:9090/api/v1/query?query=fraud_api_requests_total
   → Có metrics trả về

7. http://localhost:3002
   → Grafana dashboard hiển thị metrics
```

---

*Làm từng bước. Bước nào xong → user xác nhận → sang bước tiếp.*
