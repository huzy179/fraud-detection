# PLAN — Fraud Detection End-to-End ML Ops

## Cấu trúc hiện tại (sau dọn dẹp)

```
fraud-detection/
├── docker-compose.yml
├── mlflow.db
├── models/
│   ├── xgboost_model.json      (1.5MB, threshold=0.94)
│   ├── lgbm_model.txt          (1.0MB, threshold=0.93) ← ACTIVE
│   ├── rf_model.joblib          (6.8MB, threshold=0.89)
│   └── best_config.json         (LightGBM best)
├── data/
│   ├── raw/creditcard.csv      (98MB)
│   ├── processed/               (parquet + scalers)
│   └── scripts/download_data.py
├── services/
│   ├── ml-pipeline/            (Dockerfile, preprocess.py, train.py)
│   ├── ml-serving/             (Dockerfile, main.py, tests/)
│   └── frontend/               (Dockerfile, pages/, styles/)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/provisioning/
│       ├── dashboards/fraud-api.json + dashboard.yml
│       └── datasources/prometheus.yml
├── mlflow_artifacts/            (MLflow run artifacts)
└── .claude/settings.json
```

---

## Luồng ML Ops

```
Data (raw CSV)
    ↓ download_data.py
Preprocess (StandardScaler, split, SMOTE)
    ↓ preprocess.py
Train (XGBoost + LightGBM + RandomForest)
    ↓ train.py → models/*.json, *.txt, *.joblib
    ↓ MLflow log runs + artifacts
MLflow Server (PostgreSQL backend)
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

## Tình trạng hiện tại

### ✅ Đã xong

| Component | Trạng thái |
|---|---|
| Dọn dẹp file/folder thừa | ✅ |
| docker-compose.yml (xóa version, healthcheck, depends_on) | ✅ |
| train.py (wait_for_mlflow, log_artifact, path fix) | ✅ |
| ml-pipeline Dockerfile (libgomp1, paths) | ✅ |
| preprocess.py (path fix) | ✅ |
| XGBoost + LightGBM + RandomForest trained | ✅ |
| MLflow runs + artifacts logged | ✅ |
| best_config.json created | ✅ |
| README.md cập nhật model performance | ✅ |
| Old test runs deleted | ✅ |
| Prometheus scrape api | ✅ |
| Version field xóa khỏi docker-compose | ✅ |

### ⚠️ Cần kiểm tra thêm

| Component | Cần check |
|---|---|
| Grafana datasource | Có kết nối Prometheus chưa? (http_code 000 lúc check gần nhất) |
| Grafana dashboard | Có hiển thị metrics không? |
| API threshold | Đang dùng 0.93 hay 0.5? |
| best_config.json threshold | Có được API đọc không? |

---

## Các bước tiếp theo (nếu cần)

### Bước A — Kiểm tra Grafana
- Check http://localhost:3002 xem dashboard
- Kiểm tra datasource Prometheus có kết nối không

### Bước B — Kiểm tra API threshold
- API đọc `FRAUD_THRESHOLD=0.93` từ env
- Nhưng `main.py` hardcoded fallback 0.5
- Cần verify API đang dùng threshold nào

### Bước C — Test Prometheus metrics
- Gửi request lên API → check Prometheus scrape được không
- Check Grafana dashboard có data không

### Bước D — Commit code
- Sau khi mọi thứ ổn, commit toàn bộ thay đổi

---

## Debug checklist

```
1. docker-compose ps
   → postgres: healthy ✅
   → mlflow:    healthy ✅
   → api:        healthy ✅
   → prometheus: running ✅
   → grafana:    running ✅

2. curl http://localhost:8000/health
   → {"status": "healthy", "model_loaded": true, ...}

3. curl http://localhost:5001
   → MLflow UI → fraud_detection_improved → 3 runs (XGBoost, LightGBM, RandomForest)

4. curl http://localhost:9090/api/v1/query?query=fraud_api_requests_total
   → Có metrics trả về

5. http://localhost:3002
   → Grafana dashboard hiển thị
```

---

*Cập nhật: 2026-04-05*