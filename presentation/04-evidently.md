# 03 — Evidently: Drift Detection Microservice

## Tổng quan

Evidently là microservice độc lập (port 8002) chạy Evidently AI để so sánh phân bố reference data vs. production data, phát hiện data drift.

- **Location:** [services/evidently/](services/evidently/)
- **Framework:** FastAPI + Evidently AI
- **Port:** 8002
- **Storage:** MinIO `drift-reports` bucket (S3)

---

## Endpoints

### Health

| Method | Path | Mô tả |
|--------|------|--------|
| `GET` | `/health` | Health check |

### Drift Analysis

| Method | Path | Mô tả |
|--------|------|--------|
| `POST` | `/analyze` | Chạy Evidently analysis → trả JSON result |
| `GET` | `/reports/{filename}` | Download report file từ MinIO |
| `GET` | `/reports/list` | Danh sách các report đã lưu |

---

## Cách sử dụng

### Chạy service

```bash
# Docker
docker compose up -d evidently-service
# Truy cập: http://localhost:8002

# Local dev
cd services/evidently
pip install -r requirements.txt
uvicorn evidently_service:app --host 0.0.0.0 --port 8002 --reload
```

### Analyze drift

```bash
curl -X POST http://localhost:8002/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "reference_path": "/data/processed/X_train.parquet",
    "current_path": "/data/processed/X_test.parquet"
  }'
```

### Response

```json
{
  "drift_detected": true,
  "drift_score": 0.23,
  "n_drifted_features": 7,
  "total_features": 30,
  "retrain_recommended": false,
  "report_files": [
    "data_drift_report_20260415_143022.html",
    "data_drift_report_20260415_143022.json"
  ]
}
```

---

## Kiến trúc

```
┌──────────────────────────────────────────────────────────────┐
│  Evidently Service (port 8002)                                │
│                                                              │
│  POST /analyze                                                │
│     │                                                         │
│     ▼                                                         │
│  Evidently DataDriftPreset ── PSI, KS test, KL divergence   │
│     │                                                         │
│     ▼                                                         │
│  MinIO S3 ─── Upload HTML + JSON reports                     │
│  Bucket: drift-reports                                       │
└──────────────────────────────────────────────────────────────┘
       │
       │ HTTP GET /drift-status
       ▼
┌──────────────────────────────────────────────────────────────┐
│  API Server (port 8000)                                      │
│    /drift-status  → gọi Evidently service                  │
│    /drift-report  → đọc report từ MinIO                      │
│    /metrics      → expose fraud_drift_score (Prometheus)   │
└──────────────────────────────────────────────────────────────┘
```

---

## Environment Variables

| Variable | Default | Mô tả |
|----------|---------|--------|
| `MINIO_ENDPOINT` | `minio:9000` | MinIO S3 endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin123` | MinIO secret key |
| `MINIO_BUCKET` | `drift-reports` | Bucket cho drift reports |
| `MINIO_REGION` | `us-east-1` | MinIO region |
| `PROCESSED_DIR` | `/app/data/processed` | Đường dẫn processed data |

---

## Vì sao tách riêng microservice?

### Chạy song song với ml-pipeline
- `detect_drift.py` (ml-pipeline) chạy 1 lần rồi exit
- Evidently microservice chạy liên tục, API Server gọi được bất kỳ lúc nào
- Không cần chạy full ml-pipeline chỉ để check drift

### Independent scaling
- Evidently tính toán nặng (PSI trên 30 features) → có thể scale riêng
- Không ảnh hưởng API Server latency

### Tại sao MinIO thay vì local filesystem?
- Reports có thể được download từ bất kỳ đâu (qua `/reports/{filename}`)
- Persistence qua container restarts
- Có thể truy cập từ Grafana dashboard hoặc external tools

### Tại sao không dùng Evidently trực tiếp trong API Server?
- API Server giữ minimal — chỉ route request, không chạy Evidently computation
- Evidently dependency nặng (numpy, pandas, evidently) → tránh tăng image size
- Tách biệt concerns: API Server = inference, Evidently Service = drift analysis
