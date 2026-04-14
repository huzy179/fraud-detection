# 05 — ML Serving: FastAPI Core Service

## Tổng quan

Service **ml-serving** là trái tim của hệ thống — chạy trên FastAPI (port 8000), đóng vai trò unified API cho cả ML inference lẫn transaction management.

- **Entry point:** [services/ml-serving/main.py](services/ml-serving/main.py)
- **Database:** PostgreSQL (SQLAlchemy ORM)
- **ML Backend:** LightGBM (primary) / XGBoost (fallback) + KNN serving index
- **Metrics:** Prometheus client (Counter, Histogram, Gauge)

---

## Endpoints

### Health & Info

| Method | Path | Mô tả |
|--------|------|--------|
| `GET` | `/health` | Health check + model info + threshold |

### ML Inference

| Method | Path | Mô tả |
|--------|------|--------|
| `POST` | `/predict` | KNN-based fraud prediction |
| `POST` | `/explain` | SHAP-based explanation |

### Transaction Management

| Method | Path | Mô tả |
|--------|------|--------|
| `POST` | `/transactions` | Tạo transaction → auto-predict → save to DB |
| `GET` | `/transactions` | List recent transactions (default 100, max 1000) |
| `GET` | `/transactions/stats` | Fraud statistics |

### Drift Monitoring

| Method | Path | Mô tả |
|--------|------|--------|
| `GET` | `/drift-status` | JSON: drift status từ Evidently service (port 8002) |
| `GET` | `/drift-report` | HTML report từ Evidently (MinIO bucket) |
| `GET` | `/metrics` | Prometheus metrics (PlainText) |

> Drift endpoints gọi Evidently microservice (HTTP) và đọc reports từ MinIO.

---

## Cách sử dụng

### Chạy service

```bash
# Docker (recommended)
docker compose up -d api

# Local dev
cd services/ml-serving
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Predict fraud

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

### Create transaction (predict + store)

```bash
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{ "V1": -1.27, ..., "Amount": 0.01, "Time": 57007.00 }'
```

### Check stats

```bash
curl http://localhost:8000/transactions/stats
# Response: { total_transactions, fraud_count, fraud_rate, avg_fraud_probability }
```

### Prometheus metrics

```bash
curl http://localhost:8000/metrics
# Output: Prometheus text format
#   fraud_api_requests_total{endpoint="/predict",method="POST"}
#   fraud_api_latency_seconds_bucket{endpoint="/predict",le="0.1"}
#   fraud_predictions_total{prediction="fraud"}
#   fraud_rate_estimated
#   fraud_drift_score
```

---

## Inference Logic chi tiết

### KNN Serving Index (primary)

Hệ thống build 1 KDTree từ `X_test.parquet` (56,962 rows × 30 features) tại startup:

```python
# Tại startup:
_serving_knn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X_test)

# Mỗi request:
1. Scale Time, Amount bằng scalers đã fit
2. Tìm nearest neighbor trong feature space
3. Nếu neighbor là legit (Class=0): fraud_prob = LOW
   Nếu neighbor là fraud (Class=1): fraud_prob = HIGH
4. Confidence = 1 - distance/10
```

### LightGBM Fallback (khi KNN không khả dụng)

```python
if model_type == "lightgbm":
    raw_score = model.predict(features)[0]   # log-odds
    prob = 1 / (1 + exp(-raw_score))         # sigmoid
else:
    prob = model.predict_proba(features)[0][1]
is_fraud = prob >= FRAUD_THRESHOLD
```

### Environment Variables

| Variable | Default | Mô tả |
|----------|---------|--------|
| `FRAUD_THRESHOLD` | `0.5` (hoặc `0.93` trong Docker) | Ngưỡng phân loại fraud |
| `MODEL_PATH` | `./models` | Đường dẫn model files |
| `DATA_DIR` | `./data` | Đường dẫn processed data |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO S3 endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin123` | MinIO secret key |
| `MINIO_BUCKET_REPORTS` | `drift-reports` | MinIO bucket cho drift reports |
| `EVIDENTLY_SERVICE_URL` | `http://evidently-service:8002` | Evidently service endpoint |

---

## Vì sao dùng như vậy?

### KNN nearest-neighbor lookup
- **Không cần inference**: chỉ tìm nearest neighbor, không chạy gradient boosting mỗi request
- **O(1) prediction** cho production traffic cao
- **Interpretability tự nhiên**: nếu transaction gần với fraud cases → có khả năng cao là fraud
- **Lazy initialization**: serving index chỉ build khi có request đầu tiên, không block startup

### Tại sao `FRAUD_THRESHOLD = 0.93` trong Docker?
- Threshold được optimize bằng `find_optimal_threshold()` trong `train.py` để maximize F1 score
- Với highly imbalanced dataset (0.17% fraud), threshold mặc định 0.5 sẽ gây quá nhiều false positives
- Threshold cao hơn (0.93) chỉ classify là fraud khi model rất confident → precision cao, recall thấp hơn một chút

### Tại sao dùng SHAP nhưng output là `shap_values=[0.0] * 30`?
- SHAP `Explainer` được init với Booster nhưng với KNN serving, không có direct model để explain
- Fallback: trả về probability từ KNN và một "KNN nearest neighbor" làm top feature
- **Đây là limitation** — trong production nên dùng SHAP KernelExplainer hoặc chỉ dùng model inference thay vì KNN serving

### Prometheus metrics design
- **Counter** `fraud_api_requests_total`: đếm requests theo endpoint/method
- **Histogram** `fraud_api_latency_seconds`: phân bố latency, tính p50/p95/p99
- **Counter** `fraud_predictions_total`: đếm fraud/legit predictions
- **Gauge** `fraud_rate_estimated`: fraud rate hiện tại (từ DB)
- **Gauge** `fraud_drift_score`: drift score từ Evidently

### Database schema
- 1 table `transactions` với 30 feature columns (V1-V28, Amount, Time)
- Store cả raw features để có thể tái sử dụng cho drift detection
- `fraud_probability` và `is_fraud` được compute tại write time, không compute lại

---

## Database Schema

```sql
CREATE TABLE transactions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    V1  FLOAT, V2  FLOAT, V3  FLOAT, V4  FLOAT, V5  FLOAT,
    V6  FLOAT, V7  FLOAT, V8  FLOAT, V9  FLOAT, V10 FLOAT,
    V11 FLOAT, V12 FLOAT, V13 FLOAT, V14 FLOAT, V15 FLOAT,
    V16 FLOAT, V17 FLOAT, V18 FLOAT, V19 FLOAT, V20 FLOAT,
    V21 FLOAT, V22 FLOAT, V23 FLOAT, V24 FLOAT, V25 FLOAT,
    V26 FLOAT, V27 FLOAT, V28 FLOAT,
    Amount  FLOAT NOT NULL,
    Time    FLOAT NOT NULL,
    fraud_probability FLOAT,
    is_fraud BOOLEAN,
    confidence VARCHAR(10),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Prometheus Metrics Design (toàn bộ)

| Metric | Type | Labels | Mô tả |
|--------|------|--------|--------|
| `fraud_api_requests_total` | Counter | `endpoint`, `method` | Tổng số requests |
| `fraud_api_latency_seconds` | Histogram | `endpoint` | Phân bố latency (p50/p95/p99) |
| `fraud_predictions_total` | Counter | `prediction` | `fraud` hoặc `legit` |
| `fraud_rate_estimated` | Gauge | — | Fraud rate hiện tại (từ DB) |
| `fraud_drift_score` | Gauge | — | Drift score từ Evidently (0→1) |

### Ví dụ Prometheus query

```promql
# Requests/giây theo endpoint
rate(fraud_api_requests_total[1m])

# p95 latency
histogram_quantile(0.95, rate(fraud_api_latency_seconds_bucket[5m]))

# Tỷ lệ fraud trong 5 phút
rate(fraud_predictions_total{prediction="fraud"}[5m])
  /
rate(fraud_predictions_total[5m])
```
