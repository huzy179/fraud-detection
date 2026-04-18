# Fraud Detection — Final Summary

Đây là bản tổng hợp nâng cấp của toàn bộ tài liệu thuyết trình. Mục tiêu là giữ đủ thông tin để trình bày được trọn hệ thống, nhưng vẫn ngắn hơn bản chi tiết trong các file con.

## 1. Executive Summary

- Bài toán: phát hiện giao dịch thẻ tín dụng gian lận theo thời gian thực trên bộ dữ liệu Kaggle Credit Card Fraud.
- Đặc thù dữ liệu: 284,807 giao dịch, chỉ khoảng 0.17% là fraud, nên bài toán cực kỳ mất cân bằng.
- Kết quả chính: LightGBM đạt F1 = 0.8438 với threshold tối ưu 0.93.
- Hệ thống end-to-end gồm frontend, API serving, ML pipeline, MLflow, Evidently, Airflow, Prometheus, Grafana và CI/CD.

## 2. Kiến trúc Tổng thể

Hệ thống chạy bằng Docker Compose với 12 services, chia thành 5 lớp:

| Lớp | Thành phần |
|---|---|
| UI | Next.js Frontend |
| Serving | FastAPI API, Evidently microservice |
| ML Ops | ML pipeline, MLflow, Airflow |
| Observability | Prometheus, Grafana |
| Storage | PostgreSQL, MinIO |

### Luồng xử lý chính

1. User nhập transaction trên Frontend.
2. Frontend gọi API để predict fraud và lưu transaction.
3. API dùng KNN serving index, rồi fallback sang model booster nếu cần.
4. ML pipeline chạy batch để preprocess, train, export data và detect drift.
5. Evidently tạo report drift và lưu lên MinIO.
6. Prometheus scrape metrics, Grafana hiển thị dashboard.
7. Airflow điều phối pipeline và retraining theo lịch.

## 3. Dữ liệu & Tiền xử lý

### Dataset

- Nguồn: Kaggle Credit Card Fraud Detection.
- Quy mô: 284,807 rows, 30 features.
- Đặc trưng: V1–V28 đã PCA, thêm `Time` và `Amount`.
- Mất cân bằng mạnh: fraud rate chỉ khoảng 0.17%.

### Pipeline `preprocess.py`

| Bước | Mô tả |
|---|---|
| Load | Đọc `raw/creditcard.csv` |
| Clean | Bỏ missing values, lọc `Amount >= 0` |
| Scale | StandardScaler riêng cho `Time` và `Amount` |
| Split | Stratified train/test 80/20 |
| Balance | SMOTE trên training set với `sampling_strategy=0.5` |
| Save | Lưu Parquet + scaler joblib |

### Vì sao như vậy

- StandardScaler riêng cho `Time` và `Amount` để tránh lệch scale.
- Stratified split để giữ đúng tỷ lệ fraud trong train/test.
- SMOTE chỉ áp dụng trên train set để tránh leakage.
- Parquet được dùng vì nhanh hơn và nhỏ hơn CSV.

## 4. Mô hình ML & Kết quả

### Training workflow `train.py`

1. Load dữ liệu đã preprocess.
2. Chờ MLflow sẵn sàng, nếu không thì fallback local-only.
3. Train 3 model: LightGBM, XGBoost, RandomForest.
4. Dùng 5-fold Stratified CV để đánh giá ổn định.
5. Scan threshold từ 0.05 đến 0.95 để maximize F1.
6. Log metrics và artifacts lên MLflow.
7. Chọn model tốt nhất và lưu `best_config.json`.

### Kết quả chính

| Model | Precision | Recall | F1 | ROC-AUC | Threshold |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.8617 | 0.8265 | **0.8438** | 0.9751 | 0.93 |
| XGBoost | 0.8526 | 0.8265 | 0.8394 | 0.9792 | 0.94 |
| RandomForest | 0.9048 | 0.7755 | 0.8352 | 0.9844 | 0.89 |

LightGBM được chọn vì F1 tốt nhất, model nhỏ và inference nhanh.

## 5. Frontend & API Serving

### Frontend

- Công nghệ: Next.js 14, TypeScript, dark theme.
- Port: 3000.
- Chức năng: KPI cards, prediction form, transaction history.
- Có 2 sample data được pre-extract từ Kaggle để demo nhanh.

### API Server

- Công nghệ: FastAPI + Uvicorn + SQLAlchemy.
- Port: 8000.
- Backend DB: PostgreSQL.
- ML backend: LightGBM primary, XGBoost fallback.

### API endpoints chính

| Method | Path | Mục đích |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Dự đoán fraud |
| POST | `/explain` | Giải thích SHAP |
| POST | `/transactions` | Lưu transaction + predict |
| GET | `/transactions` | Xem lịch sử giao dịch |
| GET | `/transactions/stats` | Thống kê fraud |
| GET | `/drift-status` | Trạng thái drift |
| GET | `/drift-report` | Xem report drift |
| POST | `/run-drift` | Trigger Evidently |
| GET | `/metrics` | Prometheus metrics |

### Điểm kỹ thuật quan trọng

- API xây serving index bằng KNN trên dữ liệu test đã preprocess.
- Nếu KNN hoặc booster không sẵn sàng, hệ thống có fallback logic.
- Transaction được lưu vào PostgreSQL cùng prediction và confidence.
- `/metrics` expose counter, histogram, gauge để Prometheus scrape.

## 6. Monitoring, Drift & Orchestration

### Evidently

- Chạy như microservice riêng ở port 8002.
- So sánh reference data với production data.
- Xuất drift report HTML/JSON và lưu lên MinIO `drift-reports`.

### Prometheus + Grafana

- Prometheus scrape `/metrics` mỗi 15 giây.
- Grafana hiển thị dashboard 10 panels: requests, latency, fraud count, fraud rate, drift score.
- Có alert cho latency cao, fraud rate tăng, và drift vượt ngưỡng.

### Airflow

- DAG `fraud_ml_pipeline` chạy theo lịch `@daily`.
- Luồng: download data → preprocess → train → export transactions → detect drift.
- Dùng `trigger_rule="all_done"` cho export và drift để vẫn chạy dù train fail.

## 7. MLflow & Storage

### MLflow

- Port: 5001.
- Backend store: PostgreSQL.
- Artifact store: MinIO S3-compatible.
- Dùng để log metrics, params, model artifacts và so sánh các run.

### Storage

- PostgreSQL: lưu transaction history và metadata MLOps.
- MinIO: lưu MLflow artifacts và drift reports.
- Lợi ích: self-hosted, không phụ thuộc cloud provider.

## 8. CI/CD

### GitHub Actions

- Chạy lint, test, build frontend trên mọi push/PR.
- Chỉ build và push Docker images khi push lên main.
- Push lên GHCR với tag theo commit SHA và latest.

### 5 images được build

- `fraud-api`
- `fraud-ml-pipeline`
- `fraud-evidently`
- `fraud-airflow`
- `fraud-frontend`

## 9. Quick Demo & Vận hành

### Demo flow

1. Mở Frontend tại `http://localhost:3000`.
2. Load sample legit và sample fraud.
3. Submit để xem transaction history và KPI thay đổi.
4. Mở API metrics tại `http://localhost:8000/metrics`.
5. Mở Grafana để xem dashboard realtime.
6. Vào Airflow để trigger DAG nếu muốn trình diễn retraining.

### Quick start

```bash
docker compose up -d
```

### URL quan trọng

- Frontend: http://localhost:3000
- API: http://localhost:8000
- MLflow: http://localhost:5001
- Airflow: http://localhost:8080
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3002
- Evidently: http://localhost:8002

## 10. Tài liệu liên quan

- [README.md](README.md)
- [01-overview.md](01-overview.md)
- [02-frontend.md](02-frontend.md)
- [03-ml-pipeline.md](03-ml-pipeline.md)
- [04-evidently.md](04-evidently.md)
- [05-mlflow.md](05-mlflow.md)
- [06-ml-serving.md](06-ml-serving.md)
- [07-airflow.md](07-airflow.md)
- [08-monitoring.md](08-monitoring.md)
- [09-cicd.md](09-cicd.md)

## Kết luận

Hệ thống này là một MLOps stack end-to-end đúng nghĩa: có UI để demo, có pipeline để huấn luyện, có serving realtime, có drift monitoring, có observability, và có CI/CD để duy trì chất lượng. File này là bản nâng cấp để đọc nhanh nhưng vẫn đủ ý để thuyết trình.