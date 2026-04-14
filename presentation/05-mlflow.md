# 04 — MLflow: Experiment Tracking & Model Registry

## Tổng quan

MLflow là centralized experiment tracking server cho toàn bộ ML workflow — theo dõi metrics, parameters, artifacts qua mỗi training run.

- **Port:** 5001 (external) / 5000 (container internal)
- **Backend store:** PostgreSQL `mlflow_db`
- **Artifact root:** MinIO S3 `s3://mlflow-artifacts/` (bucket: `mlflow-artifacts`)
- **Access:** http://localhost:5001

---

## Cách sử dụng

### Truy cập MLflow UI

```bash
# Mở browser
open http://localhost:5001

# Hoặc trong Docker:
docker compose up mlflow
# Truy cập: http://localhost:5001
```

### Experiment: `fraud_detection_improved`

MLflow tự động tạo experiment `fraud_detection_improved` khi chạy `train.py`. Mỗi lần train tạo 1 run:

```
fraud_detection_improved
├── XGBoost (run_id: ...)
├── LightGBM (run_id: ...)
└── RandomForest (run_id: ...)
```

### Các metrics được log

| Metric | Mô tả |
|--------|--------|
| `precision` | Precision at optimal threshold |
| `recall` | Recall at optimal threshold |
| `f1` | F1 score at optimal threshold |
| `roc_auc` | ROC AUC score |
| `average_precision` | Average precision (AP) |

### Parameters được log

```json
{
  "max_depth": 6,
  "learning_rate": 0.05,
  "n_estimators": 300,
  "scale_pos_weight": 577.0,
  "random_state": 42
}
```

### Artifacts

```
lightgbm/
└── model.joblib    (temporary — main model in models/lgbm_model.txt)
```

---

## MLflow API (trong train.py)

```python
import mlflow

# Setup
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("fraud_detection_improved")

# Training run
with mlflow.start_run(run_name="LightGBM") as run:
    mlflow.set_tag("model_type", "LightGBM")

    # Log params (clean booster-specific keys)
    clean_cfg = {k: v for k, v in model_cfg.items() if k not in {"booster", "device"}}
    mlflow.log_params(clean_cfg)

    # Log metrics
    mlflow.log_metrics(metrics_opt)

    # Log model artifact
    mlflow.log_artifact("/tmp/model.joblib", artifact_path="lightgbm")
```

---

## MLflow Architecture

```
┌───────────────────────────────────────────────────────────────┐
│ MLflow Tracking Server (port 5001)                              │
│                                                               │
│  PostgreSQL ─── Backend Store (mlflow_db)                      │
│    Tables: experiments, runs, metrics, params, tags            │
│                                                               │
│  MinIO S3 ─── Artifact Store (s3://mlflow-artifacts/)         │
│    /run_id/model.joblib                                       │
│    Buckets: mlflow-artifacts, drift-reports                    │
└───────────────────────────────────────────────────────────────┘
        ▲
        │ HTTP API (mlflow.* SDK)
        │
        │
┌───────────────────────────────────────────────────────────────┐
│ train.py                                                        │
│   mlflow.set_tracking_uri("http://mlflow:5000")                 │
│   mlflow.start_run() → train → mlflow.log_metrics()             │
│   Artifacts uploaded to MinIO S3                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Vì sao dùng MLflow?

### So sánh với alternatives

| Feature | MLflow | Weights & Biases | Comet.ml |
|---------|--------|-------------------|----------|
| Open source | ✅ | ❌ (có free tier) | ❌ |
| Self-hosted | ✅ (free) | ❌ | ❌ |
| Model registry | ✅ | ✅ | ✅ |
| Artifact storage | ✅ | ✅ | ✅ |
| Integration | 10+ ML frameworks | PyTorch, TensorFlow | PyTorch, TensorFlow |

### Tại sao không dùng WandB/Comet?
- **MLflow là open-source, self-hosted**: không phụ thuộc cloud provider, không có chi phí
- **Không cần login**: MLflow server chạy local, team có thể truy cập qua network
- **Model registry tích hợp**: version models, stage transitions (staging → production)

### Tại sao dùng MinIO làm artifact store thay vì local filesystem?
- **Shared storage**: artifacts có thể truy cập từ bất kỳ container nào (ml-pipeline, api, airflow)
- **Persistence**: không phụ thuộc vào volume mount trong Docker
- **S3-compatible**: MLflow hỗ trợ S3 natively, upload/download tự động
- **Production-ready**: dễ dàng thay bằng AWS S3 khi deploy lên cloud

### Tại sao PostgreSQL làm backend store?
- PostgreSQL là shared database giữa tất cả services → không cần thêm database server
- MLflow backend store chỉ cần ACID guarantees, không cần OLAP capabilities

### Tại sao `log_artifact` dùng `/tmp/model.joblib` thay vì path trực tiếp?
- `mlflow.log_artifact()` log một file đã tồn tại trên disk
- File model được save tại `models/lgbm_model.txt` (production path)
- MLflow artifact là bản sao cho experiment reproducibility
- Production model = file trong `models/`, không phải trong MLflow artifacts

### Retry logic cho MLflow connection

```python
# Trong train.py: wait_for_mlflow()
# Thử kết nối mỗi 5s, timeout 60s
# Nếu không kết nối được → fallback to local-only training
USE_MLFLOW = False  # graceful degradation
```
- Không block training nếu MLflow server chưa sẵn sàng
- CI/CD không fail vì MLflow down
- Production training vẫn chạy được trong air-gapped environment
