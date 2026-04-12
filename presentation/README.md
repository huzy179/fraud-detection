# Fraud Detection — Presentation Materials

## Mục lục

| # | Phần | File | Mô tả |
|---|------|------|--------|
| 0 | [Presentation Deck](./00-presentation-deck.md) | 00-presentation-deck.md | **Kịch bản & Slide cho thuyết trình** |
| - | [Full Documentation](./COMPREHENSIVE-GUIDE.md) | COMPREHENSIVE-GUIDE.md | **Tài liệu kỹ thuật tổng thể (Mới)** |
| 1 | [Overview](./01-overview.md) | 01-overview.md | Kiến trúc tổng quan 8 services, data flow |
| 2 | [ML Pipeline](./02-ml-pipeline.md) | 02-ml-pipeline.md | Batch: preprocess → train → detect drift |
| 3 | [ML Serving](./03-ml-serving.md) | 03-ml-serving.md | FastAPI — core inference API |
| 4 | [MLflow](./05-mlflow.md) | 05-mlflow.md | Experiment tracking, model registry |
| 5 | [Frontend](./04-frontend.md) | 04-frontend.md | Next.js real-time dashboard |
| 6 | [Airflow](./06-airflow.md) | 06-airflow.md | DAG orchestration, scheduled retraining |
| 7 | [Monitoring](./07-monitoring.md) | 07-monitoring.md | Prometheus + Grafana observability |
| 8 | [CI/CD](./08-cicd.md) | 08-cicd.md | GitHub Actions: lint → test → build → push |

## Thứ tự trình bày đề xuất

1. **Presentation Deck** — Sử dụng file này làm kịch bản chính cho buổi thuyết trình.
2. **Overview** — Bắt đầu với bức tranh toàn cảnh, sơ đồ kiến trúc
3. **ML Pipeline** — Quy trình huấn luyện model: preprocessing, training, drift detection
4. **ML Serving** — Đi vào core: API nhận transaction → dự đoán fraud
5. **MLflow** — Theo dõi experiment, so sánh 3 model (LightGBM, XGBoost, RandomForest)
6. **Frontend** — Demo trực tiếp: submit transaction, xem kết quả real-time
7. **Airflow** — Tự động hóa pipeline: DAG chạy daily
8. **Monitoring** — Quan sát hệ thống: Prometheus metrics, Grafana dashboard
9. **CI/CD** — Đảm bảo chất lượng code: automated testing và Docker image publishing

## Quick Start

```bash
# Chạy toàn bộ hệ thống
docker compose up -d

# Các port quan trọng
# - Frontend:  http://localhost:3000
# - API:       http://localhost:8000
# - MLflow:    http://localhost:5001
# - Airflow:   http://localhost:8080
# - Prometheus: http://localhost:9090
# - Grafana:   http://localhost:3002 (admin/admin)
```
