# Fraud Detection — Presentation Materials

## Mục lục

| # | Phần | File | Mô tả |
|---|------|------|--------|
| - | [Full Documentation](./COMPREHENSIVE-GUIDE.md) | COMPREHENSIVE-GUIDE.md | **Tài liệu kỹ thuật tổng thể** |
| 1 | [Overview](./01-overview.md) | 01-overview.md | Kiến trúc tổng quan, data flow, tech stack, 12 services |
| 2 | [Frontend](./02-frontend.md) | 02-frontend.md | Next.js real-time dashboard — KPI cards, prediction form, transaction history |
| 3 | [ML Pipeline](./03-ml-pipeline.md) | 03-ml-pipeline.md | Batch: preprocess → train → detect drift |
| 4 | [Evidently](./04-evidently.md) | 04-evidently.md | Drift detection microservice |
| 5 | [MLflow](./05-mlflow.md) | 05-mlflow.md | Experiment tracking, model registry |
| 6 | [ML Serving](./06-ml-serving.md) | 06-ml-serving.md | FastAPI — real-time inference API |
| 7 | [Airflow](./07-airflow.md) | 07-airflow.md | DAG orchestration, scheduled retraining |
| 8 | [Monitoring](./08-monitoring.md) | 08-monitoring.md | Prometheus + Grafana observability |
| 9 | [CI/CD](./09-cicd.md) | 09-cicd.md | GitHub Actions: lint → test → build → push |

## Thứ tự trình bày đề xuất

> Sắp xếp theo luồng: **Tổng quan → Demo nhanh (Frontend) → Batch ML → Real-time Serving → Orchestration → Observability → CI/CD**

### Luồng 1: Tổng quan + Demo nhanh (slides 1–2)
1. **Overview** — Bức tranh toàn cảnh, sơ đồ kiến trúc 12 services, data flow tổng quan
2. **Frontend** — Demo nhanh dashboard: KPI cards, prediction form, transaction history, load sample

### Luồng 2: Batch ML (slides 3–5)
3. **ML Pipeline** — Chi tiết batch workflow: StandardScaler, SMOTE, 5-fold CV, 3 models, threshold tuning
4. **Evidently** — Drift detection: PSI, KS test, Evidently microservice (port 8002), MinIO reports
5. **MLflow** — Experiment tracking: metrics, parameters, artifacts, model comparison

### Luồng 3: Real-time Serving (slides 6–8)
6. **ML Serving** — FastAPI: KNN lookup + LightGBM fallback, SHAP explainability, transaction CRUD
7. **Airflow** — DAG: download → preprocess → train → export → drift, trigger_rule, retry logic
8. **Monitoring** — Prometheus scrape, Grafana dashboard (10 panels), PromQL queries

### Luồng 4: DevOps (slide 9)
9. **CI/CD** — GitHub Actions: lint + test → Docker build + push (5 images) lên GHCR

## Quick Start

```bash
# Chạy toàn bộ hệ thống
docker compose up -d

# Các port quan trọng
# - Frontend:      http://localhost:3000
# - API:           http://localhost:8000
# - MLflow:        http://localhost:5001
# - Airflow:       http://localhost:8080  (admin/admin)
# - Prometheus:    http://localhost:9090
# - Grafana:       http://localhost:3002  (admin/admin)
# - MinIO Console: http://localhost:9001  (minioadmin/minioadmin123)
# - Evidently:     http://localhost:8002
```
