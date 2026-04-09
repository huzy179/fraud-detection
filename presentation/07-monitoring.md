# 07 — Monitoring: Prometheus + Grafana

## Tổng quan

Hệ thống monitoring gồm Prometheus (metrics collection) và Grafana (visualization dashboard) để quan sát health và performance của ML API.

- **Prometheus:** port 9090 — metrics collection & storage
- **Grafana:** port 3002 — visualization (admin/admin)
- **Scrape interval:** 15 seconds

---

## Prometheus Configuration

**File:** [monitoring/prometheus.yml](monitoring/prometheus.yml)

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "api"
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]    # FastAPI /metrics endpoint

  - job_name: "prometheus"
    static_configs:
      - targets: ["prometheus:9090"]  # Self-monitoring
```

### Scrape Targets

| Target | URL | Metrics collected |
|--------|-----|-------------------|
| `api:8000` | `/metrics` | `fraud_api_requests_total`, `fraud_api_latency_seconds`, `fraud_predictions_total`, `fraud_rate_estimated`, `fraud_drift_score` |
| `prometheus:9090` | `/-/healthy` | Prometheus internal metrics |

---

## Grafana Dashboard

**Dashboard:** Fraud Detection API (`uid: fraud-detection-api`)

**File:** [monitoring/grafana/provisioning/dashboards/fraud-api.json](monitoring/grafana/provisioning/dashboards/fraud-api.json)

### 10 Panels

#### Overview Row (5 stat cards)

| Panel | Metric | Visual | Threshold |
|-------|--------|--------|-----------|
| Total API Requests | `sum(fraud_api_requests_total)` | Stat | — |
| API Latency (p95) | `histogram_quantile(0.95, rate(latency_seconds_bucket))` | Stat | >0.5s → yellow, >2s → red |
| Total Predictions | `sum(fraud_predictions_total)` | Stat | — |
| Fraud Predictions | `fraud_predictions_total{prediction="fraud"}` | Stat | >5 → red |
| Fraud Rate | `fraud_rate_estimated` | Stat | >2% → red |

#### Time Series Row

| Panel | Query | Type |
|-------|-------|------|
| Request Rate by Endpoint | `sum(rate(fraud_api_requests_total[1m])) by (endpoint)` | Line |
| Latency Percentiles | `histogram_quantile(0.50/0.95/0.99, rate(latency_seconds_bucket))` | Line |
| Fraud vs Legit Predictions | `sum(rate(fraud_predictions_total[5m])) by (prediction)` | Bar (stacked) |
| Requests per Hour | `sum(increase(fraud_api_requests_total[1h])) by (endpoint, method)` | Histogram |

#### Evidently Drift Row

| Panel | Query | Type |
|-------|-------|------|
| Drift Score | `fraud_drift_score` | Stat (0.5 threshold) |
| Data Drift Status | `fraud_drift_score >= 0.5` | Stat (0=green, 1=red) |
| Evidently Report Link | HTML panel → `/drift-report` | HTML button |

---

## Prometheus Metrics chi tiết

### Request Counting

```promql
# Counter: tổng số requests theo endpoint và method
fraud_api_requests_total{endpoint="/predict", method="POST"}
fraud_api_requests_total{endpoint="/transactions", method="GET"}
fraud_api_requests_total{endpoint="/transactions", method="POST"}

# Rate: requests per second trong 1 phút
rate(fraud_api_requests_total[1m])

# Increase: số requests trong 1 giờ
increase(fraud_api_requests_total[1h])
```

### Latency Histogram

```promql
# Latency bucket distribution
fraud_api_latency_seconds_bucket{endpoint="/predict", le="0.1"}
fraud_api_latency_seconds_bucket{endpoint="/predict", le="0.5"}
fraud_api_latency_seconds_bucket{endpoint="/predict", le="1.0"}

# Percentiles (quantiles)
histogram_quantile(0.50, rate(latency_seconds_bucket[5m]))  # p50 (median)
histogram_quantile(0.95, rate(latency_seconds_bucket[5m]))  # p95
histogram_quantile(0.99, rate(latency_seconds_bucket[5m]))  # p99
```

### Prediction Metrics

```promql
# Total fraud/legit predictions
fraud_predictions_total{prediction="fraud"}
fraud_predictions_total{prediction="legit"}

# Rate over 5 minutes
rate(fraud_predictions_total{prediction="fraud"}[5m])
```

### Drift Score

```promql
# Latest drift score (0 = no drift, 1 = full drift)
fraud_drift_score

# Status: 1 = drift detected
fraud_drift_score >= 0.5
```

---

## Cách sử dụng

### Truy cập Grafana

```bash
open http://localhost:3002
# Login: admin / admin
```

Dashboard tự động provisioned từ:
- `monitoring/grafana/provisioning/datasources/grafana-datasources.yml`
- `monitoring/grafana/provisioning/dashboards/dashboard.yml`

### Prometheus UI

```bash
open http://localhost:9090
# Graph: nhập PromQL query, xem metrics
# Status → Targets: kiểm tra scrape targets
```

### Tạo alert rule (example)

```yaml
# monitoring/prometheus_alerts.yml
groups:
  - name: fraud_api_alerts
    rules:
      - alert: HighFraudRate
        expr: fraud_rate_estimated > 0.02
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Fraud rate above 2%"
```

---

## Vì sao dùng như vậy?

### Tại sao Prometheus + Grafana thay vì built-in logging?
- **Structured metrics**: counter, histogram, gauge — không phải log lines
- **Query language**: PromQL cho aggregations, percentiles, rates
- **Visualization**: Grafana có template variables, alerting, multi-panel dashboards
- **Industry standard**: Prometheus + Grafana là stack phổ biến nhất trong Cloud-Native ecosystem
- **Auto-provisioning**: dashboards và datasources được tự động setup từ YAML/JSON files

### Tại sao Histogram cho latency thay vì Summary?
| Metric Type | Advantages | Disadvantages |
|-------------|-----------|--------------|
| **Histogram** | Percentiles computed at query time, aggregatable across instances | Bucket boundaries affect precision |
| **Summary** | Exact percentiles at write time | Not aggregatable across instances |

- **Histogram wins** vì: fraud-api có thể scale horizontally (nhiều replicas)
- Prometheus tính p50/p95/p99 từ histogram buckets khi query
- Summary không thể aggregate → không work với multi-instance deployment

### Tại sao scrape interval = 15s?
- **15s**: fine-grained enough cho real-time monitoring, không quá noisy
- **5s**: quá frequent, tăng Prometheus load
- **1m**: quá coarse, miss short spikes
- **Industry standard**: 15s là sweet spot cho most use cases

### Tại sao Grafana datasource = Prometheus (uid: prometheus)?
- Prometheus exposes `/-/healthy` endpoint → Grafana health check
- Prometheus là pull-based → Grafana chỉ cần 1 datasource URL
- Dùng `uid: prometheus` để match dashboard JSON (`"uid": "prometheus"`)

### Tại sao 10 panels?
- **5 stat cards**: overview nhanh, không cần nhìn graphs
- **4 time series**: track trends theo thời gian
- **1 drift section**: ML-specific monitoring (Evidently integration)
- Đủ để cover health mà không overwhelm

### Tại sao Evidently gauge được push qua Prometheus?
- Evidently chạy batch (detect_drift.py) → không continuous
- Nhưng Grafana dashboard cần continuous monitoring
- Solution: API endpoint `/metrics` exposes `fraud_drift_score` gauge
- `detect_drift.py` writes `drift_alert.json` → API reads on `/metrics` call → updates gauge
- coupling: drift script không cần push to Prometheus directly
