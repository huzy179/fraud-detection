# Luồng Test User — Hệ Thống Phát Hiện Gian Lận

## Tổng Quan Luồng

```
┌─────────────┐      POST /transactions       ┌──────────────┐
│   User      │ ────────────────────────────▶ │   Frontend   │
│  (Browser)  │                                │  Next.js     │
│             │ ◀── HTLM Response + Stats ──── │  :3000       │
└─────────────┘                                └──────┬───────┘
                                                      │
                                              HTTP REST
                                                      │
                                            ┌─────────▼─────────┐
                                            │   API Server      │
                                            │   FastAPI         │
                                            │   :8000           │
                                            └─────────┬─────────┘
                                                      │
                              ┌───────────────────────┼───────────────────────┐
                              │                       │                       │
                     ┌────────▼────────┐   ┌──────────▼──────────┐  ┌──────▼───────┐
                     │  ML Inference   │   │   Database          │  │ Prometheus   │
                     │  (LightGBM/     │   │   PostgreSQL         │  │ :9090        │
                     │   XGBoost)      │   │   transactions table │  │              │
                     └────────┬────────┘   └─────────────────────┘  └──────────────┘
                              │
                     ┌────────▼────────┐
                     │  SHAP Explain    │
                     │  (top 5 features)│
                     └──────────────────┘
```

---

## Chi Tiết Từng Bước

### Bước 1 — Truy Cập Dashboard (User)

```
URL: http://localhost:3000
```

User mở trình duyệt, truy cập dashboard. Giao diện hiển thị:

- **4 Stats Cards:** Tổng giao dịch, Số fraud, Tỷ lệ fraud (%), Xác suất TB
- **Form nhập giao dịch:** 30 fields (V1–V28, Amount, Time)
- **2 Quick Buttons:** "Load Legit Sample" / "Load Fraud Sample"
- **Bảng lịch sử:** 20 giao dịch gần nhất

---

### Bước 2 — Nhập Dữ Liệu Giao Dịch (User)

User nhập 30 features của giao dịch cần kiểm tra:

| Field | Mô tả |
|-------|--------|
| V1 – V28 | 28 features ẩn danh (PCA components) |
| Amount | Số tiền giao dịch ($) |
| Time | Thời gian tính từ giao dịch đầu tiên (giây) |

**Có 2 cách nhập:**
1. **Nhập thủ công** — điền 30 giá trị vào form
2. **Quick Load** — click "Load Legit Sample" hoặc "Load Fraud Sample" để điền nhanh

---

### Bước 3 — Gửi Yêu Cầu Dự Đoán (User click "Predict Fraud")

Frontend gửi **POST request** lên API:

```http
POST http://localhost:8000/transactions
Content-Type: application/json

{
  "V1": -1.359, "V2": -0.072, ..., "V28": -0.068,
  "Amount": 149.52,
  "Time": 40680
}
```

---

### Bước 4 — API Server Nhận Request

```
FastAPI main.py
  │
  ├── 1. Validate input (Pydantic) → 422 if invalid
  │
  ├── 2. Preprocess:
  │       Time → StandardScaler(time_scaler) → scaled_Time
  │       Amount → StandardScaler(amount_scaler) → scaled_Amount
  │
  ├── 3. ML Inference:
  │       feature_array → model.predict_proba() → fraud_probability
  │
  ├── 4. Classification:
  │       fraud_probability ≥ threshold (0.5) → is_fraud = true
  │       fraud_probability <  threshold (0.5) → is_fraud = false
  │
  ├── 5. Confidence:
  │       prob ≥ 0.8 or prob ≤ 0.2 → "high"
  │       prob ≥ 0.6 or prob ≤ 0.4 → "medium"
  │       otherwise                → "low"
  │
  ├── 6. Save to PostgreSQL:
  │       INSERT INTO transactions (id, V1..V28, Amount, Time,
  │         fraud_probability, is_fraud, confidence, created_at)
  │
  └── 7. Prometheus metrics:
            fraud_api_requests_total{endpoint="/transactions"}
            fraud_api_latency_seconds{endpoint="/transactions"}
            fraud_predictions_total{prediction="fraud/legit"}
```

---

### Bước 5 — Trả Kết Quả Về Frontend

```json
HTTP 201 Created
{
  "id": "a1b2c3d4-e5f6-...",
  "fraud_probability": 0.023,
  "is_fraud": false,
  "confidence": "high",
  "created_at": "2026-04-02T10:30:00Z"
}
```

---

### Bước 6 — Frontend Hiển Thị Kết Quả

- **Result Banner** hiện màu xanh (legit) hoặc đỏ (fraud)
- Hiển thị `% fraud_probability` + mức `confidence`
- Stats cards cập nhật số liệu mới
- Bảng lịch sử thêm dòng giao dịch vừa tạo

---

## Các Endpoint Liên Quan

| # | Endpoint | Method | Test Case |
|---|----------|--------|-----------|
| 1 | `/health` | GET | Kiểm tra model đã load chưa |
| 2 | `/predict` | POST | Dự đoán nhanh (không lưu DB) |
| 3 | `/explain` | POST | Xem SHAP — top 5 features ảnh hưởng nhất |
| 4 | `/transactions` | POST | Tạo giao dịch → dự đoán → lưu DB |
| 5 | `/transactions` | GET | Xem lịch sử giao dịch (`?limit=N`) |
| 6 | `/transactions/stats` | GET | Xem thống kê tổng quan |
| 7 | `/metrics` | GET | Prometheus metrics (promql format) |

---

## Test Cases Chi Tiết

### TC-01: Giao Dịch Hợp Lệ (Legit)
```
Input:  Amount=149.52, V1=-1.359, ..., V28=-0.068
Action: POST /transactions
Expected:
  - HTTP 201
  - is_fraud = false
  - fraud_probability < 0.5
  - confidence = "high"
```

### TC-02: Giao Dịch Gian Lận (Fraud)
```
Input:  Amount=999.99, V1=-3.043, ..., V28=-0.054
Action: POST /transactions
Expected:
  - HTTP 201
  - is_fraud = true
  - fraud_probability ≥ 0.5
```

### TC-03: Dữ Liệu Thiếu (Validation)
```
Input:  { "V1": 1.0 }    ← thiếu 29 fields còn lại
Action: POST /transactions
Expected: HTTP 422 Unprocessable Entity
```

### TC-04: Amount Âm (Validation)
```
Input:  { ..., "Amount": -50 }
Action: POST /transactions
Expected: HTTP 422 Unprocessable Entity
```

### TC-05: Health Check
```
Action: GET /health
Expected:
  - HTTP 200
  - model_loaded = true
  - threshold = 0.5
```

### TC-06: Thống Kê Sau Khi Tạo Giao Dịch
```
Action: GET /transactions/stats
Expected:
  - total_transactions ≥ 1
  - fraud_rate ∈ [0, 100]
  - avg_fraud_probability ∈ [0, 1]
```

### TC-07: Explainability (SHAP)
```
Action: POST /explain với legit transaction
Expected:
  - HTTP 200
  - Trả về top_5_features với SHAP values
  - Features có ảnh hưởng lớn nhất đến kết quả
```

---

## Luồng Giám Sát (Ngầm)

```
User tạo giao dịch
       │
       ▼
FastAPI export metrics ──▶ Prometheus (:9090)
       │                      │
       │              scrape every 15s
       │                      │
       │                      ▼
       │               Grafana Dashboard (:3002)
       │               ├── Request Rate (req/s)
       │               ├── Latency P50/P95/P99
       │               ├── Fraud Rate Over Time
       │               └── Model Performance
```

Prometheus tự động scrape metrics từ API mỗi 15 giây — user không cần thao tác gì thêm, chỉ mở Grafana để xem dashboard giám sát.
