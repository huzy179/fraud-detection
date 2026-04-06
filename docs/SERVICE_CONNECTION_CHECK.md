# Tài Liệu Kiểm Tra Kết Nối Giữa Các Service

## Mục Lục
1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Bảng Tra Cứu Nhanh Cổng & URL](#2-bảng-tra-cứu-nhanh-cổng--url)
3. [Kiểm Tra Từng Kết Nối](#3-kiểm-tra-từng-kết-nối)
4. [Hướng Dẫn Kiểm Tra Trên Giao Diện Mỗi Phần](#4-hướng-dẫn-kiểm-tra-trên-giao-diện-mỗi-phần)
5. [Script Kiểm Tra Tự Động](#5-script-kiểm-tra-tự-động)
6. [Python Event Loop — Uvicorn / FastAPI](#6-python-event-loop--uvicorn--fastapi)

---

## 1. Tổng Quan Kiến Trúc

```
Browser (Người dùng)
        │
        ▼
┌──────────────────┐     HTTP (axios)      ┌──────────────────────┐
│  Frontend (:3000) │ ──────────────────────▶  FastAPI (:8000)    │
│  Next.js          │                        │  ML Inference API    │
└──────────────────┘                        └──────────┬───────────┘
                                                       │
                          ┌─────────────────────────────┼──────────────────────────────┐
                          │                             │                              │
                          ▼                             ▼                              ▼
                ┌──────────────────┐       ┌──────────────────┐          ┌──────────────────┐
                │  PostgreSQL      │       │  File System     │          │  Prometheus      │
                │  (:5432)         │       │  ./models/       │          │  (:9090)         │
                │  fraud_detection │       │  lgbm_model.txt  │          └────────┬─────────┘
                │  mlflow_db       │       └──────────────────┘                   │
                │  airflow_db      │                                               ▼
                └────────┬─────────┘                                 ┌──────────────────┐
                         │                                             │  Grafana (:3002) │
                         ▼                                             └──────────────────┘
                ┌──────────────────┐
                │  MLflow (:5001)  │◀── trains & logs
                └────────┬─────────┘
                         │
                         ▼
                ┌────────────────────────────────────────────────────────┐
                │  Airflow (:8080)                                       │
                │  DAG: fraud_ml_pipeline (chạy @daily)                  │
                │  download → preprocess → train → detect_drift            │
                └────────────────────────────────────────────────────────┘
```

---

## 2. Bảng Tra Cứu Nhanh: Cổng & URL

| Service | Container Port | Host Port | URL |
|---|---|---|---|
| **Frontend** (Next.js) | 3000 | 3000 | http://localhost:3000 |
| **FastAPI** (ML Inference) | 8000 | 8000 | http://localhost:8000 |
| **MLflow** | 5000 | 5001 | http://localhost:5001 |
| **Airflow Webserver** | 8080 | 8080 | http://localhost:8080 |
| **Prometheus** | 9090 | 9090 | http://localhost:9090 |
| **Grafana** | 3000 | 3002 | http://localhost:3002 |
| **PostgreSQL** | 5432 | 5432 | postgresql://postgres:postgres@localhost:5432 |

---

## 3. Kiểm Tra Từng Kết Nối

### 3.1. Frontend → FastAPI

**Kiểm tra bằng cURL:**
```bash
# 1. Health check FastAPI (điều kiện tiên quyết)
curl -s http://localhost:8000/health

# 2. Đẩy thử 1 giao dịch
curl -s -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": 2.07, "V3": 1.1, "V4": -0.93, "V5": -0.21,
    "V6": -0.47, "V7": -0.11, "V8": 0.14, "V9": -0.16, "V10": -0.07,
    "V11": 1.47, "V12": -1.36, "V13": -0.96, "V14": -0.22, "V15": 0.41,
    "V16": -0.5, "V17": 0.6, "V18": -0.01, "V19": 0.33, "V20": 0.12,
    "V21": 0.21, "V22": -0.06, "V23": 0.15, "V24": -0.30, "V25": 0.1,
    "V26": 0.13, "V27": -0.12, "V28": 0.14,
    "Amount": 150.0, "Time": 86400
  }'

# 3. Lấy stats
curl -s http://localhost:8000/transactions/stats

# 4. Lấy danh sách giao dịch
curl -s "http://localhost:8000/transactions?limit=5"
```

**Kết quả mong đợi:**
- `/health` → `{"status":"ok","model_type":"lgbm","threshold":0.93,...}`
- `/transactions` POST → trả về giao dịch đã tạo với `fraud_probability`
- `/transactions/stats` → JSON chứa `total_transactions`, `fraud_count`, `fraud_rate`
- `/transactions` GET → danh sách giao dịch gần đây

---

### 3.2. FastAPI → PostgreSQL

**Kiểm tra qua API:**
```bash
curl -s http://localhost:8000/transactions/stats | python3 -m json.tool
curl -s "http://localhost:8000/transactions?limit=3" | python3 -m json.tool
```

**Kiểm tra trực tiếp bằng psql:**
```bash
# Tên container có thể khác - kiểm tra bằng: docker ps --format "{{.Names}}"
docker exec -it <container_name> psql -U postgres -d fraud_detection -c "\dt"
docker exec -it <container_name> psql -U postgres -d fraud_detection -c "SELECT COUNT(*) FROM transactions;"
docker exec -it <container_name> psql -U postgres -d fraud_detection -c "\d transactions"
```

**Kết quả mong đợi:**
- Bảng `transactions` tồn tại với 30 cột V-feature + Amount + Time + fraud_probability + is_fraud + ...
- Có dữ liệu sau khi chạy predict

---

### 3.3. FastAPI → File System / Models

**Kiểm tra:**
```bash
# Kiểm tra file model tồn tại
ls -lh models/lgbm_model.txt
ls -lh models/*.joblib
ls -lh models/best_config.json

# Kiểm tra qua API health endpoint (model info)
curl -s http://localhost:8000/health
```

**Kết quả mong đợi:**
- `lgbm_model.txt` tồn tại (LightGBM đang active)
- `best_config.json` tồn tại
- Scalers (`time_scaler.joblib`, `amount_scaler.joblib`) tồn tại
- Health endpoint trả về `"model_type": "lgbm"`

---

### 3.4. Prometheus → FastAPI

**Kiểm tra bằng cURL:**
```bash
# Lấy metrics từ API
curl -s http://localhost:8000/metrics

# Kiểm tra Prometheus scrape thành công
# Truy cập http://localhost:9090/targets → thấy endpoint "api:8000" có trạng thái UP
```

**Kết quả mong đợi:**
- `/metrics` trả về Prometheus-format metrics (các dòng bắt đầu bằng `# HELP` hoặc `fraud_api_`)
- Trên Prometheus UI: target `api:8000` có trạng thái **UP** (màu xanh)
- Prometheus scrape interval: 15s (mặc định)

---

### 3.5. Grafana → Prometheus

**Kiểm tra:**
```bash
# Test Prometheus query từ bên ngoài
curl -s "http://localhost:9090/api/v1/query?query=fraud_api_requests_total"
```

**Kiểm tra trên Grafana UI:**
- Đăng nhập http://localhost:3002 (admin / admin)
- Vào **Settings → Data Sources** → kiểm tra Prometheus datasource có trạng thái **OK**
- Mở dashboard "Fraud API Metrics" → xem charts có dữ liệu

---

### 3.6. MLflow → PostgreSQL

**Kiểm tra:**
```bash
# Health check MLflow
curl -s http://localhost:5001/health 2>/dev/null || echo "MLflow chưa chạy"

# Kiểm tra trong MLflow UI
# Truy cập http://localhost:5001 → vào Experiments
# Kiểm tra có experiment "fraud_detection_improved" với các run
```

**Kết quả mong đợi:**
- MLflow UI truy cập được
- Tồn tại experiment `fraud_detection_improved`
- Các run có metrics: precision, recall, f1, roc_auc

---

### 3.7. Airflow → PostgreSQL + File System

**Kiểm tra:**
```bash
# Truy cập http://localhost:8080 (admin / admin)
# Kiểm tra DAG "fraud_ml_pipeline" có hiển thị

# Chạy thủ công một DAG run để test
# Trên Airflow UI: vào DAG → Trigger DAG → kiểm tra các task
```

**Kiểm tra kết nối DB Airflow:**
```bash
# Lấy tên container Airflow trước bằng: docker ps
docker exec -it <airflow_container> airflow dags list
docker exec -it <airflow_container> airflow connections list
```

---

## 4. Hướng Dẫn Kiểm Tra Trên Giao Diện Mỗi Phần

### 4.1. Frontend (http://localhost:3000)

**Mục đích:** Kiểm tra Frontend kết nối FastAPI thành công.

**Các bước thực hiện:**

1. **Mở trình duyệt** → truy cập `http://localhost:3000`

2. **Kiểm tra phần Stats Cards** (trên cùng dashboard):
   - Thấy 4 card: **Total Transactions**, **Fraud Count**, **Fraud Rate %**, **Avg Fraud Probability**
   - Nếu thấy số liệu → Frontend ↔ FastAPI **KẾT NỐI TỐT**
   - Nếu thấy `Error loading...` hoặc spinner xoay mãi → **CÓ VẤN ĐỀ**

3. **Kiểm tra phần Transaction History Table**:
   - Danh sách giao dịch gần đây hiển thị
   - Mỗi dòng có: Transaction ID, Fraud Probability, Is Fraud, Confidence, Created At
   - Nếu bảng trống → vẫn OK (chưa có dữ liệu) nhưng không có lỗi

4. **Kiểm tra Prediction Form** (nhập thủ công):
   - Nhấn **"Load Sample"** (gần cuối form) → form tự điền dữ liệu mẫu
   - Nhấn **"Predict"**
   - Kết quả: hiển thị Fraud Probability, Confidence, Recommendation
   - → Kiểm tra thành công cả luồng: Frontend → FastAPI → DB

5. **Dấu hiệu lỗi cần chú ý:**
   - Lỗi mạng: `Network Error`, `Failed to fetch`
   - Lỗi CORS: nếu có
   - Trang trắng hoàn toàn: Next.js không build/start đúng

---

### 4.2. FastAPI / ML Inference (http://localhost:8000)

**Mục đích:** Kiểm tra API chính, database, model inference.

**Các bước thực hiện:**

1. **Mở Swagger UI** → truy cập `http://localhost:8000/docs`

2. **Test endpoint `/health`**:
   - Click **GET /health** → **Try it out** → **Execute**
   - Kết quả mong đợi:
     ```json
     {
       "status": "ok",
       "model_type": "lgbm",
       "threshold": 0.93,
       "model_loaded": true
     }
     ```
   - `status: ok` → API đang chạy tốt
   - `model_type: lgbm` → Model được load thành công

3. **Test endpoint `/predict`**:
   - Click **POST /predict** → **Try it out**
   - Điền body JSON (ví dụ):
     ```json
     {
       "V1": -1.36, "V2": 2.07, "V3": 1.1, "V4": -0.93,
       "V5": -0.21, "V6": -0.47, "V7": -0.11, "V8": 0.14,
       "V9": -0.16, "V10": -0.07, "V11": 1.47, "V12": -1.36,
       "V13": -0.96, "V14": -0.22, "V15": 0.41, "V16": -0.5,
       "V17": 0.6, "V18": -0.01, "V19": 0.33, "V20": 0.12,
       "V21": 0.21, "V22": -0.06, "V23": 0.15, "V24": -0.30,
       "V25": 0.1, "V26": 0.13, "V27": -0.12, "V28": 0.14,
       "Amount": 150.0, "Time": 86400
     }
     ```
   - **Execute** → kiểm tra response
   - Kết quả: `fraud_probability` (0-1), `is_fraud` (true/false), `confidence`
   - → Kiểm tra: API ↔ Model file ↔ Scalers **TỐT**

4. **Test endpoint `/transactions` (POST)**:
   - POST body giống trên
   - Sau khi execute → xem response → giao dịch được tạo với ID
   - → Kiểm tra: API ↔ PostgreSQL **TỐT**

5. **Test endpoint `/transactions/stats`**:
   - Click **GET /transactions/stats** → **Execute**
   - Kết quả: JSON với `total_transactions`, `fraud_count`, `fraud_rate`, `avg_fraud_prob`
   - → Kiểm tra: API đọc dữ liệu từ PostgreSQL **TỐT**

6. **Test endpoint `/metrics`**:
   - Click **GET /metrics**
   - Kết quả: Prometheus-format metrics (nhiều dòng text)
   - → Kiểm tra: Prometheus scrape endpoint **SẴN SÀNG**

---

### 4.3. MLflow (http://localhost:5001)

**Mục đích:** Kiểm tra MLflow kết nối PostgreSQL, tracking experiments.

**Các bước thực hiện:**

1. **Mở trình duyệt** → `http://localhost:5001`

2. **Kiểm tra Experiments:**
   - Click **"fraud_detection_improved"** experiment
   - Thấy danh sách các **Run** (mỗi lần train tạo 1 run)
   - Click vào run → xem **Metrics**: precision, recall, f1, roc_auc
   - Click → xem **Artifacts**: lgbm_model.txt, best_config.json
   - → MLflow ↔ PostgreSQL (mlflow_db) **TỐT**

3. **Kiểm tra Model Registry** (nếu có):
   - Vào tab **Models** → kiểm tra model versions

---

### 4.4. Airflow (http://localhost:8080)

**Mục đích:** Kiểm tra Airflow kết nối PostgreSQL, DAG workflow.

**Các bước thực hiện:**

1. **Đăng nhập** → `http://localhost:8080`
   - Username: `airflow` / Password: `airflow` (hoặc admin/admin tùy cấu hình)

2. **Kiểm tra DAG `fraud_ml_pipeline`:**
   - Trên menu trái → **DAGs** → thấy `fraud_ml_pipeline`
   - Click vào → xem **Graph View**
   - 4 task: `download_data` → `preprocess_data` → `train_model` → `detect_drift`
   - → Airflow ↔ PostgreSQL (airflow_db) **TỐT**

3. **Trigger thủ công để test toàn pipeline:**
   - Click nút **Play/Trigger** (▶) → **Trigger DAG**
   - Theo dõi **Grid View**: từng task chuyển từ `queued` → `running` → `success` (🟢)
   - Nếu tất cả 4 task đều xanh → Airflow ↔ ML Pipeline ↔ File System **TỐT**
   - Nếu có task đỏ → click vào task → xem **Log** để chuẩn đoán

4. **Kiểm tra task logs:**
   - Click task failed → **Logs** → đọc stack trace
   - Các lỗi thường gặp:
     - `FileNotFoundError`: thiếu script hoặc data
     - `Connection refused`: MLflow chưa chạy
     - `Postgres connection failed`: DB chưa ready

5. **Kiểm tra Connections:**
   - Menu → **Admin → Connections**
   - Kiểm tra có connection `postgres_default` không

---

### 4.5. Prometheus (http://localhost:9090)

**Mục đích:** Kiểm tra Prometheus scrape FastAPI metrics.

**Các bước thực hiện:**

1. **Mở trình duyệt** → `http://localhost:9090`

2. **Kiểm tra Targets:**
   - Menu → **Status → Targets**
   - Tìm endpoint **`api:8000`**
   - Trạng thái: **UP** (màu xanh lá)
   - State: **HEALTHY**
   - → Prometheus → FastAPI **KẾT NỐI TỐT**
   - Nếu **DOWN** → Prometheus không scrape được, kiểm tra network

3. **Kiểm tra Metrics:**
   - Menu → **Graph**
   - Gõ vào ô query: `fraud_api_requests_total` → nhấn **Execute**
   - Chuyển sang tab **Graph** → thấy đường cong metrics
   - → Prometheus nhận metrics từ API **TỐT**

4. **Các metrics quan trọng cần thấy:**
   - `fraud_api_requests_total` — tổng số request
   - `fraud_api_latency_seconds` — độ trễ
   - `fraud_predictions_total` — số lần predict fraud/legit
   - `fraud_rate_estimated` — tỷ lệ fraud ước tính

5. **Kiểm tra cấu hình scrape:**
   - Menu → **Status → Configuration**
   - Tìm section `scrape_configs` → thấy `api:8000` với path `/metrics`

---

### 4.6. Grafana (http://localhost:3002)

**Mục đích:** Kiểm tra Grafana kết nối Prometheus, dashboard hiển thị metrics.

**Các bước thực hiện:**

1. **Đăng nhập** → `http://localhost:3002`
   - Username: `admin` / Password: `admin`

2. **Kiểm tra Data Source:**
   - 🔍 Dùng thanh tìm kiếm: gõ **"Connections"** hoặc **"Data sources"**
   - Cách nhanh nhất: vào `http://localhost:3002/connections/datasources`
   - Thấy **Prometheus** datasource
   - Click vào → nhấn **Save & Test**
   - Kết quả: ✅ **"Data source is working"**
   - → Grafana ↔ Prometheus **TỐT**
   - *(Lưu ý: Menu Settings ⚙️ → Data Sources ở Grafana 10.x đã chuyển sang **Connections** ở sidebar trái)*

3. **Kiểm tra Dashboard:**
   - 🏠 **Dashboards → Browse**
   - Mở dashboard **"Fraud API Metrics"**
   - Các panel cần có dữ liệu:
     - **Request Rate** — số request/giây
     - **Latency P50/P95/P99** — độ trễ
     - **Fraud Prediction Distribution** — phân bố fraud/legit
     - **Estimated Fraud Rate** — tỷ lệ fraud gauge
   - Nếu panels hiển thị "No data" → metrics chưa được scrape

4. **Tạo Alert thử** (tùy chọn):
   - Click panel → **Edit → Alert** → tạo alert rule
   - Ví dụ: fraud_rate > 10% → gửi notification
   - → Kiểm tra hệ thống alerting hoạt động

---

### 4.7. PostgreSQL (kiểm tra trực tiếp)

**Mục đích:** Kiểm tra tất cả database kết nối (fraud_detection, mlflow_db, airflow_db).

**Các bước thực hiện:**

```bash
# Tìm tên container Postgres (khác nhau tùy project)
docker ps --format "{{.Names}}" | grep postgres

# Liệt kê databases
docker exec -it <postgres_container> psql -U postgres -c "\l"

# Kết nối vào fraud_detection DB
docker exec -it <postgres_container> psql -U postgres -d fraud_detection -c "\dt"
docker exec -it <postgres_container> psql -U postgres -d fraud_detection -c \
  "SELECT COUNT(*) as total, SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count FROM transactions;"

# Kết nối vào mlflow_db
docker exec -it <postgres_container> psql -U postgres -d mlflow_db -c "\dt" | head -20

# Kết nối vào airflow_db
docker exec -it <postgres_container> psql -U postgres -d airflow_db -c "\dt" | head -10
```

**Kết quả mong đợi:**
- 3 databases: `fraud_detection`, `mlflow_db`, `airflow_db` đều tồn tại
- `fraud_detection.transactions` có dữ liệu sau khi predict
- `mlflow_db` có bảng `experiments`, `runs`, `metrics`
- `airflow_db` có bảng `dag`, `job`, `serialized_dag`

---

## 5. Script Kiểm Tra Tự Động

Chạy script này để kiểm tra tất cả kết nối cùng lúc:

```bash
#!/bin/bash
# ================================================================
#  Check All Service Connections — Fraud Detection System
# ================================================================
#  Cách dùng: chmod +x health_check.sh && ./health_check.sh
# ================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check() {
  local name="$1"
  local cmd="$2"
  echo -n "[$name] "
  if eval "$cmd" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS${NC}"
  else
    echo -e "${RED}❌ FAIL${NC}"
  fi
}

echo "=========================================="
echo "  Service Connection Health Check"
echo "=========================================="
echo ""

echo "--- Core Services ---"
check "Frontend (3000)"        "curl -sf http://localhost:3000 > /dev/null"
check "FastAPI Health (8000)"  "curl -sf http://localhost:8000/health > /dev/null"
check "FastAPI Metrics (8000)" "curl -sf http://localhost:8000/metrics > /dev/null"
check "MLflow (5001)"          "curl -sf http://localhost:5001 > /dev/null"
check "Airflow (8080)"         "curl -sf http://localhost:8080 > /dev/null"
check "Prometheus (9090)"      "curl -sf http://localhost:9090 > /dev/null"
check "Grafana (3002)"         "curl -sf http://localhost:3002 > /dev/null"

echo ""
echo "--- Connection Verification ---"
check "Frontend → FastAPI"      "curl -sf http://localhost:8000/transactions/stats > /dev/null"
check "FastAPI → DB (read)"     "curl -sf http://localhost:8000/transactions?limit=1 > /dev/null"
check "FastAPI → Prometheus"    "curl -sf http://localhost:8000/metrics | grep -q fraud_api"
check "Prometheus → API scrape" "curl -sf http://localhost:9090/api/v1/query?query=up | grep -q api"
check "Grafana → Prometheus DS" "curl -sf http://localhost:9090/api/v1/status/config > /dev/null"

echo ""
echo "--- Database Check ---"
PG_CONTAINER=$(docker ps --format "{{.Names}}" | grep postgres | head -1)
if [ -n "$PG_CONTAINER" ]; then
  docker exec -it "$PG_CONTAINER" psql -U postgres -d fraud_detection -c "SELECT 1;" \
    > /dev/null 2>&1 && echo -e "[DB fraud_detection] ${GREEN}✅ PASS${NC}" \
    || echo -e "[DB fraud_detection] ${RED}❌ FAIL${NC}"

  docker exec -it "$PG_CONTAINER" psql -U postgres -d mlflow_db -c "SELECT 1;" \
    > /dev/null 2>&1 && echo -e "[DB mlflow_db]       ${GREEN}✅ PASS${NC}" \
    || echo -e "[DB mlflow_db]       ${RED}❌ FAIL${NC}"

  docker exec -it "$PG_CONTAINER" psql -U postgres -d airflow_db -c "SELECT 1;" \
    > /dev/null 2>&1 && echo -e "[DB airflow_db]      ${GREEN}✅ PASS${NC}" \
    || echo -e "[DB airflow_db]      ${RED}❌ FAIL${NC}"
else
  echo -e "${YELLOW}⚠️  Không tìm thấy Postgres container — bỏ qua DB check${NC}"
fi

echo ""
echo "--- Model Files ---"
check "lgbm_model.txt"     "test -f models/lgbm_model.txt"
check "xgb_model.json"     "test -f models/xgboost_model.json"
check "Scalers (.joblib)" "test -f models/time_scaler.joblib && test -f models/amount_scaler.joblib"

echo ""
echo "=========================================="
echo "  Done"
echo "=========================================="
```

> **Lưu file** → `docs/health_check.sh` → `chmod +x docs/health_check.sh` → chạy `./docs/health_check.sh`

---

## 6. Python Event Loop — Uvicorn / FastAPI

### 6.1. Cách Python Event Loop Hoạt Động Trong FastAPI

FastAPI chạy trên **Uvicorn** — một ASGI server dùng **uvloop** (trên Linux/macOS) hoặc event loop mặc định của Python. Dưới đây là cách event loop xử lý một request:

```
Client gửi request HTTP
         │
         ▼
  Uvicorn Event Loop (uvloop / asyncio)
         │
         ├─ Nhận connection (non-blocking)
         │
         ├─ Gọi ASGI app (FastAPI)
         │    │
         │    ├─ Routing → chọn endpoint handler
         │    │
         │    └─ Gọi async def endpoint(req):
         │         │
         │         ├─ Nếu có I/O:  await db.query() / await http_call()
         │         │    → event loop SUSPEND task, chạy task khác
         │         │    → khi I/O xong, RESUME task
         │         │
         │         └─ Nếu CPU-bound: chạy trực tiếp (blocking event loop!)
         │
         ├─ Trả response về cho client
         │
         ▼
  Event loop tiếp tục xử lý request tiếp theo
```

**Điểm quan trọng:**
- Tất cả các endpoint trong `main.py` đều là `async def` → dùng `await` cho I/O operations
- `get_db()` trả về **sync** Session nhưng được gọi qua `Depends(get_db)` → FastAPI tự động chạy trong thread pool (không block event loop)
- Nếu viết **CPU-bound** code (model inference nặng) trong async def mà không dùng `await` → **event loop bị block**, throughput giảm mạnh

---

### 6.2. Kiểm Tra Python Event Loop Đang Chạy

```bash
# Xem uvicorn đang chạy với event loop gì
ps aux | grep uvicorn

# Test event loop không bị block (gửi nhiều request song song)
for i in {1..10}; do
  curl -s http://localhost:8000/health &
done
wait
echo "Done — event loop xử lý song song tốt"

# Hoặc dùng Python asyncio test
python3 -c "
import asyncio, aiohttp
async def check():
    async with aiohttp.ClientSession() as s:
        tasks = [s.get('http://localhost:8000/health') for _ in range(10)]
        results = await asyncio.gather(*tasks)
        for r in results:
            print(await r.json())
asyncio.run(check())
"
```

**Kết quả mong đợi:**
- 10 requests hoàn thành gần như **cùng lúc** (concurrency)
- Event loop không bị block

---

### 6.3. Dấu Hiệu Event Loop Bị Block & Cách Xử Lý

**Dấu hiệu nhận biết:**
```
# Symptom 1: 1 request chậm → tất cả request khác đều chờ
curl http://localhost:8000/transactions/stats  # chậm > 5s
# → Tất cả request khác đều pending

# Symptom 2: Uvicorn log hiện warning
# WARNING:  Detected concurrency on a single OS thread
# Hoặc:SlowAPI rate limiting warnings
```

**Nguyên nhân phổ biến:**

| Nguyên nhân | Mô tả | Cách xử lý |
|---|---|---|
| CPU-bound trong `async def` | Model inference nặng chạy trực tiếp trong event loop | Dùng `run_in_executor()` hoặc chạy sync endpoint |
| Blocking DB query | SQLAlchemy sync query trong async handler | Dùng `AsyncSession` + `asyncpg` hoặc `Depends` tự động thread-pool |
| Sync file I/O | `open()`, `joblib.load()` chặn event loop | Dùng `aiofiles` hoặc `run_in_executor()` |
| Heavy SHAP computation | `shap.Explainer()` tính toán lâu trong `/explain` | Chạy trong background task (`BackgroundTasks`) |

**Cách xử lý cụ thể cho FastAPI sync → async:**

```python
# ❌ SAI — blocking event loop
@app.get("/transactions/stats")
async def get_stats():
    result = db.query(TransactionDB).count()  # sync → block!
    return result

# ✅ ĐÚNG — FastAPI tự động chạy sync Depends trong thread pool
@app.get("/transactions/stats")
async def get_stats(db: Session = Depends(get_db)):
    # SQLAlchemy sync session vẫn OK vì được gọi qua Depends
    total = db.query(TransactionDB).count()
    return {"total": total}

# ✅ TỐT HƠN — Nếu có nhiều CPU-bound work:
from fastapi import BackgroundTasks

@app.post("/explain")
async def explain(req: ExplainRequest, background_tasks: BackgroundTasks):
    def heavy_shap():
        # SHAP computation ở đây, không block event loop
        explainer = shap.Explainer(model)
        return explainer.shap_values(features)
    background_tasks.add_task(heavy_shap)
    return {"status": "processing"}
```

---

### 6.4. Lifespan — Startup & Shutdown Event Loop

FastAPI có `@asynccontextmanager` lifespan để quản lý resource lifecycle:

```python
# main.py — xem phần Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Chạy TRƯỚC khi server bắt đầu nhận request
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables ready.")
    except Exception as e:
        print(f"Database not available: {e}.")
    yield  # ← Server bắt đầu nhận request tại đây
    # SHUTDOWN: Chạy SAU khi server dừng
    engine.dispose()
    print("Database connection closed.")
```

**Kiểm tra lifespan hoạt động:**
```bash
# Khi start container/service:
# Log phải hiện: "Database tables ready."
docker logs fraud_api 2>&1 | grep "tables ready"
docker logs fraud_api 2>&1 | grep "closed"

# Hoặc test startup/shutdown bằng curl liên tục:
while true; do
  curl -s http://localhost:8000/health | jq .status
  sleep 1
done
# → Khi shutdown: health check trả về Connection refused
```

---

### 6.5. Kiểm Tra Event Loop Concurrency

```bash
# Dùng Apache Bench (ab) để stress test
ab -n 100 -c 10 http://localhost:8000/transactions/stats

# Hoặc dùng Python httpx (async)
python3 -c "
import asyncio, httpx, time

async def stress_test():
    async with httpx.AsyncClient(timeout=30) as client:
        start = time.time()
        tasks = [client.get('http://localhost:8000/transactions/stats') for _ in range(50)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f'50 requests trong {elapsed:.2f}s — avg {elapsed/50*1000:.0f}ms/req')
        print(f'Success: {sum(1 for r in results if r.status_code == 200)}')

asyncio.run(stress_test())
"
```

**Kết quả mong đợi:**
- 50 requests concurrency hoàn thành trong thời gian ngắn (event loop không bị block)
- Tốc độ trung bình: < 100ms/req cho `/health`, < 500ms cho `/transactions`

---

## Bảng Tổng Hợp: Dấu Hiệu Lỗi & Cách Xử Lý

| Dấu hiệu | Nguyên nhân thường | Cách xử lý |
|---|---|---|
| Frontend trả `Network Error` | FastAPI chưa chạy hoặc sai URL | Kiểm tra `NEXT_PUBLIC_API_URL` = `http://localhost:8000` |
| `/health` trả `model_loaded: false` | File model bị thiếu hoặc path sai | Kiểm tra `MODEL_PATH` env var và `models/` directory |
| Prometheus target DOWN | Network blocked hoặc API chưa expose port | Kiểm tra `docker-compose.yml` port mapping |
| Grafana "No data" trên dashboard | Prometheus chưa scrape được metrics | Vào Prometheus `/targets` kiểm tra target status |
| Airflow task failed với `Connection refused` | MLflow/DB chưa ready khi task chạy | Thêm healthcheck hoặc tăng retry trong DAG |
| MLflow không có experiment | `MLFLOW_TRACKING_URI` sai | Kiểm tra env var trong ml-pipeline |
| Stats cards trên Frontend = 0 | DB trống hoặc API không đọc được | POST vài giao dịch, kiểm tra DB trực tiếp |

---

## Checklist Cuối Cùng

- [ ] Frontend load được dashboard tại `http://localhost:3000`
- [ ] FastAPI Swagger UI tại `http://localhost:8000/docs` mở được
- [ ] `/health` trả `status: ok`
- [ ] `/transactions` POST + GET hoạt động (dữ liệu lưu vào DB)
- [ ] Prometheus targets: `api:8000` → **UP**
- [ ] Grafana datasource Prometheus → **OK**
- [ ] Grafana dashboard có charts với dữ liệu
- [ ] MLflow có experiment `fraud_detection_improved`
- [ ] Airflow DAG `fraud_ml_pipeline` hiển thị đầy đủ 4 task
- [ ] PostgreSQL: cả 3 databases đều có dữ liệu
