# Fraud Detection System: Master Presentation & Operations Handbook

Tài liệu này là bản đặc tả kỹ thuật chuyên sâu (Technical Master Spec), được cấu trúc theo 6 Trụ cột chiến lược để phục vụ cho cả mục đích thuyết trình (Presentation) và vận hành thực tế (Production Operations).

---

## 🏛️ Trụ cột 1: Bối cảnh chiến lược & Bài toán "Kim đáy bể"
*Mục tiêu: Thiết lập bối cảnh và độ khó của bài toán để tạo ấn tượng về giá trị giải pháp.*

### 1.1. Phân tích Dữ liệu (The Dataset)
- **Nguồn:** Kaggle Credit Card Fraud Detection (Châu Âu).
- **Đặc trưng PCA (V1-V28):** Hệ thống phải làm việc với các đặc trưng đã được ẩn danh hóa. Điều này đòi hỏi mô hình phải học các tương quan phi tuyến tính thay vì các hành vi tường minh (như tên cửa hàng hay địa chỉ).
- **Độ mất cân bằng (Imbalance):** 492 gian lận / 284,807 giao dịch (**0.17%**).
- **Thách thức toán học:** Trong một không gian 30 chiều, các điểm gian lận nằm cực kỳ phân tán và dễ bị nhiễu bởi các giao dịch hợp lệ có giá trị cao.

### 1.2. Mục tiêu tối thượng (Strategic Goals)
- **Tối ưu hóa F1-Score:** Thay vì Accuracy, chúng ta cân bằng giữa **Precision** (giảm thiểu báo động giả gây khó chịu cho khách hàng) và **Recall** (giảm thiểu thiệt hại tài chính do bỏ sót gian lận).
- **Độ trễ (Latency):** < 500ms cho toàn bộ chu trình (nhận yêu cầu -> chuẩn hóa -> dự đoán -> SHAP -> phản hồi).

---

## 🏛️ Trụ cột 2: Hệ sinh thái hạ tầng (8-Service Ecosystem)
*Mục tiêu: Chứng minh tính hiện đại, độc lập và khả năng mở rộng của hệ thống.*

### 2.1. Kiến trúc Mạng & Giao tiếp (Communication Layer)
Hệ thống sử dụng Docker Bridge Network, cho phép các dịch vụ gọi nhau qua Service Name (Internal DNS):
- `api` liên kết trực tiếp với `postgres` và `mlflow`.
- `airflow` điều phối các task chạy trên container `ml-pipeline` tạm thời.
- `prometheus` định kỳ quét (scrape) endpoint `/metrics` của `api` mỗi 15 giây.

### 2.2. Bảng thông số kỹ thuật (Infrastructure Specs)
| Service | Công nghệ | Port | Phụ thuộc | Vai trò |
| :--- | :--- | :--- | :--- | :--- |
| **API** | FastAPI/Uvicorn | `8000` | `postgres`, `mlflow` | Cổng tiếp nhận giao dịch & Inference |
| **Pipeline**| Python/Sklearn | N/A | `mlflow` | Huấn luyện mô hình & Xử lý Batch |
| **DB** | PostgreSQL 15 | `5432` | N/A | Lưu trữ lịch sử & Metadata MLOps |
| **Airflow** | Apache Airflow | `8080` | `postgres` | Nhạc trưởng điều phối 5-step DAG |
| **Registry**| MLflow | `5001` | `postgres` | Quản lý phiên bản & So sánh Model |
| **Monitoring**| Prometheus | `9090` | `api` | Thu thập chỉ số sức khỏe hệ thống |
| **Viz** | Grafana | `3002` | `prometheus` | Dashboard trực quan cho vận hành |

---

## 🏛️ Trụ cột 3: Hạt nhân AI & Khoa học Dữ liệu (The Science)
*Mục tiêu: Đào sâu vào các quyết định kỹ thuật giúp đạt độ chính xác cao.*

### 3.1. Tiền xử lý & Cân bằng (Math Deep-dive)
- **Chuẩn hóa (Scaling):** Sử dụng `StandardScaler` để đưa `Amount` và `Time` về phân phối $\mathcal{N}(0, 1)$. Công thức: $z = \frac{x - \mu}{\sigma}$. Điều này ngăn chặn việc `Amount` (giá trị lớn) gây nhiễu cho các trọng số của mô hình.
- **SMOTE (Oversampling):** Để khắc phục tỉ lệ 0.17%, thuật toán SMOTE tìm 5 lân cận gần nhất (k=5) của mỗi ca gian lận và tạo ra các ca gian lận mới bằng cách nội suy tuyến tính: $x_{new} = x_i + \lambda \times (x_j - x_i)$ với $\lambda \in [0, 1]$.

### 3.2. Chiến thuật Huấn luyện (Modeling Tactics)
Hệ thống thực hiện so sánh song song 3 mô hình hàng đầu:
1.  **LightGBM:** Sử dụng GOSS (Gradient-based One-Side Sampling) để tăng tốc độ huấn luyện trên tập dữ liệu lớn.
2.  **XGBoost:** Tối ưu hóa cho Precision-Recall thông qua hàm mất mát (loss function) tùy chỉnh.
3.  **Threshold Tuning:** Tìm kiếm ngưỡng xác suất tối ưu trong dải [0.05, 0.95] để đạt **F1-Score MAX**.

---

## 🏛️ Trụ cột 4: Multi-Layer Serving & Trí tuệ AI minh bạch
*Mục tiêu: Trình bày cơ chế phục vụ thời gian thực và lòng tin từ giải thích AI.*

### 4.1. Cơ chế Dự đoán Lai (Hybrid Strategy)
Không chỉ đơn thuần là gọi model, hệ thống thực hiện 2 lớp kiểm tra:
- **Lớp 1 (KNN Fast-Lookup):** So sánh vectơ giao dịch hiện tại với 1,000 ca gian lận "kinh điển". Nếu khoảng cách Euclidean cực thấp -> Cảnh báo ngay lập tức.
- **Lớp 2 (Booster Inference):** Nếu Lớp 1 không chắc chắn, LightGBM sẽ tính toán xác suất rủi ro dựa trên cấu trúc cây quyết định.

### 4.2. Giải thích AI (Explainability)
Mỗi dự đoán gian lận được đi kèm với biểu đồ **SHAP (SHapley Additive exPlanations)**:
- Chỉ rõ chính xác biến số nào (ví dụ: V14 - Giao dịch tại quốc gia lạ) đã đẩy mức rủi ro lên cao.
- Điều này giúp nhân viên ngân hàng có thể giải thích lý do từ chối giao dịch cho khách hàng một cách tự tin.

---

## 🏛️ Trụ cột 5: Hệ sinh thái MLOps Automation
*Mục tiêu: Chứng minh tính tự hành và bền bỉ của hệ thống.*

### 5.1. Luồng tự động hóa (The Pipeline)
Được điều khiển bởi Airflow DAG `fraud_ml_pipeline`:
1.  **Step 1:** Thu thập dữ liệu giao dịch thực tế từ Database.
2.  **Step 2:** Kiểm tra lệch dữ liệu (**Data Drift**) bằng Evidently AI thông qua chỉ số PSI.
3.  **Step 3:** Nếu `drift_score > 0.5`, tự động trigger huấn luyện lại (Retrain).
4.  **Step 4:** Kết quả huấn luyện được đẩy lên MLflow để so sánh với bản "Champion" hiện tại.
5.  **Step 5:** Cập nhật phiên bản mô hình tốt nhất vào `api` mà không gây gián đoạn (Zero-downtime update).

---

## 🏛️ Trụ cột 6: Giám sát toàn vẹn & Phụ lục Kỹ thuật
*Mục tiêu: Đảm bảo khả năng kiểm soát và xử lý sự cố.*

### 6.1. Metrics quan trọng (Prometheus)
- `fraud_rate_estimated`: Tần suất xuất hiện giao dịch gian lận dự kiến.
- `api_inference_latency`: Theo dõi độ trễ (latency) của quá trình AI.
- `model_drift_index`: Chỉ số cảnh báo khi mô hình cũ không còn khớp với thói quen tiêu dùng mới.

### 6.2. Cẩm nang lệnh CLI (Ops Cheat Sheet)
- **Check health:** `curl http://localhost:8000/health`
- **Xem log dự đoán:** `docker logs -f fraud_api`
- **Re-run Pipeline:** Truy cập `localhost:8080`, unpause và trigger `fraud_ml_pipeline`.

---

## 💡 Mẹo thuyết trình (Presentation Tips)
- **Câu chuyện:** Hãy kể về việc một giao dịch mất chưa đầy 1 giây để đi qua "tấm lưới" 6 trụ cột này để bảo vệ túi tiền của khách hàng.
- **Minh họa:** Hãy mở Dashboard Grafana trong lúc thuyết trình để cho thấy các con số đang nhảy múa theo thời gian thực.
