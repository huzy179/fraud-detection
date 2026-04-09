# 02 — ML Pipeline: Batch ML Workflow

## Tổng quan

ML Pipeline là batch job chạy các scripts xử lý dữ liệu, huấn luyện model, và phát hiện data drift. Trong Docker Compose, service `ml-pipeline` chạy một lần rồi exit (`restart: "no"`).

- **Location:** [services/ml-pipeline/scripts/](services/ml-pipeline/scripts/)
- **Scripts:** `preprocess.py` → `train.py` → `detect_drift.py` → `export_transactions.py`

---

## Scripts

### 1. preprocess.py — Data Preprocessing

**Chạy:**
```bash
cd services/ml-pipeline
python scripts/preprocess.py
```

**Pipeline steps:**

```
1. load_data()       ── Load raw/creditcard.csv (284,807 rows)
2. clean_data()      ── Drop missing values, filter Amount >= 0
3. scale_features()  ── StandardScaler cho Time và Amount RIÊNG BIỆT
4. split_data()      ── Stratified train/test (80/20)
5. handle_imbalance()── SMOTE (sampling_strategy=0.5) trên training set
6. save_processed_data() ── Save .parquet files
```

**Outputs:**
- `data/processed/X_train.parquet` — training features (SMOTE-resampled)
- `data/processed/X_test.parquet` — test features (unchanged)
- `data/processed/y_train.parquet` — training labels
- `data/processed/y_test.parquet` — test labels
- `data/processed/time_scaler.joblib` — StandardScaler cho Time
- `data/processed/amount_scaler.joblib` — StandardScaler cho Amount

**Tại sao SMOTE chỉ áp dụng cho training set?**
- SMOTE chỉ oversample minority class (fraud) trên training data
- Test set phải giữ nguyên phân bố thực tế để đánh giá model không bị optimistic bias
- `sampling_strategy=0.5` nghĩa là fraud:số còn lại = 1:2 (tăng từ 0.17% lên ~33%)

---

### 2. train.py — Model Training

**Chạy:**
```bash
python scripts/train.py
```

**Pipeline steps:**

```
1. load_data()              ── Load X_train.parquet, X_test.parquet
2. train_with_cv() × 3      ── Train 3 models với 5-fold Stratified CV
   ├── XGBoost    (max_depth=6, lr=0.05, n_estimators=300)
   ├── LightGBM   (max_depth=6, lr=0.05, n_estimators=300)
   └── RandomForest (n_estimators=200, max_depth=12, balanced)
3. find_optimal_threshold()  ── Optimize threshold for F1 score
4. Log to MLflow             ── Metrics, parameters, model artifacts
5. Save best model + config  ── models/lgbm_model.txt + best_config.json
```

**5-fold Stratified CV:**
```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
- **Stratified**: giữ nguyên tỷ lệ fraud trong mỗi fold
- **5 splits**: mỗi fold train 80% → validate 20%, lặp 5 lần
- Kết quả: mean ± std của precision, recall, F1, ROC-AUC, Avg Precision

**Threshold optimization:**
```python
find_optimal_threshold(y_test, y_proba)
# Thử tất cả thresholds từ 0.05 → 0.95 (step 0.01)
# Chọn threshold có F1 score cao nhất
```

**Kết quả mẫu (sau khi train):**
```
Model           Precision    Recall      F1       Avg Prec    Thresh
XGBoost            0.8450    0.8426    0.8438     0.8647     0.53
LightGBM           0.8400    0.8390    0.8395     0.8600     0.55    ⭐ BEST
RandomForest      0.8200    0.8100    0.8150     0.8400     0.60
```

**Outputs:**
- `models/lgbm_model.txt` — active LightGBM model (hoặc `xgboost_model.json`)
- `models/xgboost_model.json` — XGBoost model
- `models/rf_model.joblib` — RandomForest model
- `models/best_config.json` — best model config

---

### 3. detect_drift.py — Data Drift Detection

**Chạy:**
```bash
python scripts/detect_drift.py
```

**Logic:**
```
1. load_reference_data() ── X_train.parquet (training distribution)
2. load_current_data()   ── current.parquet (production) hoặc X_test.parquet (fallback)
3. detect_data_drift()    ── Evidently DataDriftPreset
   └── Population Stability Index (PSI) trên từng feature
4. should_retrain()       ── Nếu drift_score >= 0.5 → recommend retrain
5. save drift_alert.json  ── Flag file cho Airflow/API
```

**Drift score interpretation:**
- `0.0`: không có drift
- `0.5` (threshold): 50% columns có drift → recommend retrain
- `1.0`: full drift → cần retrain ngay

**Outputs:**
- `monitoring/reports/data_drift_report.html` — Evidently HTML report
- `monitoring/reports/drift_alert.json` — `{ drift_detected, drift_score, retrain_recommended }`

---

### 4. export_transactions.py — Export Production Data

**Chạy:**
```bash
python scripts/export_transactions.py
```

Export transactions từ PostgreSQL để so sánh với training data cho drift detection.

---

## Vì sao dùng như vậy?

### Tại sao StandardScaler cho Time và Amount RIÊNG BIỆT?
- Time và Amount có distribution khác nhau (Time: seconds từ 0→172800, Amount: $0→$25,691)
- Scale riêng để mỗi feature có mean=0, std=1 trong không gian feature riêng
- V1-V28 đã được PCA-transformed (từ Kaggle dataset), đã ở cùng scale

### Tại sao Stratified split?
- Với 0.17% fraud rate, random split có thể bỏ sót fraud cases vào test set
- Stratified đảm bảo tỷ lệ fraud giống nhau trong cả train và test

### Tại sao SMOTE oversampling?
- Accuracy không phù hợp với imbalanced data (model có thể predict tất cả là legit, accuracy = 99.83%)
- SMOTE tạo synthetic fraud samples bằng cách interpolate giữa các fraud samples gần nhau
- `sampling_strategy=0.5`: giữ 50% majority class → fraud tăng lên ~33% trong training

### Tại sao 5-fold CV thay vì train/test đơn giản?
- 5-fold CV cho 5 estimates khác nhau của metrics → robust
- Sử dụng full training data (không hold out validation set riêng)
- Tốt hơn train/test split đơn giản vì đánh giá stable hơn

### Tại sao F1 score cho threshold optimization?
- Precision: trong fraud detection, false positive = legitimate transaction bị đánh dấu fraud → bad user experience
- Recall: fraud không detected = financial loss
- **F1 = 2×(P×R)/(P+R)** là harmonic mean, balance giữa Precision và Recall
- Accuracy không phù hợp vì imbalanced data

### Tại sao LightGBM là best model?
- LightGBM sử dụng gradient-based one-side sampling (GOSS) → fast training
- Leaf-wise tree growth → better accuracy than level-wise (XGBoost default)
- Hỗ trợ `scale_pos_weight` để handle class imbalance natively
- F1 score cao nhất trong 3 model candidates

### Tại sao dùng Evidently cho drift detection?
- Chuyên biệt cho ML: PSI (Population Stability Index), KS test, KL divergence
- Feature-level drift detection (biết CHÍNH XÁC feature nào drifted)
- Tích hợp Prometheus gauge cho monitoring dashboard
- HTML report cho non-technical stakeholders
