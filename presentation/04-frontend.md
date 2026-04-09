# 04 — Frontend: Next.js Dashboard

## Tổng quan

Frontend là Next.js 13 web app (port 3000) với dark theme, cung cấp giao diện real-time để test transaction và xem fraud predictions.

- **Location:** [services/frontend/](services/frontend/)
- **Framework:** Next.js (React, TypeScript)
- **Styling:** Custom CSS (dark theme, no Tailwind)
- **Data fetching:** Axios + useState/useEffect
- **API base:** `http://localhost:8000` (hoặc `NEXT_PUBLIC_API_URL` env)

---

## Giao diện chính

### 1. KPI Cards (4 cards)

| Card | Dữ liệu | Icon |
|------|---------|------|
| Total Transactions | `stats.total_transactions` | Activity |
| Fraud Detected | `stats.fraud_count` | AlertTriangle |
| Fraud Rate | `stats.fraud_rate` % | TrendingUp |
| Avg Fraud Probability | `stats.avg_fraud_probability` % | — |

### 2. Prediction Form

- 30 input fields (V1–V28, Amount, Time)
- 2 buttons: "Load Legit Sample" / "Load Fraud Sample"
- Submit → POST `/transactions` → real-time result display
- Result hiển thị: Fraud/Legit, probability %, confidence level

### 3. Transaction History Table

- Columns: ID, Amount, Fraud Prob, Status (badge), Confidence, Time
- Auto-refresh khi submit new transaction
- Max 20 rows (API limit), paginated via `?limit=20`

---

## Cách sử dụng

### Chạy frontend

```bash
# Docker (recommended)
docker compose up -d frontend
# Truy cập: http://localhost:3000

# Local dev
cd services/frontend
npm install
npm run dev
# Truy cập: http://localhost:3000
```

### Build production

```bash
cd services/frontend
npm run build   # Next.js production build
npm start       # Production server
```

---

## Source code chính

**File:** [services/frontend/pages/index.tsx](services/frontend/pages/index.tsx)

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Pre-extracted samples từ Kaggle test set
const SAMPLE_LEGIT = { V1: -0.67, ..., Amount: 23.00, Time: 160760.00 };
const SAMPLE_FRAUD = { V1: -1.27, ..., Amount: 0.01, Time: 57007.00 };

// Fetch data on mount
useEffect(() => {
  const [txRes, statsRes] = await Promise.all([
    axios.get(`${API_URL}/transactions?limit=20`),
    axios.get(`${API_URL}/transactions/stats`),
  ]);
  setTransactions(txRes.data);
  setStats(statsRes.data);
}, []);

// Submit transaction
const handleSubmit = async (e) => {
  const res = await axios.post(`${API_URL}/transactions`, formData);
  setResult(res.data);
  fetchData(); // Refresh table
};
```

---

## API Integration

### Endpoints used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/transactions?limit=20` | GET | Recent transactions |
| `/transactions/stats` | GET | KPI stats |
| `/transactions` | POST | Submit new transaction + predict |

### Response handling

```typescript
// Transaction POST response
{
  id: "uuid-string",
  amount: 23.00,
  fraud_probability: 0.15,        // 15% fraud probability
  is_fraud: false,                // below threshold
  confidence: "medium",           // "low" | "medium" | "high"
  created_at: "2026-04-09T12:00:00Z"
}
```

---

## Styling

**File:** [services/frontend/styles/globals.css](services/frontend/styles/globals.css)

Theme: Dark mode `#0a0f1e` background, high contrast text.

```css
:root {
  --bg-primary: #0a0f1e;
  --bg-card: #111827;
  --text-primary: #f3f4f6;
  --text-secondary: #9ca3af;
  --accent-green: #10b981;   /* legit */
  --accent-red: #ef4444;      /* fraud */
  --accent-blue: #3b82f6;    /* primary actions */
}
```

---

## Vì sao dùng như vậy?

### Tại sao Next.js thay vì plain React?
- **Server-side rendering (SSR)**: initial page load nhanh hơn, SEO-friendly
- **File-based routing**: không cần React Router
- **API routes**: có thể đặt backend logic ngay trong Next.js (cho demo đơn giản)
- **Built-in optimization**: image, font, script optimization tự động

### Tại sao Axios thay vì fetch()?
- **Interceptors**: có thể thêm auth headers, error handling centrally
- **Automatic JSON transformation**: không cần `.json()` call
- **Testability**: Axios dễ mock trong unit tests
- **Timeout handling**: built-in timeout config

### Tại sao pre-extracted samples?
- Kaggle dataset V1-V28 là PCA-transformed features — không intuitive để user tự nhập
- 2 sample transactions (1 legit, 1 fraud) được extract từ test set, đại diện cho real cases
- User có thể modify các giá trị để thử nghiệm
- **Amount = 0.01** cho fraud sample: test case điển hình (small fraud amounts để tránh detection)

### Tại sao Promise.all() cho initial fetch?
- 2 API calls độc lập → fetch song song
- Nếu stats fail, transactions vẫn hiển thị
- Faster perceived load time

### Tại sao auto-refresh không dùng polling?
- Frontend chỉ refresh khi user submit transaction mới
- Không có background polling → tiết kiệm server resources
- Transaction table không thay đổi nếu không có data mới
- Trong production, có thể thêm WebSocket/SSE cho real-time updates

### Tại sao dùng custom CSS thay vì Tailwind/MUI?
- Dataset nhỏ (fraud-detection demo), không cần full component library
- Custom CSS cho dark theme trực tiếp, control hoàn toàn
- Không có dependency overhead
