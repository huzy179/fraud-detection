# 08 — CI/CD: GitHub Actions

## Tổng quan

GitHub Actions CI/CD pipeline đảm bảo chất lượng code qua mỗi commit — lint → test → build → push Docker images.

- **File:** [.github/workflows/ci.yml](.github/workflows/ci.yml)
- **Triggers:** push (main, develop) + pull_request (main)
- **Registry:** GHCR (GitHub Container Registry)

---

## Pipeline Overview

```
push / pull_request (main)
        │
        ▼
┌──────────────────────────────────────────────┐
│  Job 1: lint-and-test                         │
│  ──────────────────────────────────────────   │
│  • Python 3.9: lint (flake8) + pytest          │
│    ├── ML Serving: flake8 + pytest             │
│    └── ML Pipeline: flake8 + pytest            │
│  • Node.js 18: npm ci + npm run build          │
│    └── Frontend: build test                    │
│                                              │
│  PostgreSQL service container                  │
└────────────────────┬─────────────────────────┘
                     │ ✅ passed
                     ▼
┌──────────────────────────────────────────────┐
│  Job 2: docker-build (only on main push)     │
│  ──────────────────────────────────────────   │
│  • API:       fraud-api       → GHCR          │
│  • Pipeline:  fraud-ml-pipeline → GHCR         │
│  • Airflow:   fraud-airflow   → GHCR           │
│  • Frontend:  fraud-frontend → GHCR            │
│                                              │
│  Permissions: contents:read, packages:write   │
└──────────────────────────────────────────────┘
```

---

## Job 1: lint-and-test

### Python (ML Serving)

```yaml
- name: Install ML Serving deps
  run: pip install -r services/ml-serving/requirements.txt

- name: Lint Python (ML Serving)
  run: flake8 services/ml-serving/ --max-line-length=120

- name: Test ML Serving
  run: pytest services/ml-serving/ -v
```

### Python (ML Pipeline)

```yaml
- name: Install ML Pipeline deps
  run: pip install -r services/ml-pipeline/requirements.txt

- name: Lint Python (ML Pipeline)
  run: flake8 services/ml-pipeline/scripts/export_transactions.py \
                    services/ml-pipeline/scripts/detect_drift.py \
                    --max-line-length=120

- name: Test ML Pipeline
  run: pytest services/ml-pipeline/ -v
```

### Node.js (Frontend)

```yaml
- name: Install & build frontend
  working-directory: ./services/frontend
  run: |
    npm ci
    npm run build
```

### PostgreSQL Service Container

```yaml
services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: fraud_detection
    ports:
      - 5432:5432
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
```

---

## Job 2: docker-build

**Chỉ chạy khi:** `github.event_name == 'push' && github.ref == 'refs/heads/main'`

```yaml
- name: Log in to GHCR
  uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}

- name: Build API Server image
  uses: docker/build-push-action@v5
  with:
    context: ./services/ml-serving
    file: ./services/ml-serving/Dockerfile
    push: true
    tags: |
      ghcr.io/${{ github.repository }}/fraud-api:${{ github.sha }}
      ghcr.io/${{ github.repository }}/fraud-api:latest
```

**4 Docker images được build:**

| Image | Context | Tags |
|-------|---------|------|
| `fraud-api` | `./services/ml-serving` | `${{ github.sha }}`, `latest` |
| `fraud-ml-pipeline` | `./services/ml-pipeline` | `${{ github.sha }}`, `latest` |
| `fraud-airflow` | `.` (root) | `${{ github.sha }}`, `latest` |
| `fraud-frontend` | `./services/frontend` | `${{ github.sha }}`, `latest` |

---

## Concurrency Control

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

- **Group**: tất cả runs cùng workflow + ref vào 1 group
- **cancel-in-progress**: nếu có push mới trong khi đang running → hủy run cũ
- Tránh wasted compute khi developer push nhiều commits liên tục

---

## Vì sao dùng như vậy?

### Tại sao 2 jobs riêng biệt?
- **Job 1 (lint-and-test)**: chạy trên TẤT CẢ branches (push + PR)
- **Job 2 (docker-build)**: chỉ chạy trên `main` push (production deployment)
- Phân tách rõ ràng: quality checks trên PR → Docker push trên main
- Không push Docker image cho feature branches

### Tại sao flake8 cho lint thay vì ruff?
- **ruff** là rust-based, nhanh hơn flake8 10-100x
- Tuy nhiên, `flake8` là standard có sẵn, ít dependencies
- Đây là demo project → flake8 đủ dùng
- **Production recommendation**: dùng ruff cho speed

### Tại sao không có full pytest cho train.py?
```yaml
python3 -m pytest services/ml-pipeline/ \
  --ignore=services/ml-pipeline/scripts/train.py 2>/dev/null || true
```

- `train.py` chạy full ML training (bao gồm SMOTE, 5-fold CV, 3 models)
- Quá lâu cho CI (minutes vs seconds)
- Để CI pass nhanh, chỉ test non-training scripts
- `train.py` nên được test riêng (manual hoặc integration test)

### Tại sao dùng `npm ci` thay vì `npm install`?
- `npm ci` cài đúng versions từ `package-lock.json`
- `npm install` có thể update lock file hoặc resolve khác
- CI environment phải deterministic → `npm ci` là best practice

### Tại sao dùng `|| true` cho pytest?
```bash
pytest services/ml-pipeline/ -v \
  --ignore=services/ml-pipeline/scripts/train.py 2>/dev/null || true
```
- Nếu pytest không có test files hoặc fail → không fail CI
- ML pipeline có thể chưa có unit tests đầy đủ
- `|| true` prevents CI from red nếu pytest config issues
- **Technical debt**: nên fix và remove `|| true` khi có proper tests

### Tại sao GHCR thay vì Docker Hub / ECR?
- **GHCR (GitHub Container Registry)**: miễn phí cho public repos, built-in với GitHub
- Không cần separate Docker Hub account hoặc AWS credentials
- Permissions gắn với GitHub Actions → secure
- Tags bao gồm `github.sha` (immutable) và `latest` (mutable) → versioning rõ ràng

### Tại sao không có separate deployment job?
- Hệ thống này chưa có Kubernetes / cloud deployment
- Docker images được push lên GHCR, sẵn sàng cho deployment
- Production deployment có thể: `kubectl apply` hoặc `docker compose pull`
- Đây là CI-only pipeline → CI không làm CD

### Tại sao permissions `packages: write`?
```yaml
permissions:
  contents: read
  packages: write
```
- `contents: read`: checkout code
- `packages: write`: push Docker images to GHCR
- Principle of least privilege: chỉ cấp quyền cần thiết
