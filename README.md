# Credit Card Fraud Detection System

A real-time fraud detection system for credit card transactions, built with a clean microservices architecture.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Frontend   │────▶│  FastAPI     │────▶│  MLflow      │
│  (Next.js)  │     │  (API Server)│     │ (Model Reg)  │
│  :3000      │     │  :8000       │     └──────────────┘
└─────────────┘     └──────┬───────┘
                           │ PostgreSQL
                    ┌──────▼───────┐
                    │ Prometheus    │
                    └──────┬───────┘
                           │ Grafana :3002
```

## Tech Stack

| Component     | Technology                     |
|--------------|-------------------------------|
| API Server   | FastAPI + SQLAlchemy + XGBoost |
| ML Pipeline  | Python, scikit-learn, XGBoost  |
| Experiment Track | MLflow                      |
| Database     | PostgreSQL                    |
| Frontend     | Next.js                       |
| Monitoring   | Prometheus + Grafana          |
| CI/CD        | GitHub Actions + Docker        |

## Quick Start

### Step 1: Download Dataset

```bash
# Use direct download (no API key needed)
python data/scripts/download_data.py
```

The dataset will be saved to `data/raw/creditcard.csv`.

### Step 2: Install Dependencies

```bash
# ML Pipeline
cd services/ml-pipeline
pip install -r requirements.txt

# API Server
cd services/ml-serving
pip install -r requirements.txt

# Frontend
cd services/frontend
npm install
```

### Step 3: Train the Model

```bash
cd services/ml-pipeline

# Download dataset first
python ../../data/scripts/download_data.py

# Preprocess data
python scripts/preprocess.py

# Train models (starts MLflow on port 5000)
python scripts/train.py
```

### Step 4: Run Services

```bash
# Start MLflow (in one terminal)
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

# Start API Server (another terminal)
cd services/ml-serving
uvicorn main:app --reload --port 8000

# Start Frontend (another terminal)
cd services/frontend
npm run dev
```

### Step 5: Open Dashboard

- **Frontend:** http://localhost:3000
- **FastAPI Docs:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000
- **Grafana:** http://localhost:3002

## Docker Mode (Recommended)

```bash
docker-compose up --build
docker-compose up -d
```

## API Endpoints

| Method | Endpoint              | Description                        |
|--------|-----------------------|------------------------------------|
| GET    | `/health`             | Health check                       |
| POST   | `/predict`            | Predict fraud probability          |
| POST   | `/explain`            | SHAP-based explanation             |
| POST   | `/transactions`       | Create transaction → ML → save DB |
| GET    | `/transactions`       | List recent transactions           |
| GET    | `/transactions/stats` | Get fraud statistics               |
| GET    | `/metrics`            | Prometheus metrics                 |

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378,
      "V5": -0.338, "V6": 0.462, "V7": 0.239, "V8": 0.098,
      "V9": -0.664, "V10": 0.463, "V11": -0.931, "V12": -2.304,
      "V13": 0.772, "V14": -1.576, "V15": -0.230, "V16": -0.050,
      "V17": -0.844, "V18": -0.380, "V19": 0.597, "V20": -0.697,
      "V21": -0.055, "V22": -0.270, "V23": -0.233, "V24": 0.140,
      "V25": -0.052, "V26": 0.265, "V27": 0.825, "V28": -0.068,
      "Amount": 149.52, "Time": 40680
    }
  }'
```

## Environment Variables

| Variable               | Default                                        | Description          |
|-----------------------|------------------------------------------------|----------------------|
| `FRAUD_THRESHOLD`      | `0.5`                                          | Fraud threshold      |
| `DATABASE_URL`         | `postgresql://postgres:postgres@localhost:5432/` | PostgreSQL URL      |
| `MLFLOW_TRACKING_URI`  | `http://localhost:5000`                        | MLflow server URL    |
| `NEXT_PUBLIC_API_URL`  | `http://localhost:8000`                        | FastAPI URL          |

## Project Structure

```
fraud-detection/
├── docker-compose.yml           # All services orchestration
├── .github/workflows/ci.yml     # CI/CD pipeline
├── data/
│   ├── raw/                     # Raw dataset
│   ├── processed/               # Preprocessed data
│   └── scripts/download_data.py
├── notebooks/01_eda.ipynb       # EDA notebook
├── services/
│   ├── ml-pipeline/             # Data processing + training
│   │   ├── scripts/
│   │   │   ├── preprocess.py
│   │   │   └── train.py
│   │   └── Dockerfile
│   ├── ml-serving/              # FastAPI API Server
│   │   ├── main.py             # ML inference + DB + REST
│   │   └── Dockerfile
│   └── frontend/               # Next.js Dashboard
│       ├── pages/
│       └── Dockerfile
└── monitoring/
    └── prometheus.yml
```

## License

MIT
