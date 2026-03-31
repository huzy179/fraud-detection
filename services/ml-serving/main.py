"""
FastAPI — Credit Card Fraud Detection
Hợp nhất ML Inference + Transaction API + Database
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import shap

from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool

# ── Prometheus Metrics ─────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraud_api_requests_total", "Total API requests", ["endpoint", "method"]
)
REQUEST_LATENCY = Histogram(
    "fraud_api_latency_seconds", "API latency in seconds", ["endpoint"]
)
PREDICTION_GAUGE = Gauge("fraud_predictions_total", "Total predictions", ["prediction"])
FRAUD_RATE_GAUGE = Gauge("fraud_rate_estimated", "Estimated fraud rate")

# ── Database Setup ─────────────────────────────────────────────────────────────
def _get_database_url():
    url = os.getenv("DATABASE_URL", "")
    if url:
        return url
    # SQLite fallback for local dev (no Docker)
    import pathlib
    db_path = pathlib.Path(__file__).parent.parent.parent / "fraud_detection.db"
    return f"sqlite:///{db_path}"

DATABASE_URL = _get_database_url()
_use_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool if not _use_sqlite else None,
    echo=False,
    connect_args={"check_same_thread": False} if _use_sqlite else {},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class TransactionDB(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    amount = Column(Float, nullable=False)
    V1 = Column(Float); V2 = Column(Float); V3 = Column(Float); V4 = Column(Float)
    V5 = Column(Float); V6 = Column(Float); V7 = Column(Float); V8 = Column(Float)
    V9 = Column(Float); V10 = Column(Float); V11 = Column(Float); V12 = Column(Float)
    V13 = Column(Float); V14 = Column(Float); V15 = Column(Float); V16 = Column(Float)
    V17 = Column(Float); V18 = Column(Float); V19 = Column(Float); V20 = Column(Float)
    V21 = Column(Float); V22 = Column(Float); V23 = Column(Float); V24 = Column(Float)
    V25 = Column(Float); V26 = Column(Float); V27 = Column(Float); V28 = Column(Float)
    fraud_probability = Column(Float, default=0.0)
    is_fraud = Column(Boolean, default=False)
    confidence = Column(String(20), default="low")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── ML Model Loading ───────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "models"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))


def load_model():
    model_file = os.path.join(MODEL_PATH, "xgboost_model.json")
    if os.path.exists(model_file):
        model = xgb.XGBClassifier()
        model.load_model(model_file)
        print(f"Model loaded from {model_file}")
        return model
    raise RuntimeError(f"Model not found at {model_file}. Run train.py first.")


def load_scalers():
    """Load separate Time and Amount scalers."""
    time_scaler_file = os.path.join(DATA_DIR, "processed", "time_scaler.joblib")
    amount_scaler_file = os.path.join(DATA_DIR, "processed", "amount_scaler.joblib")
    time_scaler = joblib.load(time_scaler_file) if os.path.exists(time_scaler_file) else None
    amount_scaler = joblib.load(amount_scaler_file) if os.path.exists(amount_scaler_file) else None
    return time_scaler, amount_scaler


model = None
time_scaler = None
amount_scaler = None
EXPLAINER = None

try:
    model = load_model()
    time_scaler, amount_scaler = load_scalers()
    EXPLAINER = shap.Explainer(model) if model else None
    print(f"Model + scalers loaded. time_scaler={time_scaler is not None}, amount_scaler={amount_scaler is not None}")
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")


# ── Pydantic Models ────────────────────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0)
    Time: float = Field(default=0, ge=0)


class TransactionCreate(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0)
    Time: Optional[float] = Field(default=0, ge=0)


class TransactionResponse(BaseModel):
    id: str
    amount: float
    fraud_probability: float
    is_fraud: bool
    confidence: str
    created_at: datetime

    class Config:
        from_attributes = True


class TransactionStats(BaseModel):
    total_transactions: int
    fraud_count: int
    fraud_rate: float
    avg_fraud_probability: float


class PredictionRequest(BaseModel):
    transaction: TransactionFeatures


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    confidence: str


class ExplainRequest(BaseModel):
    transaction: TransactionFeatures


class ExplainResponse(BaseModel):
    fraud_probability: float
    shap_values: List[float]
    top_features: List[dict]


# ── ML Helper Functions ────────────────────────────────────────────────────────
def preprocess(tx: TransactionFeatures | TransactionCreate) -> np.ndarray:
    """Scale Time/Amount and build feature array."""
    d = tx.model_dump()
    time_val = d.pop("Time")
    amount_val = d.pop("Amount")

    if time_scaler and amount_scaler:
        time_scaled = float(time_scaler.transform([[time_val]])[0][0])
        amount_scaled = float(amount_scaler.transform([[amount_val]])[0][0])
    else:
        time_scaled = time_val
        amount_scaled = amount_val

    vals = [d[f"V{i}"] for i in range(1, 29)]
    vals += [time_scaled, amount_scaled]
    return np.array([vals])


def predict_fraud(features: np.ndarray) -> tuple[float, bool, str]:
    """Run model inference, return (probability, is_fraud, confidence)."""
    prob = float(model.predict_proba(features)[0][1])
    is_fraud = prob >= FRAUD_THRESHOLD
    confidence = "high" if prob >= 0.8 else "medium" if prob >= 0.5 else "low"
    return prob, is_fraud, confidence


# ── Lifespan (startup/shutdown) ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables ready.")
    except Exception as e:
        print(f"Database not available: {e}. Running without DB.")
    yield
    engine.dispose()
    print("Database connection closed.")


# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Unified API: ML Inference + Transaction Management",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ML Inference Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True, "threshold": FRAUD_THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """Predict fraud probability for a transaction."""
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()
    start = time.time()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = preprocess(req.transaction)
        prob, is_fraud, confidence = predict_fraud(features)
        PREDICTION_GAUGE.labels(prediction="fraud" if is_fraud else "legit").inc()
        return PredictionResponse(
            fraud_probability=round(prob, 6),
            is_fraud=is_fraud,
            threshold=FRAUD_THRESHOLD,
            confidence=confidence,
        )
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """SHAP-based explanation for a prediction."""
    if EXPLAINER is None or model is None:
        raise HTTPException(status_code=503, detail="SHAP Explainer not available")

    try:
        features = preprocess(req.transaction)
        prob = float(model.predict_proba(features)[0][1])
        shap_vals = EXPLAINER(features).values[0].tolist()
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]
        ranked = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
        return ExplainResponse(
            fraud_probability=round(prob, 6),
            shap_values=[round(v, 6) for v in shap_vals],
            top_features=[{"feature": f, "shap_value": round(s, 6)} for f, s in ranked[:5]],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Transaction Endpoints ──────────────────────────────────────────────────────
@app.post("/transactions", response_model=TransactionResponse, status_code=201)
async def create_transaction(tx: TransactionCreate, db: Session = Depends(get_db)):
    """Create a transaction → run fraud prediction → save to database."""
    REQUEST_COUNT.labels(endpoint="/transactions", method="POST").inc()
    start = time.time()

    # Run ML prediction
    fraud_prob = 0.0
    is_fraud = False
    confidence = "low"

    if model is not None:
        try:
            features = preprocess(tx)
            fraud_prob, is_fraud, confidence = predict_fraud(features)
            PREDICTION_GAUGE.labels(prediction="fraud" if is_fraud else "legit").inc()
        except Exception as e:
            print(f"ML prediction failed: {e}")

    # Save to DB
    db_tx = TransactionDB(
        id=str(uuid.uuid4()),
        amount=tx.Amount,
        V1=tx.V1, V2=tx.V2, V3=tx.V3, V4=tx.V4, V5=tx.V5,
        V6=tx.V6, V7=tx.V7, V8=tx.V8, V9=tx.V9, V10=tx.V10,
        V11=tx.V11, V12=tx.V12, V13=tx.V13, V14=tx.V14, V15=tx.V15,
        V16=tx.V16, V17=tx.V17, V18=tx.V18, V19=tx.V19, V20=tx.V20,
        V21=tx.V21, V22=tx.V22, V23=tx.V23, V24=tx.V24, V25=tx.V25,
        V26=tx.V26, V27=tx.V27, V28=tx.V28,
        fraud_probability=fraud_prob,
        is_fraud=is_fraud,
        confidence=confidence,
    )

    try:
        db.add(db_tx)
        db.commit()
        db.refresh(db_tx)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        db.close()

    REQUEST_LATENCY.labels(endpoint="/transactions").observe(time.time() - start)
    return db_tx


@app.get("/transactions", response_model=List[TransactionResponse])
async def list_transactions(
    limit: int = Query(default=100, le=1000),
    db: Session = Depends(get_db),
):
    """List recent transactions."""
    REQUEST_COUNT.labels(endpoint="/transactions", method="GET").inc()
    txs = db.query(TransactionDB)\
        .order_by(TransactionDB.created_at.desc())\
        .limit(limit)\
        .all()
    return txs


@app.get("/transactions/stats", response_model=TransactionStats)
async def get_stats(db: Session = Depends(get_db)):
    """Get fraud statistics."""
    REQUEST_COUNT.labels(endpoint="/transactions/stats", method="GET").inc()
    total = db.query(TransactionDB).count()
    fraud = db.query(TransactionDB).filter(TransactionDB.is_fraud == True).count()
    result = db.query(TransactionDB.fraud_probability).all()
    avg_prob = sum(r[0] for r in result) / len(result) if result else 0.0
    return TransactionStats(
        total_transactions=total,
        fraud_count=fraud,
        fraud_rate=round((fraud / total * 100), 4) if total > 0 else 0.0,
        avg_fraud_probability=round(avg_prob, 6),
    )


# ── Prometheus Metrics ──────────────────────────────────────────────────────────
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
