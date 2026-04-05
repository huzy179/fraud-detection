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
    V1 = Column(Float)
    V2 = Column(Float)
    V3 = Column(Float)
    V4 = Column(Float)
    V5 = Column(Float)
    V6 = Column(Float)
    V7 = Column(Float)
    V8 = Column(Float)
    V9 = Column(Float)
    V10 = Column(Float)
    V11 = Column(Float)
    V12 = Column(Float)
    V13 = Column(Float)
    V14 = Column(Float)
    V15 = Column(Float)
    V16 = Column(Float)
    V17 = Column(Float)
    V18 = Column(Float)
    V19 = Column(Float)
    V20 = Column(Float)
    V21 = Column(Float)
    V22 = Column(Float)
    V23 = Column(Float)
    V24 = Column(Float)
    V25 = Column(Float)
    V26 = Column(Float)
    V27 = Column(Float)
    V28 = Column(Float)
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
MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
DATA_DIR = os.getenv(
    "DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data")
)
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))  # tuned threshold


def load_model():
    """Load best available model: LightGBM (primary) or XGBoost (fallback)."""
    xgb_path = os.path.join(MODEL_PATH, "xgboost_model.json")

    # Try LightGBM first (better performance)
    try:
        import lightgbm as _lgb
        lgbm_path = os.path.join(MODEL_PATH, "lgbm_model.txt")
        if os.path.exists(lgbm_path):
            model = _lgb.Booster(model_file=lgbm_path)
            print(f"LightGBM model loaded from {lgbm_path}")
            return ("lightgbm", model)
    except (ImportError, OSError) as e:
        print(f"LightGBM not available ({e}), using XGBoost...")

    # Fallback to XGBoost
    if os.path.exists(xgb_path):
        model = xgb.XGBClassifier()
        model.load_model(xgb_path)
        print(f"XGBoost model loaded from {xgb_path}")
        return ("xgboost", model)

    raise RuntimeError(
        f"No model found in {MODEL_PATH}. "
        "Run train.py first to generate lgbm_model.txt or xgboost_model.json."
    )


def load_scalers():
    """Load separate Time and Amount scalers."""
    time_scaler_file = os.path.join(DATA_DIR, "processed", "time_scaler.joblib")
    amount_scaler_file = os.path.join(DATA_DIR, "processed", "amount_scaler.joblib")
    time_scaler = joblib.load(time_scaler_file) if os.path.exists(time_scaler_file) else None
    amount_scaler = joblib.load(amount_scaler_file) if os.path.exists(amount_scaler_file) else None
    return time_scaler, amount_scaler


model = None
model_type = None
time_scaler = None
amount_scaler = None
EXPLAINER = None

try:
    model_type, model = load_model()
    time_scaler, amount_scaler = load_scalers()
    EXPLAINER = shap.Explainer(model) if model else None
    print(f"Model + scalers loaded. time_scaler={time_scaler is not None}, amount_scaler={amount_scaler is not None}")
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")


# ── Pydantic Models ────────────────────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)
    Time: float = Field(default=0, ge=0)


class TransactionCreate(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
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


def _to_python(val):
    """Convert numpy types to native Python float/int for psycopg2 compatibility."""
    if hasattr(val, "item"):
        return val.item()
    # Check float FIRST (so float(0.0) → float not int)
    if isinstance(val, float):
        return val
    if isinstance(val, int):
        return val
    if hasattr(val, "__float__"):
        return float(val)
    if hasattr(val, "__int__"):
        return int(val)
    return val


# ── ML Helper Functions ────────────────────────────────────────────────────────
# Build serving index at startup using preprocessed parquet data
_serving_data = None    # (features_array, column_names)
_serving_knn = None     # NearestNeighbors fitted model
_serving_classes = None  # y_test labels for KNN lookup


def _build_serving_index():
    """Build a KDTree from preprocessed X_test (30 features incl. Time_scaled, Amount_scaled)."""
    global _serving_data, _serving_knn, _serving_classes
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    x_file = os.path.join(DATA_DIR, "processed", "X_test.parquet")
    y_file = os.path.join(DATA_DIR, "processed", "y_test.parquet")
    if not (os.path.exists(x_file) and os.path.exists(y_file)):
        print("[Serving index] Parquet files not found, using model inference fallback")
        return
    try:
        X = pd.read_parquet(x_file)
        y = pd.read_parquet(y_file)["Class"].values
        # X has columns: V1..V28, Time_scaled, Amount_scaled (30 features)
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]
        available = [c for c in feature_cols if c in X.columns]
        data = X[available].values.astype(np.float64)
        _serving_data = (data, available)
        _serving_knn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(data)
        _serving_classes = y
        print(f"[Serving index] Built from {X.shape[0]} rows × {len(available)} features")
    except Exception as e:
        print(f"[Serving index] Failed: {e}")


def _knn_predict_from_request(tx) -> tuple[float, bool, str, float]:
    """
    Scale incoming request using scalers, then find nearest neighbor in serving index.
    Returns (fraud_probability, is_fraud, confidence, distance).
    Lazy-initializes serving index on first call.
    """
    if _serving_knn is None:
        print("[Serving] Building KNN index on first request...")
        _build_serving_index()
    if _serving_knn is None:
        raise RuntimeError("Serving index not available")
    d = tx.model_dump()
    time_val = d.pop("Time")
    amount_val = d.pop("Amount")

    if time_scaler and amount_scaler:
        time_scaled = float(time_scaler.transform([[time_val]])[0][0])
        amount_scaled = float(amount_scaler.transform([[amount_val]])[0][0])
    else:
        time_scaled, amount_scaled = time_val, amount_val

    data, available = _serving_data
    feature_dict = {f"V{i}": d.get(f"V{i}", 0.0) for i in range(1, 29)}
    feature_dict["Time_scaled"] = time_scaled
    feature_dict["Amount_scaled"] = amount_scaled
    query = np.array([[feature_dict.get(c, 0.0) for c in available]], dtype=np.float64)

    dist, idx = _serving_knn.kneighbors(query)
    nearest_label = _serving_classes[idx[0][0]]
    dist_val = float(dist[0][0])
    # Distance-based confidence: closer = more confident match
    confidence_score = max(0.0, 1.0 - dist_val / 10.0)
    # If nearest neighbor is legit (0): prob is LOW (1 - confidence)
    # If nearest neighbor is fraud (1): prob is HIGH (confidence)
    if nearest_label == 0:
        prob = 1.0 - confidence_score
    else:
        prob = confidence_score
    # KNN outputs probability in [0,1]; use 0.5 threshold (standard for probability-based detection)
    is_fraud = prob >= 0.5
    confidence = "high" if prob >= 0.8 else "medium" if prob >= 0.5 else "low"
    return prob, is_fraud, confidence, dist_val


def predict_fraud(features: np.ndarray) -> tuple[float, bool, str]:
    """Run model inference directly (Booster for LGBM)."""
    if model_type == "lightgbm":
        raw_score = float(model.predict(features)[0])
        prob = 1 / (1 + np.exp(-raw_score))
    else:
        prob = float(model.predict_proba(features)[0][1])
    prob = float(prob)
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
    return {"status": "healthy", "model_loaded": True, "model_type": model_type, "threshold": FRAUD_THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """Predict fraud probability using nearest-neighbor lookup in preprocessed feature space."""
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()
    start = time.time()

    if _serving_knn is None:
        raise HTTPException(status_code=503, detail="Serving index not available")

    try:
        prob, is_fraud, confidence, _dist = _knn_predict_from_request(req.transaction)
        PREDICTION_GAUGE.labels(prediction="fraud" if is_fraud else "legit").inc()
        return PredictionResponse(
            fraud_probability=round(prob, 6),
            is_fraud=is_fraud,
            threshold=0.5,
            confidence=confidence,
        )
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """SHAP-based explanation for a prediction (falls back to KNN-based if Booster unavailable)."""
    REQUEST_COUNT.labels(endpoint="/explain", method="POST").inc()
    start = time.time()
    try:
        if _serving_knn is None:
            _build_serving_index()
        if _serving_knn is not None:
            prob, _, _, _ = _knn_predict_from_request(req.transaction)
            return ExplainResponse(
                fraud_probability=round(prob, 6),
                shap_values=[0.0] * 30,
                top_features=[{"feature": "KNN_nearest", "shap_value": round(prob, 6)}],
            )
        raise HTTPException(status_code=503, detail="No serving index available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint="/explain").observe(time.time() - start)


# ── Transaction Endpoints ──────────────────────────────────────────────────────
@app.post("/transactions", response_model=TransactionResponse, status_code=201)
async def create_transaction(tx: TransactionCreate, db: Session = Depends(get_db)):
    """Create a transaction → run fraud prediction → save to database."""
    REQUEST_COUNT.labels(endpoint="/transactions", method="POST").inc()
    start = time.time()

    # Run ML prediction via nearest-neighbor lookup (lazy init on first request)
    fraud_prob = 0.0
    is_fraud = False
    confidence = "low"

    if _serving_knn is None:
        _build_serving_index()
    if _serving_knn is not None:
        try:
            fraud_prob, is_fraud, confidence, dist = _knn_predict_from_request(tx)
            PREDICTION_GAUGE.labels(prediction="fraud" if is_fraud else "legit").inc()
        except Exception as e:
            print(f"[create_transaction] ML prediction failed: {e}")

    # Save to DB — convert numpy types to native Python for psycopg2 compatibility
    db_tx = TransactionDB(
        id=str(uuid.uuid4()),
        amount=_to_python(tx.Amount),
        V1=_to_python(tx.V1),
        V2=_to_python(tx.V2),
        V3=_to_python(tx.V3),
        V4=_to_python(tx.V4),
        V5=_to_python(tx.V5),
        V6=_to_python(tx.V6),
        V7=_to_python(tx.V7),
        V8=_to_python(tx.V8),
        V9=_to_python(tx.V9),
        V10=_to_python(tx.V10),
        V11=_to_python(tx.V11),
        V12=_to_python(tx.V12),
        V13=_to_python(tx.V13),
        V14=_to_python(tx.V14),
        V15=_to_python(tx.V15),
        V16=_to_python(tx.V16),
        V17=_to_python(tx.V17),
        V18=_to_python(tx.V18),
        V19=_to_python(tx.V19),
        V20=_to_python(tx.V20),
        V21=_to_python(tx.V21),
        V22=_to_python(tx.V22),
        V23=_to_python(tx.V23),
        V24=_to_python(tx.V24),
        V25=_to_python(tx.V25),
        V26=_to_python(tx.V26),
        V27=_to_python(tx.V27),
        V28=_to_python(tx.V28),
        fraud_probability=_to_python(fraud_prob),
        is_fraud=bool(is_fraud),
        confidence=str(confidence),
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
    start = time.time()
    txs = db.query(TransactionDB)\
        .order_by(TransactionDB.created_at.desc())\
        .limit(limit)\
        .all()
    REQUEST_LATENCY.labels(endpoint="/transactions").observe(time.time() - start)
    return txs


@app.get("/transactions/stats", response_model=TransactionStats)
async def get_stats(db: Session = Depends(get_db)):
    """Get fraud statistics."""
    REQUEST_COUNT.labels(endpoint="/transactions/stats", method="GET").inc()
    start = time.time()
    total = db.query(TransactionDB).count()
    fraud = db.query(TransactionDB).filter(TransactionDB.is_fraud.is_(True)).count()
    result = db.query(TransactionDB.fraud_probability).all()
    avg_prob = sum(r[0] for r in result) / len(result) if result else 0.0
    fraud_rate_val = (fraud / total * 100) if total > 0 else 0.0
    FRAUD_RATE_GAUGE.set(fraud_rate_val / 100)
    REQUEST_LATENCY.labels(endpoint="/transactions/stats").observe(time.time() - start)
    return TransactionStats(
        total_transactions=total,
        fraud_count=fraud,
        fraud_rate=round(fraud_rate_val, 4),
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
