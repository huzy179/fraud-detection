"""
Unit tests for Fraud Detection API.
Run: pytest services/ml-serving/tests/ -v
Uses TestClient (in-memory, no real DB / model files needed).
"""

import os
import sys
import uuid
from unittest.mock import MagicMock

import pytest

# Set test DB path before importing app
os.environ["DATABASE_URL"] = f"sqlite:///./test_{uuid.uuid4().hex}.db"
os.environ["FRAUD_THRESHOLD"] = "0.5"

# ── Mock serving index so tests don't need data/processed/*.parquet ─────────────
mock_data = MagicMock()
mock_data.__getitem__ = lambda s, k: (
    [[0.0] * 30] * 5,   # data
    [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"],  # columns
)
mock_knn = MagicMock()
mock_knn.kneighbors.return_value = ([[0.5]], [[0]])  # distance, nearest_idx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch serving index BEFORE importing app
import main as _main_module  # noqa: E402
_main_module._serving_knn = mock_knn
_main_module._serving_data = mock_data
_main_module._serving_classes = [0, 0, 0, 1, 0]  # legit majority

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)

# ─── Helpers ───────────────────────────────────────────────────────────────────
VALID_TX = {
    "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378,
    "V5": -0.338, "V6": 0.462, "V7": 0.239, "V8": 0.098,
    "V9": -0.664, "V10": 0.463, "V11": -0.931, "V12": -2.304,
    "V13": 0.772, "V14": -1.576, "V15": -0.230, "V16": -0.050,
    "V17": -0.844, "V18": -0.380, "V19": 0.597, "V20": -0.697,
    "V21": -0.055, "V22": -0.270, "V23": -0.233, "V24": 0.140,
    "V25": -0.052, "V26": 0.265, "V27": 0.825, "V28": -0.068,
    "Amount": 149.52, "Time": 40680,
}

FRAUD_TX = {
    "V1": -3.043, "V2": 3.033, "V3": -1.833, "V4": 2.531,
    "V5": -3.136, "V6": 1.874, "V7": -4.731, "V8": 3.201,
    "V9": 1.716, "V10": -1.223, "V11": 2.301, "V12": -2.842,
    "V13": -2.011, "V14": 2.671, "V15": -1.099, "V16": -3.219,
    "V17": 3.914, "V18": -1.915, "V19": 1.199, "V20": -0.434,
    "V21": 0.570, "V22": -0.055, "V23": -2.074, "V24": 0.808,
    "V25": -0.253, "V26": 1.110, "V27": 0.921, "V28": -0.054,
    "Amount": 999.99, "Time": 85400,
}


# ─── Health ───────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_fields(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "threshold" in data
        assert data["threshold"] == 0.5

    def test_health_model_loaded(self):
        assert client.get("/health").json()["model_loaded"] is True


# ─── Predict ──────────────────────────────────────────────────────────────────
class TestPredict:
    def test_predict_returns_200(self):
        r = client.post("/predict", json={"transaction": VALID_TX})
        assert r.status_code == 200

    def test_predict_required_fields(self):
        data = client.post("/predict", json={"transaction": VALID_TX}).json()
        for field in ("fraud_probability", "is_fraud", "threshold", "confidence"):
            assert field in data

    def test_predict_prob_in_range(self):
        prob = client.post("/predict", json={"transaction": VALID_TX}).json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_confidence_valid(self):
        conf = client.post("/predict", json={"transaction": VALID_TX}).json()["confidence"]
        assert conf in ("high", "medium", "low")

    def test_predict_legit_low_prob(self):
        """Legit transaction should have fraud_prob < 0.5."""
        prob = client.post("/predict", json={"transaction": VALID_TX}).json()["fraud_probability"]
        assert prob < 0.5

    def test_predict_fraud_tx_responds(self):
        """Fraud transaction should return a valid prediction (not crash)."""
        prob = client.post("/predict", json={"transaction": FRAUD_TX}).json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_missing_field_returns_422(self):
        bad = {"transaction": {"V1": 1.0}}
        assert client.post("/predict", json=bad).status_code == 422

    def test_predict_negative_amount_returns_422(self):
        bad = {"transaction": {**VALID_TX, "Amount": -50}}
        assert client.post("/predict", json=bad).status_code == 422

    def test_predict_missing_time_uses_default(self):
        tx = {k: v for k, v in VALID_TX.items() if k != "Time"}
        r = client.post("/predict", json={"transaction": tx})
        assert r.status_code == 200

    def test_predict_large_amount(self):
        tx = {**VALID_TX, "Amount": 10000}
        r = client.post("/predict", json={"transaction": tx})
        assert r.status_code == 200
        assert "fraud_probability" in r.json()


# ─── Transactions (DB required) ────────────────────────────────────────────────
class TestTransactions:
    @pytest.fixture(autouse=True, scope="class")
    def ensure_db(self):
        """Ensure DB tables exist before running DB tests."""
        from main import Base, engine
        Base.metadata.create_all(bind=engine)

    def test_create_transaction_returns_201(self):
        r = client.post("/transactions", json=VALID_TX)
        assert r.status_code == 201

    def test_create_transaction_fields(self):
        data = client.post("/transactions", json=VALID_TX).json()
        for field in ("id", "fraud_probability", "is_fraud", "confidence", "created_at"):
            assert field in data

    def test_create_transaction_prob_range(self):
        prob = client.post("/transactions", json=VALID_TX).json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_list_transactions_returns_200(self):
        r = client.get("/transactions")
        assert r.status_code == 200

    def test_list_transactions_returns_list(self):
        assert isinstance(client.get("/transactions").json(), list)

    def test_list_transactions_limit(self):
        r = client.get("/transactions?limit=5")
        assert r.status_code == 200
        assert len(r.json()) <= 5

    def test_stats_returns_200(self):
        r = client.get("/transactions/stats")
        assert r.status_code == 200

    def test_stats_fields(self):
        data = client.get("/transactions/stats").json()
        for field in ("total_transactions", "fraud_count", "fraud_rate", "avg_fraud_probability"):
            assert field in data

    def test_stats_fraud_rate_is_pct(self):
        rate = client.get("/transactions/stats").json()["fraud_rate"]
        assert 0.0 <= rate <= 100.0

    def test_stats_positive_values(self):
        data = client.get("/transactions/stats").json()
        assert data["total_transactions"] >= 0
        assert data["fraud_count"] >= 0
        assert data["fraud_rate"] >= 0.0
        assert 0.0 <= data["avg_fraud_probability"] <= 1.0


# ─── Metrics ───────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_metrics_returns_200(self):
        assert client.get("/metrics").status_code == 200

    def test_metrics_is_prometheus_format(self):
        text = client.get("/metrics").text
        assert "fraud_api_requests_total" in text or "# HELP" in text

    def test_metrics_content_type(self):
        assert client.get("/metrics").headers["content-type"].startswith("text/plain")


# ─── Integration: Full Flow ──────────────────────────────────────────────────
class TestFullFlow:
    """Test the complete flow: create → predict → stats updated."""

    def test_flow_create_and_stats(self):
        # Create a transaction
        create_r = client.post("/transactions", json=VALID_TX)
        assert create_r.status_code == 201

        # Check stats updated
        stats = client.get("/transactions/stats").json()
        assert stats["total_transactions"] >= 1

    def test_flow_predict_vs_created_match(self):
        """Predicted probability should match created transaction probability."""
        create_r = client.post("/transactions", json=VALID_TX)
        assert create_r.status_code == 201
        created_prob = create_r.json()["fraud_probability"]

        predict_r = client.post("/predict", json={"transaction": VALID_TX})
        assert predict_r.status_code == 200
        predicted_prob = predict_r.json()["fraud_probability"]

        # Should be very close (same model, same data)
        assert abs(created_prob - predicted_prob) < 0.0001
