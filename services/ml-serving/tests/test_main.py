"""
Unit tests for FastAPI main.py
Run: pytest services/ml-serving/tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import sys, os

# Add parent dir to path so we can import main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock model loading before importing app
os.environ["FRAUD_THRESHOLD"] = "0.5"

from main import app

client = TestClient(app)


# ─── Health ──────────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_required_fields(self):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "threshold" in data

    def test_health_threshold(self):
        r = client.get("/health")
        assert r.json()["threshold"] == 0.5


# ─── Predict ───────────────────────────────────────────────────────────────────
VALID_TX = {
    "transaction": {
        "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378,
        "V5": -0.338, "V6": 0.462, "V7": 0.239, "V8": 0.098,
        "V9": -0.664, "V10": 0.463, "V11": -0.931, "V12": -2.304,
        "V13": 0.772, "V14": -1.576, "V15": -0.230, "V16": -0.050,
        "V17": -0.844, "V18": -0.380, "V19": 0.597, "V20": -0.697,
        "V21": -0.055, "V22": -0.270, "V23": -0.233, "V24": 0.140,
        "V25": -0.052, "V26": 0.265, "V27": 0.825, "V28": -0.068,
        "Amount": 149.52, "Time": 40680,
    }
}

LEGIT_TX = {
    "transaction": {
        "V1": 0.5, "V2": 0.3, "V3": -0.1, "V4": 0.8,
        "V5": 0.1, "V6": -0.2, "V7": 0.1, "V8": 0.05,
        "V9": 0.2, "V10": -0.1, "V11": 0.3, "V12": -0.1,
        "V13": 0.1, "V14": 0.2, "V15": -0.1, "V16": 0.1,
        "V17": -0.2, "V18": 0.1, "V19": -0.1, "V20": 0.05,
        "V21": -0.05, "V22": 0.02, "V23": 0.01, "V24": -0.02,
        "V25": 0.01, "V26": -0.01, "V27": 0.005, "V28": -0.005,
        "Amount": 50.00, "Time": 10000,
    }
}

FRAUD_TX = {
    "transaction": {
        "V1": -3.043, "V2": 3.033, "V3": -1.833, "V4": 2.531,
        "V5": -3.136, "V6": 1.874, "V7": -4.731, "V8": 3.201,
        "V9": 1.716, "V10": -1.223, "V11": 2.301, "V12": -2.842,
        "V13": -2.011, "V14": 2.671, "V15": -1.099, "V16": -3.219,
        "V17": 3.914, "V18": -1.915, "V19": 1.199, "V20": -0.434,
        "V21": 0.570, "V22": -0.055, "V23": -2.074, "V24": 0.808,
        "V25": -0.253, "V26": 1.110, "V27": 0.921, "V28": -0.054,
        "Amount": 999.99, "Time": 85400,
    }
}


class TestPredict:
    def test_predict_returns_200(self):
        r = client.post("/predict", json=VALID_TX)
        assert r.status_code == 200

    def test_predict_has_required_fields(self):
        r = client.post("/predict", json=VALID_TX)
        data = r.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "threshold" in data
        assert "confidence" in data

    def test_predict_probability_range(self):
        r = client.post("/predict", json=VALID_TX)
        prob = r.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_confidence_valid(self):
        r = client.post("/predict", json=VALID_TX)
        conf = r.json()["confidence"]
        assert conf in ("high", "medium", "low")

    def test_legit_transaction_low_probability(self):
        r = client.post("/predict", json=LEGIT_TX)
        prob = r.json()["fraud_probability"]
        # Legit transaction should have low fraud probability
        assert prob < 0.5

    def test_predict_missing_field_returns_422(self):
        bad = {"transaction": {"V1": 1.0}}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_invalid_amount_returns_422(self):
        bad = {**VALID_TX}
        bad["transaction"]["Amount"] = -100  # negative not allowed
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


# ─── Transactions ───────────────────────────────────────────────────────────────
class TestTransactions:
    def test_create_transaction_returns_201(self):
        r = client.post("/transactions", json=VALID_TX["transaction"])
        assert r.status_code == 201

    def test_create_transaction_response_fields(self):
        r = client.post("/transactions", json=VALID_TX["transaction"])
        data = r.json()
        assert "id" in data
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "confidence" in data
        assert "created_at" in data

    def test_create_transaction_probability_range(self):
        r = client.post("/transactions", json=VALID_TX["transaction"])
        prob = r.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_list_transactions_returns_200(self):
        r = client.get("/transactions")
        assert r.status_code == 200

    def test_list_transactions_returns_array(self):
        r = client.get("/transactions")
        assert isinstance(r.json(), list)

    def test_list_transactions_limit_param(self):
        r = client.get("/transactions?limit=5")
        assert r.status_code == 200
        assert len(r.json()) <= 5

    def test_stats_returns_200(self):
        r = client.get("/transactions/stats")
        assert r.status_code == 200

    def test_stats_has_required_fields(self):
        r = client.get("/transactions/stats")
        data = r.json()
        assert "total_transactions" in data
        assert "fraud_count" in data
        assert "fraud_rate" in data
        assert "avg_fraud_probability" in data

    def test_stats_fraud_rate_is_percentage(self):
        r = client.get("/transactions/stats")
        rate = r.json()["fraud_rate"]
        assert 0.0 <= rate <= 100.0


# ─── Metrics ────────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_metrics_returns_200(self):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_is_prometheus_format(self):
        r = client.get("/metrics")
        text = r.text
        assert "fraud_api_requests_total" in text or "# HELP" in text
