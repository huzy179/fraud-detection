"""
Credit Card Fraud Detection - Model Training.
Trains XGBoost and RandomForest, logs metrics, saves best model locally.
MLflow tracking is optional — gracefully skips if server is unavailable.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import mlflow
    import mlflow.xgboost
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
USE_MLFLOW = MLFLOW_TRACKING_URI not in ("", "false", "None")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def mlflow_safe(fn):
    """Decorator: run mlflow block only if USE_MLFLOW and server reachable."""
    def wrapper(*args, **kwargs):
        if not USE_MLFLOW or not MLFLOW_AVAILABLE:
            return None  # no-op
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"MLflow skipped: {e}")
            return None
    return wrapper


def load_processed_data():
    """Load preprocessed train/test data."""
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet")).squeeze()
    y_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet")).squeeze()
    logger.info(f"Loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred, y_proba, model_name: str) -> dict:
    """Compute and print evaluation metrics."""
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }
    logger.info(f"\n{'='*40}")
    logger.info(f"{model_name} Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    logger.info(f"{'='*40}")
    return metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost classifier."""
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "scale_pos_weight": pos_weight,
        "eval_metric": "aucpr",
        "random_state": 42,
    }

    run_name = "xgboost_fraud"
    model = xgb.XGBClassifier(**params)

    if USE_MLFLOW and MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("fraud_detection")
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)
                metrics = evaluate_model(y_test, y_pred, y_proba, "XGBoost")
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                model_path = os.path.join(MODELS_DIR, "xgboost_model.json")
                model.save_model(model_path)
                mlflow.log_artifact(model_path)
                mlflow.xgboost.log_model(model, "xgboost_model", registered_model_name="FraudDetector")
                logger.info("XGBoost model logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow unavailable ({e}), training without tracking...")
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            model_path = os.path.join(MODELS_DIR, "xgboost_model.json")
            model.save_model(model_path)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = evaluate_model(y_test, y_pred, y_proba, "XGBoost")
    else:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        model_path = os.path.join(MODELS_DIR, "xgboost_model.json")
        model.save_model(model_path)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = evaluate_model(y_test, y_pred, y_proba, "XGBoost")

    logger.info(f"XGBoost model saved to {model_path}")
    return model, metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train RandomForest classifier."""
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)

    if USE_MLFLOW and MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name="random_forest_fraud"):
                mlflow.log_params(params)
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                metrics = evaluate_model(y_test, y_pred, y_proba, "RandomForest")
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                model_path = os.path.join(MODELS_DIR, "rf_model.joblib")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)
                mlflow.sklearn.log_model(model, "rf_model", registered_model_name="FraudDetector")
                logger.info("RandomForest model logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow unavailable ({e}), training without tracking...")
            model.fit(X_train, y_train)
            model_path = os.path.join(MODELS_DIR, "rf_model.joblib")
            joblib.dump(model, model_path)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred, y_proba, "RandomForest")
    else:
        model.fit(X_train, y_train)
        model_path = os.path.join(MODELS_DIR, "rf_model.joblib")
        joblib.dump(model, model_path)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, y_proba, "RandomForest")

    logger.info(f"RandomForest model saved to {model_path}")
    return model, metrics


def main():
    logger.info("=" * 50)
    logger.info("Credit Card Fraud Detection — Model Training")
    logger.info(f"MLflow tracking: {'ON (' + MLFLOW_TRACKING_URI + ')' if USE_MLFLOW else 'OFF (local only)'}")
    logger.info("=" * 50)

    X_train, X_test, y_train, y_test = load_processed_data()

    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)

    best = "XGBoost" if xgb_metrics["average_precision"] >= rf_metrics["average_precision"] else "RandomForest"
    best_ap = max(xgb_metrics["average_precision"], rf_metrics["average_precision"])

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULT: Best model = {best} (average_precision={best_ap:.4f})")
    logger.info(f"Models saved in: {MODELS_DIR}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
