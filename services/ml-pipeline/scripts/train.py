"""
Credit Card Fraud Detection - Improved Model Training.
Features:
  - XGBoost + LightGBM + RandomForest
  - 5-fold Cross-Validation
  - Threshold tuning (optimal F1)
  - MLflow tracking (optional)
Run: python scripts/train.py
"""

import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from sklearn.metrics import precision_recall_curve

try:
    import mlflow
    import mlflow.xgboost
    import mlflow.sklearn
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
USE_MLFLOW = MLFLOW_TRACKING_URI not in ("", "false", "None") and MLFLOW_AVAILABLE

_artifact_root = os.getenv(
    "MLFLOW_ARTIFACT_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mlflow_artifacts")
)
os.environ["MLFLOW_ARTIFACT_ROOT"] = _artifact_root

_script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/app/data/processed"):
    PROCESSED_DIR = "/app/data/processed"
    MODELS_DIR = "/app/models"
else:
    PROCESSED_DIR = os.path.normpath(os.path.join(_script_dir, "..", "..", "data", "processed"))
    MODELS_DIR = os.path.normpath(os.path.join(_script_dir, "..", "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)


def wait_for_mlflow(uri: str, timeout: int = 60, interval: int = 5) -> bool:
    """
    Retry connecting to MLflow server. Returns True if connected, False otherwise.
    """
    if not USE_MLFLOW:
        return False

    import urllib.request
    endpoint = uri.rstrip("/") + "/health"
    logger.info(f"Waiting for MLflow server at {endpoint} ...")

    for attempt in range(1, timeout // interval + 1):
        try:
            urllib.request.urlopen(endpoint, timeout=5)
            logger.info(f"✅ MLflow server connected ({endpoint})")
            return True
        except Exception:
            pass
        remaining = (timeout // interval - attempt) * interval
        logger.info(f"  Attempt {attempt}: MLflow not ready, retrying in {interval}s ... ({remaining}s left)")
        time.sleep(interval)

    logger.error(f"❌ MLflow server not reachable after {timeout}s at {endpoint}")
    logger.warning("   Falling back to local-only training (no MLflow tracking)")
    return False


def load_data():
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet")).squeeze()
    y_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet")).squeeze()
    logger.info(f"Loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1 score."""
    thresholds, precisions, recalls, f1s = [], [], [], []
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_proba >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        thresholds.append(t)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    logger.info(f"Optimal threshold: {best_t:.2f} (F1={best_f1:.4f})")
    return best_t


def evaluate(y_true, y_pred, y_proba, name, threshold=None):
    """Compute and log metrics."""
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }
    logger.info(f"\n{'='*50}")
    logger.info(f"  {name} {'(threshold=' + str(round(threshold, 2)) + ')' if threshold else ''}")
    for k, v in metrics.items():
        logger.info(f"  {k:20s}: {v:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    logger.info(f"{'='*50}\n")
    return metrics


def train_with_cv(model_cfg, model_name, X_train, y_train, X_test, y_test):
    """Train with 5-fold CV + threshold tuning."""
    logger.info(f"\n{'#'*60}")
    logger.info(f"# Training {model_name} with 5-fold CV")
    logger.info(f"{'#'*60}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {k: [] for k in ["precision", "recall", "f1", "roc_auc", "avg_precision"]}

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        if model_name == "XGBoost":
            model = xgb.XGBClassifier(**model_cfg, eval_metric="aucpr")
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == "LightGBM":
            lgb_cfg = {k: v for k, v in model_cfg.items() if k != "verbose"}
            model = lgb.LGBMClassifier(**lgb_cfg, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        else:
            model = RandomForestClassifier(**model_cfg)
            model.fit(X_tr, y_tr)

        y_proba_val = model.predict_proba(X_val)[:, 1]
        y_pred_val = (y_proba_val >= 0.5).astype(int)

        for k, fn in [("precision", precision_score), ("recall", recall_score),
                       ("f1", f1_score), ("roc_auc", roc_auc_score),
                       ("avg_precision", average_precision_score)]:
            try:
                cv_scores[k].append(fn(y_val, y_pred_val, zero_division=0))
            except TypeError:
                cv_scores[k].append(fn(y_val, y_pred_val))

    logger.info(f"  CV Results (mean ± std):")
    for k, v in cv_scores.items():
        logger.info(f"    {k:20s}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    # Final model on full training data
    if model_name == "XGBoost":
        final_model = xgb.XGBClassifier(**model_cfg, eval_metric="aucpr")
        final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elif model_name == "LightGBM":
        lgb_cfg = {k: v for k, v in model_cfg.items() if k != "verbose"}
        final_model = lgb.LGBMClassifier(**lgb_cfg, verbose=-1)
        final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    else:
        final_model = RandomForestClassifier(**model_cfg)
        final_model.fit(X_train, y_train)

    # Predict on test set
    y_proba_test = final_model.predict_proba(X_test)[:, 1]

    # Default eval at 0.5
    y_pred_test = (y_proba_test >= 0.5).astype(int)
    metrics_default = evaluate(y_test, y_pred_test, y_proba_test,
                               f"{model_name} (threshold=0.5)", threshold=0.5)

    # Optimal threshold
    optimal_t = find_optimal_threshold(y_test, y_proba_test)
    y_pred_opt = (y_proba_test >= optimal_t).astype(int)
    metrics_opt = evaluate(y_test, y_pred_opt, y_proba_test,
                          f"{model_name} (optimal threshold)", threshold=optimal_t)

    # Save model
    if model_name == "XGBoost":
        path = os.path.join(MODELS_DIR, "xgboost_model.json")
        final_model.save_model(path)
    elif model_name == "LightGBM":
        path = os.path.join(MODELS_DIR, "lgbm_model.txt")
        final_model.booster_.save_model(path)
    else:
        path = os.path.join(MODELS_DIR, "rf_model.joblib")
        joblib.dump(final_model, path)

    logger.info(f"Saved: {path}")
    return final_model, metrics_opt, optimal_t


def _train_with_mlflow(model_cfg, model_name, X_train, y_train, X_test, y_test):
    """Wrapper: runs train_with_cv inside an MLflow active run, logs model + metrics."""
    if USE_MLFLOW:
        try:
            with mlflow.start_run(run_name=model_name) as run:
                mlflow.set_tag("model_type", model_name)

                # Train first to get the model object
                _, metrics_opt, optimal_t = train_with_cv(model_cfg, model_name, X_train, y_train, X_test, y_test)

                # Re-train inside MLflow run to get a proper model object
                if model_name == "XGBoost":
                    model = xgb.XGBClassifier(**model_cfg, eval_metric="aucpr")
                    model.fit(X_train, y_train, verbose=False)
                elif model_name == "LightGBM":
                    lgb_cfg = {k: v for k, v in model_cfg.items() if k != "verbose"}
                    model = lgb.LGBMClassifier(**lgb_cfg, verbose=-1)
                    model.fit(X_train, y_train)
                else:
                    model = RandomForestClassifier(**model_cfg)
                    model.fit(X_train, y_train)

                # Log params (skip booster-specific keys)
                skip_keys = {"booster", "device"}
                clean_cfg = {k: v for k, v in model_cfg.items() if k not in skip_keys}
                mlflow.log_params(clean_cfg)

                # Log best metrics
                mlflow.log_metrics(metrics_opt)

                # Log model artifact using log_artifact (writes directly to run's artifact directory)
                try:
                    import joblib as _jlib
                    _tmp_path = "/tmp/model.joblib"
                    _jlib.dump(model, _tmp_path)
                    mlflow.log_artifact(_tmp_path, artifact_path=model_name.lower())
                    os.remove(_tmp_path)
                    logger.info(f"  ✅ MLflow: {model_name} artifact saved to run directory")
                except Exception as art_err:
                    import traceback as _tb
                    logger.warning(f"  ⚠️  Artifact logging failed for {model_name}: {art_err}")
                    logger.warning(f"     Model file saved locally at {MODELS_DIR}")


                logger.info(f"  ✅ MLflow: {model_name} run complete")
                return model, metrics_opt, optimal_t
        except Exception as e:
            logger.error(f"  ❌ MLflow logging failed for {model_name}: {e}")
            logger.warning(f"     Falling back to local training for {model_name}")
    return train_with_cv(model_cfg, model_name, X_train, y_train, X_test, y_test)


def main():
    logger.info("=" * 60)
    logger.info("  Credit Card Fraud Detection — Model Training")
    logger.info("=" * 60)
    logger.info(f"  MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"  MLflow available locally: {MLFLOW_AVAILABLE}")

    if USE_MLFLOW:
        logger.info(f"  MLflow enabled — attempting to connect ...")
        mlflow_ready = wait_for_mlflow(MLFLOW_TRACKING_URI, timeout=60, interval=5)
        if mlflow_ready:
            try:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                mlflow.set_experiment("fraud_detection_improved")
                logger.info(f"✅ MLflow tracking active — runs will be logged")
            except Exception as e:
                logger.warning(f"❌ MLflow connection failed: {e}")
                globals()["USE_MLFLOW"] = False
        else:
            logger.warning("⚠️  MLflow server not reachable — training local only")
            globals()["USE_MLFLOW"] = False
    else:
        logger.info("  MLflow disabled (USE_MLFLOW=False) — training local only")

    X_train, X_test, y_train, y_test = load_data()

    # ── XGBoost ──
    xgb_params = dict(
        max_depth=6, learning_rate=0.05, n_estimators=300,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42,
    )
    xgb_model, xgb_metrics, xgb_t = _train_with_mlflow(
        xgb_params, "XGBoost", X_train, y_train, X_test, y_test
    )

    # ── LightGBM ──
    lgb_params = dict(
        max_depth=6, learning_rate=0.05, n_estimators=300,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42,
    )
    lgb_model, lgb_metrics, lgb_t = _train_with_mlflow(
        lgb_params, "LightGBM", X_train, y_train, X_test, y_test
    )

    # ── RandomForest ──
    rf_params = dict(
        n_estimators=200, max_depth=12, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_model, rf_metrics, rf_t = _train_with_mlflow(
        rf_params, "RandomForest", X_train, y_train, X_test, y_test
    )

    # ── Summary ──
    all_models = [
        ("XGBoost", xgb_metrics, xgb_t),
        ("LightGBM", lgb_metrics, lgb_t),
        ("RandomForest", rf_metrics, rf_t),
    ]
    best_name, best_metrics, best_t = max(all_models, key=lambda x: x[1]["f1"])

    logger.info("=" * 60)
    logger.info("  FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg Prec':>10} {'Thresh':>8}")
    logger.info("-" * 68)
    for name, m, t in all_models:
        marker = " ⭐ BEST" if name == best_name else ""
        logger.info(f"{name:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                    f"{m['f1']:>10.4f} {m['average_precision']:>10.4f} {t:>8.2f}{marker}")

    # Save best config
    best_config = {
        "model": best_name,
        "threshold": round(best_t, 2),
        "f1": round(best_metrics["f1"], 4),
        "precision": round(best_metrics["precision"], 4),
        "recall": round(best_metrics["recall"], 4),
        "roc_auc": round(best_metrics["roc_auc"], 4),
        "average_precision": round(best_metrics["average_precision"], 4),
    }
    with open(os.path.join(MODELS_DIR, "best_config.json"), "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"\nBest config saved: {best_config}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
