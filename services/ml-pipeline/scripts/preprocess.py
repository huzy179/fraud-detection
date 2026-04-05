"""
Credit Card Fraud Detection - Preprocessing Pipeline
Handles missing values, feature scaling, and class imbalance.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "raw", "creditcard.csv")


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw dataset."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and outliers."""
    # Check missing
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"Found {missing} missing values, dropping...")
        df = df.dropna()
    else:
        logger.info("No missing values found")

    # Amount should be non-negative
    df = df[df["Amount"] >= 0]
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale Time and Amount features separately (1 scaler each)."""
    time_scaler = StandardScaler()
    df["Time_scaled"] = time_scaler.fit_transform(df[["Time"]])

    amount_scaler = StandardScaler()
    df["Amount_scaled"] = amount_scaler.fit_transform(df[["Amount"]])

    # Save both scalers
    joblib.dump(time_scaler, os.path.join(PROCESSED_DIR, "time_scaler.joblib"))
    joblib.dump(amount_scaler, os.path.join(PROCESSED_DIR, "amount_scaler.joblib"))
    logger.info("Scalers saved.")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets, stratified on Class."""
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Train: {len(X_train)} ({y_train.sum()} fraud), "
        f"Test: {len(X_test)} ({y_test.sum()} fraud)"
    )
    return X_train, X_test, y_train, y_test


def handle_imbalance(X_train, y_train, strategy: str = "smote", random_state: int = 42):
    """
    Handle class imbalance using SMOTE or RandomUnderSampler.
    Returns resampled X and y.
    """
    if strategy == "smote":
        logger.info("Applying SMOTE oversampling...")
        sampler = SMOTE(sampling_strategy=0.5, random_state=random_state)
    elif strategy == "rus":
        logger.info("Applying RandomUnderSampler...")
        sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    logger.info(
        f"After resampling: {len(X_res)} samples "
        f"({y_res.sum()} fraud, {y_res.sum()/len(y_res)*100:.2f}%)"
    )
    return X_res, y_res


def save_processed_data(X_train, X_test, y_train, y_test):
    """Save processed datasets as Parquet."""
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.to_frame(),
        "y_test": y_test.to_frame(),
    }
    for name, df in data.items():
        path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        df.to_parquet(path)
        logger.info(f"Saved {name} to {path}")


def main():
    # 1. Load
    df = load_data()

    # 2. Clean
    df = clean_data(df)

    # 3. Scale
    df = scale_features(df)

    # 4. Drop original Time/Amount, keep scaled versions
    df = df.drop(columns=["Time", "Amount"])

    # 5. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 6. Handle imbalance (SMOTE)
    X_train_res, y_train_res = handle_imbalance(X_train, y_train, strategy="smote")

    # 7. Save
    save_processed_data(X_train_res, X_test, y_train_res, y_test)
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
