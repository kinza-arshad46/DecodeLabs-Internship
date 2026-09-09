"""
utils/data_loader.py
---------------------
Loads the credit card transactions dataset.

Looks for a real dataset at:
    data/raw/creditcard.csv        (Kaggle's standard filename), or
    data/raw/fraud_dataset.csv     (alternate name)

Falls back to a synthetic dataset with the identical schema and imbalance
ratio (~0.17% fraud) if neither file is found, so the app always runs.

NOTE ON CACHING: this module works whether or not Streamlit is installed.
When Streamlit is available, `st.cache_data` is used so the dataset is only
loaded/generated once per session. When it isn't (e.g. running `pytest` in
a plain Python environment), caching is skipped and the functions still
work normally. This keeps the data layer independently testable.
"""

import os
import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:  # pragma: no cover - only hit outside Streamlit runtime
    def cache_data(*dargs, **dkwargs):
        if dargs and callable(dargs[0]):
            return dargs[0]

        def decorator(func):
            return func
        return decorator

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
CANDIDATE_FILENAMES = ["creditcard.csv", "fraud_dataset.csv"]
N_FEATURES = 28  # V1..V28


@cache_data(show_spinner=False)
def _generate_synthetic_dataset(n_rows: int = 12000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset mimicking the real creditcard.csv schema:
    Time, V1..V28, Amount, Class - with the same ~0.17% fraud ratio.
    Fraud rows are shifted in feature space so models have real signal to
    learn from (mirrors the separability of V14/V17/V12 in the real data).
    """
    rng = np.random.default_rng(random_state)

    fraud_ratio = 0.00172
    n_fraud = max(10, int(n_rows * fraud_ratio))
    n_normal = n_rows - n_fraud

    time_col = np.sort(rng.integers(0, 172792, size=n_rows)).astype(float)

    normal_features = rng.normal(loc=0.0, scale=1.0, size=(n_normal, N_FEATURES))
    fraud_features = rng.normal(loc=2.5, scale=2.2, size=(n_fraud, N_FEATURES))
    fraud_features[:, 13] -= 6.0   # mimics V14
    fraud_features[:, 16] -= 4.0   # mimics V17
    fraud_features[:, 11] -= 3.5   # mimics V12

    normal_amount = np.round(np.abs(rng.gamma(shape=1.2, scale=45, size=n_normal)), 2)
    fraud_amount = np.round(np.abs(rng.gamma(shape=1.0, scale=110, size=n_fraud)), 2)

    features = np.vstack([normal_features, fraud_features])
    amounts = np.concatenate([normal_amount, fraud_amount])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

    columns = [f"V{i}" for i in range(1, N_FEATURES + 1)]
    df = pd.DataFrame(features, columns=columns)
    df["Amount"] = amounts
    df["Class"] = labels.astype(int)
    df.insert(0, "Time", time_col)

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


@cache_data(show_spinner=False)
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _find_raw_file():
    for name in CANDIDATE_FILENAMES:
        path = os.path.join(RAW_DATA_DIR, name)
        if os.path.exists(path):
            return path
    return None


def load_data():
    """Returns (df, is_real_data: bool)."""
    path = _find_raw_file()
    if path:
        try:
            df = _load_csv(path)
            expected_cols = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, N_FEATURES + 1)}
            if expected_cols.issubset(set(df.columns)):
                return df, True
        except Exception:
            pass
    return _generate_synthetic_dataset(), False


def dataset_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    fraud = int(df["Class"].sum())
    normal = total - fraud
    return {
        "total": total,
        "fraud": fraud,
        "normal": normal,
        "fraud_pct": round(fraud / total * 100, 2) if total else 0,
        "normal_pct": round(normal / total * 100, 2) if total else 0,
        "features": df.shape[1] - 1,
        "missing": int(df.isnull().sum().sum()),
    }


def save_processed_split(X_train, X_test, y_train, y_test):
    """
    Persists the train/test split to data/processed/processed_data.csv with
    a 'split' column, so the processed data is inspectable outside the app
    (e.g. for the tests, or for loading into a notebook).
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_df = X_train.copy()
    train_df["Class"] = y_train.values
    train_df["split"] = "train"
    test_df = X_test.copy()
    test_df["Class"] = y_test.values
    test_df["split"] = "test"
    combined = pd.concat([train_df, test_df], ignore_index=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")
    combined.to_csv(out_path, index=False)
    return out_path
