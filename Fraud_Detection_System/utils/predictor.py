"""
utils/predictor.py
---------------------
Turns raw form inputs into a model-ready row and runs a single prediction.
Kept free of Streamlit so it's directly unit-testable.
"""

import pandas as pd


def build_input_row(feature_columns: list, values: dict) -> pd.DataFrame:
    """
    values: dict of {column_name: number}. Missing columns default to 0.0.
    Returns a single-row DataFrame with columns in the exact order the
    model was trained on.
    """
    row = {col: float(values.get(col, 0.0)) for col in feature_columns}
    return pd.DataFrame([row])[feature_columns]


def predict_transaction(model, input_df: pd.DataFrame, threshold: float = 0.5):
    """
    Returns (is_fraud: bool, probability: float).
    """
    probability = float(model.predict_proba(input_df)[0][1])
    is_fraud = probability >= threshold
    return is_fraud, probability
