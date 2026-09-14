"""
Data loading and basic preprocessing utilities for Credit Card Fraud Detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_creditcard_data(data_path: str | Path = "data/creditcard.csv") -> pd.DataFrame:
    """
    Load the credit card fraud dataset and clean the Class column.
    OpenML export sometimes stores Class as quoted strings.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please ensure creditcard.csv is present in the data/ folder."
        )

    df = pd.read_csv(data_path)

    # Clean Class column (handle '0' / '1' strings from OpenML export)
    # OpenML CSV stores Class as the literal characters '0' and '1' (with single quotes)
    df["Class"] = (
        df["Class"]
        .astype(str)
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
        .astype(int)
    )

    # Basic validation
    if not df["Class"].isin([0, 1]).all():
        raise ValueError(f"Class column cleaning failed. Unique values: {df['Class'].unique()}")
    assert df.shape[1] == 31, f"Expected 31 columns, got {df.shape[1]}"

    return df


def get_features_target(df: pd.DataFrame):
    """Separate features (X) and target (y). Drop Time as it is not predictive in the usual sense."""
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]
    return X, y


def stratified_train_test_split(
    X, y, test_size: float = 0.2, random_state: int = 42
):
    """
    Perform stratified train-test split to preserve class ratios.
    This is critical for highly imbalanced data.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def print_class_distribution(y, title: str = "Class Distribution"):
    """Pretty print class balance."""
    counts = y.value_counts().sort_index()
    total = len(y)
    print(f"\n{title}")
    print("-" * 40)
    print(f"Legitimate (0): {counts.get(0, 0):,}  ({100 * counts.get(0, 0) / total:.4f}%)")
    print(f"Fraudulent (1): {counts.get(1, 0):,}  ({100 * counts.get(1, 0) / total:.4f}%)")
    print(f"Total         : {total:,}")
    print("-" * 40)
