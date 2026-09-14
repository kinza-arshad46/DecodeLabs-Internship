"""
Example: Load a trained model and score new transactions.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_creditcard_data, get_features_target


def main():
    model_path = ROOT / "models" / "random_forest_best.joblib"
    if not model_path.exists():
        print("Trained model not found. Please run src/train.py first.")
        return

    print("Loading best Random Forest model...")
    model = joblib.load(model_path)

    # Load a few real samples from the dataset for demonstration
    df = load_creditcard_data(ROOT / "data" / "creditcard.csv")
    X, y = get_features_target(df)

    # Take 5 legitimate + 5 fraud samples
    legit = X[y == 0].sample(5, random_state=42)
    fraud = X[y == 1].sample(5, random_state=42)
    samples = pd.concat([legit, fraud])
    true_labels = [0] * 5 + [1] * 5

    preds = model.predict(samples)
    probas = model.predict_proba(samples)[:, 1]

    print("\nPrediction Demo (10 samples):")
    print("-" * 55)
    print(f"{'True':>6}  {'Pred':>6}  {'Fraud Prob':>12}")
    print("-" * 55)
    for t, p, pr in zip(true_labels, preds, probas):
        print(f"{t:>6}  {p:>6}  {pr:>12.4f}")
    print("-" * 55)
    print("\nDone. In production you would pass new transaction feature vectors here.")


if __name__ == "__main__":
    main()
