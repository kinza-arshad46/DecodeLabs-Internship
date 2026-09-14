"""
Fast training script with fixed good hyperparameters (no GridSearch).
Useful for low-resource environments or quick verification.
Still fully respects the Zero-Leakage Protocol.
"""

import sys
from pathlib import Path
import warnings
import joblib
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import (
    load_creditcard_data,
    get_features_target,
    stratified_train_test_split,
    print_class_distribution,
)
from src.pipelines import build_logistic_pipeline, build_random_forest_pipeline
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"

for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("  FRAUD DETECTION – FAST VERSION (Fixed Hyperparameters)")
    print("  Still 100% Leak-Free + SMOTE + Imblearn Pipeline")
    print("=" * 70)

    df = load_creditcard_data(DATA_PATH)
    print(f"Full shape: {df.shape}")
    print_class_distribution(df["Class"])

    # Practical subset for low-RAM machines (keep ALL fraud + 25% legitimate)
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(frac=0.25, random_state=42)
    df_sample = pd.concat([legit, fraud], ignore_index=True)
    X, y = get_features_target(df_sample)
    print(f"\nWorking subset: {X.shape}")
    print_class_distribution(y)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print_class_distribution(y_train, "Train")
    print_class_distribution(y_test, "Test (untouched)")

    results = {}

    for name, pipe in [
        ("Logistic Regression", build_logistic_pipeline(42)),
        ("Random Forest", build_random_forest_pipeline(42)),
    ]:
        print(f"\n>>> Training {name} ...")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba, name)
        results[name] = {k: float(v) if v is not None else None for k, v in metrics.items()}

        plot_confusion_matrix(y_test, y_pred, name, FIGURES_DIR)
        plot_roc_curve(y_test, y_proba, name, FIGURES_DIR)
        plot_precision_recall_curve(y_test, y_proba, name, FIGURES_DIR)

        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}_fast.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved → {model_path}")

    with open(REPORTS_DIR / "metrics_fast.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FAST TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
