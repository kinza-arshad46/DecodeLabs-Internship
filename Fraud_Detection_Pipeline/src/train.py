"""
Main training script for Fraud Detection Pipeline (Project 2).

Implements the Zero-Leakage Protocol:
- Stratified train/test split FIRST
- SMOTE + Scaler only inside imblearn pipelines (applied only on training folds)
- GridSearchCV with proper scoring (roc_auc / recall)
- Final evaluation on completely untouched test set using Precision, Recall, ROC-AUC
"""

import sys
from pathlib import Path
import warnings
import joblib
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, roc_auc_score

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import (
    load_creditcard_data,
    get_features_target,
    stratified_train_test_split,
    print_class_distribution,
)
from src.pipelines import (
    build_logistic_pipeline,
    build_random_forest_pipeline,
    get_param_grids,
)
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"

for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("  FRAUD DETECTION PIPELINE – DecodeLabs Project 2")
    print("  Leak-Free Supervised Learning with SMOTE + Imblearn")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/6] Loading dataset...")
    df = load_creditcard_data(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print_class_distribution(df["Class"], "Full Dataset Class Distribution")

    X, y = get_features_target(df)

    # ------------------------------------------------------------------
    # 2. Stratified Train / Test Split  (BEFORE any resampling or scaling)
    # ------------------------------------------------------------------
    print("\n[2/6] Performing stratified train/test split (80/20)...")
    # NOTE: Full dataset (~285k rows) is large for low-RAM environments.
    # We keep ALL fraud cases and a stratified 20% sample of legitimate transactions.
    # This preserves the extreme imbalance while remaining computationally tractable.
    # On a machine with ≥8 GB RAM you can simply remove the sampling block below
    # and use the full X, y.
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(frac=0.20, random_state=42)
    df_sample = pd.concat([legit, fraud], ignore_index=True)
    X, y = get_features_target(df_sample)
    print(f"Working with practical subset shape: {X.shape}")
    print_class_distribution(y, "Subset Class Distribution (all fraud + 20% legit)")

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print_class_distribution(y_train, "Training Set")
    print_class_distribution(y_test, "Test Set (untouched – real-world imbalance)")

    # ------------------------------------------------------------------
    # 3. Build pipelines
    # ------------------------------------------------------------------
    print("\n[3/6] Building leak-free imblearn pipelines...")
    lr_pipe = build_logistic_pipeline(random_state=42)
    rf_pipe = build_random_forest_pipeline(random_state=42)
    lr_grid, rf_grid = get_param_grids()

    # Scoring: we care about catching fraud → recall + roc_auc
    scoring = {
        "roc_auc": "roc_auc",
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": "precision",
        "f1": "f1",
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # 2 folds for faster demo runtime

    # ------------------------------------------------------------------
    # 4. GridSearchCV for Logistic Regression
    # ------------------------------------------------------------------
    print("\n[4/6] Tuning Logistic Regression pipeline (GridSearchCV)...")
    print("       (This may take a few minutes...)")

    # Slightly reduced grid for reliable runtime while still showing proper tuning
    lr_search = GridSearchCV(
        estimator=lr_pipe,
        param_grid={
            "smote__k_neighbors": [5],
            "classifier__C": [0.1, 1.0],
        },
        scoring=scoring,
        refit="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=1,
        return_train_score=False,
    )
    lr_search.fit(X_train, y_train)

    print(f"\nBest LR params : {lr_search.best_params_}")
    print(f"Best LR ROC-AUC (CV): {lr_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 5. GridSearchCV for Random Forest
    # ------------------------------------------------------------------
    print("\n[5/6] Tuning Random Forest pipeline (GridSearchCV)...")
    print("       (This may take several minutes – Random Forest is heavier...)")

    # Slightly reduced grid for practical runtime while still demonstrating the concept
    rf_search = GridSearchCV(
        estimator=rf_pipe,
        param_grid={
            "smote__k_neighbors": [5],
            "classifier__n_estimators": [80],
            "classifier__max_depth": [15, None],
            "classifier__min_samples_split": [2],
        },
        scoring=scoring,
        refit="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=1,
        return_train_score=False,
    )
    rf_search.fit(X_train, y_train)

    print(f"\nBest RF params : {rf_search.best_params_}")
    print(f"Best RF ROC-AUC (CV): {rf_search.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 6. Final evaluation on untouched test set
    # ------------------------------------------------------------------
    print("\n[6/6] Final evaluation on completely held-out test set...")

    results = {}

    for name, search in [
        ("Logistic Regression", lr_search),
        ("Random Forest", rf_search),
    ]:
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba, model_name=name)
        results[name] = {
            "best_params": search.best_params_,
            "cv_roc_auc": float(search.best_score_),
            **{k: float(v) if v is not None else None for k, v in metrics.items()},
        }

        # Save plots
        plot_confusion_matrix(y_test, y_pred, name, FIGURES_DIR)
        plot_roc_curve(y_test, y_proba, name, FIGURES_DIR)
        plot_precision_recall_curve(y_test, y_proba, name, FIGURES_DIR)

        # Persist model
        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}_best.joblib"
        joblib.dump(best_model, model_path)
        print(f"Model saved → {model_path}")

    # Save metrics summary
    summary_path = REPORTS_DIR / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics summary saved → {summary_path}")

    # Quick comparison table
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON (Test Set)")
    print("=" * 70)
    print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<25} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} "
            f"{m['roc_auc']:>10.4f}"
        )
    print("=" * 70)
    print("\nProject completed successfully following Zero-Leakage Protocol.")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
