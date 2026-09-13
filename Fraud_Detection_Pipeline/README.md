# Fraud Detection Pipeline – DecodeLabs Data Science Project 2

**Supervised Learning | Highly Imbalanced Classification | Leak-Free Design**

This project implements a production-grade **Credit Card Fraud Detection** system following the exact requirements of the DecodeLabs Industrial Training Kit (Batch 2026).

## Project Goal

Build and tune classification models that correctly identify fraudulent transactions in a highly imbalanced dataset (≈ 0.17% fraud rate) while strictly avoiding data leakage.

## Key Requirements Implemented

| Requirement                       | Status | Implementation                                                                      |
| --------------------------------- | ------ | ----------------------------------------------------------------------------------- |
| Handle extreme class imbalance    | ✅     | SMOTE (Synthetic Minority Over-sampling)                                            |
| Train multiple algorithms         | ✅     | Logistic Regression + Random Forest                                                 |
| Discard Accuracy                  | ✅     | Evaluation focused on**Precision, Recall, F1, ROC-AUC**                       |
| Scikit-Learn + Imblearn pipelines | ✅     | `imblearn.pipeline.Pipeline`                                                      |
| Zero data leakage                 | ✅     | Stratified split**before** any resampling/scaling; SMOTE only inside CV folds |
| Hyperparameter tuning             | ✅     | `GridSearchCV` with nested SMOTE                                                  |
| Feature scaling                   | ✅     | `StandardScaler` only for Logistic Regression (inside pipeline)                   |

## Dataset

- **Source**: Credit Card Fraud Detection (ULB / Kaggle classic)
- **Transactions**: 284,807
- **Fraud rate**: 0.172% (492 frauds)
- **Features**: V1–V28 (PCA components), Amount (Time is dropped as non-predictive for this exercise)
- File location: `data/creditcard.csv`

## Project Structure

```
fraud_detection_project/
├── data/
│   └── creditcard.csv              # Dataset (already included)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Loading + stratified split
│   ├── pipelines.py                # Leak-free imblearn pipelines
│   ├── evaluation.py               # Metrics + plots (no Accuracy focus)
│   ├── train.py                    # Main end-to-end training script
│   └── predict_example.py          # Inference demo
├── models/                         # Saved best models (.joblib)
├── reports/                        # metrics_summary.json
├── figures/                        # Confusion matrices, ROC & PR curves
├── requirements.txt
└── README.md
```

## Zero-Leakage Protocol (Strictly Followed)

1. **Ditch Accuracy** → Optimize on Recall / F1 / ROC-AUC
2. Use **SMOTE** to synthesize, never simple duplication
3. **NEVER** apply SMOTE or Scalers before the Train/Test split
4. **ALWAYS** use `imblearn.pipeline.Pipeline` so resampling happens only on training folds inside Cross-Validation
5. Tune preprocessing + model hyperparameters **holistically** inside `GridSearchCV`

## Quick Start

### 1. Install dependencies

```bash
cd fraud_detection_project
pip install -r requirements.txt
```

### 2. Download the dataset (if missing)

```bash
python download_data.py
```

(Or place `creditcard.csv` manually into the `data/` folder)

### 3. Run training

**Fast & recommended (fixed good hyperparameters, low RAM):**

```bash
python src/train_fast.py
```

**Full GridSearchCV version (slower, needs more RAM ≥ 8 GB):**

```bash
python -m src.train
# or
python src/train.py
```

### 4. Inspect results

- Models → `models/`
- Metrics JSON → `reports/`
- Plots → `figures/`

### 5. Run inference example

```bash
python src/predict_example.py
```

## Evaluation Metrics (Why not Accuracy?)

In this dataset a naïve model that always predicts “Legitimate” achieves **99.83% Accuracy** while catching **zero fraud**. That is catastrophic for a bank.

We therefore report:

- **Precision**: When we flag fraud, how often are we correct? (reduces customer friction)
- **Recall**: Did we catch the actual fraud cases? (direct financial loss)
- **F1-Score**: Harmonic balance of the two
- **ROC-AUC**: Overall ranking ability of the model
- **PR-AUC**: Especially informative under extreme imbalance

## Technical Highlights

- Stratified 80/20 split preserving the original 0.17% fraud rate in both sets
- `imblearn.pipeline.Pipeline` correctly handles the `fit_resample` interface
- `StandardScaler` lives **inside** the Logistic Regression pipeline only
- Random Forest needs no scaling (tree splits are ordinal)
- 3-fold StratifiedKFold inside GridSearchCV
- All synthetic samples generated **only** on the training portion of each fold

## Expected Performance (Approximate)

After proper tuning you should typically observe:

| Model               | Precision  | Recall     | ROC-AUC |
| ------------------- | ---------- | ---------- | ------- |
| Logistic Regression | 0.70–0.90 | 0.75–0.90 | > 0.95  |
| Random Forest       | 0.85–0.95 | 0.80–0.92 | > 0.97  |

(Exact numbers vary slightly with random seeds and hardware.)

## How to Extend

- Try different samplers: `ADASYN`, `BorderlineSMOTE`, `SMOTETomek`
- Add XGBoost / LightGBM pipelines
- Experiment with cost-sensitive learning (`class_weight="balanced"`)
- Deploy the best model with FastAPI + Docker
- Add SHAP explanations for interpretability

## Author Notes

This project was built to fully satisfy the DecodeLabs Project 2 Industrial Training requirements while remaining clean, modular, and production-oriented.

**Happy Fraud Hunting!** 🛡️

---

DecodeLabs | Batch 2026 | Data Science Industrial Training Kit
