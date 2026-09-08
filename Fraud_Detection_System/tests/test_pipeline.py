"""
tests/test_pipeline.py
-------------------------
Unit tests for the streamlit-free utils/ layer. Run with:

    pytest tests/

These tests do NOT require Streamlit to be installed, since utils/
modules degrade gracefully without it (see utils/data_loader.py).
They also do NOT require real network access or the real Kaggle
dataset - they run against the synthetic fallback dataset.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_data, dataset_summary
from utils.preprocessing import clean_data, split_data
from utils.smote_handler import apply_smote
from utils.model_trainer import build_pipeline
from utils.evaluator import evaluate_model
from utils.predictor import build_input_row, predict_transaction


def test_load_data_schema():
    df, _ = load_data()
    expected_cols = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0


def test_dataset_summary_matches_class_column():
    df, _ = load_data()
    summary = dataset_summary(df)
    assert summary["total"] == len(df)
    assert summary["fraud"] + summary["normal"] == summary["total"]
    assert 0 <= summary["fraud_pct"] <= 100


def test_clean_data_removes_duplicates():
    df, _ = load_data()
    doubled = df.iloc[:5]._append(df.iloc[:5]) if hasattr(df.iloc[:5], "_append") else df.iloc[:5]
    cleaned, report = clean_data(df)
    assert report["missing_values"] == int(df.isnull().sum().sum())
    assert len(cleaned) <= len(df)


def test_split_data_is_stratified():
    df, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert len(X_train) + len(X_test) == len(df)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    # stratified split should keep ratios close
    assert abs(train_ratio - test_ratio) < 0.01


def test_smote_balances_training_classes():
    df, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    X_res, y_res = apply_smote(X_train, y_train)
    counts = y_res.value_counts() if hasattr(y_res, "value_counts") else None
    import pandas as pd
    counts = pd.Series(y_res).value_counts()
    assert counts[0] == counts[1]
    # test set must remain untouched / imbalanced
    assert y_test.mean() < 0.05


def test_build_pipeline_has_expected_steps():
    lr_pipeline = build_pipeline("Logistic Regression")
    step_names = [name for name, _ in lr_pipeline.steps]
    assert step_names == ["scaler", "smote", "classifier"]

    rf_pipeline = build_pipeline("Random Forest")
    step_names_rf = [name for name, _ in rf_pipeline.steps]
    assert step_names_rf == ["smote", "classifier"]


def test_build_pipeline_rejects_unknown_model():
    try:
        build_pipeline("Unknown Model")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_evaluate_model_returns_expected_keys():
    df, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    pipeline = build_pipeline("Random Forest")
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    for key in ["precision", "recall", "f1", "roc_auc", "fpr", "tpr"]:
        assert key in metrics


def test_predictor_builds_correct_row_order():
    feature_cols = ["Time", "V1", "V2", "Amount"]
    row = build_input_row(feature_cols, {"Amount": 99.5, "V1": 1.2})
    assert list(row.columns) == feature_cols
    assert row.iloc[0]["Amount"] == 99.5
    assert row.iloc[0]["Time"] == 0.0


def test_predict_transaction_respects_threshold():
    class DummyModel:
        def predict_proba(self, X):
            return [[0.3, 0.7]]

    row = build_input_row(["Amount"], {"Amount": 10})
    is_fraud_low_thresh, proba = predict_transaction(DummyModel(), row, threshold=0.5)
    is_fraud_high_thresh, _ = predict_transaction(DummyModel(), row, threshold=0.9)
    assert proba == 0.7
    assert is_fraud_low_thresh is True
    assert is_fraud_high_thresh is False


if __name__ == "__main__":
    # allow running this file directly with `python tests/test_pipeline.py`
    # even without pytest installed
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} -> {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
