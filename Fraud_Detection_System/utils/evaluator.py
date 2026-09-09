"""
utils/evaluator.py
--------------------
Test-set evaluation. The test set is NEVER resampled - it always reflects
the true, extreme class imbalance, which is the only honest way to measure
a fraud detector. Accuracy is intentionally not reported here: on a 99.83%
/ 0.17% split it rewards a model that predicts "normal" every time.
"""

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "fpr": fpr,
        "tpr": tpr,
    }
