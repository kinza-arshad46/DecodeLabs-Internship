"""
Evaluation utilities focused on Precision, Recall, F1 and ROC-AUC.
Accuracy is deliberately de-emphasized as required by the project brief.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)


def evaluate_model(y_true, y_pred, y_proba=None, model_name: str = "Model"):
    """
    Compute and print the metrics required by the project:
    Precision, Recall, F1, ROC-AUC.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Evaluation Report: {model_name}")
    print(f"{'='*60}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")
    else:
        roc_auc = None

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"], digits=4))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: Path):
    """Save a nicely formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        cbar=False,
    )
    plt.title(f"Confusion Matrix – {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    out = save_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_roc_curve(y_true, y_proba, model_name: str, save_dir: Path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = save_dir / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_precision_recall_curve(y_true, y_proba, model_name: str, save_dir: Path):
    """Plot and save Precision-Recall curve (especially useful for imbalanced data)."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="purple", lw=2, label=f"PR curve (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve – {model_name}")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = save_dir / f"pr_curve_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")
