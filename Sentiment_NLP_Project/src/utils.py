"""
Utility functions: data loading, visualization, NLTK setup.
"""

import os
import warnings
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def setup_nltk(quiet: bool = True) -> None:
    """Download all required NLTK resources (safe for proxy environments)."""
    import nltk
    import os as _os

    # Allow proxied downloads if needed
    _os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"

    resources = [
        "movie_reviews",
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "omw-1.4",
    ]
    for res in resources:
        try:
            nltk.download(res, quiet=quiet)
        except Exception as e:
            print(f"[!] Warning downloading {res}: {e}")


def load_movie_reviews(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Load the classic NLTK movie_reviews corpus (2000 reviews, balanced).
    Returns: X_train, X_test, y_train, y_test
    Labels: 0 = Negative, 1 = Positive
    """
    from nltk.corpus import movie_reviews

    documents = []
    labels = []

    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            # raw text
            text = movie_reviews.raw(fileid)
            documents.append(text)
            labels.append(1 if category == "pos" else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        documents,
        np.array(labels),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test


def load_csv_reviews(
    csv_path: str,
    text_col: str = "review",
    label_col: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Generic CSV loader.
    Expects binary labels: positive/negative or 1/0 or pos/neg.
    """
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()

    raw_labels = df[label_col].astype(str).str.lower().str.strip()
    label_map = {
        "positive": 1, "pos": 1, "1": 1, "yes": 1,
        "negative": 0, "neg": 0, "0": 0, "no": 0,
    }
    y = np.array([label_map.get(l, 0) for l in raw_labels])

    return train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )


def plot_confusion_matrix(
    cm: List[List[int]],
    labels: List[str] = ["Negative", "Positive"],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Pretty confusion matrix plot."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[+] Confusion matrix saved → {save_path}")
    plt.close()


def print_evaluation(results: dict) -> None:
    """Pretty print of evaluation metrics."""
    print("\n" + "=" * 55)
    print("MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"Accuracy      : {results['accuracy']:.4f}")
    print(f"F1 (macro)    : {results['f1_macro']:.4f}")
    print(f"F1 (weighted) : {results['f1_weighted']:.4f}")
    print("\nClassification Report:")
    report = results["classification_report"]
    print(f"{'':15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    for label in ["Negative", "Positive"]:
        r = report[label]
        print(
            f"{label:15} {r['precision']:10.4f} {r['recall']:10.4f} "
            f"{r['f1-score']:10.4f} {r['support']:10.0f}"
        )
    print(f"{'accuracy':15} {'':10} {'':10} {report['accuracy']:10.4f} {report['macro avg']['support']:10.0f}")
    print("=" * 55)
