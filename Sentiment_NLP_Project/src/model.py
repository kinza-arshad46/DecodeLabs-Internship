"""
Sentiment Classifier
Supports MultinomialNB (balanced data) and ComplementNB (imbalanced).
Includes Laplace smoothing (alpha=1.0) to solve Zero Frequency Problem.
"""

from typing import Optional, Dict, Any, List, Tuple
import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from scipy.sparse import csr_matrix


class SentimentClassifier:
    """
    Wrapper around Naive Bayes / Linear SVM for sentiment.
    """

    SUPPORTED_MODELS = {
        "multinomial": MultinomialNB,
        "complement": ComplementNB,
        "svm": LinearSVC,
    }

    def __init__(
        self,
        model_type: str = "multinomial",
        alpha: float = 1.0,
        class_weight: Optional[str] = None,
        **kwargs,
    ):
        """
        model_type:
            - "multinomial" → best for balanced datasets (50/50)
            - "complement"  → best for imbalanced (e.g. 95% pos)
            - "svm"         → Linear Support Vector Machine (strong baseline)
        alpha: Laplace smoothing (prevents zero probability)
        """
        model_type = model_type.lower()
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {list(self.SUPPORTED_MODELS)}")

        self.model_type = model_type
        self.alpha = alpha

        if model_type in ("multinomial", "complement"):
            self.model = self.SUPPORTED_MODELS[model_type](alpha=alpha, **kwargs)
        else:  # svm
            self.model = LinearSVC(
                class_weight=class_weight or "balanced",
                max_iter=5000,
                dual="auto",
                **kwargs,
            )

        self._is_fitted = False
        self.label_map = {0: "Negative", 1: "Positive"}
        self.inv_label_map = {"Negative": 0, "Positive": 1, "neg": 0, "pos": 1}

    def fit(self, X: csr_matrix, y: np.ndarray) -> "SentimentClassifier":
        """
        y should be binary: 0 = Negative, 1 = Positive
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: csr_matrix) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> Optional[np.ndarray]:
        """Available only for Naive Bayes models."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def evaluate(self, X: csr_matrix, y_true: np.ndarray) -> Dict[str, Any]:
        y_pred = self.predict(X)
        report = classification_report(
            y_true, y_pred, target_names=["Negative", "Positive"], output_dict=True
        )
        cm = confusion_matrix(y_true, y_pred)

        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "y_pred": y_pred,
        }
        return results

    def save(self, path: str) -> None:
        payload = {
            "model": self.model,
            "model_type": self.model_type,
            "alpha": self.alpha,
            "label_map": self.label_map,
        }
        joblib.dump(payload, path)
        print(f"[+] Model saved → {path}")

    def load(self, path: str) -> "SentimentClassifier":
        payload = joblib.load(path)
        self.model = payload["model"]
        self.model_type = payload["model_type"]
        self.alpha = payload.get("alpha", 1.0)
        self.label_map = payload.get("label_map", {0: "Negative", 1: "Positive"})
        self._is_fitted = True
        print(f"[+] Model loaded ← {path}")
        return self

    def predict_label(self, X: csr_matrix) -> List[str]:
        preds = self.predict(X)
        return [self.label_map[int(p)] for p in preds]
