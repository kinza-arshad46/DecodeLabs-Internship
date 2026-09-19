"""
TF-IDF Vectorization with N-grams + Sparse CSR storage
Follows Project 4 rules:
- Unigrams + Bigrams to capture negations ("not good")
- max_features to control dimensionality
- min_df to exclude rare typos
- Output is SciPy CSR sparse matrix (industry standard)
"""

from typing import Optional, Tuple, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


class SentimentVectorizer:
    """
    Production TF-IDF vectorizer tuned for sentiment analysis.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        """
        Parameters chosen according to the engineering rules in the PDF:
        - ngram_range=(1,2)  → captures "not good"
        - max_features=10000 → prevents feature explosion
        - min_df=2           → excludes rare typos / noise
        - sublinear_tf=True  → dampens the effect of very frequent terms
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            dtype="float32",  # memory friendly
        )
        self._is_fitted = False

    def fit(self, texts: List[str]) -> "SentimentVectorizer":
        """Fit on preprocessed texts."""
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts → sparse CSR matrix.
        CSR is mandatory for memory efficiency (99% zeros otherwise).
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transform.")
        X = self.vectorizer.transform(texts)
        # Ensure CSR (sklearn already returns CSR for TfidfVectorizer)
        if not isinstance(X, csr_matrix):
            X = X.tocsr()
        return X

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def save(self, path: str) -> None:
        joblib.dump(self.vectorizer, path)
        print(f"[+] Vectorizer saved → {path}")

    def load(self, path: str) -> "SentimentVectorizer":
        self.vectorizer = joblib.load(path)
        self._is_fitted = True
        print(f"[+] Vectorizer loaded ← {path}")
        return self

    @property
    def n_features(self) -> int:
        if not self._is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)
