"""
Project 4 – NLP & Sentiment Analysis
Complete Training Pipeline following DecodeLabs Blueprint.

Steps:
1. Load balanced movie_reviews corpus (NLTK)
2. Pre-process with custom stop-words + POS lemmatization
3. TF-IDF Vectorization (unigrams + bigrams, max_features, min_df)
4. Sparse CSR matrix (memory efficient)
5. MultinomialNB + ComplementNB with Laplace smoothing (alpha=1.0)
6. Evaluation + save model artifacts
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.corpus import movie_reviews
from scipy.sparse import csr_matrix

from preprocess import clean_text, ensure_nltk_data

ensure_nltk_data()


def load_movie_reviews():
    """Load NLTK movie_reviews – balanced positive / negative."""
    documents = []
    labels = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            text = movie_reviews.raw(fileid)
            documents.append(text)
            labels.append(1 if category == "pos" else 0)  # 1=Positive, 0=Negative
    return documents, np.array(labels)


def main():
    print("=" * 60)
    print("Project 4: NLP & Sentiment Analysis – Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    print("\n[1] Loading movie_reviews corpus ...")
    docs, y = load_movie_reviews()
    print(f"    Total reviews : {len(docs)}")
    print(f"    Positive      : {sum(y)}")
    print(f"    Negative      : {len(y) - sum(y)}")

    # ------------------------------------------------------------------
    # 2. Pre-Process
    # ------------------------------------------------------------------
    print("\n[2] Pre-processing (tokenization → stop-word filter → POS lemmatization) ...")
    cleaned_docs = [clean_text(doc) for doc in docs]
    print("    Done.")

    # ------------------------------------------------------------------
    # 3. Train / Test Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_docs, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[3] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ------------------------------------------------------------------
    # 4. Vectorize – TF-IDF with Unigrams + Bigrams
    #    Engineering Rule: max_features + min_df to control dimensionality
    # ------------------------------------------------------------------
    print("\n[4] TF-IDF Vectorization (ngram_range=(1,2), max_features=10000, min_df=2) ...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams → captures "not good"
        max_features=10000,      # prevent feature explosion
        min_df=2,                # ignore rare typos
        sublinear_tf=True,       # 1 + log(tf)
        use_idf=True,
        smooth_idf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Force CSR (industry standard for linear classifiers)
    X_train_tfidf = csr_matrix(X_train_tfidf)
    X_test_tfidf = csr_matrix(X_test_tfidf)

    print(f"    Vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"    Train matrix    : {X_train_tfidf.shape} (CSR, density={X_train_tfidf.nnz / (X_train_tfidf.shape[0]*X_train_tfidf.shape[1]):.4f})")

    # ------------------------------------------------------------------
    # 5. Train Classifiers (Laplace smoothing alpha=1.0)
    # ------------------------------------------------------------------
    print("\n[5] Training MultinomialNB and ComplementNB ...")

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_tfidf, y_train)

    cnb = ComplementNB(alpha=1.0)
    cnb.fit(X_train_tfidf, y_train)

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    print("\n[6] Evaluation on held-out test set")
    print("-" * 50)

    for name, model in [("MultinomialNB", mnb), ("ComplementNB", cnb)]:
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n>>> {name}")
        print(f"Accuracy : {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------------
    # 7. Save artifacts
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    joblib.dump(mnb, "models/multinomial_nb.joblib")
    joblib.dump(cnb, "models/complement_nb.joblib")
    print("\n[7] Models saved to ./models/")
    print("    - tfidf_vectorizer.joblib")
    print("    - multinomial_nb.joblib")
    print("    - complement_nb.joblib")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
