#!/usr/bin/env python3
"""
DecodeLabs Project 4 – NLP & Sentiment Analysis
Complete Training Pipeline

Usage:
    python train.py
    python train.py --model complement --max-features 8000
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import TextPreprocessor
from src.vectorizer import SentimentVectorizer
from src.model import SentimentClassifier
from src.utils import (
    setup_nltk,
    load_movie_reviews,
    plot_confusion_matrix,
    print_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Sentiment Analysis model (DecodeLabs Project 4)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="multinomial",
        choices=["multinomial", "complement", "svm"],
        help="Classifier type (default: multinomial)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Max TF-IDF features (default: 10000)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model & vectorizer",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving confusion matrix plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DecodeLabs Project 4 – NLP & Sentiment Analysis")
    print("  Optional Mastery Phase: Mathematical Linguistics")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    print("\n[1/6] Setting up NLTK resources...")
    setup_nltk(quiet=True)

    # ------------------------------------------------------------------
    # 2. Load Data
    # ------------------------------------------------------------------
    print("[2/6] Loading movie reviews corpus (2000 balanced samples)...")
    X_train_raw, X_test_raw, y_train, y_test = load_movie_reviews(
        test_size=args.test_size, random_state=42
    )
    print(f"       Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")
    print(f"       Class balance (train): Neg={sum(y_train==0)} Pos={sum(y_train==1)}")

    # ------------------------------------------------------------------
    # 3. Pre-processing (strict pipeline)
    # ------------------------------------------------------------------
    print("\n[3/6] Building Text Pre-Processing pipeline...")
    print("       • Character Normalization + Lowercasing")
    print("       • Tokenization")
    print("       • Custom Stop-Word Removal (NEGATIONS PRESERVED)")
    print("       • POS-Guided WordNet Lemmatization")

    preprocessor = TextPreprocessor()
    start = time.time()
    X_train_clean = preprocessor.preprocess_batch(X_train_raw, show_progress=True)
    X_test_clean = preprocessor.preprocess_batch(X_test_raw, show_progress=True)
    print(f"       Preprocessing done in {time.time()-start:.1f}s")

    # Quick sanity check
    print("\n       Sample preprocessed text:")
    print(f"       → {X_train_clean[0][:120]}...")

    # ------------------------------------------------------------------
    # 4. Vectorization (TF-IDF + N-grams + CSR)
    # ------------------------------------------------------------------
    print("\n[4/6] Vectorizing with TF-IDF (Unigrams + Bigrams)...")
    print(f"       max_features={args.max_features}, min_df=2, ngram_range=(1,2)")

    vectorizer = SentimentVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_clean)
    X_test = vectorizer.transform(X_test_clean)

    print(f"       Sparse matrix shape (train): {X_train.shape}")
    print(f"       Non-zero density           : {X_train.nnz / (X_train.shape[0]*X_train.shape[1]):.4%}")
    print(f"       Memory saved by CSR        : ~{(1 - X_train.nnz / (X_train.shape[0]*X_train.shape[1]))*100:.1f}%")

    # ------------------------------------------------------------------
    # 5. Train Classifier
    # ------------------------------------------------------------------
    print(f"\n[5/6] Training {args.model.upper()} classifier (Laplace α=1.0)...")
    clf = SentimentClassifier(model_type=args.model, alpha=1.0)
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Evaluate & Save
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating on held-out test set...")
    results = clf.evaluate(X_test, y_test)
    print_evaluation(results)

    # Save artifacts
    model_path = output_dir / f"sentiment_{args.model}.joblib"
    vec_path = output_dir / "tfidf_vectorizer.joblib"
    clf.save(str(model_path))
    vectorizer.save(str(vec_path))

    if not args.no_plot:
        cm_path = output_dir / f"confusion_matrix_{args.model}.png"
        plot_confusion_matrix(
            results["confusion_matrix"],
            save_path=str(cm_path),
            title=f"Confusion Matrix – {args.model.upper()}",
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE – Production Pipeline Ready")
    print("=" * 60)
    print(f"Model type      : {args.model}")
    print(f"Features        : {vectorizer.n_features}")
    print(f"Test Accuracy   : {results['accuracy']:.4f}")
    print(f"Artifacts saved : {output_dir}/")
    print("\nNext steps:")
    print("  python predict.py --text \"I am not happy with this product\"")
    print("  python predict.py --interactive")
    print("=" * 60)


if __name__ == "__main__":
    main()
