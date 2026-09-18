#!/usr/bin/env python3
"""
Interactive / CLI Prediction for Sentiment Analysis
Usage:
    python predict.py --text "This product is terrible!!!"
    python predict.py --interactive
    python predict.py --file reviews.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import TextPreprocessor
from src.vectorizer import SentimentVectorizer
from src.model import SentimentClassifier
from src.utils import setup_nltk


def load_artifacts(model_dir: str = "models", model_name: str = "multinomial"):
    model_path = Path(model_dir) / f"sentiment_{model_name}.joblib"
    vec_path = Path(model_dir) / "tfidf_vectorizer.joblib"

    if not model_path.exists() or not vec_path.exists():
        print("[!] Model artifacts not found.")
        print("    Please run: python train.py")
        sys.exit(1)

    vectorizer = SentimentVectorizer().load(str(vec_path))
    classifier = SentimentClassifier().load(str(model_path))
    preprocessor = TextPreprocessor()
    return preprocessor, vectorizer, classifier


def predict_single(text: str, preprocessor, vectorizer, classifier) -> dict:
    cleaned = preprocessor.preprocess(text)
    X = vectorizer.transform([cleaned])
    label = classifier.predict_label(X)[0]
    proba = classifier.predict_proba(X)

    result = {
        "original": text,
        "cleaned": cleaned,
        "prediction": label,
        "confidence": None,
    }
    if proba is not None:
        # proba shape (1, 2) → [neg, pos]
        conf = float(max(proba[0]))
        result["confidence"] = conf
        result["prob_negative"] = float(proba[0][0])
        result["prob_positive"] = float(proba[0][1])
    return result


def print_result(res: dict):
    print("\n" + "─" * 50)
    print(f"Original   : {res['original']}")
    print(f"Cleaned    : {res['cleaned'][:100]}{'...' if len(res['cleaned'])>100 else ''}")
    emoji = "😊" if res["prediction"] == "Positive" else "😞"
    print(f"Prediction : {res['prediction']} {emoji}")
    if res["confidence"] is not None:
        print(f"Confidence : {res['confidence']*100:.1f}%")
        print(f"   P(Neg)  : {res['prob_negative']*100:.1f}%")
        print(f"   P(Pos)  : {res['prob_positive']*100:.1f}%")
    print("─" * 50)


def interactive_mode(preprocessor, vectorizer, classifier):
    print("\n" + "=" * 50)
    print("  INTERACTIVE SENTIMENT PREDICTOR")
    print("  Type a review and press Enter (or 'quit' to exit)")
    print("=" * 50)
    while True:
        try:
            text = input("\nReview > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if text.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if not text:
            continue
        res = predict_single(text, preprocessor, vectorizer, classifier)
        print_result(res)


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of text")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--file", type=str, help="File with one review per line")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--model", type=str, default="multinomial",
                        choices=["multinomial", "complement", "svm"])
    parser.add_argument("--model-dir", type=str, default="models")
    args = parser.parse_args()

    setup_nltk(quiet=True)
    preprocessor, vectorizer, classifier = load_artifacts(args.model_dir, args.model)

    if args.interactive or (not args.text and not args.file):
        interactive_mode(preprocessor, vectorizer, classifier)
        return

    if args.text:
        res = predict_single(args.text, preprocessor, vectorizer, classifier)
        print_result(res)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"[!] File not found: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"\nPredicting {len(lines)} reviews from {path}...\n")
        for line in lines:
            res = predict_single(line, preprocessor, vectorizer, classifier)
            print_result(res)


if __name__ == "__main__":
    main()
