"""
Inference script – demonstrates that negative sentences are correctly classified.
Uses the trained ComplementNB (more robust) + TF-IDF vectorizer.
"""

import joblib
from preprocess import clean_text

# Load artifacts
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
model = joblib.load("models/complement_nb.joblib")   # ComplementNB preferred for robustness


def predict_sentiment(text: str) -> dict:
    """
    Returns prediction + confidence for a single review.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {"label": "Unknown", "confidence": 0.0, "cleaned": ""}

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    label = "Positive" if pred == 1 else "Negative"
    confidence = float(max(proba))
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "cleaned": cleaned,
        "proba_negative": round(float(proba[0]), 4),
        "proba_positive": round(float(proba[1]), 4)
    }


if __name__ == "__main__":
    # Critical test cases that previously failed (negative → positive bias)
    test_sentences = [
        "I am not happy with this product.",
        "This product is TERRIBLE!!!",
        "The service was not good at all.",
        "I never want to buy this again.",
        "Absolutely awful experience, do not recommend.",
        "This is the worst movie I have ever seen.",
        "Not worth the money.",
        "I am very happy and satisfied with the quality.",
        "Excellent product, highly recommended!",
        "The food was delicious and the staff was friendly."
    ]

    print("=" * 70)
    print("Sentiment Prediction – Bias Check (Negative must stay Negative)")
    print("=" * 70)

    for sent in test_sentences:
        result = predict_sentiment(sent)
        print(f"\nOriginal : {sent}")
        print(f"Cleaned  : {result['cleaned']}")
        print(f"→ {result['label']}  (confidence={result['confidence']:.2%})")
        print(f"  P(Neg)={result['proba_negative']:.3f}  |  P(Pos)={result['proba_positive']:.3f}")
