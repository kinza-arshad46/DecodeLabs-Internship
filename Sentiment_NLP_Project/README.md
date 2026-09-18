# DecodeLabs Project 4 – NLP & Sentiment Analysis

**Optional Mastery Phase: Mathematical Linguistics**  
Batch 2026 | Powered by DecodeLabs

> Program a machine to read and mathematically categorize unstructured human text (product / movie reviews).

---

## Project Goals (from official brief)

| Requirement | Implementation |
|-------------|----------------|
| Strict Text Pre-Processing pipeline | Tokenization → Custom Stop-Word removal → POS-guided Lemmatization |
| Convert text → mathematical arrays | TF-IDF (Term Frequency-Inverse Document Frequency) |
| Train classifier | MultinomialNB / ComplementNB / LinearSVM |
| Key Skills | NLTK, text vectorization, unstructured data handling |

---

## Engineering Rules Implemented Exactly

1. **Stop-Word Trap**  
   Default NLTK stopwords **exclude** negations (`not`, `no`, `never`, `don't`…).  
   → `"I am not happy"` correctly stays negative.

2. **POS-Guided Lemmatization**  
   Treebank tags are mapped to WordNet POS before `WordNetLemmatizer`.  
   → `"went"` + verb tag → `"go"` (correct).

3. **N-Grams**  
   Unigrams + Bigrams so `"not good"` is captured as a single feature.

4. **Dimensionality Control**  
   `max_features=10000`, `min_df=2` → prevents vocabulary explosion.

5. **Memory Optimization**  
   All matrices stored as **SciPy CSR** sparse format.

6. **Zero-Frequency Problem**  
   Laplace smoothing (`alpha=1.0`) on Naive Bayes.

7. **Model Choice**  
   - `MultinomialNB` → balanced data  
   - `ComplementNB` → imbalanced data  
   - `LinearSVC` → strong linear baseline

---

## Project Structure

```
sentiment_nlp_project/
├── README.md
├── requirements.txt
├── train.py                 # Full training pipeline
├── predict.py               # CLI + interactive prediction
├── demo_preprocessing.py    # Demo of stop-word + lemmatization rules
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Strict text pipeline
│   ├── vectorizer.py        # TF-IDF + CSR
│   ├── model.py             # Naive Bayes / SVM wrapper
│   └── utils.py             # Data loading, plots, NLTK setup
├── models/                  # Saved artifacts (created after training)
│   ├── sentiment_multinomial.joblib
│   ├── tfidf_vectorizer.joblib
│   └── confusion_matrix_*.png
└── data/                    # Optional custom CSVs
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

Optional flags:
```bash
python train.py --model complement --max-features 8000
python train.py --model svm
```

### 3. Predict
```bash
# Single text
python predict.py --text "I am not happy with this product at all"

# Interactive mode
python predict.py --interactive

# From file (one review per line)
python predict.py --file my_reviews.txt
```

### 4. Demo the critical rules
```bash
python demo_preprocessing.py
```

---

## Expected Performance

On the classic NLTK `movie_reviews` corpus (2000 balanced reviews):

| Model            | Typical Accuracy | Notes                          |
|------------------|------------------|--------------------------------|
| MultinomialNB    | ~82–85%          | Fast, good baseline            |
| ComplementNB     | ~83–86%          | Better on slight imbalance     |
| LinearSVC        | ~85–88%          | Strongest linear model         |

---

## How the Pipeline Works (Data Flow)

```
Raw Review
    │
    ▼
[1] Character Normalization + Lowercasing
    │  (remove HTML, URLs, punctuation…)
    ▼
[2] Tokenization
    │
    ▼
[3] Custom Stop-Word Filter
    │  (negations KEPT)
    ▼
[4] POS Tagging → WordNet Lemmatization
    │
    ▼
[5] TF-IDF Vectorizer (1-2 grams, max 10k features)
    │  → SciPy CSR Sparse Matrix
    ▼
[6] Multinomial / Complement Naive Bayes
    │  (Laplace smoothing)
    ▼
Positive / Negative  +  confidence score
```

---

## Enhancing Further (Ideas already prepared)

- Add your own CSV dataset via `src/utils.load_csv_reviews`
- Swap to a larger corpus (Amazon reviews, IMDB 50k)
- Add Gradio / Streamlit UI on top of `predict.py`
- Experiment with `ngram_range=(1,3)` or different `alpha`

---

## Author Notes

This project strictly follows the **DecodeLabs Project 4 Engineering Blueprint** while adding production-quality structure, proper package layout, CLI interfaces, evaluation plots, and model persistence.

**Certificate-ready.**  
Keep decoding the data of the future!

---

Contact (DecodeLabs):  
+91 9236011887 | decodelabs.tech@gmail.com | www.decodelabs.tech
