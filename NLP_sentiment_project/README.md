# Project 4: NLP & Sentiment Analysis
**DecodeLabs Industrial Training Kit – Batch 2026**

## Goal
Build a robust pipeline that converts unstructured human text (product / movie reviews) into mathematical vectors and classifies sentiment as **Positive** or **Negative**.

## Key Fixes Applied (previous version problems)
| Problem | Root Cause | Solution Implemented |
|---------|------------|----------------------|
| Negative sentences classified as Positive | Default NLTK stopwords remove `"not"`, `"no"`, `"never"` | Custom stop-word list that **explicitly keeps all negations** |
| Lost negation context | Only unigrams used | TF-IDF with `ngram_range=(1,2)` (bigrams capture `"not good"`) |
| Incorrect verb/adjective reduction | Lemmatizer called without POS tag | POS tagging → map Treebank → WordNet → correct lemma |
| Memory crash risk | Dense matrices | SciPy **CSR sparse** format forced after vectorization |
| Zero-frequency crash | No smoothing | Laplace smoothing (`alpha=1.0`) on both NB models |

## Pipeline (5 Engineering Steps)
1. **Stop-Words** – exclude negations from default NLTK list
2. **Lemmatization** – POS-guided WordNetLemmatizer
3. **Vectorization** – TF-IDF (unigrams + bigrams, `max_features=10000`, `min_df=2`)
4. **Memory** – CSR sparse matrices
5. **Inference** – MultinomialNB + ComplementNB with Laplace smoothing

## Project Structure
```
NLP_Sentiment_Project/
├── preprocess.py          # Text cleaning + custom stop-words + POS lemmatization
├── train_model.py         # Full training pipeline + evaluation
├── predict.py             # Inference script with bias-check test cases
├── requirements.txt
├── README.md
└── models/                # Created after training
    ├── tfidf_vectorizer.joblib
    ├── multinomial_nb.joblib
    └── complement_nb.joblib
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data (first time only)
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('movie_reviews')
"

# 3. Train the models
python train_model.py

# 4. Run predictions (includes negative-sentence bias check)
python predict.py
```

## Expected Behaviour
- `"I am not happy with this product."` → **Negative**
- `"This product is TERRIBLE!!!"` → **Negative**
- `"Excellent product, highly recommended!"` → **Positive**

## Dataset
NLTK `movie_reviews` corpus (2000 reviews, perfectly balanced 1000 pos / 1000 neg).

## Core Language
**Python 3.8+**
