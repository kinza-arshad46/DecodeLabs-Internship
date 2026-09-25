"""
Text Pre-Processing Pipeline for Project 4: NLP & Sentiment Analysis
Follows DecodeLabs Blueprint:
1. Character normalization + lowercasing + tokenization
2. Custom stop-word removal (NEGATIONS EXPLICITLY PRESERVED)
3. POS-guided Lemmatization using WordNetLemmatizer
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV

# Ensure required NLTK resources are available
def ensure_nltk_data():
    resources = [
        "punkt", "punkt_tab", "stopwords", "wordnet",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"
    ]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}" if "punkt" in res else
                           f"taggers/{res}" if "tagger" in res else
                           f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


ensure_nltk_data()

# ------------------------------------------------------------------
# 1. Custom Stop-Word List (CRITICAL FIX for bias)
# Default NLTK stopwords contain "not", "no", "never", "nor" etc.
# Removing them turns "I am not happy" → "happy" → POSITIVE (wrong)
# We explicitly keep all negation words.
# ------------------------------------------------------------------
NEGATION_WORDS = {
    "not", "no", "never", "nor", "neither", "nobody", "nothing",
    "nowhere", "none", "n't", "cannot", "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "doesn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "don't", "didn't"
}

# Base English stopwords minus the negations we must keep
base_stopwords = set(stopwords.words("english"))
CUSTOM_STOPWORDS = base_stopwords - NEGATION_WORDS


def get_wordnet_pos(treebank_tag: str):
    """
    Map Treebank POS tags → WordNet POS tags.
    Mandatory for correct morphological reduction of verbs/adjectives.
    """
    if treebank_tag.startswith("J"):
        return ADJ
    elif treebank_tag.startswith("V"):
        return VERB
    elif treebank_tag.startswith("N"):
        return NOUN
    elif treebank_tag.startswith("R"):
        return ADV
    else:
        return NOUN  # default


lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Full pre-processing pipeline:
    - Character normalization (remove HTML, special chars, digits)
    - Lowercasing
    - Tokenization
    - Custom stop-word removal (negations kept)
    - POS-guided Lemmatization
    Returns a single cleaned string ready for TF-IDF.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Character Normalization
    text = re.sub(r"<.*?>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s']", " ", text)   # keep only letters + apostrophe
    text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
    text = text.lower()

    # 2. Tokenization
    tokens = word_tokenize(text)

    # 3. POS tagging + Lemmatization + Stop-word filtering
    tagged = pos_tag(tokens)
    cleaned = []
    for word, tag in tagged:
        if word in CUSTOM_STOPWORDS:
            continue
        if len(word) < 2:                       # drop single-letter noise
            continue
        wn_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        cleaned.append(lemma)

    return " ".join(cleaned)


# Quick self-test for the critical negation case
if __name__ == "__main__":
    samples = [
        "I am not happy with this product!!!",
        "This movie is terrible and boring.",
        "Absolutely wonderful experience, highly recommended.",
        "The service was not good at all.",
        "I never want to buy this again."
    ]
    print("=== Pre-processing Self-Test ===")
    for s in samples:
        print(f"\nOriginal : {s}")
        print(f"Cleaned  : {clean_text(s)}")
