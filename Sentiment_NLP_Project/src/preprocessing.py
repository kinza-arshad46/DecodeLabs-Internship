"""
Text Pre-Processing Pipeline for Sentiment Analysis
Follows DecodeLabs Project 4 Engineering Blueprint:

1. Character Normalization + Lowercasing
2. Tokenization
3. Custom Stop-Word Removal (NEGATIONS PRESERVED)
4. POS-Guided Lemmatization using WordNetLemmatizer
"""

import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Ensure required NLTK data is available (caller should download)
# nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])


class TextPreprocessor:
    """
    Strict text pre-processing pipeline that preserves negation words.
    """

    # Negation words that MUST be kept (critical for sentiment)
    NEGATION_WORDS = {
        "not", "no", "never", "none", "nobody", "nothing", "neither",
        "nowhere", "hardly", "scarcely", "barely", "doesn't", "isn't",
        "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't",
        "don't", "aren't", "didn't", "nor"
    }

    def __init__(self, language: str = "english"):
        self.lemmatizer = WordNetLemmatizer()
        # Build custom stopwords: default NLTK minus negations
        default_stops = set(stopwords.words(language))
        self.stop_words = default_stops - self.NEGATION_WORDS

        # Compile regex patterns for speed
        self.html_pattern = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#\w+")
        self.non_alpha_pattern = re.compile(r"[^a-zA-Z\s]")
        self.multi_space = re.compile(r"\s+")

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        Map Treebank POS tags to WordNet POS tags.
        This is MANDATORY for correct morphological reduction.
        """
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # default

    def clean_text(self, text: str) -> str:
        """
        Character-level normalization:
        - Remove HTML tags
        - Remove URLs, mentions, hashtags
        - Lowercase
        - Remove non-alphabetic characters (keep spaces)
        - Collapse multiple spaces
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.html_pattern.sub(" ", text)
        text = self.url_pattern.sub(" ", text)
        text = self.mention_pattern.sub(" ", text)
        text = self.hashtag_pattern.sub(" ", text)
        text = text.lower()

        # Expand common contractions so negations stay intact
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "can't": "can not",
            "couldn't": "could not", "shouldn't": "should not", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "i'm": "i am", "you're": "you are", "it's": "it is", "that's": "that is",
        }
        for cont, exp in contractions.items():
            text = text.replace(cont, exp)

        text = self.non_alpha_pattern.sub(" ", text)
        text = self.multi_space.sub(" ", text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Full pipeline:
        1. Clean
        2. Tokenize
        3. POS tag
        4. Remove stopwords (negations kept)
        5. Lemmatize with correct POS
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        tokens = word_tokenize(cleaned)
        # POS tagging is mandatory for accurate lemmatization
        tagged = pos_tag(tokens)

        lemmas = []
        for word, tag in tagged:
            if word in self.stop_words:
                continue
            if len(word) < 2:  # skip single letters
                continue
            wn_tag = self._get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
            lemmas.append(lemma)

        return lemmas

    def preprocess(self, text: str) -> str:
        """
        Return space-joined lemmatized tokens ready for vectorization.
        """
        tokens = self.tokenize_and_lemmatize(text)
        return " ".join(tokens)

    def preprocess_batch(self, texts: List[str], show_progress: bool = False) -> List[str]:
        """
        Batch preprocessing with optional progress bar.
        """
        from tqdm import tqdm
        iterator = tqdm(texts, desc="Preprocessing") if show_progress else texts
        return [self.preprocess(t) for t in iterator]


def demonstrate_pipeline():
    """Quick demo of the preprocessing rules from the PDF."""
    pp = TextPreprocessor()

    examples = [
        "This product is TERRIBLE!!! <br>",
        "I am not happy with this purchase.",
        "The movie was absolutely wonderful and amazing!",
        "I don't like it at all. Never buying again.",
    ]

    print("=" * 60)
    print("PREPROCESSING PIPELINE DEMONSTRATION")
    print("=" * 60)
    for ex in examples:
        print(f"\nOriginal : {ex}")
        print(f"Cleaned  : {pp.clean_text(ex)}")
        print(f"Tokens   : {pp.tokenize_and_lemmatize(ex)}")
        print(f"Final    : {pp.preprocess(ex)}")


if __name__ == "__main__":
    demonstrate_pipeline()
