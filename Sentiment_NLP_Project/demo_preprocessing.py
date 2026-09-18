#!/usr/bin/env python3
"""
Quick demo of the critical engineering rules from the PDF:
- Stop-word trap (negations preserved)
- POS-guided lemmatization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_nltk
from src.preprocessing import TextPreprocessor

setup_nltk(quiet=True)

pp = TextPreprocessor()

print("=" * 65)
print("  DEMO: The Stop-Word Trap (from Project 4 PDF)")
print("=" * 65)

examples = [
    "I am not happy with this product.",
    "This movie is not good at all.",
    "I never want to buy this again.",
    "The product is terrible!!! <br> Don't buy it.",
    "Absolutely wonderful experience, highly recommended!",
]

for text in examples:
    print(f"\nOriginal : {text}")
    print(f"Cleaned  : {pp.clean_text(text)}")
    tokens = pp.tokenize_and_lemmatize(text)
    print(f"Tokens   : {tokens}")
    print(f"Final    : {pp.preprocess(text)}")

print("\n" + "=" * 65)
print("  Notice: 'not', 'never', 'don't' are KEPT → sentiment preserved")
print("=" * 65)
