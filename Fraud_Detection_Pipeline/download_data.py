"""
Download the Credit Card Fraud Detection dataset (OpenML mirror).
Run this once if data/creditcard.csv is missing.
"""

from pathlib import Path
import urllib.request
import sys

URL = "https://www.openml.org/data/get_csv/1673544/phpKo8OWT"
OUT = Path("data/creditcard.csv")

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 100_000_000:
        print(f"Dataset already present: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"Downloading from OpenML (~144 MB) ...")
    print(f"URL: {URL}")
    try:
        urllib.request.urlretrieve(URL, OUT)
        print(f"Saved to {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative options:")
        print("1. Download manually from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Or from Figshare: https://figshare.com/articles/dataset/creditcard_csv/27989750")
        print("   Then place the file as data/creditcard.csv")
        sys.exit(1)

if __name__ == "__main__":
    main()
