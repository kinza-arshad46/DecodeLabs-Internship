#!/usr/bin/env python3
"""
DecodeLabs – Data Science Project 3
Unsupervised Learning: Customer Segmentation
Industrial Training Kit – Batch 2026

Usage:
    python main.py                  # full pipeline
    python main.py --k 4            # force specific K
    streamlit run app.py            # interactive dashboard
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import run_full_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Customer Segmentation Pipeline (PCA + K-Means + Personas)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/marketing_campaign.csv",
        help="Path to raw customer CSV",
    )
    parser.add_argument(
        "--variance",
        type=float,
        default=0.95,
        help="PCA cumulative explained variance threshold (default 0.95)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Force a specific number of clusters (skip auto selection)",
    )
    parser.add_argument(
        "--kmin",
        type=int,
        default=2,
        help="Minimum K for search range",
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=10,
        help="Maximum K for search range",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(
        """
╔══════════════════════════════════════════════════════════════╗
║   DecodeLabs  ·  Project 3  ·  Customer Segmentation         ║
║   Unsupervised Learning  ·  PCA + K-Means + Personas         ║
║   Batch 2026  ·  Industrial Training Kit                     ║
╚══════════════════════════════════════════════════════════════╝
        """
    )

    run_full_pipeline(
        data_path=args.data,
        variance_threshold=args.variance,
        k_range=range(args.kmin, args.kmax + 1),
        force_k=args.k,
        output_dir="outputs",
    )


if __name__ == "__main__":
    main()
