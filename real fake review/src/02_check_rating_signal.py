"""
Step 2: Check whether star rating correlates with the fake/real label.

Run this BEFORE deciding whether to add rating as an SVM feature.

Usage:
    python src\\02_check_rating_signal.py
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/processed"


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    print("=" * 60)
    print("Rating statistics by label")
    print("=" * 60)
    print(train.groupby("label")["rating"].describe())

    print("\n" + "=" * 60)
    print("Rating distribution (%) by label")
    print("=" * 60)
    print(pd.crosstab(train["rating"], train["label"], normalize="columns") * 100)

    corr = np.corrcoef(train["rating"], train["label"])[0, 1]
    print(f"\n{'='*60}")
    print(f"Correlation between rating and label: {corr:.4f}")
    print(f"{'='*60}")

    if abs(corr) > 0.15:
        print("\nResult: Rating shows a MEANINGFUL correlation with the label.")
        print("Recommendation: Add rating as an extra numeric feature alongside TF-IDF.")
    elif abs(corr) > 0.05:
        print("\nResult: Rating shows a WEAK correlation with the label.")
        print("Recommendation: Adding it might give a small boost. Test both ways.")
    else:
        print("\nResult: Rating shows NEGLIGIBLE correlation with the label.")
        print("Recommendation: Adding it likely won't help. Text-only is fine.")


if __name__ == "__main__":
    main()
