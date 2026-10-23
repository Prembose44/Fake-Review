"""
Step 1: Preprocess the Maxwell dataset ONLY.

We use only the Maxwell Fake Reviews Dataset because it has validated,
transparent CG (Computer Generated = fake) / OR (Original = real) labels.
The Amazon Review Manipulation Dataset was excluded after determining its
product IDs were specifically selected as targets of paid fake-review
campaigns, making any real/fake label we assigned to it untrustworthy.

Usage (from project root, with venv activated):
    python src\\01_preprocess.py
"""

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

MEXWELL_PATH = "data/raw/archive (6)/fake reviews dataset.csv"
OUTPUT_DIR = "data/processed"

RANDOM_SEED = 42


def clean_text(text):
    """Lowercase, strip HTML/URLs, normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)              # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # strip URLs
    text = re.sub(r"[^\x00-\x7f]", " ", text)          # strip non-ASCII
    text = re.sub(r"\s+", " ", text).strip()            # normalise whitespace
    return text


def load_mexwell():
    print("Loading Maxwell dataset (full file)...")
    df = pd.read_csv(MEXWELL_PATH)
    print(f"  Total rows: {len(df)}")
    print(f"  Categories found: {df['category'].nunique()}")
    print(df["category"].value_counts())

    df = df.rename(columns={"text_": "text"})
    df["label"] = df["label"].map({"CG": 1, "OR": 0})  # 1 = fake, 0 = real
    df = df[["text", "label", "rating", "category"]]

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_mexwell()

    print("\nCleaning text...")
    df["text"] = df["text"].fillna("").astype(str).apply(clean_text)

    # Drop empty / too-short reviews after cleaning
    before = len(df)
    df = df[df["text"].str.split().str.len() >= 3]
    print(f"  Dropped {before - len(df)} rows with <3 words after cleaning")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset="text")
    print(f"  Dropped {before - len(df)} duplicate rows")

    print(f"\nFinal dataset size: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts())
    print(f"\nLabel distribution (%):")
    print(df["label"].value_counts(normalize=True) * 100)

    # Stratified split: train / val / test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_SEED
    )

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"\nSaved train.csv, val.csv, test.csv to {OUTPUT_DIR}/")
    print("\nDone. This is Maxwell-only data with validated labels.")
    print("Next step: run check_rating_signal.py to test if 'rating' is a useful feature.")


if __name__ == "__main__":
    main()
