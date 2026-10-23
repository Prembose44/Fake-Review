"""
Step 3: Train TF-IDF + SVM baseline (Tier 1 of the cascade).

Text-only features. Rating was tested and showed negligible correlation
with the label (-0.0002), so it is not included as a feature.

Usage (from project root, with venv activated):
    python src\\03_train_svm.py
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

DATA_DIR = "data/processed"
MODEL_DIR = "models"

RANDOM_SEED = 42


def load_splits():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    for name, df in [("train", train), ("val", val), ("test", test)]:
        before = len(df)
        df.dropna(subset=["text"], inplace=True)
        after = len(df)
        if before != after:
            print(f"  Dropped {before - after} empty rows from {name}")

    return train, val, test


def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} results ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 score:  {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["real (0)", "fake (1)"]))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data splits...")
    train, val, test = load_splits()
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    print("\nFitting TF-IDF vectoriser on training text...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(val["text"])
    X_test = vectorizer.transform(test["text"])

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    print("\nTraining SVM (LinearSVC + Platt scaling via CalibratedClassifierCV)...")
    base_svm = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000)
    svm = CalibratedClassifierCV(base_svm, method="sigmoid", cv=5)
    svm.fit(X_train, y_train)

    val_pred = svm.predict(X_val)
    evaluate("Validation", y_val, val_pred)

    test_pred = svm.predict(X_test)
    evaluate("Test", y_test, test_pred)

    val_proba = svm.predict_proba(X_val)
    val_confidence = val_proba.max(axis=1)

    print("\n--- Confidence score distribution (validation set) ---")
    print(f"Mean confidence:   {val_confidence.mean():.3f}")
    print(f"Median confidence: {np.median(val_confidence):.3f}")
    for pct in [50, 60, 70, 75, 80, 85, 90, 95]:
        frac_above = (val_confidence >= pct / 100).mean()
        print(f"  Reviews with confidence >= {pct}%: {frac_above*100:.1f}%")

    joblib.dump(svm, os.path.join(MODEL_DIR, "svm_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    print(f"\nSaved svm_model.joblib and tfidf_vectorizer.joblib to {MODEL_DIR}/")

    print("\nDone. Next step: 04_train_deberta.py")


if __name__ == "__main__":
    main()
