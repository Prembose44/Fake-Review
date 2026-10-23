"""
Step 5: Evaluate the full Gated Cascaded Classification Architecture.

Maxwell-only data. Combines Tier 1 (SVM) and Tier 2 (DeBERTa), tests
multiple gate thresholds, measures accuracy/F1, escalation rate, and
real per-review latency.

Usage (from project root, with venv activated):
    python src\\05_cascade_eval.py
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

DATA_DIR = "data/processed"
MODEL_DIR = "models"
DEBERTA_PATH = os.path.join(MODEL_DIR, "deberta_v3_base", "final")
OUTPUT_DIR = "outputs"
MAX_LENGTH = 256

THRESHOLDS = [0.80, 0.85, 0.90, 0.95]


def load_test_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).dropna(subset=["text"])
    return df.reset_index(drop=True)


def load_svm():
    svm = joblib.load(os.path.join(MODEL_DIR, "svm_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    return svm, vectorizer


def load_deberta():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_PATH, dtype=torch.float32
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def svm_predict_all(svm, vectorizer, texts):
    X = vectorizer.transform(texts)
    proba = svm.predict_proba(X)
    preds = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    return preds, confidence


def deberta_predict_batch(model, tokenizer, device, texts, batch_size=16):
    all_preds, all_conf = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                            padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        all_preds.extend(preds.tolist())
        all_conf.extend(conf.tolist())
    return np.array(all_preds), np.array(all_conf)


def measure_latency_svm(svm, vectorizer, texts, n_samples=500):
    sample = texts[:n_samples]
    start = time.perf_counter()
    for t in sample:
        X = vectorizer.transform([t])
        svm.predict_proba(X)
    elapsed = time.perf_counter() - start
    return (elapsed / len(sample)) * 1000


def measure_latency_deberta(model, tokenizer, device, texts, n_samples=500):
    sample = texts[:n_samples]
    warm_inputs = tokenizer([sample[0]], truncation=True, max_length=MAX_LENGTH,
                             padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**warm_inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for t in sample:
        inputs = tokenizer([t], truncation=True, max_length=MAX_LENGTH,
                            padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / len(sample)) * 1000


def evaluate_predictions(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Confusion matrix:\n{cm}")

    return {"name": name, "accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading test data...")
    test_df = load_test_data()
    texts = test_df["text"].tolist()
    y_true = test_df["label"].values
    print(f"Test set size: {len(texts)}")

    print("\nLoading SVM (Tier 1)...")
    svm, vectorizer = load_svm()

    print("Loading DeBERTa (Tier 2)...")
    deberta_model, deberta_tokenizer, device = load_deberta()
    print(f"DeBERTa running on: {device}")

    print("\n" + "="*70)
    print("STEP 1: Pure baseline predictions (full test set)")
    print("="*70)

    print("\nRunning SVM on full test set...")
    svm_preds, svm_conf = svm_predict_all(svm, vectorizer, texts)
    svm_result = evaluate_predictions("Pure SVM (Tier 1 only)", y_true, svm_preds)

    print("\nRunning DeBERTa on full test set...")
    deberta_preds, deberta_conf = deberta_predict_batch(
        deberta_model, deberta_tokenizer, device, texts
    )
    deberta_result = evaluate_predictions("Pure DeBERTa (Tier 2 only)", y_true, deberta_preds)

    print("\n" + "="*70)
    print("STEP 2: Cascade evaluation at multiple gate thresholds")
    print("="*70)

    cascade_results = []
    for threshold in THRESHOLDS:
        print(f"\n--- Threshold: {threshold} ---")
        escalate_mask = svm_conf < threshold
        n_escalated = escalate_mask.sum()
        escalation_rate = n_escalated / len(texts)

        cascade_preds = np.where(escalate_mask, deberta_preds, svm_preds)
        result = evaluate_predictions(f"Cascade @ threshold={threshold}", y_true, cascade_preds)
        result["threshold"] = threshold
        result["escalation_rate"] = escalation_rate
        result["n_escalated"] = int(n_escalated)
        result["n_total"] = len(texts)

        print(f"Escalation rate: {escalation_rate*100:.1f}% "
              f"({n_escalated}/{len(texts)} reviews sent to Tier 2)")

        cascade_results.append(result)

    print("\n" + "="*70)
    print("STEP 3: Latency benchmarking (per-review, milliseconds)")
    print("="*70)

    n_latency_samples = 500
    print(f"\nMeasuring SVM latency over {n_latency_samples} reviews...")
    svm_latency_ms = measure_latency_svm(svm, vectorizer, texts, n_latency_samples)
    print(f"SVM avg latency: {svm_latency_ms:.3f} ms/review")

    print(f"\nMeasuring DeBERTa latency over {n_latency_samples} reviews...")
    deberta_latency_ms = measure_latency_deberta(
        deberta_model, deberta_tokenizer, device, texts, n_latency_samples
    )
    print(f"DeBERTa avg latency: {deberta_latency_ms:.3f} ms/review")

    print("\nEstimated cascade latency per threshold:")
    for result in cascade_results:
        rate = result["escalation_rate"]
        est_latency = svm_latency_ms + (rate * deberta_latency_ms)
        result["estimated_latency_ms"] = est_latency
        print(f"  threshold={result['threshold']}: ~{est_latency:.3f} ms/review "
              f"({rate*100:.1f}% escalated)")

    print("\n" + "="*70)
    print("FINAL SUMMARY TABLE")
    print("="*70)

    summary_rows = [{
        "model": "Pure SVM", "accuracy": svm_result["accuracy"], "f1": svm_result["f1"],
        "escalation_rate": 0.0, "latency_ms": svm_latency_ms
    }]
    for result in cascade_results:
        summary_rows.append({
            "model": f"Cascade (t={result['threshold']})",
            "accuracy": result["accuracy"], "f1": result["f1"],
            "escalation_rate": result["escalation_rate"],
            "latency_ms": result["estimated_latency_ms"]
        })
    summary_rows.append({
        "model": "Pure DeBERTa", "accuracy": deberta_result["accuracy"],
        "f1": deberta_result["f1"], "escalation_rate": 1.0,
        "latency_ms": deberta_latency_ms
    })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cascade_summary.csv"), index=False)
    full_results = {
        "svm_baseline": svm_result, "deberta_baseline": deberta_result,
        "cascade_results": cascade_results,
        "svm_latency_ms": svm_latency_ms, "deberta_latency_ms": deberta_latency_ms,
    }
    with open(os.path.join(OUTPUT_DIR, "cascade_full_results.json"), "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\nSaved cascade_summary.csv and cascade_full_results.json to {OUTPUT_DIR}/")
    print("\nDone. The summary table above is your paper's central results table.")


if __name__ == "__main__":
    main()
