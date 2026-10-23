"""
Step 6: Explainable AI (XAI) layer for the cascade.

- Tier 1 (SVM): LIME explains which words pushed the TF-IDF/SVM decision
- Tier 2 (DeBERTa): SHAP explains which tokens the transformer focused on

Trained on Maxwell-only data (clean, validated CG/OR labels).

For each example we run through the cascade, this script produces:
  1. A human-readable console explanation
  2. HTML files saved to outputs/xai/ for visual inspection (LIME only)

Usage (from project root, with venv activated):
    python src\\06_xai_explain.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = "data/processed"
MODEL_DIR = "models"
DEBERTA_PATH = os.path.join(MODEL_DIR, "deberta_v3_base", "final")
OUTPUT_DIR = "outputs/xai"
MAX_LENGTH = 256
CASCADE_THRESHOLD = 0.95   # using the updated recommended threshold
N_LIME_FEATURES = 10
N_EXAMPLES = 10


def load_everything():
    svm = joblib.load(os.path.join(MODEL_DIR, "svm_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_PATH, dtype=torch.float32
    )
    model.to(device)
    model.eval()

    return svm, vectorizer, model, tokenizer, device


def svm_predict_proba(svm, vectorizer, texts):
    X = vectorizer.transform(texts)
    return svm.predict_proba(X)


def deberta_predict_proba(model, tokenizer, device, texts):
    all_probs = []
    for text in texts:
        inputs = tokenizer([text], truncation=True, max_length=MAX_LENGTH,
                            padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        all_probs.append(probs[0])
    return np.array(all_probs)


def explain_with_lime(svm, vectorizer, text, label, pred, confidence):
    print(f"\n  [LIME -- Tier 1 handled this review, confidence={confidence:.3f}]")
    print(f"  True label: {'FAKE' if label == 1 else 'REAL'} | "
          f"Predicted: {'FAKE' if pred == 1 else 'REAL'}")
    print(f"  Review: {text[:200]}{'...' if len(text) > 200 else ''}")

    def predict_fn(texts):
        return svm_predict_proba(svm, vectorizer, texts)

    explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])
    explanation = explainer.explain_instance(
        text, predict_fn, num_features=N_LIME_FEATURES, num_samples=500, labels=[1]
    )

    print(f"\n  Top {N_LIME_FEATURES} words pushing toward FAKE (positive) "
          f"or REAL (negative):")
    for word, weight in explanation.as_list(label=1):
        direction = "-> FAKE" if weight > 0 else "-> REAL"
        bar = "#" * int(abs(weight) * 30)
        print(f"    {word:20s}  {direction}  {bar} ({weight:+.4f})")

    return explanation


def explain_with_shap(model, tokenizer, device, text, label, pred, confidence):
    print(f"\n  [SHAP -- Tier 2 handled this review, confidence={confidence:.3f}]")
    print(f"  True label: {'FAKE' if label == 1 else 'REAL'} | "
          f"Predicted: {'FAKE' if pred == 1 else 'REAL'}")
    print(f"  Review: {text[:200]}{'...' if len(text) > 200 else ''}")

    def predict_fn(texts):
        return deberta_predict_proba(model, tokenizer, device, texts)

    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(predict_fn, masker, output_names=["REAL", "FAKE"])

    words = text.split()
    if len(words) > 60:
        text_for_shap = " ".join(words[:60])
        print(f"  (Text truncated to 60 words for SHAP computation speed)")
    else:
        text_for_shap = text

    shap_values = explainer([text_for_shap])

    print(f"\n  Top tokens by SHAP attribution (class: FAKE):")
    tokens = text_for_shap.split()
    fake_shap = shap_values.values[0, :len(tokens), 1]

    token_importance = sorted(
        zip(tokens, fake_shap), key=lambda x: abs(x[1]), reverse=True
    )[:N_LIME_FEATURES]

    for token, sv in token_importance:
        direction = "-> FAKE" if sv > 0 else "-> REAL"
        bar = "#" * int(abs(sv) * 80)
        print(f"    {token:20s}  {direction}  {bar} ({sv:+.4f})")

    return shap_values


def save_lime_html(explanation, path, label, pred):
    html = explanation.as_html()
    true_str = "FAKE" if label == 1 else "REAL"
    pred_str = "FAKE" if pred == 1 else "REAL"
    correct = "CORRECT" if label == pred else "WRONG"
    header = f"""
    <div style="font-family:Arial; padding:15px; background:#f8f9fa; border-radius:8px; margin-bottom:10px;">
        <h3>Tier 1 (SVM) -- LIME Explanation</h3>
        <p><b>True label:</b> {true_str} &nbsp;|&nbsp;
           <b>Predicted:</b> {pred_str} &nbsp;|&nbsp;
           <b>{correct}</b></p>
        <p style="color:#666; font-size:0.9em;">
            Green words push toward FAKE. Orange/red words push toward REAL.</p>
    </div>
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + html)
    print(f"  Saved LIME HTML -> {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading all models...")
    svm, vectorizer, deberta_model, deberta_tokenizer, device = load_everything()
    print(f"DeBERTa on: {device}")

    print("\nLoading test data...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).dropna(subset=["text"])
    test_df = test_df.reset_index(drop=True)

    X_test = vectorizer.transform(test_df["text"].tolist())
    svm_proba = svm.predict_proba(X_test)
    svm_preds = np.argmax(svm_proba, axis=1)
    svm_conf = np.max(svm_proba, axis=1)

    tier1_mask = svm_conf >= CASCADE_THRESHOLD
    tier2_mask = svm_conf < CASCADE_THRESHOLD

    print(f"\nCascade @ threshold {CASCADE_THRESHOLD}:")
    print(f"  Tier 1 (SVM handles): {tier1_mask.sum()} reviews")
    print(f"  Tier 2 (DeBERTa handles): {tier2_mask.sum()} reviews")

    tier1_df = test_df[tier1_mask].copy()
    tier1_df["svm_pred"] = svm_preds[tier1_mask]
    tier1_df["svm_conf"] = svm_conf[tier1_mask]

    tier2_df = test_df[tier2_mask].copy()
    tier2_df["svm_pred"] = svm_preds[tier2_mask]
    tier2_df["svm_conf"] = svm_conf[tier2_mask]

    def pick_examples(df, n):
        examples = []
        for label in [0, 1]:
            subset = df[df["label"] == label]
            if len(subset) >= (n // 2):
                examples.append(subset.sample(n=n // 2, random_state=42))
        if not examples:
            return df.head(0)
        return pd.concat(examples).sample(frac=1, random_state=42).head(n)

    lime_examples = pick_examples(tier1_df, N_EXAMPLES)
    shap_examples = pick_examples(tier2_df, min(N_EXAMPLES, len(tier2_df)))

    print(f"\n{'='*70}")
    print(f"LIME EXPLANATIONS -- {len(lime_examples)} Tier 1 reviews")
    print(f"{'='*70}")

    for i, (idx, row) in enumerate(lime_examples.iterrows()):
        print(f"\n[Example {i+1}/{len(lime_examples)}]")
        exp = explain_with_lime(
            svm, vectorizer, text=row["text"], label=int(row["label"]),
            pred=int(row["svm_pred"]), confidence=float(row["svm_conf"])
        )
        html_path = os.path.join(OUTPUT_DIR, f"lime_example_{i+1}.html")
        save_lime_html(exp, html_path, int(row["label"]), int(row["svm_pred"]))

    print(f"\n{'='*70}")
    print(f"SHAP EXPLANATIONS -- {len(shap_examples)} Tier 2 reviews")
    print(f"{'='*70}")
    print("Note: SHAP is slow per-review even on GPU. This may take several minutes.")

    for i, (idx, row) in enumerate(shap_examples.iterrows()):
        print(f"\n[Example {i+1}/{len(shap_examples)}]")
        explain_with_shap(
            deberta_model, deberta_tokenizer, device,
            text=row["text"], label=int(row["label"]),
            pred=int(row["svm_pred"]), confidence=float(row["svm_conf"])
        )

    print(f"\n{'='*70}")
    print("XAI layer complete.")
    print(f"LIME HTML files saved to {OUTPUT_DIR}/")
    print("Open any lime_example_N.html in your browser to see visual word highlighting.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
