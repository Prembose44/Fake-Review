"""
Step 8: Interactive review tester.

Type any review and the cascade will classify it with a full explanation
of which tier handled it and why.

Usage (from project root, with venv activated):
    python src\\08_test_review.py
"""

import os
import re
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap

MODEL_DIR = "models"
DEBERTA_PATH = os.path.join(MODEL_DIR, "deberta_v3_base", "final")
CASCADE_THRESHOLD = 0.95
MAX_LENGTH = 256


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_models():
    print("Loading models (first run may take ~30 seconds)...")
    svm = joblib.load(os.path.join(MODEL_DIR, "svm_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_PATH, dtype=torch.float32
    )
    model.to(device)
    model.eval()
    print(f"Models loaded. DeBERTa on: {device}\n")
    return svm, vectorizer, model, tokenizer, device


def svm_predict(svm, vectorizer, text):
    X = vectorizer.transform([text])
    proba = svm.predict_proba(X)[0]
    pred = int(np.argmax(proba))
    confidence = float(np.max(proba))
    return pred, confidence, proba


def deberta_predict(model, tokenizer, device, text):
    inputs = tokenizer([text], truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred, confidence, probs


def explain_lime(svm, vectorizer, text):
    def predict_fn(texts):
        X = vectorizer.transform(texts)
        return svm.predict_proba(X)
    explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])
    exp = explainer.explain_instance(text, predict_fn, num_features=8,
                                      num_samples=300, labels=[1])
    print("\n  Key words driving this decision (LIME):")
    for word, weight in exp.as_list(label=1):
        direction = "-> FAKE" if weight > 0 else "-> REAL"
        bar = "#" * max(1, int(abs(weight) * 25))
        print(f"    {word:20s}  {direction}  {bar} ({weight:+.3f})")


def explain_shap(model, tokenizer, device, text):
    words = text.split()
    if len(words) > 50:
        text = " ".join(words[:50])
        print("  (Truncated to 50 words for SHAP speed)")

    def predict_fn(texts):
        all_probs = []
        for t in texts:
            inputs = tokenizer([t], truncation=True, max_length=MAX_LENGTH,
                                padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
                probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
        return np.array(all_probs)

    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(predict_fn, masker, output_names=["REAL", "FAKE"])
    shap_values = explainer([text])

    tokens = text.split()
    fake_shap = shap_values.values[0, :len(tokens), 1]
    token_importance = sorted(zip(tokens, fake_shap), key=lambda x: abs(x[1]), reverse=True)[:8]

    print("\n  Key tokens driving this decision (SHAP):")
    for token, sv in token_importance:
        direction = "-> FAKE" if sv > 0 else "-> REAL"
        bar = "#" * max(1, int(abs(sv) * 60))
        print(f"    {token:20s}  {direction}  {bar} ({sv:+.3f})")


def classify_review(text, svm, vectorizer, model, tokenizer, device):
    print("\n" + "=" * 65)
    print(f"Review: \"{text[:120]}{'...' if len(text) > 120 else ''}\"")
    print("=" * 65)

    cleaned = clean_text(text)

    svm_pred, svm_conf, svm_proba = svm_predict(svm, vectorizer, cleaned)
    print(f"\n[Tier 1 -- TF-IDF + SVM]")
    print(f"  Prediction:  {'FAKE' if svm_pred == 1 else 'REAL'}")
    print(f"  Confidence:  {svm_conf:.1%}")
    print(f"  P(REAL)={svm_proba[0]:.3f}  P(FAKE)={svm_proba[1]:.3f}")

    if svm_conf >= CASCADE_THRESHOLD:
        print(f"\n  Confidence {svm_conf:.1%} >= {CASCADE_THRESHOLD:.0%} threshold")
        print(f"  -> GATE: Tier 1 handles this review (no escalation needed)")
        explain_lime(svm, vectorizer, cleaned)

        print(f"\n{'-'*65}")
        print(f"FINAL VERDICT:  {'FAKE REVIEW' if svm_pred == 1 else 'GENUINE REVIEW'}")
        print(f"Handled by:     Tier 1 (SVM) -- fast path")
        print(f"Confidence:     {svm_conf:.1%}")

    else:
        print(f"\n  Confidence {svm_conf:.1%} < {CASCADE_THRESHOLD:.0%} threshold")
        print(f"  -> GATE: Uncertain -- escalating to Tier 2 (DeBERTa)...")

        deb_pred, deb_conf, deb_proba = deberta_predict(model, tokenizer, device, cleaned)
        print(f"\n[Tier 2 -- DeBERTa-v3-base]")
        print(f"  Prediction:  {'FAKE' if deb_pred == 1 else 'REAL'}")
        print(f"  Confidence:  {deb_conf:.1%}")
        print(f"  P(REAL)={deb_proba[0]:.3f}  P(FAKE)={deb_proba[1]:.3f}")

        print(f"\n  Running SHAP explanation (may take ~60 seconds)...")
        explain_shap(model, tokenizer, device, cleaned)

        print(f"\n{'-'*65}")
        print(f"FINAL VERDICT:  {'FAKE REVIEW' if deb_pred == 1 else 'GENUINE REVIEW'}")
        print(f"Handled by:     Tier 2 (DeBERTa) -- deep analysis path")
        print(f"Confidence:     {deb_conf:.1%}")

        if svm_pred != deb_pred:
            print(f"\n  Note: SVM and DeBERTa disagreed on this review.")
            print(f"  SVM said {'FAKE' if svm_pred==1 else 'REAL'}, "
                  f"DeBERTa said {'FAKE' if deb_pred==1 else 'REAL'}.")
            print(f"  DeBERTa's answer is used (deeper semantic understanding).")


def main():
    svm, vectorizer, model, tokenizer, device = load_models()

    print("=" * 65)
    print("  Fake Review Detector -- Gated Cascade")
    print(f"  Gate threshold: {CASCADE_THRESHOLD:.0%} confidence")
    print("  Type a review to classify, or 'quit' to exit.")
    print("=" * 65)

    test_reviews = [
        "I absolutely love this product! It has completely changed my life and I would recommend it to everyone. The quality is outstanding and the price is very reasonable.",
        "Broke after 2 weeks. Contacted support and they never responded. Total waste of money, avoid this.",
        "decent product, does what it says on the box. nothing special but gets the job done.",
        "I have been using this for a few years now and it still works great. I love it so much that I bought one for my sister and she loves it too. I will definitely buy again.",
    ]

    print("\nRunning 4 built-in test examples first...\n")
    for review in test_reviews:
        classify_review(review, svm, vectorizer, model, tokenizer, device)
        print()

    print("\n" + "=" * 65)
    print("Now enter your own reviews:")
    print("=" * 65)
    while True:
        print()
        review = input("Enter review (or 'quit'): ").strip()
        if review.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        if not review:
            continue
        classify_review(review, svm, vectorizer, model, tokenizer, device)


if __name__ == "__main__":
    main()
