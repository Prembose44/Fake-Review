"""
model_utils.py
----------------
Loads your trained SVM + TF-IDF (Tier 1) and DeBERTa-v3-base (Tier 2) models,
and implements the Gated Cascade prediction logic:

    review -> TF-IDF -> SVM -> confident? -> return fast label
                                  |
                                  no (uncertain)
                                  v
                          DeBERTa-v3-base -> final label

EDIT THE THREE PATHS BELOW to point at your actual saved model files.
Everything else (the predict functions) should work as-is once the paths
are correct, since they just call standard sklearn / HuggingFace APIs.

If the model files are not found, the app runs in DEMO_MODE using a simple
keyword heuristic instead of crashing -- so you can build/test the frontend
and API wiring today, then drop in your real models later with zero
frontend changes.
"""

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../real fake review/app
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # .../real fake review

SVM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "svm_model.joblib")
TFIDF_VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.joblib")
DEBERTA_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "deberta_v3_base", "final")

CONFIDENCE_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# 2. Try to load real models. Fall back to DEMO_MODE if anything is missing.
# ---------------------------------------------------------------------------
DEMO_MODE = False
svm_model = None
tfidf_vectorizer = None
deberta_tokenizer = None
deberta_model = None
torch = None

try:
    if os.path.exists(SVM_MODEL_PATH) and os.path.exists(TFIDF_VECTORIZER_PATH):
        svm_model = joblib.load(SVM_MODEL_PATH)
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    else:
        raise FileNotFoundError("SVM/TF-IDF files not found")

    if os.path.isdir(DEBERTA_MODEL_DIR):
        import torch as _torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        torch = _torch
        deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_DIR)
        deberta_model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL_DIR)
        deberta_model.eval()
    else:
        raise FileNotFoundError("DeBERTa model dir not found")

    print("[model_utils] Loaded real SVM + TF-IDF + DeBERTa-v3-base models.")

except Exception as e:
    DEMO_MODE = True
    print(f"[model_utils] Running in DEMO_MODE (real models not loaded): {e}")


# ---------------------------------------------------------------------------
# 3. Tier 1: SVM + TF-IDF
# ---------------------------------------------------------------------------
def _svm_predict(text: str):
    """Returns (label, confidence) from the Tier 1 SVM classifier."""
    if DEMO_MODE:
        return _demo_heuristic(text)

    X = tfidf_vectorizer.transform([text])
    probs = svm_model.predict_proba(X)[0]          # requires SVC(probability=True)
    classes = svm_model.classes_
    top_idx = probs.argmax()
    raw_label = str(classes[top_idx])
    confidence = float(probs[top_idx])
    # Normalize to Genuine/Fake regardless of training label format
    label_map = {"0": "Genuine", "1": "Fake", "genuine": "Genuine", "fake": "Fake",
             "label_0": "Genuine", "label_1": "Fake"}
    label = label_map.get(raw_label.lower(), raw_label)
    return label, confidence


# ---------------------------------------------------------------------------
# 4. Tier 2: DeBERTa-v3-base (only called when SVM is uncertain)
# ---------------------------------------------------------------------------
def _deberta_predict(text: str):
    """Returns (label, confidence) from the Tier 2 DeBERTa-v3-base model."""
    if DEMO_MODE:
        return _demo_heuristic(text, bias=True)

    inputs = deberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = deberta_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top_idx = int(torch.argmax(probs))
    confidence = float(probs[top_idx])

    # EDIT if your label mapping differs (check your model's config.json id2label)
    raw_label = deberta_model.config.id2label.get(top_idx, str(top_idx)) if hasattr(deberta_model.config, "id2label") else str(top_idx)
    label_map = {"label_0": "Genuine", "label_1": "Fake", "0": "Genuine", "1": "Fake",
             "genuine": "Genuine", "fake": "Fake"}
    label = label_map.get(raw_label.lower(), raw_label)
    return label, confidence


# ---------------------------------------------------------------------------
# 5. Demo heuristic (only used when real models aren't found yet)
# ---------------------------------------------------------------------------
_FAKE_HINTS = ["amazing", "best ever", "must buy", "100%", "highly recommend", "perfect product", "excellent quality"]

def _demo_heuristic(text: str, bias: bool = False):
    t = text.lower()
    hint_hits = sum(1 for h in _FAKE_HINTS if h in t)
    exclaim_density = t.count("!") / max(len(t), 1)

    if hint_hits >= 2 or exclaim_density > 0.02:
        score = min(0.78 + 0.05 * hint_hits, 0.97)
        label = "Fake"
    elif hint_hits == 1:
        score = 0.58
        label = "Fake"
    else:
        score = 0.70 + min(len(t) % 20, 15) / 100  # varies a bit by text, stays genuine-leaning
        label = "Genuine"

    if bias and label == "Fake":
        score = min(score + 0.08, 0.98)

    return label, round(score, 2)


# ---------------------------------------------------------------------------
# 6. The Gated Cascade -- single review prediction (used by both endpoints)
# ---------------------------------------------------------------------------
def predict_single(text: str) -> dict:
    label, confidence = _svm_predict(text)

    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "label": label,
            "confidence": round(confidence, 3),
            "tier_used": "SVM",
            "explanation": {
                "type": "lime",
                # TODO: call your LIME explainer here and return real (token, weight) pairs
                "tokens": []
            }
        }

    # Uncertain -> escalate to Tier 2
    label, confidence = _deberta_predict(text)
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "tier_used": "DeBERTa",
        "explanation": {
            "type": "shap",
            # TODO: call your SHAP explainer here and return real (token, weight) pairs
            "tokens": []
        }
    }


def predict_batch(texts: list) -> list:
    """Runs predict_single over a list of reviews. Used by the product-link feature."""
    return [predict_single(t) for t in texts]
