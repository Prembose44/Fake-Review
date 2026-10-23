"""
main.py
--------
The single backend server for the demo.

Run with:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000

Endpoints:
   - POST /api/predict/text      Feature 1: single review
   - POST /api/predict/product   Feature 2: product link (scrape + analyze)
   - POST /api/predict/reviews   Feature 2b: bookmarklet (reviews sent directly)
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os

import model_utils
import scraper

app = FastAPI(title="Fake Review Detection API")

# Allow bookmarklet (running on Flipkart's domain) to POST to our backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    review: str


class ProductRequest(BaseModel):
    url: str


class ReviewsRequest(BaseModel):
    reviews: List[str]
    product_name: str = "Unknown Product"
    product_url: str = ""


# ---------------------------------------------------------------------------
# Shared helper: run cascade on a list of reviews and build response
# ---------------------------------------------------------------------------
def _analyze_reviews(reviews: list, source: str, product_name: str = "", product_url: str = ""):
    predictions = model_utils.predict_batch(reviews)

    fake_count = sum(1 for p in predictions if p["label"] == "Fake")
    genuine_count = len(predictions) - fake_count
    total = len(predictions)

    avg_genuine_confidence = sum(
        (1 - p["confidence"]) if p["label"] == "Fake" else p["confidence"]
        for p in predictions
    ) / total if total else 0

    trust_score = round(avg_genuine_confidence * 100)

    if trust_score >= 75:
        verdict = "Trustworthy"
    elif trust_score >= 50:
        verdict = "Mostly Trustworthy"
    elif trust_score >= 30:
        verdict = "Use Caution"
    else:
        verdict = "Likely Manipulated"

    svm_only = sum(1 for p in predictions if p["tier_used"] == "SVM")
    escalated = total - svm_only

    flagged = sorted(
        [
            {"text": r, **p}
            for r, p in zip(reviews, predictions)
            if p["label"] == "Fake"
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )[:5]

    return {
        "source": source,
        "demo_mode": model_utils.DEMO_MODE,
        "product_name": product_name,
        "product_url": product_url,
        "total_reviews": total,
        "fake_count": fake_count,
        "genuine_count": genuine_count,
        "trust_score": trust_score,
        "verdict": verdict,
        "tier_breakdown": {"svm_only": svm_only, "escalated_to_deberta": escalated},
        "flagged_reviews": flagged,
    }


# ---------------------------------------------------------------------------
# Feature 1: single review
# ---------------------------------------------------------------------------
@app.post("/api/predict/text")
def predict_text(req: ReviewRequest):
    if not req.review or not req.review.strip():
        raise HTTPException(status_code=400, detail="Review text is empty.")
    result = model_utils.predict_single(req.review)
    result["demo_mode"] = model_utils.DEMO_MODE
    return result


# ---------------------------------------------------------------------------
# Feature 2a: product link (scrape then analyze)
# ---------------------------------------------------------------------------
@app.post("/api/predict/product")
def predict_product(req: ProductRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="URL is empty.")
    try:
        reviews, source = scraper.get_reviews_for_product(req.url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return _analyze_reviews(reviews, source, product_url=req.url)


# ---------------------------------------------------------------------------
# Feature 2b: bookmarklet direct POST (reviews extracted in-browser)
# ---------------------------------------------------------------------------
@app.post("/api/predict/reviews")
def predict_reviews(req: ReviewsRequest):
    if not req.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided.")
    reviews = [r.strip() for r in req.reviews if r.strip()]
    if not reviews:
        raise HTTPException(status_code=400, detail="All reviews were empty.")
    return _analyze_reviews(reviews, "live", product_name=req.product_name, product_url=req.product_url)


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))