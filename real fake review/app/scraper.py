"""
scraper.py
-----------
Scrapes reviews from a Flipkart/Meesho product URL using Playwright.

Live scraping is tried first with multiple CSS selector fallbacks.
If all live attempts fail (anti-bot, DOM change, timeout), falls back
to a cached JSON file so your demo never breaks on stage.

HOW TO ADD CACHED PRODUCTS FOR YOUR DEMO:
1. Find a Flipkart/Meesho product you plan to demo
2. Save 10-20 reviews as a JSON array of strings to cache/<name>.json
3. Add an entry to CACHE_MAP mapping a URL substring to that filename

CACHE_MAP example:
    "samsung-galaxy-f15" : "galaxy_f15.json",
    "boat-airdopes"      : "boat_airdopes.json",
"""

import os
import json
import random
import time

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# Map a recognizable substring of the product URL -> cached JSON filename
CACHE_MAP = {
    "sample-product": "sample_product.json",
    # Add your demo products here, e.g.:
    # "samsung-galaxy": "galaxy.json",
    # "boat-airdopes": "boat.json",
}

SCRAPE_TIMEOUT_MS = 20000  # 20 seconds

# Flipkart CSS selectors for review text, tried in order.
# If the first breaks (Flipkart redeploys), the next ones are attempted.
FLIPKART_SELECTORS = [
    "span.css-1jxf684",          # current (inspected June 2025)
    "div.ZmyHeo",                # older Flipkart class
    "div.t-ZTKy",                # another known variant
    "p.g3ENOe",                  # fallback
    "[class*='review'] p",       # generic fuzzy fallback
    "[class*='Review'] span",    # generic fuzzy fallback
]

MEESHO_SELECTORS = [
    "p.sc-eDvSVe",               # known Meesho review text class
    "div.review-text",
    "[class*='review'] p",
]

SCRAPE_TIMEOUT_MS = 20000

def _load_cache(filename: str):
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_cache_for_url(url: str):
    for key, filename in CACHE_MAP.items():
        if key in url:
            return _load_cache(filename)
    return None


def _is_meesho(url: str) -> bool:
    return "meesho.com" in url.lower()


def scrape_live(url: str):
    """
    Attempts a live scrape using Playwright with anti-bot evasion.
    Tries multiple CSS selectors in sequence.
    Returns a list of review strings, or raises on failure.
    """
    from playwright.sync_api import sync_playwright

    selectors = MEESHO_SELECTORS if _is_meesho(url) else FLIPKART_SELECTORS

    reviews = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ]
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 768},
            locale="en-IN",
            extra_http_headers={
                "Accept-Language": "en-IN,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

        # Hide webdriver fingerprint
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        page = context.new_page()
        page.goto(url, timeout=SCRAPE_TIMEOUT_MS, wait_until="domcontentloaded")

        # Random human-like delay
        time.sleep(random.uniform(2.5, 4.0))

        # Scroll down to trigger lazy-loaded reviews
        page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        time.sleep(1.5)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1.0)

        # Try each selector until we get reviews
        for selector in selectors:
            try:
                elements = page.query_selector_all(selector)
                found = []
                for el in elements:
                    text = el.inner_text().strip()
                    # Filter out nav/UI text — reviews are usually >30 chars
                    if len(text) > 30:
                        found.append(text)
                if found:
                    print(f"[scraper] Selector '{selector}' found {len(found)} reviews")
                    reviews = found
                    break
                else:
                    print(f"[scraper] Selector '{selector}' matched 0 usable reviews, trying next...")
            except Exception as sel_err:
                print(f"[scraper] Selector '{selector}' error: {sel_err}, trying next...")

        browser.close()

    if not reviews:
        raise ValueError(
            "No reviews found with any selector. "
            "The page may have blocked the scraper or changed its DOM. "
            "Falling back to cache."
        )

    return reviews


def get_reviews_for_product(url: str):
    """
    Main entry point used by the API.
    1. Try live scraping
    2. Fall back to cache
    3. Raise if neither works

    Returns: (reviews: list[str], source: "live" | "cached")
    """
    try:
        reviews = scrape_live(url)
        if reviews:
            return reviews, "live"
    except Exception as e:
        print(f"[scraper] Live scrape failed: {e}")
        print("[scraper] Trying cache fallback...")

    cached = _find_cache_for_url(url)
    if cached:
        print(f"[scraper] Serving from cache for URL: {url}")
        return cached, "cached"

    raise RuntimeError(
        "Could not scrape this URL live and no cached fallback exists. "
        "Either the site blocked the scraper, or you need to add a cache "
        "entry in scraper.py CACHE_MAP for this product URL."
    )