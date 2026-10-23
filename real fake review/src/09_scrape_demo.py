"""
Step 9: One-time live demo scraper for e-commerce review pages.

WARNING: This is a DEMO-ONLY proof of concept, not production code.
Scraping e-commerce sites may violate their Terms of Service. This script
is intended to be run ONCE, live, during an academic presentation to show
feasibility -- not deployed as a running service or used repeatedly.

Targets Flipkart first, since its product/review pages are more often
server-side rendered (reviews appear in raw HTML) compared to Amazon.in
or Meesho, which load reviews via JavaScript and are much harder to scrape
with simple requests + BeautifulSoup.

Usage:
    python src\\09_scrape_demo.py "<flipkart_product_review_url>"
"""

import sys
import re
import requests
from bs4 import BeautifulSoup
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}


def fetch_page(url):
    print(f"Fetching: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    print(f"Status code: {resp.status_code}")
    if resp.status_code != 200:
        print("Non-200 response -- site may be blocking automated requests.")
    return resp.text


def extract_flipkart_reviews(html):
    """
    Flipkart review text typically sits in <div> elements with classes
    containing 'review' or similar. Flipkart changes class names often,
    so this uses a few fallback strategies.
    """
    soup = BeautifulSoup(html, "html.parser")
    reviews = []

    # Strategy 1: common Flipkart review text container classes (may break
    # if Flipkart changes their frontend -- that's expected and OK for a demo)
    candidates = soup.find_all("div", class_=re.compile(r"(ZmyHeo|qwjRop|t-ZTKy)"))
    for c in candidates:
        text = c.get_text(separator=" ", strip=True)
        if len(text.split()) >= 4:
            reviews.append(text)

    # Strategy 2 fallback: any <div> with a decent amount of plain text
    # that looks review-like (heuristic, last resort)
    if not reviews:
        print("Primary selectors found nothing -- trying fallback heuristic...")
        for div in soup.find_all("div"):
            text = div.get_text(separator=" ", strip=True)
            words = text.split()
            if 8 <= len(words) <= 150 and not div.find("div"):
                reviews.append(text)

    return reviews


def main():
    if len(sys.argv) < 2:
        print('Usage: python src\\09_scrape_demo.py "<flipkart_review_url>"')
        sys.exit(1)

    url = sys.argv[1]
    html = fetch_page(url)

    print("\nExtracting reviews...")
    reviews = extract_flipkart_reviews(html)

    print(f"\nFound {len(reviews)} candidate review texts.\n")
    for i, r in enumerate(reviews[:15], 1):
        print(f"[{i}] {r[:150]}{'...' if len(r) > 150 else ''}")

    if not reviews:
        print("\nNo reviews extracted. This likely means:")
        print("  - The page requires JavaScript rendering (reviews loaded dynamically)")
        print("  - Flipkart's bot detection blocked or altered the response")
        print("  - The CSS class names have changed since this script was written")
        print("\nFor the demo, consider using Selenium (headless browser) instead,")
        print("or fall back to a pre-saved set of reviews if live scraping fails.")
    else:
        # Save to a file so it can be fed into the cascade
        with open("outputs/scraped_reviews.txt", "w", encoding="utf-8") as f:
            for r in reviews:
                f.write(r + "\n---\n")
        print(f"\nSaved {len(reviews)} reviews to outputs/scraped_reviews.txt")
        print("Next: feed these into 08_test_review.py or a batch scoring script.")


if __name__ == "__main__":
    main()
