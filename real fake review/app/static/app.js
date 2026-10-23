// ---------- Tab switching ----------
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
  });
});

// ---------- On load: check if bookmarklet passed reviews via ?bm= param ----------
window.addEventListener("DOMContentLoaded", () => {
  const params = new URLSearchParams(window.location.search);
  const bm = params.get("bm");
  if (!bm) return;

  // Clean URL immediately
  window.history.replaceState({}, "", "/");

  // Switch to Tab 2
  tabs.forEach((t) => t.classList.remove("active"));
  panels.forEach((p) => p.classList.remove("active"));
  document.querySelector('[data-tab="product"]').classList.add("active");
  document.getElementById("panel-product").classList.add("active");

  const productStatus = document.getElementById("product-status");
  productStatus.textContent = "Analyzing reviews from Flipkart...";

  let payload;
  try {
    payload = JSON.parse(decodeURIComponent(bm));
  } catch (e) {
    productStatus.textContent = "Error reading bookmarklet data.";
    productStatus.classList.add("error");
    return;
  }

  // POST to our own backend — same origin, no CORS issue
  fetch("/api/predict/reviews", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((r) => r.json())
    .then((data) => {
      renderProductResult(data);
      productStatus.textContent = "Done.";
    })
    .catch((err) => {
      productStatus.textContent = "Error: " + err.message;
      productStatus.classList.add("error");
    });
});

// ---------- Pipeline animation helper ----------
function resetPipeline() {
  ["stage-input", "stage-svm", "stage-gate", "stage-deberta"].forEach((id) =>
    document.getElementById(id).classList.remove("active", "escalated")
  );
  ["link1", "link2", "link3"].forEach((id) =>
    document.getElementById(id).classList.remove("active", "escalated")
  );
}

function animatePipeline(tierUsed) {
  resetPipeline();
  const seq = ["stage-input", "link1", "stage-svm", "link2", "stage-gate"];
  if (tierUsed === "DeBERTa") {
    seq.push("link3", "stage-deberta");
  }
  seq.forEach((id, i) => {
    setTimeout(() => {
      const el = document.getElementById(id);
      el.classList.add(tierUsed === "DeBERTa" ? "escalated" : "active");
    }, i * 160);
  });
}

// ---------- Shared: render LIME / saliency explanation tokens as chips ----------
function renderExplanation(container, explanation) {
  container.innerHTML = "";
  if (!explanation || !explanation.tokens || explanation.tokens.length === 0) {
    container.textContent = "No explanation tokens available.";
    return;
  }
  const label = document.createElement("div");
  label.className = "explain-type-label";
  label.textContent = `${explanation.type.toUpperCase()} attribution`;
  container.appendChild(label);

  const wrap = document.createElement("div");
  wrap.className = "explain-words";
  explanation.tokens.forEach(({ word, weight }) => {
    const chip = document.createElement("span");
    chip.className = "word-chip " + (weight > 0 ? "toward-fake" : "toward-genuine");
    chip.textContent = word;
    chip.title = weight.toFixed(3);
    wrap.appendChild(chip);
  });
  container.appendChild(wrap);
}

// ---------- Feature 1: single review ----------
const runSingleBtn = document.getElementById("run-single");
const reviewInput = document.getElementById("review-input");
const singleStatus = document.getElementById("single-status");
const resultCard = document.getElementById("single-result");

runSingleBtn.addEventListener("click", async () => {
  const text = reviewInput.value.trim();
  if (!text) {
    singleStatus.textContent = "Paste a review first.";
    singleStatus.classList.add("error");
    return;
  }
  singleStatus.classList.remove("error");
  singleStatus.textContent = "Running through cascade...";
  runSingleBtn.disabled = true;
  resultCard.hidden = true;

  try {
    const res = await fetch("/api/predict/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review: text }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Request failed");
    const data = await res.json();

    animatePipeline(data.tier_used);

    document.getElementById("result-label").textContent = data.label;
    document.getElementById("result-label").setAttribute("data-v", data.label);
    document.getElementById("result-label").className =
      "verdict-value " + (data.label === "Fake" ? "fake" : "genuine");

    const pct = Math.round(data.confidence * 100);
    document.getElementById("confidence-fill").style.width = pct + "%";
    document.getElementById("confidence-fill").style.background =
      data.label === "Fake" ? "var(--fake)" : "var(--genuine)";
    document.getElementById("confidence-pct").textContent = pct + "%";
    document.getElementById("tier-value").textContent = data.tier_used;

    const explainRow = document.getElementById("explain-row");
    if (data.demo_mode) {
      explainRow.textContent = "Running in DEMO_MODE — connect your trained model files in backend/models/ for real predictions.";
    } else {
      renderExplanation(explainRow, data.explanation);
    }

    resultCard.hidden = false;
    singleStatus.textContent = "Done.";
  } catch (err) {
    singleStatus.textContent = "Error: " + err.message;
    singleStatus.classList.add("error");
  } finally {
    runSingleBtn.disabled = false;
  }
});

// ---------- Feature 2: product link ----------
const runProductBtn = document.getElementById("run-product");
const urlInput = document.getElementById("url-input");
const productStatus = document.getElementById("product-status");
const productResult = document.getElementById("product-result");

const GAUGE_CIRCUMFERENCE = 327;

runProductBtn.addEventListener("click", async () => {
  const url = urlInput.value.trim();
  if (!url) {
    productStatus.textContent = "Paste a product link first.";
    productStatus.classList.add("error");
    return;
  }
  productStatus.classList.remove("error");
  productStatus.textContent = "Fetching reviews and running cascade...";
  runProductBtn.disabled = true;
  productResult.hidden = true;

  try {
    const res = await fetch("/api/predict/product", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Request failed");
    const data = await res.json();
    renderProductResult(data);
    productStatus.textContent = data.demo_mode
      ? "Done. (DEMO_MODE — connect real models in backend/models/)"
      : "Done.";
  } catch (err) {
    productStatus.textContent = "Error: " + err.message;
    productStatus.classList.add("error");
  } finally {
    runProductBtn.disabled = false;
  }
});

// ---------- Shared result renderer (used by both URL and bookmarklet paths) ----------
function renderProductResult(data) {
  const nameRow = document.getElementById("product-name-row");
  if (data.product_name && data.product_name !== "Unknown Product") {
    nameRow.textContent = data.product_name;
    nameRow.hidden = false;
  } else {
    nameRow.hidden = true;
  }

  document.getElementById("trust-score").textContent = data.trust_score;
  document.getElementById("product-verdict").textContent = data.verdict;
  document.getElementById("product-verdict").className =
    "verdict-value " + (data.trust_score >= 50 ? "genuine" : "fake");

  const offset = GAUGE_CIRCUMFERENCE * (1 - data.trust_score / 100);
  const gaugeFill = document.getElementById("gauge-fill");
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = data.trust_score >= 50 ? "var(--genuine)" : "var(--fake)";

  document.getElementById("genuine-count").textContent = data.genuine_count;
  document.getElementById("fake-count").textContent = data.fake_count;
  document.getElementById("total-count").textContent = data.total_reviews;

  document.getElementById("tier-breakdown").textContent =
    `${data.tier_breakdown.svm_only} resolved at Tier 1 (SVM) · ${data.tier_breakdown.escalated_to_deberta} escalated to Tier 2 (DeBERTa)`;

  document.getElementById("source-tag").textContent =
    data.source === "live" ? "● LIVE DATA" : "● CACHED DATA (fallback)";

  const flaggedList = document.getElementById("flagged-list");
  flaggedList.innerHTML = "";
  if (data.flagged_reviews.length === 0) {
    flaggedList.innerHTML = '<div class="hint">No fake reviews flagged.</div>';
  } else {
    data.flagged_reviews.forEach((r) => {
      const div = document.createElement("div");
      div.className = "flagged-item";

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `FAKE · ${Math.round(r.confidence * 100)}% confidence · ${r.tier_used}`;
      div.appendChild(meta);

      const textEl = document.createElement("div");
      textEl.textContent = r.text;
      div.appendChild(textEl);

      if (r.explanation && r.explanation.tokens && r.explanation.tokens.length) {
        const wrap = document.createElement("div");
        wrap.className = "explain-words";
        r.explanation.tokens.slice(0, 6).forEach(({ word, weight }) => {
          const chip = document.createElement("span");
          chip.className = "word-chip " + (weight > 0 ? "toward-fake" : "toward-genuine");
          chip.textContent = word;
          wrap.appendChild(chip);
        });
        div.appendChild(wrap);
      }

      flaggedList.appendChild(div);
    });
  }

  productResult.hidden = false;
}