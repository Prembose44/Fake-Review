"""
Step 7: Generate results visualization charts for the internship report.

Uses the Maxwell-only cascade results (clean, validated labels).

Usage (from project root, with venv activated):
    python src\\07_visualize_results.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = "outputs"

# Maxwell-only results from 05_cascade_eval.py
RESULTS = {
    "models":     ["Pure SVM", "Cascade\n(t=0.80)", "Cascade\n(t=0.85)",
                   "Cascade\n(t=0.90)", "Cascade\n(t=0.95)", "Pure DeBERTa"],
    "accuracy":   [0.9389, 0.9648, 0.9680, 0.9693, 0.9713, 0.9690],
    "f1":         [0.9385, 0.9649, 0.9681, 0.9695, 0.9717, 0.9697],
    "latency_ms": [2.137,  3.903,  4.475,  5.161,  6.475,  16.686],
    "escalation": [0.000,  0.106,  0.140,  0.181,  0.260,  1.000],
}

C_SVM      = "#4C72B0"
C_CASCADE  = "#55A868"
C_DEBERTA  = "#C44E52"
C_HIGHLIGHT= "#DD8452"   # recommended threshold


def make_charts():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Gated Cascade: Accuracy vs. Efficiency Tradeoff\n"
        "Fake Review Detection -- TF-IDF+SVM -> DeBERTa-v3-base (Maxwell dataset)",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    models     = RESULTS["models"]
    accuracy   = RESULTS["accuracy"]
    f1         = RESULTS["f1"]
    latency    = RESULTS["latency_ms"]

    # t=0.95 is now the recommended threshold (index 4)
    bar_colors = [C_SVM, C_CASCADE, C_CASCADE, C_CASCADE, C_HIGHLIGHT, C_DEBERTA]

    # Chart 1: F1 comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(models, [v * 100 for v in f1], color=bar_colors,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax1.set_ylim(92, 99)
    ax1.set_ylabel("F1 Score (%)", fontsize=11)
    ax1.set_title("F1 Score by Model", fontsize=12, fontweight="bold")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="x", labelsize=8.5)
    for bar, val in zip(bars, f1):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val*100:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Chart 2: Latency comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(models, latency, color=bar_colors,
                    edgecolor="white", linewidth=0.8, zorder=3)
    ax2.set_ylabel("Avg Latency (ms/review)", fontsize=11)
    ax2.set_title("Inference Latency by Model", fontsize=12, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="x", labelsize=8.5)
    for bar, val in zip(bars2, latency):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}ms", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Chart 3: Accuracy vs Latency tradeoff (main result)
    ax3 = fig.add_subplot(gs[1, :])

    ax3.scatter([latency[0]], [accuracy[0] * 100], color=C_SVM, s=200,
                zorder=5, marker="D", label="Pure SVM")
    ax3.scatter([latency[-1]], [accuracy[-1] * 100], color=C_DEBERTA, s=200,
                zorder=5, marker="D", label="Pure DeBERTa")

    casc_latency  = latency[1:5]
    casc_accuracy = [v * 100 for v in accuracy[1:5]]
    casc_labels   = ["t=0.80", "t=0.85", "t=0.90", "t=0.95*"]
    casc_colors   = [C_CASCADE, C_CASCADE, C_CASCADE, C_HIGHLIGHT]

    ax3.plot(casc_latency, casc_accuracy, color=C_CASCADE, linewidth=1.5,
             linestyle="--", alpha=0.6, zorder=3)
    for lx, ay, lbl, col in zip(casc_latency, casc_accuracy, casc_labels, casc_colors):
        ax3.scatter([lx], [ay], color=col, s=180, zorder=6,
                    edgecolors="white", linewidth=1.5)
        ax3.annotate(lbl, (lx, ay), textcoords="offset points", xytext=(6, 8),
                     fontsize=9, color=col, fontweight="bold")

    ax3.annotate(
        "  Recommended\n  threshold (t=0.95)\n  97.13% acc @ 6.48ms",
        xy=(latency[4], accuracy[4] * 100),
        xytext=(latency[4] + 1.0, accuracy[4] * 100 - 0.35),
        fontsize=9, color=C_HIGHLIGHT,
        arrowprops=dict(arrowstyle="->", color=C_HIGHLIGHT, lw=1.5)
    )

    speedup = latency[-1] / latency[4]
    ax3.annotate(f"  {speedup:.1f}x faster\n  than pure DeBERTa",
                 xy=((latency[4] + latency[-1]) / 2, (accuracy[4] + accuracy[-1]) / 2 * 100 - 0.1),
                 fontsize=9, color="#666666", ha="center")
    ax3.annotate("", xy=(latency[-1], accuracy[-1] * 100), xytext=(latency[4], accuracy[4] * 100),
                 arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.2))

    ax3.set_xlabel("Avg Latency (ms/review)", fontsize=12)
    ax3.set_ylabel("Accuracy (%)", fontsize=12)
    ax3.set_title("Accuracy vs. Inference Latency -- Efficiency-Accuracy Tradeoff",
                  fontsize=12, fontweight="bold")
    ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax3.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax3.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=C_SVM, label="Pure SVM (Tier 1 only)"),
        mpatches.Patch(color=C_CASCADE, label="Cascade (various thresholds)"),
        mpatches.Patch(color=C_HIGHLIGHT, label="Cascade t=0.95 (recommended)"),
        mpatches.Patch(color=C_DEBERTA, label="Pure DeBERTa (Tier 2 only)"),
    ]
    ax3.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.savefig(os.path.join(OUTPUT_DIR, "cascade_results_chart.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "cascade_results_chart.pdf"), bbox_inches="tight")
    print(f"Saved cascade_results_chart.png and .pdf to {OUTPUT_DIR}/")
    plt.show()


if __name__ == "__main__":
    make_charts()
