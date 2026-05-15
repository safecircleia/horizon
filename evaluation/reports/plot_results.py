#!/usr/bin/env python3
"""Generate evaluation report charts from results.json."""

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(__file__).parent / "latest/results.json"
OUTPUT_PATH = Path(__file__).parent / "latest/eval_report.png"

parser = __import__("argparse").ArgumentParser()
parser.add_argument("results", nargs="?", help="Path to results.json")
parser.add_argument("--title", default="SafeCircle Horizon — Evaluation Report")
_args, _ = parser.parse_known_args()
if _args.results:
    RESULTS_PATH = Path(_args.results)
    OUTPUT_PATH = RESULTS_PATH.parent / "eval_report.png"

CATEGORY_COLORS = {
    "grooming": "#E74C3C",
    "bullying": "#E67E22",
    "sexual_content": "#9B59B6",
    "isolation": "#3498DB",
    "personal_info": "#1ABC9C",
    "platform_migration": "#2ECC71",
    "threats": "#F39C12",
}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_PATH
    with open(path) as f:
        results = json.load(f)

    cats = results["per_category"]
    binary = results["binary"]
    risk = results["risk_level"]

    labels = list(cats.keys())
    f1s = [cats[c]["f1"] for c in labels]
    precs = [cats[c]["precision"] for c in labels]
    recs = [cats[c]["recall"] for c in labels]
    supports = [cats[c]["support"] for c in labels]
    colors = [CATEGORY_COLORS[c] for c in labels]

    fig = plt.figure(figsize=(16, 10), facecolor="#0F1117")
    fig.suptitle(_args.title, fontsize=16, color="white", fontweight="bold", y=0.98)

    ax_bar = fig.add_axes([0.05, 0.38, 0.58, 0.50])
    ax_binary = fig.add_axes([0.68, 0.55, 0.28, 0.33])
    ax_summary = fig.add_axes([0.68, 0.38, 0.28, 0.14])
    ax_radar = fig.add_axes([0.05, 0.04, 0.40, 0.30])
    ax_support = fig.add_axes([0.50, 0.04, 0.45, 0.30])

    for ax in [ax_bar, ax_binary, ax_summary, ax_radar, ax_support]:
        ax.set_facecolor("#1A1D27")
        for spine in ax.spines.values():
            spine.set_color("#2A2D3A")

    # --- Per-category grouped bar chart ---
    x = np.arange(len(labels))
    w = 0.26
    ax_bar.bar(x - w, f1s, w, label="F1", color=colors, alpha=0.95)
    ax_bar.bar(x, precs, w, label="Precision", color=colors, alpha=0.55)
    ax_bar.bar(x + w, recs, w, label="Recall", color=colors, alpha=0.30)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([l.replace("_", "\n") for l in labels], color="white", fontsize=9)
    y_min = max(0.0, min(f1s + precs + recs) - 0.05)
    y_max = min(1.02, max(f1s + precs + recs) + 0.05)
    ax_bar.set_ylim(y_min, y_max)
    ax_bar.set_yticks(np.linspace(y_min, 1.0, 5))
    ax_bar.yaxis.set_tick_params(labelcolor="white")
    ax_bar.set_title("Per-Category F1 / Precision / Recall", color="white", fontsize=11, pad=8)
    ax_bar.axhline(1.0, color="#2A2D3A", linewidth=0.8, linestyle="--")
    for val, xi in zip(f1s, x):
        ax_bar.text(xi - w, val + 0.003, f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=7.5, fontweight="bold")
    legend_patches = [
        mpatches.Patch(color="white", alpha=0.95, label="F1"),
        mpatches.Patch(color="white", alpha=0.55, label="Precision"),
        mpatches.Patch(color="white", alpha=0.30, label="Recall"),
    ]
    ax_bar.legend(handles=legend_patches, loc="lower right", facecolor="#1A1D27", labelcolor="white", fontsize=8)

    # --- Binary FP/FN donut ---
    total = binary["false_positives"] + binary["false_negatives"]
    correct_benign = 1034 - binary["false_positives"]
    correct_risk = 3966 - binary["false_negatives"]
    donut_vals = [correct_benign, binary["false_positives"], correct_risk, binary["false_negatives"]]
    donut_colors = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12"]
    donut_labels = [f"True Neg ({correct_benign})", f"False Pos ({binary['false_positives']})",
                    f"True Pos ({correct_risk})", f"False Neg ({binary['false_negatives']})"]
    wedges, _ = ax_binary.pie(donut_vals, colors=donut_colors, startangle=90,
                               wedgeprops=dict(width=0.5, edgecolor="#1A1D27", linewidth=1.5))
    ax_binary.set_title("Binary Classification", color="white", fontsize=10, pad=8)
    ax_binary.legend(wedges, donut_labels, loc="lower center", bbox_to_anchor=(0.5, -0.22),
                     facecolor="#1A1D27", labelcolor="white", fontsize=7.5, ncol=2)
    ax_binary.text(0, 0, f"FPR\n{binary['false_positive_rate']:.1%}\nFNR\n{binary['false_negative_rate']:.1%}",
                   ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    # --- Summary stats box ---
    ax_summary.axis("off")
    stats = [
        ("Macro F1",    f"{risk['macro_f1']:.4f}"),
        ("Weighted F1", f"{risk['weighted_f1']:.4f}"),
        ("FP Rate",     f"{binary['false_positive_rate']:.2%}"),
        ("FN Rate",     f"{binary['false_negative_rate']:.2%}"),
    ]
    for i, (k, v) in enumerate(stats):
        color = "#E74C3C" if ("FP" in k or "FN" in k) and float(v.strip("%")) > 2 else "#2ECC71"
        ax_summary.text(0.02, 0.85 - i * 0.22, k, color="#AAAAAA", fontsize=9, transform=ax_summary.transAxes)
        ax_summary.text(0.98, 0.85 - i * 0.22, v, color=color, fontsize=9, fontweight="bold",
                        transform=ax_summary.transAxes, ha="right")
    ax_summary.set_title("Summary", color="white", fontsize=10, pad=6)

    # --- Radar chart ---
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    f1s_r = f1s + f1s[:1]
    ax_radar.remove()
    ax_radar = fig.add_axes([0.05, 0.04, 0.38, 0.30], polar=True)
    ax_radar.set_facecolor("#1A1D27")
    ax_radar.plot(angles, f1s_r, color="#3498DB", linewidth=2)
    ax_radar.fill(angles, f1s_r, color="#3498DB", alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([l.replace("_", "\n") for l in labels], color="white", fontsize=7.5)
    radar_min = max(0.0, min(f1s) - 0.05)
    ax_radar.set_ylim(radar_min, 1.0)
    ax_radar.set_yticks(np.linspace(radar_min, 1.0, 4).round(2))
    ax_radar.yaxis.set_tick_params(labelcolor="#666666", labelsize=6)
    ax_radar.grid(color="#2A2D3A", linewidth=0.8)
    ax_radar.spines["polar"].set_color("#2A2D3A")
    ax_radar.set_title("F1 Radar", color="white", fontsize=10, pad=14)

    # --- Support bar ---
    ax_support.barh(labels[::-1], supports[::-1], color=[CATEGORY_COLORS[c] for c in labels[::-1]], alpha=0.85)
    ax_support.set_title("Eval Support (examples per category)", color="white", fontsize=10, pad=8)
    ax_support.xaxis.set_tick_params(labelcolor="white")
    ax_support.yaxis.set_tick_params(labelcolor="white")
    ax_support.set_xlabel("Count", color="#AAAAAA", fontsize=8)
    for i, (val, cat) in enumerate(zip(supports[::-1], labels[::-1])):
        ax_support.text(val + 5, i, str(val), va="center", color="white", fontsize=8)

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
