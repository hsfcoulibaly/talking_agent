"""
Generate Figure 08: 3-way grouped bar chart (Human vs Gemini vs Groq)
across all 7 Adams dimensions — for the Discussion section.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Data ────────────────────────────────────────────────────────────────────
dimensions = ["Grounding", "Coherence", "Credibility", "Repetition",
              "Arb.\nContent", "Emotional\nRichness", "Engagement"]

human  = [4.67, 4.33, 4.17, 5.00, 5.00, 3.33, 4.00]
gemini = [4.67, 5.00, 4.83, 5.00, 5.00, 5.00, 5.00]
groq   = [4.17, 5.00, 4.17, 4.83, 5.00, 4.50, 4.50]

# ── Layout ──────────────────────────────────────────────────────────────────
x     = np.arange(len(dimensions))
width = 0.26
fig, ax = plt.subplots(figsize=(11, 5))

bars_h = ax.bar(x - width, human,  width, label="Human raters",       color="#4C72B0", zorder=3)
bars_g = ax.bar(x,         gemini, width, label="Gemini 2.5 Flash",   color="#55A868", zorder=3)
bars_r = ax.bar(x + width, groq,   width, label="Groq / Llama-3.3-70B", color="#C44E52", zorder=3)

# value labels on top of every bar
for bars in (bars_h, bars_g, bars_r):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

# ── Axes formatting ──────────────────────────────────────────────────────────
ax.set_ylim(2.5, 5.55)
ax.set_xticks(x)
ax.set_xticklabels(dimensions, fontsize=10)
ax.set_ylabel("Average score (1–5 scale)", fontsize=11)
ax.set_title(
    "Three-way Evaluation Comparison — Human Raters vs. LLM Judges\n"
    "(Adams 7 dimensions · averaged over 6 Minecraft runs)",
    fontsize=12, pad=10
)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

# highlight the Emotional Richness column — biggest gap
ax.axvspan(x[5] - 0.45, x[5] + 0.45, color="gold", alpha=0.15, zorder=1,
           label="_nolegend_")
ax.text(x[5], 2.6, "largest gap", ha="center", va="bottom",
        fontsize=8, style="italic", color="goldenrod")

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "analysis", "figures", "08_three_way_comparison.png"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(out_path)}")
