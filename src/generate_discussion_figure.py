"""
Generate evaluation figures for the MLSP paper / Master's report:

  06_stage1_confusion_matrices.png  — per-condition TP/TN/FP/FN grids
  07_judge_radar.png                — 5-judge radar across 7 Adams dimensions
  08_three_way_comparison.png       — grouped bar: Human vs Gemini vs Groq

Run from the repo root:
    python src/generate_discussion_figure.py
"""

import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES      = PROJECT_ROOT / "data" / "analysis" / "figures"
GT_CSV       = PROJECT_ROOT / "evaluation" / "ground_truth_labels.csv"

FIGURES.mkdir(parents=True, exist_ok=True)

# ── shared dimension labels ──────────────────────────────────────────────────
DIM_LABELS = ["Grounding", "Coherence", "Credibility", "Repetition",
              "Arb.\nContent", "Emotional\nRichness", "Engagement"]

DIM_SHORT  = ["Ground", "Coher", "Cred", "Repet", "Arb", "EmRich", "Engage"]


# ============================================================
#  Figure 06 — Stage-1 confusion matrices
# ============================================================
def gen_confusion_matrices():
    if not GT_CSV.is_file():
        print(f"  [06] SKIP — {GT_CSV} not found")
        return

    df = pd.read_csv(GT_CSV)
    conditions = [
        "location_detectable",
        "action_described",
        "entities_identified",
        "player_status_described",
    ]
    short = {
        "location_detectable":      "location_detectable",
        "action_described":         "action_described",
        "entities_identified":      "entities_identified",
        "player_status_described":  "player_status_described",
    }

    metrics = {}
    for cond in conditions:
        sub = df[df["Condition_Tested"] == cond]
        tp = ((sub["Ground_Truth"] == 1) & (sub["VLM_Prediction"] == 1)).sum()
        tn = ((sub["Ground_Truth"] == 0) & (sub["VLM_Prediction"] == 0)).sum()
        fp = ((sub["Ground_Truth"] == 0) & (sub["VLM_Prediction"] == 1)).sum()
        fn = ((sub["Ground_Truth"] == 1) & (sub["VLM_Prediction"] == 0)).sum()
        total = tp + tn + fp + fn
        metrics[cond] = dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn), n=int(total))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        "Stage 1 VLM Extraction — Confusion Matrices (360 frames × 4 conditions)",
        fontsize=13, y=1.02
    )

    for ax, cond in zip(axes, conditions):
        m   = metrics[cond]
        mat = np.array([[m["tn"], m["fp"]],
                        [m["fn"], m["tp"]]])
        pct = mat / m["n"] * 100

        colors = [["#55A868", "#C44E52"],
                  ["#C44E52", "#55A868"]]

        for r in range(2):
            for c in range(2):
                ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1,
                                           color=colors[r][c], alpha=0.75))
                ax.text(c + 0.5, 1 - r + 0.55,
                        str(mat[r, c]),
                        ha="center", va="center", fontsize=14, fontweight="bold", color="white")
                ax.text(c + 0.5, 1 - r + 0.35,
                        f"({pct[r, c]:.1f}%)",
                        ha="center", va="center", fontsize=9, color="white")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["GT 1", "GT 0"], fontsize=9)
        ax.set_title(cond.replace("_", "\n"), fontsize=9, pad=6)

        prec = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0
        rec  = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        ax.set_xlabel(f"P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}", fontsize=8)

    green_p = mpatches.Patch(color="#55A868", alpha=0.75, label="Correct (TP/TN)")
    red_p   = mpatches.Patch(color="#C44E52", alpha=0.75, label="Error (FP/FN)")
    fig.legend(handles=[green_p, red_p], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.08))

    out = FIGURES / "06_stage1_confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
#  Figure 07 — 5-judge radar chart
# ============================================================
def gen_radar():
    # 6-run averages (20260420_* Minecraft runs only)
    judges = {
        "Gemini 2.5 Flash":    [4.67, 5.00, 4.83, 5.00, 5.00, 5.00, 5.00],
        "Llama-3.3-70B":       [4.17, 5.00, 4.17, 4.67, 5.00, 4.33, 4.33],
        "Qwen3-32B":           [4.33, 5.00, 4.83, 5.00, 5.00, 4.33, 4.83],
        "Llama-4-Scout":       [5.00, 5.00, 5.00, 5.00, 5.00, 4.83, 4.83],
        "Claude-Haiku":        [4.17, 4.67, 3.67, 4.50, 5.00, 4.17, 3.67],
    }
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728"]

    N      = len(DIM_SHORT)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for (label, scores), color in zip(judges.items(), colors):
        vals = scores + scores[:1]
        ax.plot(angles, vals, "-o", label=label, color=color, linewidth=2, markersize=5)
        ax.fill(angles, vals, alpha=0.06, color=color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), DIM_SHORT, fontsize=11)
    ax.set_ylim(2.5, 5.2)
    ax.set_yticks([3.0, 3.5, 4.0, 4.5, 5.0])
    ax.set_yticklabels(["3.0", "3.5", "4.0", "4.5", "5.0"], fontsize=8)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_title(
        "Inter-Judge Comparison — 5 Judges × 7 Adams Dimensions\n"
        "(averaged over 6 Minecraft runs)",
        fontsize=12, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    out = FIGURES / "07_judge_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
#  Figure 08 — 6-way grouped bar chart (Human + 5 LLM judges)
# ============================================================
def gen_three_way_bar():
    # 6-run averages (20260420_* Minecraft runs only)
    evaluators = [
        ("Human raters",       [4.67, 4.33, 4.17, 5.00, 5.00, 3.33, 4.00], "#4C72B0"),
        ("Gemini 2.5 Flash",   [4.67, 5.00, 4.83, 5.00, 5.00, 5.00, 5.00], "#55A868"),
        ("Llama-3.3-70B",      [4.17, 5.00, 4.17, 4.67, 5.00, 4.33, 4.33], "#C44E52"),
        ("Qwen3-32B",          [4.33, 5.00, 4.83, 5.00, 5.00, 4.33, 4.83], "#FF7F0E"),
        ("Llama-4-Scout",      [5.00, 5.00, 5.00, 5.00, 5.00, 4.83, 4.83], "#9467BD"),
        ("Claude-Haiku",       [4.17, 4.67, 3.67, 4.50, 5.00, 4.17, 3.67], "#8C564B"),
    ]

    n_groups = len(evaluators)
    x        = np.arange(len(DIM_LABELS))
    width    = 0.13
    offsets  = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width

    fig, ax = plt.subplots(figsize=(15, 6))

    for (label, scores, color), offset in zip(evaluators, offsets):
        bars = ax.bar(x + offset, scores, width, label=label, color=color, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_ylim(2.5, 5.75)
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_LABELS, fontsize=10)
    ax.set_ylabel("Average score (1–5 scale)", fontsize=11)
    ax.set_title(
        "Evaluation Comparison — Human Raters vs. 5 LLM Judges\n"
        "(Adams 7 dimensions · averaged over 6 Minecraft runs)",
        fontsize=12, pad=10
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)

    # highlight Emotional Richness column — biggest human-vs-machine gap
    ax.axvspan(x[5] - 0.5, x[5] + 0.5, color="gold", alpha=0.15, zorder=1)
    ax.text(x[5], 2.62, "largest gap", ha="center", va="bottom",
            fontsize=8, style="italic", color="goldenrod")

    plt.tight_layout()

    out = FIGURES / "08_three_way_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
if __name__ == "__main__":
    print("Generating evaluation figures...")
    gen_confusion_matrices()
    gen_radar()
    gen_three_way_bar()
    print("Done.")
