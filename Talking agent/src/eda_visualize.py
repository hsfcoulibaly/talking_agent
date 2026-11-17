"""
Exploratory analysis and figures from talking_agent pipeline outputs.

Reads data/extracted_state/*_frames.json, *_scene_graph.json, optional
data/generated_stories/*_npc_dialogue.txt, and evaluation/ground_truth_labels.csv.

Usage (from repo root):
  pip install -r requirements.txt
  python src/eda_visualize.py
  python src/eda_visualize.py --run-id 20260408_182648
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = PROJECT_ROOT / "data" / "extracted_state"
STORIES = PROJECT_ROOT / "data" / "generated_stories"
EVAL_CSV = PROJECT_ROOT / "evaluation" / "ground_truth_labels.csv"
FIGURES = PROJECT_ROOT / "data" / "analysis" / "figures"


def discover_run_ids() -> list[str]:
    """Return all run IDs that have at least a scene_graph.json."""
    if not EXTRACTED.is_dir():
        return []
    ids = set()
    for p in EXTRACTED.glob("*_scene_graph.json"):
        ids.add(p.name.removesuffix("_scene_graph.json"))
    return sorted(ids)


def load_frames(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def load_graph(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frames_summary_table(frames: list[dict]) -> pd.DataFrame:
    rows = []
    for fr in frames:
        obs = fr.get("observation") or {}
        if "error" in obs:
            rows.append(
                {
                    "frame_index": fr.get("frame_index"),
                    "has_error": True,
                    "visible_n": 0,
                    "recent_action_len": 0,
                    "location_len": 0,
                    "player_status_len": 0,
                }
            )
            continue
        ve = obs.get("visible_entities") or []
        if isinstance(ve, str):
            ve = [ve]
        ra = obs.get("recent_action") or ""
        loc = obs.get("location") or ""
        ps = obs.get("player_status") or ""
        rows.append(
            {
                "frame_index": fr.get("frame_index"),
                "has_error": False,
                "visible_n": len(ve),
                "recent_action_len": len(str(ra)),
                "location_len": len(str(loc)),
                "player_status_len": len(str(ps)),
            }
        )
    return pd.DataFrame(rows)


def plot_frame_metrics(df: pd.DataFrame, out_path: Path, title_suffix: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle(f"Per-frame VLM text shape {title_suffix}", fontsize=12)

    if df.empty:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    x = df["frame_index"].astype(int)
    axes[0].bar(x, df["visible_n"], color="#2c7fb8", edgecolor="white")
    axes[0].set_ylabel("visible_entities count")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(x, df["recent_action_len"], "o-", label="recent_action chars", color="#2ca25f")
    axes[1].plot(x, df["location_len"], "s--", label="location chars", color="#e34a33")
    axes[1].set_ylabel("character length")
    axes[1].set_xlabel("frame_index")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entity_types(graph: dict, out_path: Path, title_suffix: str) -> None:
    entities = graph.get("entities") or []
    types = [e.get("type") or "unknown" for e in entities]
    c = Counter(types)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(c.keys())
    vals = [c[k] for k in labels]
    ax.barh(labels[::-1], vals[::-1], color="#8856a7", edgecolor="white")
    ax.set_xlabel("count")
    ax.set_title(f"Scene graph entity types {title_suffix}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entity_spans(graph: dict, out_path: Path, title_suffix: str) -> None:
    entities = graph.get("entities") or []
    if not entities:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(entities))))
    ymax = 0
    for i, e in enumerate(entities):
        y = len(entities) - 1 - i
        a = int(e.get("first_frame", 0))
        b = int(e.get("last_frame", a))
        ax.barh(y, b - a + 1, left=a, height=0.6, align="center", color="#3182bd", alpha=0.85)
        label = (e.get("name_or_description") or e.get("id") or "?")[:42]
        ax.text(a, y, f"  {label}", va="center", fontsize=8)
        ymax = max(ymax, b)
    ax.set_yticks(range(len(entities)))
    ax.set_yticklabels([e.get("id", "") for e in reversed(entities)], fontsize=8)
    ax.set_xlabel("frame_index")
    ax.set_title(f"Entity span (first_frame–last_frame) {title_suffix}")
    ax.set_xlim(-0.5, ymax + 1.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_events_timeline(graph: dict, out_path: Path, title_suffix: str) -> None:
    events = graph.get("events") or []
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.55 * len(events))))
    for i, ev in enumerate(events):
        fi = int(ev.get("frame_index", 0))
        desc = (ev.get("description") or "")[:70]
        ax.scatter([fi], [i], s=120, c="#d95f0e", zorder=3)
        ax.text(fi + 0.08, i, desc, va="center", fontsize=8)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([f"e{i}" for i in range(len(events))], fontsize=8)
    ax.set_xlabel("frame_index")
    ax.set_title(f"Events keyed to frames {title_suffix}")
    if events:
        ax.set_xlim(-0.5, max(int(e.get("frame_index", 0)) for e in events) + 1.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_run_overview(run_ids: list[str], out_path: Path) -> None:
    counts = []
    for rid in run_ids:
        p = EXTRACTED / f"{rid}_frames.json"
        if p.is_file():
            n = len(load_frames(p))
        else:
            # LLM-only re-run: infer frame count from scene graph events
            sg_path = EXTRACTED / f"{rid}_scene_graph.json"
            sg = load_graph(sg_path) if sg_path.is_file() else {}
            n = len(sg.get("events") or [])
        counts.append({"run_id": rid, "n_frames": n})
    if not counts:
        return
    df = pd.DataFrame(counts)
    fig, ax = plt.subplots(figsize=(max(8, len(run_ids) * 0.4), 4))
    ax.bar(df["run_id"].astype(str), df["n_frames"], color="#54278f", edgecolor="white")
    ax.set_ylabel("frames per run")
    ax.set_title("Pipeline runs in data/extracted_state")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluation_summary_plot(csv_path: Path, out_path: Path) -> None:
    if not csv_path.is_file():
        return
    df = pd.read_csv(csv_path)
    reqs = {"Image_ID", "Condition_Tested", "Ground_Truth", "VLM_Prediction"}
    if not reqs.issubset(df.columns):
        return
    rows = []
    for cond, g in df.groupby("Condition_Tested"):
        tp = fp = tn = fn = 0
        for _, r in g.iterrows():
            t, p = int(r["Ground_Truth"]), int(r["VLM_Prediction"])
            if t == 1 and p == 1:
                tp += 1
            elif t == 0 and p == 0:
                tn += 1
            elif t == 0 and p == 1:
                fp += 1
            else:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        rows.append({"condition": cond, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc})
    if not rows:
        return
    ev = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(ev))
    w = 0.22
    ax.bar([i - 1.5 * w for i in x], ev["precision"], width=w, label="precision")
    ax.bar([i - 0.5 * w for i in x], ev["recall"], width=w, label="recall")
    ax.bar([i + 0.5 * w for i in x], ev["f1"], width=w, label="F1")
    ax.bar([i + 1.5 * w for i in x], ev["accuracy"], width=w, label="accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ev["condition"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("evaluation/ground_truth_labels.csv (current rows)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_console_summary(run_id: str, frames: list[dict], graph: dict) -> None:
    print(f"\n=== Run {run_id} ===")
    print(f"Frames: {len(frames)}")
    df = frames_summary_table(frames)
    if not df.empty and not df["has_error"].all():
        print(df.describe().to_string())
    ents = graph.get("entities") or []
    evs = graph.get("events") or []
    print(f"Scene graph: {len(ents)} entities, {len(evs)} events, consolidation={graph.get('consolidation_method')!r}")
    ts = (graph.get("timeline_summary") or "")[:200]
    if ts:
        print(f"timeline_summary (trim): {ts}...")
    story_path = STORIES / f"{run_id}_npc_dialogue.txt"
    if story_path.is_file():
        text = story_path.read_text(encoding="utf-8").strip()
        print(f"Dialogue: {len(text)} chars, ~{len(text.split())} words")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA on talking_agent JSON/CSV outputs.")
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Analyze this run id (default: latest available).",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES,
        help="Where to save PNG figures.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_ids = discover_run_ids()
    if not run_ids:
        print(f"No *_frames.json + *_scene_graph.json pairs under {EXTRACTED}. Run the pipeline first.")
        return

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_run_overview(run_ids, args.figures_dir / "00_run_overview.png")

    run_id = args.run_id or run_ids[-1]
    if run_id not in run_ids:
        print(f"Unknown run-id {run_id!r}. Available: {run_ids}")
        return

    frames_path = EXTRACTED / f"{run_id}_frames.json"
    graph_path = EXTRACTED / f"{run_id}_scene_graph.json"
    has_frames = frames_path.is_file()
    frames = load_frames(frames_path) if has_frames else []
    graph = load_graph(graph_path)
    suf = f"({run_id})"

    if not has_frames:
        print(f"  Note: no _frames.json for {run_id} (LLM-only re-run). Skipping per-frame plot.")

    print_console_summary(run_id, frames, graph)

    if has_frames:
        df = frames_summary_table(frames)
        plot_frame_metrics(df, args.figures_dir / f"01_frame_metrics_{run_id}.png", suf)
    plot_entity_types(graph, args.figures_dir / f"02_entity_types_{run_id}.png", suf)
    plot_entity_spans(graph, args.figures_dir / f"03_entity_spans_{run_id}.png", suf)
    plot_events_timeline(graph, args.figures_dir / f"04_events_timeline_{run_id}.png", suf)

    evaluation_summary_plot(EVAL_CSV, args.figures_dir / "05_evaluation_metrics.png")

    print(f"\nFigures written under: {args.figures_dir.resolve()}")
    print("  00_run_overview.png")
    print(f"  01_frame_metrics_{run_id}.png through 04_events_timeline_{run_id}.png")
    print("  05_evaluation_metrics.png (if CSV is valid)")


if __name__ == "__main__":
    main()
