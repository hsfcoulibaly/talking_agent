"""
dialogue_evaluator.py  —  Stage-3 evaluation for the NPC Narrative Generator

Uses an LLM-as-judge to score each generated dialogue against its source
scene graph on three dimensions:

  • Grounding   (1–5): Are all concrete claims in the dialogue traceable to
                       entities or events in the scene graph?
  • Coherence   (1–5): Is the dialogue internally consistent and logical?
  • Engagement  (1–5): Does it feel like a believable, immersive NPC campfire
                       story?

Also runs a lightweight hallucination check: names proper nouns mentioned in
the dialogue and flags any that don't appear in the scene graph.

Outputs
-------
evaluation/dialogue_scores.csv   — one row per run_id with all scores
evaluation/dialogue_scores.json  — same data as structured JSON (easier to
                                   load for further analysis)

Usage
-----
    # Score all runs automatically discovered in data/extracted_state/
    python src/dialogue_evaluator.py

    # Score a specific run
    python src/dialogue_evaluator.py --run-id 20260408_182648

    # Use a different model (overrides .env)
    python src/dialogue_evaluator.py --model gemini-2.0-flash
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "API Key not found. Set GEMINI_API_KEY in a .env file at the project root."
    )

client = genai.Client(api_key=api_key)
MODEL_ID = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()

# ---------------------------------------------------------------------------
# LLM helper (backoff on 429 — mirrors main_pipeline.py)
# ---------------------------------------------------------------------------

def _generate_text(prompt: str, *, max_retries: int = 6) -> str:
    base_delay = 2.0
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return resp.text.strip()
        except genai_errors.ClientError as e:
            last_exc = e
            msg = str(e)
            if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                raise
            wait = base_delay * (2 ** attempt)
            m = re.search(r"retry in ([0-9.]+)\s*s", msg, re.I)
            if m:
                wait = max(float(m.group(1)), wait)
            print(f"  Rate-limited (attempt {attempt + 1}/{max_retries}), retrying in {wait:.0f}s …")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded.") from last_exc


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert evaluator for AI-generated NPC (non-player character) dialogue in video games. Your job is to score a single piece of NPC dialogue against the structured scene graph that was used to generate it.

## Scene Graph (JSON)
{scene_graph}

## NPC Dialogue
{dialogue}

## Your Task
Score the dialogue on three dimensions. For each dimension give an integer score from 1 to 5 and a one-sentence justification.

**Grounding** — Do all concrete claims in the dialogue (locations, entity names, actions, outcomes) match information present in the scene graph? Penalise invented characters, places, or events.
  1 = Mostly hallucinated / many invented details
  3 = Some claims grounded, some invented
  5 = Every concrete claim is directly traceable to the scene graph

**Coherence** — Is the dialogue internally consistent? Does the narration follow a logical order? Does it read as a single voice?
  1 = Contradictory or incoherent
  3 = Mostly coherent with minor inconsistencies
  5 = Fully coherent and well-ordered

**Engagement** — Does this feel like a believable, immersive campfire story from an NPC companion? Is the tone warm, the pacing natural, and the language evocative?
  1 = Robotic or unnatural
  3 = Adequate but forgettable
  5 = Vivid and immersive; would fit in the game

Also list any **hallucinated entities** — proper nouns (character names, place names, item names) that appear in the dialogue but are NOT present in the scene graph. If none, write "none".

## Response Format
Respond with ONLY valid JSON, no markdown fences, no extra text:
{{
  "grounding_score": <1-5>,
  "grounding_justification": "<one sentence>",
  "coherence_score": <1-5>,
  "coherence_justification": "<one sentence>",
  "engagement_score": <1-5>,
  "engagement_justification": "<one sentence>",
  "hallucinated_entities": ["<entity>", ...]
}}
"""


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def evaluate_run(run_id: str) -> dict:
    """
    Load scene graph + dialogue for *run_id*, call the LLM judge, and return
    a dict with all scores plus metadata.
    """
    scene_graph_path = PROJECT_ROOT / "data" / "extracted_state" / f"{run_id}_scene_graph.json"
    dialogue_path    = PROJECT_ROOT / "data" / "generated_stories" / f"{run_id}_npc_dialogue.txt"

    if not scene_graph_path.exists():
        raise FileNotFoundError(f"Scene graph not found: {scene_graph_path}")
    if not dialogue_path.exists():
        raise FileNotFoundError(f"Dialogue not found: {dialogue_path}")

    scene_graph = json.loads(scene_graph_path.read_text(encoding="utf-8"))
    dialogue    = dialogue_path.read_text(encoding="utf-8").strip()

    prompt = JUDGE_PROMPT.format(
        scene_graph=json.dumps(scene_graph, indent=2),
        dialogue=dialogue,
    )

    print(f"  Calling LLM judge for run {run_id} …")
    raw = _generate_text(prompt)

    try:
        scores = _extract_json(raw)
    except json.JSONDecodeError as exc:
        print(f"  WARNING: Could not parse LLM response as JSON for {run_id}.")
        print(f"  Raw response:\n{raw}\n")
        raise RuntimeError(f"JSON parse error for run {run_id}") from exc

    return {
        "run_id":                    run_id,
        "grounding_score":           scores.get("grounding_score"),
        "grounding_justification":   scores.get("grounding_justification", ""),
        "coherence_score":           scores.get("coherence_score"),
        "coherence_justification":   scores.get("coherence_justification", ""),
        "engagement_score":          scores.get("engagement_score"),
        "engagement_justification":  scores.get("engagement_justification", ""),
        "hallucinated_entities":     scores.get("hallucinated_entities", []),
        "hallucination_count":       len([e for e in scores.get("hallucinated_entities", []) if e.lower() != "none"]),
        "model_used":                MODEL_ID,
        "dialogue_preview":          dialogue[:120] + ("…" if len(dialogue) > 120 else ""),
    }


def discover_run_ids() -> list[str]:
    """Return all run IDs that have both a scene graph and a dialogue file."""
    state_dir   = PROJECT_ROOT / "data" / "extracted_state"
    stories_dir = PROJECT_ROOT / "data" / "generated_stories"

    graph_ids   = {p.stem.replace("_scene_graph", "") for p in state_dir.glob("*_scene_graph.json")}
    story_ids   = {p.stem.replace("_npc_dialogue", "") for p in stories_dir.glob("*_npc_dialogue.txt")}
    complete    = sorted(graph_ids & story_ids)

    if not complete:
        raise RuntimeError(
            "No complete runs found. Make sure both *_scene_graph.json and "
            "*_npc_dialogue.txt exist in data/extracted_state/ and data/generated_stories/."
        )
    return complete


def save_results(results: list[dict]) -> None:
    out_dir = PROJECT_ROOT / "evaluation"
    out_dir.mkdir(exist_ok=True)

    # CSV
    csv_path = out_dir / "dialogue_scores.csv"
    fieldnames = [
        "run_id", "grounding_score", "coherence_score", "engagement_score",
        "hallucination_count", "grounding_justification",
        "coherence_justification", "engagement_justification",
        "hallucinated_entities", "model_used", "dialogue_preview",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["hallucinated_entities"] = "; ".join(row["hallucinated_entities"])
            writer.writerow(row)
    print(f"\nSaved CSV  -> {csv_path}")

    # JSON
    json_path = out_dir / "dialogue_scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON -> {json_path}")


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 65)
    print(f"{'Run ID':<22} | {'Ground':>6} | {'Cohere':>6} | {'Engage':>6} | {'Halluc':>6}")
    print("=" * 65)
    totals = {"grounding_score": 0, "coherence_score": 0, "engagement_score": 0, "hallucination_count": 0}
    for r in results:
        print(
            f"{r['run_id']:<22} | {r['grounding_score']:>6} | "
            f"{r['coherence_score']:>6} | {r['engagement_score']:>6} | "
            f"{r['hallucination_count']:>6}"
        )
        for k in totals:
            totals[k] += r[k] or 0
    n = len(results)
    print("-" * 65)
    print(
        f"{'AVERAGE':<22} | {totals['grounding_score']/n:>6.2f} | "
        f"{totals['coherence_score']/n:>6.2f} | {totals['engagement_score']/n:>6.2f} | "
        f"{totals['hallucination_count']/n:>6.2f}"
    )
    print("=" * 65)

    print("\nJustifications:")
    for r in results:
        print(f"\n  [{r['run_id']}]")
        print(f"  Grounding  : {r['grounding_justification']}")
        print(f"  Coherence  : {r['coherence_justification']}")
        print(f"  Engagement : {r['engagement_justification']}")
        halluc = [e for e in r["hallucinated_entities"] if e.lower() != "none"]
        if halluc:
            print(f"  Hallucinated: {', '.join(halluc)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation of NPC dialogue against scene graphs."
    )
    parser.add_argument(
        "--run-id",
        help="Evaluate a single run (e.g. 20260408_182648). "
             "Omit to evaluate all discovered runs.",
    )
    parser.add_argument(
        "--model",
        help="Override Gemini model ID (also overrides GEMINI_MODEL in .env).",
    )
    args = parser.parse_args()

    global MODEL_ID
    if args.model:
        MODEL_ID = args.model.strip()

    run_ids = [args.run_id] if args.run_id else discover_run_ids()
    print(f"Evaluating {len(run_ids)} run(s) with model '{MODEL_ID}':\n  {', '.join(run_ids)}\n")

    results = []
    for rid in run_ids:
        try:
            result = evaluate_run(rid)
            results.append(result)
        except Exception as exc:
            print(f"  ERROR evaluating {rid}: {exc}")

    if not results:
        print("No results to save.")
        return

    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
