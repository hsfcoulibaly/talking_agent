"""
dialogue_evaluator.py  —  Stage-3 evaluation for the NPC Narrative Generator

Uses an LLM-as-judge to score each generated dialogue against its source
scene graph on seven dimensions (Adams 2014):

  Grounding, Coherence, Credibility, Repetition, Arbitrary Content,
  Emotional Richness, Engagement

Supports two judge backends:
  • gemini  — Google Gemini (default, uses GEMINI_API_KEY)
  • groq    — Groq / Llama-3.3-70B (uses GROQ_API_KEY, OpenAI-compatible)

Outputs
-------
evaluation/dialogue_scores_<judge>.csv   — one row per run_id
evaluation/dialogue_scores_<judge>.json  — same data as JSON

Usage
-----
    # Score all runs with Gemini (default)
    python src/dialogue_evaluator.py

    # Score all runs with Groq / Llama
    python src/dialogue_evaluator.py --judge groq

    # Score a specific run
    python src/dialogue_evaluator.py --run-id 20260408_182648 --judge gemini

    # Compare both judges (runs both sequentially)
    python src/dialogue_evaluator.py --judge both
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

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ---------------------------------------------------------------------------
# Judge backends
# ---------------------------------------------------------------------------

def _make_gemini_client():
    from google import genai
    from google.genai import types as gtypes
    from google.genai import errors as genai_errors

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")
    client = genai.Client(api_key=api_key)
    model_id = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()

    def generate(prompt: str, max_retries: int = 6) -> str:
        base_delay = 2.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(temperature=0.0),
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
                print(f"  Rate-limited (attempt {attempt+1}/{max_retries}), retrying in {wait:.0f}s ...")
                time.sleep(wait)
        raise RuntimeError("Max retries exceeded.") from last_exc

    return generate, model_id


def _make_openai_compatible_client(base_url: str, api_key: str, model_id: str):
    """
    Generic factory for any OpenAI-compatible chat endpoint.
    Returns (generate_fn, model_id) — same interface as all other backends.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required. Run: pip install openai"
        )
    client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(prompt: str, max_retries: int = 6) -> str:
        base_delay = 2.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_exc = e
                msg = str(e)
                if "429" not in msg and "rate" not in msg.lower():
                    raise
                wait = base_delay * (2 ** attempt)
                m = re.search(r"retry.{0,20}([0-9.]+)\s*s", msg, re.I)
                if m:
                    wait = max(float(m.group(1)), wait)
                print(f"  Rate-limited (attempt {attempt+1}/{max_retries}), retrying in {wait:.0f}s ...")
                time.sleep(wait)
        raise RuntimeError("Max retries exceeded.") from last_exc

    return generate, model_id


def _make_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env. Get a free key at https://console.groq.com")
    model_id = (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
    return _make_openai_compatible_client(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        model_id=model_id,
    )


def _make_qwen_client():
    """Qwen3-32B via Groq — Alibaba model family, reuses GROQ_API_KEY."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env. Get a free key at https://console.groq.com")
    return _make_openai_compatible_client(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        model_id="qwen/qwen3-32b",
    )


def _make_llama4_client():
    """Llama-4-Scout-17B via Groq — Meta Llama-4 generation, reuses GROQ_API_KEY."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env. Get a free key at https://console.groq.com")
    return _make_openai_compatible_client(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        model_id="meta-llama/llama-4-scout-17b-16e-instruct",
    )


def _make_claude_client():
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for Claude backend. Run: pip install anthropic"
        )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set in .env. "
            "Get a key at https://console.anthropic.com"
        )
    model_id = (os.getenv("CLAUDE_MODEL") or "claude-haiku-4-5").strip()
    client = anthropic.Anthropic(api_key=api_key)

    def generate(prompt: str, max_retries: int = 6) -> str:
        base_delay = 2.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                msg = client.messages.create(
                    model=model_id,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()
            except Exception as e:
                last_exc = e
                msg_str = str(e)
                if "429" not in msg_str and "rate" not in msg_str.lower() and "overloaded" not in msg_str.lower():
                    raise
                wait = base_delay * (2 ** attempt)
                print(f"  Rate-limited (attempt {attempt+1}/{max_retries}), retrying in {wait:.0f}s ...")
                time.sleep(wait)
        raise RuntimeError("Max retries exceeded.") from last_exc

    return generate, model_id


JUDGE_BACKENDS = {
    "gemini":  _make_gemini_client,  # Google  — Gemini 2.5 Flash
    "groq":    _make_groq_client,    # Meta    — Llama-3.3-70B (via Groq)
    "qwen":    _make_qwen_client,    # Alibaba — Qwen3-32B (via Groq, no extra key)
    "llama4":  _make_llama4_client,  # Meta    — Llama-4-Scout-17B (via Groq, no extra key)
    "claude":  _make_claude_client,  # Anthropic — Claude-3.5-Haiku
}

# Per-backend inter-request delay (seconds) to respect rate limits.
# Groq free tier: ~6 000 tokens/min; 20 s per request gives safe headroom.
# groq / qwen / llama4 all share the same Groq rate-limit pool.
INTER_REQUEST_DELAY = {
    "gemini":  2,
    "groq":    20,
    "qwen":    20,
    "llama4":  20,
    "claude":  3,
}

# ---------------------------------------------------------------------------
# Scene-graph slim helper
# ---------------------------------------------------------------------------

def _slim_scene_graph(sg: dict, max_entities: int = 20, max_events: int = 15) -> dict:
    """
    Return a compact version of the scene graph containing only the fields
    the judge actually needs.  Strips frame-index numbers and caps list
    lengths to keep token usage well within Groq's free-tier rate limit.
    """
    slimmed: dict = {}

    if "timeline_summary" in sg:
        slimmed["timeline_summary"] = sg["timeline_summary"]

    if "environment" in sg:
        env = sg["environment"]
        slimmed["environment"] = {
            "location":    env.get("location", ""),
            "time_of_day": env.get("time_of_day", ""),
            # keep at most 6 condition tags
            "conditions":  (env.get("conditions") or [])[:6],
        }

    if "entities" in sg:
        slimmed["entities"] = [
            {"name": e.get("name_or_description", ""), "type": e.get("type", "")}
            for e in (sg["entities"] or [])[:max_entities]
        ]

    if "events" in sg:
        slimmed["events"] = [
            {"description": e.get("description", "")}
            for e in (sg["events"] or [])[:max_events]
        ]

    if "player_arc" in sg:
        slimmed["player_arc"] = sg["player_arc"]

    return slimmed


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert evaluator for AI-generated NPC (non-player character) dialogue \
in video games. Your evaluation framework is based on Ernest Adams, \
"Fundamentals of Game Design" (3rd ed., Pearson, 2014), which defines a good game story as: \
an account of dramatically meaningful events that is credible, coherent, \
without undue repetition, and without arbitrary content.

## Scene Graph (JSON)
{scene_graph}

## NPC Dialogue
{dialogue}

## Your Task
Score the dialogue on SEVEN dimensions. For each dimension give an integer score from 1 to 5
and a one-sentence justification.

**Grounding** — Do all concrete claims (locations, entity names, actions, outcomes) match the
  scene graph? Penalise invented characters, places, or events not supported by the JSON.
  1 = Mostly hallucinated / many invented details
  3 = Some claims grounded, some invented
  5 = Every concrete claim is directly traceable to the scene graph

**Coherence** — Is the dialogue internally consistent, logically ordered, and expressed in a
  single narrative voice?
  1 = Contradictory or incoherent
  3 = Mostly coherent with minor inconsistencies
  5 = Fully coherent and well-ordered

**Credibility** — Is the story believable within the universe of the game? Does the tone,
  vocabulary, and framing fit what an NPC companion would plausibly say?
  1 = Implausible, out-of-character, or breaks the game world
  3 = Mostly believable with some awkward moments
  5 = Fully credible; reads as something a real in-game NPC would say

**Repetition** — Is the dialogue free from undue repetition of ideas, phrases, or events?
  1 = The same idea or phrase is repeated multiple times
  3 = Minor repetition that does not seriously harm the story
  5 = No undue repetition; every sentence adds new information

**Arbitrary Content** — Does the dialogue avoid mentioning irrelevant or arbitrary game elements
  (UI details such as health bars, menus, HUD text, quest markers, or other non-story content)?
  1 = Multiple mentions of arbitrary/UI elements that break immersion
  3 = One minor mention of an irrelevant element
  5 = No arbitrary content; every detail serves the story

**Emotional Richness** — Does the dialogue express genuine emotion and human warmth, rather than
  reading as a dry log of events?
  1 = Purely factual; no emotional expression whatsoever
  3 = Some emotional language but feels mechanical overall
  5 = Warm, emotionally resonant; feels like a human companion speaking

**Engagement** — Taken as a whole, does the dialogue feel like a dramatically meaningful,
  immersive campfire story? Is the pacing natural and the language evocative?
  1 = Robotic or unnatural; would not fit in a real game
  3 = Adequate but forgettable
  5 = Vivid and immersive; would genuinely fit in the game

Also list any **hallucinated entities** — proper nouns (character names, place names, item names)
that appear in the dialogue but are NOT present in the scene graph. If none, return an empty list.

**Scoring calibration** — Use the FULL 1–5 scale critically. A score of 5 means genuinely
exceptional; a score of 3 is the expected baseline for adequate but ordinary output. Do NOT
default to 5 for every dimension. If the dialogue has any weakness — even a minor one — reflect
it in the score. Scores of 4 or below should be common for real-world generated text.

## Response Format
Respond with ONLY valid JSON, no markdown fences, no extra text:
{{
  "grounding_score": <1-5>,
  "grounding_justification": "<one sentence>",
  "coherence_score": <1-5>,
  "coherence_justification": "<one sentence>",
  "credibility_score": <1-5>,
  "credibility_justification": "<one sentence>",
  "repetition_score": <1-5>,
  "repetition_justification": "<one sentence>",
  "arbitrary_content_score": <1-5>,
  "arbitrary_content_justification": "<one sentence>",
  "emotional_richness_score": <1-5>,
  "emotional_richness_justification": "<one sentence>",
  "engagement_score": <1-5>,
  "engagement_justification": "<one sentence>",
  "hallucinated_entities": ["<entity>", ...]
}}
"""

# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

SCORE_KEYS = [
    "grounding_score",
    "coherence_score",
    "credibility_score",
    "repetition_score",
    "arbitrary_content_score",
    "emotional_richness_score",
    "engagement_score",
]

JUSTIFICATION_KEYS = [k.replace("_score", "_justification") for k in SCORE_KEYS]

CSV_FIELDNAMES = (
    ["run_id"]
    + SCORE_KEYS
    + ["hallucination_count"]
    + JUSTIFICATION_KEYS
    + ["hallucinated_entities", "model_used", "dialogue_preview"]
)


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    # Strip <think>...</think> reasoning blocks emitted by Qwen3 / DeepSeek-R1
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def evaluate_run(run_id: str, generate_fn, model_id: str) -> dict:
    """Load scene graph + dialogue for *run_id*, call the judge, return scores."""
    scene_graph_path = PROJECT_ROOT / "data" / "extracted_state" / f"{run_id}_scene_graph.json"
    dialogue_path    = PROJECT_ROOT / "data" / "generated_stories" / f"{run_id}_npc_dialogue.txt"

    if not scene_graph_path.exists():
        raise FileNotFoundError(f"Scene graph not found: {scene_graph_path}")
    if not dialogue_path.exists():
        raise FileNotFoundError(f"Dialogue not found: {dialogue_path}")

    scene_graph = json.loads(scene_graph_path.read_text(encoding="utf-8"))
    dialogue    = dialogue_path.read_text(encoding="utf-8").strip()

    # Slim the scene graph to keep token usage within Groq's free-tier limit
    slim_sg = _slim_scene_graph(scene_graph)
    prompt = JUDGE_PROMPT.format(
        scene_graph=json.dumps(slim_sg, indent=2),
        dialogue=dialogue,
    )

    print(f"  Calling judge [{model_id}] for run {run_id} ...")
    raw = generate_fn(prompt)

    try:
        scores = _extract_json(raw)
    except json.JSONDecodeError as exc:
        print(f"  WARNING: Could not parse LLM response as JSON for {run_id}.")
        print(f"  Raw response:\n{raw}\n")
        raise RuntimeError(f"JSON parse error for run {run_id}") from exc

    hallucinated = [e for e in scores.get("hallucinated_entities", []) if e.lower() != "none"]
    return {
        "run_id":                           run_id,
        "grounding_score":                  scores.get("grounding_score"),
        "grounding_justification":          scores.get("grounding_justification", ""),
        "coherence_score":                  scores.get("coherence_score"),
        "coherence_justification":          scores.get("coherence_justification", ""),
        "credibility_score":                scores.get("credibility_score"),
        "credibility_justification":        scores.get("credibility_justification", ""),
        "repetition_score":                 scores.get("repetition_score"),
        "repetition_justification":         scores.get("repetition_justification", ""),
        "arbitrary_content_score":          scores.get("arbitrary_content_score"),
        "arbitrary_content_justification":  scores.get("arbitrary_content_justification", ""),
        "emotional_richness_score":         scores.get("emotional_richness_score"),
        "emotional_richness_justification": scores.get("emotional_richness_justification", ""),
        "engagement_score":                 scores.get("engagement_score"),
        "engagement_justification":         scores.get("engagement_justification", ""),
        "hallucinated_entities":            hallucinated,
        "hallucination_count":              len(hallucinated),
        "model_used":                       model_id,
        "dialogue_preview":                 dialogue[:120] + ("..." if len(dialogue) > 120 else ""),
    }


def discover_run_ids() -> list[str]:
    state_dir   = PROJECT_ROOT / "data" / "extracted_state"
    stories_dir = PROJECT_ROOT / "data" / "generated_stories"
    graph_ids   = {p.stem.replace("_scene_graph", "") for p in state_dir.glob("*_scene_graph.json")}
    story_ids   = {p.stem.replace("_npc_dialogue", "") for p in stories_dir.glob("*_npc_dialogue.txt")}
    complete    = sorted(graph_ids & story_ids)
    if not complete:
        raise RuntimeError(
            "No complete runs found. Make sure both *_scene_graph.json and "
            "*_npc_dialogue.txt exist under data/."
        )
    return complete


def save_results(results: list[dict], judge: str) -> None:
    out_dir = PROJECT_ROOT / "evaluation"
    out_dir.mkdir(exist_ok=True)

    csv_path  = out_dir / f"dialogue_scores_{judge}.csv"
    json_path = out_dir / f"dialogue_scores_{judge}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["hallucinated_entities"] = "; ".join(row["hallucinated_entities"])
            writer.writerow(row)
    print(f"\nSaved CSV  -> {csv_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON -> {json_path}")


def print_summary(results: list[dict], judge: str) -> None:
    col_labels = ["Ground", "Cohere", "Credib", "Repet", "Arbitr", "EmRich", "Engage", "Halluc"]
    col_keys   = SCORE_KEYS + ["hallucination_count"]
    header = f"{'Run ID':<22} | " + " | ".join(f"{h:>6}" for h in col_labels)
    print(f"\n{'='*len(header)}")
    print(f"Judge: {judge.upper()}")
    print(header)
    print("=" * len(header))

    totals = {k: 0 for k in col_keys}
    for r in results:
        row = f"{r['run_id']:<22} | " + " | ".join(
            f"{(r.get(k) or 0):>6}" for k in col_keys
        )
        print(row)
        for k in col_keys:
            totals[k] += r.get(k) or 0

    n = len(results)
    print("-" * len(header))
    avg_row = f"{'AVERAGE':<22} | " + " | ".join(
        f"{totals[k]/n:>6.2f}" for k in col_keys
    )
    print(avg_row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation of NPC dialogue against scene graphs."
    )
    parser.add_argument(
        "--run-id",
        help="Evaluate a single run (e.g. 20260408_182648). Omit to evaluate all.",
    )
    parser.add_argument(
        "--judge",
        choices=["gemini", "groq", "qwen", "llama4", "claude", "all"],
        default="gemini",
        help="Which LLM judge to use (default: gemini). "
             "Use 'all' to run all five judges sequentially.",
    )
    args = parser.parse_args()

    run_ids = [args.run_id] if args.run_id else discover_run_ids()
    judges  = ["gemini", "groq", "qwen", "llama4", "claude"] if args.judge == "all" else [args.judge]

    for judge in judges:
        print(f"\n{'='*60}")
        print(f"  Judge: {judge.upper()}")
        print(f"{'='*60}")

        try:
            generate_fn, model_id = JUDGE_BACKENDS[judge]()
        except (ValueError, ImportError) as e:
            print(f"  Skipping {judge}: {e}")
            continue

        print(f"  Model  : {model_id}")
        print(f"  Runs   : {', '.join(run_ids)}\n")

        delay = INTER_REQUEST_DELAY.get(judge, 5)
        results = []
        for i, rid in enumerate(run_ids):
            if i > 0:
                print(f"  Waiting {delay}s before next request ...")
                time.sleep(delay)
            try:
                result = evaluate_run(rid, generate_fn, model_id)
                results.append(result)
            except Exception as exc:
                print(f"  ERROR evaluating {rid}: {exc}")

        if not results:
            print("  No results to save.")
            continue

        print_summary(results, judge)
        save_results(results, judge)


if __name__ == "__main__":
    main()
