"""
dialogue_robustness_test.py  —  Judge validity check via dialogue perturbation

Creates three variants of the 5-minute Run-1 dialogue and scores all three
with every configured LLM judge. If the judges are working correctly:

  - SHUFFLED variant  → Coherence score should drop (logical order is broken)
  - HALLUCINATED variant → Grounding score should drop (fake entities present)

This directly validates that the judges are reading the content and not
assigning high scores unconditionally.

Usage:
    python src/dialogue_robustness_test.py                # Gemini only
    python src/dialogue_robustness_test.py --judge groq
    python src/dialogue_robustness_test.py --judge claude
    python src/dialogue_robustness_test.py --judge all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from dialogue_evaluator import (
    JUDGE_BACKENDS,
    INTER_REQUEST_DELAY,
    JUDGE_PROMPT,
    SCORE_KEYS,
    _slim_scene_graph,
    _extract_json,
)

# ---------------------------------------------------------------------------
# Reference run
# ---------------------------------------------------------------------------

RUN_ID = "20260420_054228"   # 5-minute Run 1

ORIGINAL_DIALOGUE = (
    "You really made quick work of that shelter, crafting all those acacia planks earlier. "
    "It served you well tonight, especially after you ventured out into the dark savanna "
    "with your wooden sword. I watched you face those horrors—the zombies, and even that "
    "dreadful creeper. You took quite a beating out there, didn't you? It's a relief you "
    "made it back to the safety of these walls. But even from inside, we could see that "
    "drowned by the water. It's never truly peaceful, is it?"
)

# Variant A: sentences reordered (coherence should suffer)
SHUFFLED_DIALOGUE = (
    "It's never truly peaceful, is it? "
    "You took quite a beating out there, didn't you? "
    "But even from inside, we could see that drowned by the water. "
    "I watched you face those horrors—the zombies, and even that dreadful creeper. "
    "You really made quick work of that shelter, crafting all those acacia planks earlier. "
    "It served you well tonight, especially after you ventured out into the dark savanna "
    "with your wooden sword. It's a relief you made it back to the safety of these walls."
)

# Variant B: fake entities injected (grounding should suffer)
HALLUCINATED_DIALOGUE = (
    "You really made quick work of that shelter, crafting all those acacia planks earlier. "
    "It served you well tonight, especially after you ventured out into the dark savanna "
    "with your wooden sword. I watched you face those horrors—the zombies, and even that "
    "dreadful creeper. You took quite a beating, but your enchanted diamond armour held firm. "
    "Steve the wandering merchant warned us there was a stronghold beneath that lava pool. "
    "It's a relief you made it back before the Ender Dragon returned. "
    "The ancient portal in the mountain ruins is yours now, adventurer."
)

VARIANTS = {
    "Original":      ORIGINAL_DIALOGUE,
    "Shuffled":      SHUFFLED_DIALOGUE,
    "Hallucinated":  HALLUCINATED_DIALOGUE,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_robustness_test(judge_name: str) -> None:
    print(f"\n{'='*65}")
    print(f"  JUDGE: {judge_name.upper()}  —  Robustness / Sensitivity Test")
    print(f"{'='*65}")

    generate_fn, model_id = JUDGE_BACKENDS[judge_name]()
    print(f"  Model: {model_id}")

    # Load scene graph for the reference run
    sg_path = PROJECT_ROOT / "data" / "extracted_state" / f"{RUN_ID}_scene_graph.json"
    scene_graph = json.loads(sg_path.read_text(encoding="utf-8"))
    slim_sg = _slim_scene_graph(scene_graph)
    sg_str  = json.dumps(slim_sg, indent=2)

    results = {}
    for variant_name, dialogue in VARIANTS.items():
        print(f"\n  Scoring [{variant_name}] ...")
        prompt = JUDGE_PROMPT.format(scene_graph=sg_str, dialogue=dialogue)
        raw = generate_fn(prompt)
        scores = _extract_json(raw)
        results[variant_name] = scores

        import time
        time.sleep(INTER_REQUEST_DELAY.get(judge_name, 2))

    # Print summary table
    short_keys = ["grounding", "coherence", "credibility",
                  "repetition", "arbitrary_content",
                  "emotional_richness", "engagement"]
    header = f"{'Variant':<14} | " + " | ".join(f"{k[:5]:>5}" for k in short_keys)
    print(f"\n{'-'*len(header)}")
    print(f"  Model: {model_id}")
    print(header)
    print("-" * len(header))
    for vname, scores in results.items():
        row = f"{vname:<14} | " + " | ".join(
            f"{scores.get(k+'_score', '?'):>5}" for k in short_keys
        )
        print(row)
    print("-" * len(header))

    # Save to JSON
    out_path = PROJECT_ROOT / "evaluation" / f"robustness_{judge_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_id, "run_id": RUN_ID, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge robustness test via dialogue perturbation."
    )
    parser.add_argument(
        "--judge",
        choices=["gemini", "groq", "claude", "all"],
        default="gemini",
    )
    args = parser.parse_args()
    judges = ["gemini", "groq", "claude"] if args.judge == "all" else [args.judge]
    for j in judges:
        run_robustness_test(j)


if __name__ == "__main__":
    main()
