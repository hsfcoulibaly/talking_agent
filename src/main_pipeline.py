# Install required libraries if not already installed:
# pip install -q -U google-genai python-dotenv pillow

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "API Key not found. Set GEMINI_API_KEY in a .env file at the project root "
        "(see .env.example)."
    )

client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-2.0-flash"


def discover_image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    paths = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name.lower())


def default_images_directory() -> Path:
    preferred = PROJECT_ROOT / "data" / "test_screenshots"
    legacy = PROJECT_ROOT / "test_images"
    if preferred.is_dir() and discover_image_paths(preferred):
        return preferred
    return legacy


def extract_game_state(image_path: str | Path, frame_index: int = 0) -> dict:
    """Per-frame VLM extraction: structured observation from one screenshot."""
    path = Path(image_path)
    print(f"Analyzing frame {frame_index}: {path.name}")

    try:
        image = Image.open(path)
    except FileNotFoundError:
        return {"error": f"Image not found at {path}"}

    prompt = f"""
    This is frame index {frame_index} in a time-ordered gameplay sequence (lower index = earlier).
    Analyze this gameplay screenshot. Extract details an NPC companion would notice.
    Return EXACTLY one JSON object with these keys:
    - "location": general environment (e.g. "dark cave", "forest path").
    - "time_of_day": e.g. "night", "day", "dusk", or "unclear".
    - "player_status": visible health, armor, or status UI if any; else "unclear".
    - "recent_action": what is happening or just happened in this frame.
    - "notable_enemies_or_objects": creatures or important objects/items (string or short list).
    - "visible_entities": array of short strings naming distinct people/creatures/objects you can name.
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        print("Warning: VLM did not return valid JSON.")
        return {"raw_text": response.text}


def consolidate_scene_graph_llm(frames: list[dict]) -> dict:
    """Merge ordered per-frame observations into one chronological scene graph (JSON)."""
    payload = json.dumps(frames, indent=2)
    schema_hint = """
    Return EXACTLY one JSON object with:
    - "timeline_summary": one or two sentences ordering what happened across frames.
    - "environment": object with "location", "time_of_day", "conditions" (array of short strings).
    - "entities": array of objects, each with "id" (e.g. e1), "name_or_description", "type"
      (one of: player_hint, npc, creature, object, environment, unknown), "first_frame", "last_frame".
    - "events": array of objects, each with "frame_index", "description",
      "involved_entity_ids" (array of ids from entities).
    - "player_arc": object with "status_notes" and "notable_actions" (array of short strings).
    Do not invent unknown enemies or locations; ground everything in the input observations.
    """
    prompt = f"""
    You are merging per-frame gameplay observations into a single structured scene graph.
    Observations are already in chronological order.

    {schema_hint}

    Per-frame data:
    {payload}
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    try:
        graph = json.loads(response.text)
        graph["consolidation_method"] = "llm"
        return graph
    except json.JSONDecodeError:
        print("Warning: consolidator did not return valid JSON; falling back to simple stitch.")
        return consolidate_scene_graph_simple(frames)


def _split_notable(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    parts = re.split(r"[,;]", str(val))
    return [p.strip() for p in parts if p.strip()]


def consolidate_scene_graph_simple(frames: list[dict]) -> dict:
    """Deterministic merge when skipping the LLM consolidation call (no extra API usage)."""
    events: list[dict] = []
    entity_names: dict[str, None] = {}
    last_env = {"location": "", "time_of_day": ""}
    status_notes: list[str] = []
    actions: list[str] = []

    for fr in frames:
        obs = fr.get("observation") or {}
        if "error" in obs:
            continue
        idx = fr.get("frame_index", 0)
        last_env["location"] = str(obs.get("location") or last_env["location"])
        last_env["time_of_day"] = str(
            obs.get("time_of_day") or last_env["time_of_day"]
        )
        act = str(obs.get("recent_action") or "").strip()
        if act:
            events.append(
                {
                    "frame_index": idx,
                    "description": act,
                    "involved_entity_ids": [],
                }
            )
            actions.append(act)
        st = str(obs.get("player_status") or "").strip()
        if st and st.lower() != "unclear":
            status_notes.append(st)
        for name in _split_notable(obs.get("visible_entities")):
            entity_names.setdefault(name, None)
        for name in _split_notable(obs.get("notable_enemies_or_objects")):
            entity_names.setdefault(name, None)

    entities = [
        {
            "id": f"e{i}",
            "name_or_description": name,
            "type": "unknown",
            "first_frame": 0,
            "last_frame": max((fr.get("frame_index", 0) for fr in frames), default=0),
        }
        for i, name in enumerate(sorted(entity_names.keys()))
    ]

    summary = (
        " ".join(e["description"] for e in events)
        if events
        else "Limited observations across frames."
    )

    return {
        "timeline_summary": summary[:500],
        "environment": {
            "location": last_env["location"],
            "time_of_day": last_env["time_of_day"],
            "conditions": [],
        },
        "entities": entities,
        "events": events,
        "player_arc": {
            "status_notes": "; ".join(dict.fromkeys(status_notes)),
            "notable_actions": actions,
        },
        "consolidation_method": "simple_stitch",
    }


def generate_npc_story(scene_graph: dict) -> str:
    """LLM dialogue grounded in the consolidated scene graph with explicit story constraints."""
    print("Generating NPC dialogue...")

    if not scene_graph or (
        scene_graph.get("consolidation_method") == "simple_stitch"
        and not scene_graph.get("events")
        and not scene_graph.get("entities")
    ):
        return (
            "NPC Companion: \"The trail's muddled—I can scarcely piece what happened. "
            "Walk me through it when you can.\""
        )

    constraints = """
    Storytelling constraints (follow strictly):
    - Ground every concrete claim in the provided JSON. Do not invent quests, characters,
      or locations not supported by entities, events, or environment.
    - If information is thin, stay in-character but vague; do not fill gaps with specific lore.
    - One short paragraph (3-6 sentences). Speak to the player in second person ("you").
    - Campfire recap tone: reflective, immediate aftermath—not omniscient narration.
    """

    prompt = f"""
    You are a loyal NPC companion at a campfire recounting what just unfolded in-game.

    {constraints}

    Scene graph (only source of truth):
    {json.dumps(scene_graph, indent=2)}

    Write the companion's spoken dialogue only, no preface or stage directions.
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    return response.text or ""


def run_pipeline(
    images_dir: Path,
    *,
    use_llm_consolidation: bool = True,
    run_id: str | None = None,
) -> dict:
    """Run full pipeline: VLM per frame -> scene graph -> NPC line. Saves JSON and story text."""
    paths = discover_image_paths(images_dir)
    if not paths:
        raise FileNotFoundError(
            f"No images in {images_dir}. Add .jpg/.png screenshots, or pass --images-dir."
        )

    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    extracted_dir = PROJECT_ROOT / "data" / "extracted_state"
    stories_dir = PROJECT_ROOT / "data" / "generated_stories"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    stories_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict] = []
    for i, path in enumerate(paths):
        obs = extract_game_state(path, frame_index=i)
        try:
            rel = path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = path
        frames.append(
            {
                "frame_index": i,
                "source_image": str(rel).replace("\\\\", "/"),
                "observation": obs,
            }
        )

    frames_path = extracted_dir / f"{rid}_frames.json"
    frames_path.write_text(json.dumps(frames, indent=2), encoding="utf-8")
    print(f"Wrote per-frame extractions: {frames_path.relative_to(PROJECT_ROOT)}")

    if use_llm_consolidation:
        scene_graph = consolidate_scene_graph_llm(frames)
    else:
        scene_graph = consolidate_scene_graph_simple(frames)

    graph_path = extracted_dir / f"{rid}_scene_graph.json"
    graph_path.write_text(json.dumps(scene_graph, indent=2), encoding="utf-8")
    print(f"Wrote scene graph: {graph_path.relative_to(PROJECT_ROOT)}")

    story = generate_npc_story(scene_graph)
    story_path = stories_dir / f"{rid}_npc_dialogue.txt"
    story_path.write_text(story.strip() + "\n", encoding="utf-8")
    print(f"Wrote dialogue: {story_path.relative_to(PROJECT_ROOT)}")

    print("\n--- Scene graph (summary) ---")
    print(json.dumps(scene_graph, indent=2)[:4000])
    if len(json.dumps(scene_graph)) > 4000:
        print("... (truncated in console; see JSON file)")

    print("\n--- NPC dialogue ---")
    print(story)

    return {
        "run_id": rid,
        "frames_path": str(frames_path),
        "scene_graph_path": str(graph_path),
        "story_path": str(story_path),
        "scene_graph": scene_graph,
        "story": story,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential VLM extraction -> scene graph -> grounded NPC dialogue."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Folder of screenshots (default: data/test_screenshots if non-empty, else test_images).",
    )
    parser.add_argument(
        "--simple-consolidate",
        action="store_true",
        help="Stitch scene graph without an extra LLM call (cheaper; less structured).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    img_dir = args.images_dir
    if img_dir is None:
        img_dir = default_images_directory()
    img_dir = img_dir.resolve()

    run_pipeline(
        img_dir,
        use_llm_consolidation=not args.simple_consolidate,
    )
