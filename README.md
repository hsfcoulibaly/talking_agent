

This is the project repo for "talking_agent".
Get an agent to talk about game-world related things.

-MCW Feb 2026

see https://arxiv.org/pdf/2305.14314

# Dynamic NPC Narrative Generator

**Georgia State University - Master's Project (Non-Thesis)** — *Automated multimodal storytelling using Vision-Language Models (VLMs) and Large Language Models (LLMs).*

## Project overview

The pipeline asks whether an LLM can produce engaging, grounded NPC dialogue from **sequential** gameplay screenshots. Screenshots are processed in order: each frame is summarized by a VLM, observations are merged into a **single structured scene graph** (JSON), then an LLM generates campfire-style dialogue under explicit **grounding constraints** (no entities or plot beats invented beyond the graph).

## Architecture

1. **Per-frame visual extraction (VLM):** `gemini-2.0-flash` reads each image and returns strict JSON (`location`, `recent_action`, `visible_entities`, etc.).
2. **Temporal consolidation:** Either an LLM merges frames into a chronological scene graph (`entities`, `events`, `player_arc`, …) or a deterministic **simple stitch** (`--simple-consolidate`) for cheaper runs.
3. **Narrative generation (LLM):** The same model role-plays an NPC and speaks only from the scene graph, with rules that limit hallucination.

## Repository layout

```text
talking_agent/
├── data/
│   ├── test_screenshots/   # Preferred input folder (screenshots for batch runs)
│   ├── extracted_state/    # Written: *_frames.json, *_scene_graph.json
│   └── generated_stories/   # Written: *_npc_dialogue.txt
├── test_images/            # Legacy sample screenshots (used if test_screenshots is empty)
├── evaluation/
│   ├── ground_truth_labels.csv   # Template + sample rows for F1 evaluation
│   └── human_scoring_template.csv
├── src/
│   ├── main_pipeline.py
│   └── evaluator.py
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. Python 3.10+ recommended.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Do not commit `.env`.

## Run the pipeline

From the **repository root**:

```bash
python src/main_pipeline.py
```

Defaults: uses `data/test_screenshots` if it contains images, otherwise `test_images`. Outputs are timestamped under `data/extracted_state/` and `data/generated_stories/`.

```bash
python src/main_pipeline.py --images-dir path/to/screenshots
python src/main_pipeline.py --simple-consolidate
```

`--simple-consolidate` skips the LLM scene-graph merge (one VLM call per frame plus one dialogue LLM call).

## Evaluation (VLM binary labels)

Fill in `evaluation/ground_truth_labels.csv` with your own `VLM_Prediction` column from experiments, then:

```bash
python src/evaluator.py
```

Human study ratings can be recorded from `evaluation/human_scoring_template.csv`.
