

This is the project repo for "talking_agent".
Get an agent to talk about game-world related things.

-MCW Feb 2026

see https://arxiv.org/pdf/2305.14314

# Dynamic NPC Narrative Generator

**Georgia State University - Master's Project (Non-Thesis)** — *Automated multimodal storytelling using Vision-Language Models (VLMs) and Large Language Models (LLMs).*

## Project overview

The pipeline asks whether an LLM can produce engaging, grounded NPC dialogue from **sequential** gameplay screenshots. Screenshots are processed in order: each frame is summarized by a VLM, observations are merged into a **single structured scene graph** (JSON), then an LLM generates campfire-style dialogue under explicit **grounding constraints** (no entities or plot beats invented beyond the graph).

## Architecture

1. **Per-frame visual extraction (VLM):** A Gemini model (default `gemini-2.0-flash`; override with `GEMINI_MODEL` in `.env` or `--model`) reads each image and returns strict JSON (`location`, `recent_action`, `visible_entities`, etc.).
2. **Temporal consolidation:** Either an LLM merges frames into a chronological scene graph (`entities`, `events`, `player_arc`, …) or a deterministic **simple stitch** (`--simple-consolidate`) for cheaper runs.
3. **Narrative generation (LLM):** The same model role-plays an NPC and speaks only from the scene graph, with rules that limit hallucination.

## Repository layout

```text
talking_agent/
├── data/
│   ├── test_screenshots/   # Preferred input folder (screenshots for batch runs)
│   ├── extracted_state/    # Written: *_frames.json, *_scene_graph.json
│   ├── generated_stories/   # Written: *_npc_dialogue.txt
│   └── analysis/figures/   # Written by src/eda_visualize.py (PNG exports)
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
python src/main_pipeline.py --model gemini-3-flash-preview --max-frames 15 --max-image-side 1024
```

`--simple-consolidate` skips the LLM scene-graph merge (one VLM call per frame plus one dialogue LLM call). Use `--max-frames`, `--sleep`, and `--max-image-side` to limit cost and ease API rate limits on large folders.

If VLM finished but merge/dialogue failed, resume without redoing vision calls:

```bash
python src/main_pipeline.py --from-frames-json data/extracted_state/<run_id>_frames.json --model gemini-3-flash-preview
```

## Evaluation (VLM binary labels)

`evaluation/ground_truth_labels.csv` lists one row per **frame** (`Image_ID`, e.g. `frame_0001.png`) and **condition** (binary checks such as `location_detectable`, `action_described`). Replace the sample **Ground_Truth** and **VLM_Prediction** (0/1) with labels from your rubric and pipeline outputs, then:

```bash
python src/evaluator.py
```

Human study ratings can be recorded from `evaluation/human_scoring_template.csv`.

## EDA / figures from saved runs

After you have `data/extracted_state/*_frames.json` (and matching `*_scene_graph.json`), generate plots and summary stats:

```bash
pip install -r requirements.txt
python src/eda_visualize.py
python src/eda_visualize.py --run-id 20260408_182648
```

PNGs are written to `data/analysis/figures/` (run overview, per-frame metrics, entity types/spans, event timeline, evaluation bars when the CSV is filled).
