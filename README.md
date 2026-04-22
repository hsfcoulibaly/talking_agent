# Dynamic NPC Narrative Generator

**Georgia State University - Master's Project (Non-Thesis)** — *Automated multimodal storytelling using Vision-Language Models (VLMs) and Large Language Models (LLMs).*

See also: https://arxiv.org/pdf/2305.14314

---

## Project overview

The pipeline asks whether an LLM can produce engaging, grounded NPC dialogue from **sequential** gameplay screenshots. Screenshots are processed in order: each frame is summarized by a VLM, observations are merged into a single structured **scene graph** (JSON), then an LLM generates campfire-style dialogue under explicit **grounding constraints** — no entities, locations, or plot beats may be invented beyond what appears in the scene graph.

---

## Architecture

<<<<<<< HEAD
1. **Per-frame visual extraction (VLM):** A Gemini model (default `gemini-2.0-flash`; override with `GEMINI_MODEL` in `.env` or `--model`) reads each image and returns strict JSON (`location`, `recent_action`, `visible_entities`, etc.).
2. **Temporal consolidation:** Either an LLM merges frames into a chronological scene graph (`entities`, `events`, `player_arc`, …) or a deterministic **simple stitch** (`--simple-consolidate`) for cheaper runs.
3. **Narrative generation (LLM):** The same model role-plays an NPC and speaks only from the scene graph, with rules that limit hallucination.
=======
The pipeline runs in three sequential stages:

```
Screenshots
    |
    v
[Stage 1] Per-frame VLM extraction   --> *_frames.json
    |
    v
[Stage 2] Scene graph consolidation  --> *_scene_graph.json
    |
    v
[Stage 3] NPC dialogue generation    --> *_npc_dialogue.txt
```

**Stage 1 — Per-frame visual extraction (VLM)**
A Gemini model reads each screenshot and returns strict JSON per frame: `location`, `time_of_day`, `player_status`, `recent_action`, `visible_entities`, `notable_enemies_or_objects`.

**Stage 2 — Temporal consolidation**
Either an LLM merges all frame observations into a chronological scene graph with `entities`, `events`, `player_arc`, and `environment` fields, or a deterministic **simple stitch** (`--simple-consolidate`) does the same without an LLM call.

**Stage 3 — Narrative generation (LLM)**
The same model role-plays an NPC companion speaking at a campfire. The prompt constrains the output to 80–120 words, forbids inventing characters/locations/quests not in the scene graph, prohibits mention of game UI elements (health bars, menus, HUD text), requires emotional expression, and disallows repetition of ideas.

---
>>>>>>> 9b2823caeea4531b83eb1eaacc4d6e8cebe1f7ba

## Repository layout

```text
talking_agent/
<<<<<<< HEAD
├── data/
│   ├── test_screenshots/   # Preferred input folder (screenshots for batch runs)
│   ├── extracted_state/    # Written: *_frames.json, *_scene_graph.json
│   ├── generated_stories/   # Written: *_npc_dialogue.txt
│   └── analysis/figures/   # Written by src/eda_visualize.py (PNG exports)
├── test_images/            # Legacy sample screenshots (used if test_screenshots is empty)
├── evaluation/
│   ├── ground_truth_labels.csv   # Template + sample rows for F1 evaluation
│   └── human_scoring_template.csv
=======
>>>>>>> 9b2823caeea4531b83eb1eaacc4d6e8cebe1f7ba
├── src/
│   ├── main_pipeline.py         # Full 3-stage pipeline
│   ├── evaluator.py             # Stage 1 — F1 scoring (VLM extraction)
│   ├── dialogue_evaluator.py    # Stage 3 — LLM-as-judge scoring (dialogue quality)
│   └── eda_visualize.py         # EDA plots and figures from saved runs
├── data/
│   ├── test_screenshots/        # Input: place gameplay screenshots here
│   ├── extracted_state/         # Output: *_frames.json, *_scene_graph.json
│   ├── generated_stories/       # Output: *_npc_dialogue.txt
│   └── analysis/figures/        # Output: PNG plots from eda_visualize.py
├── test_images/                 # Legacy sample screenshots (fallback if test_screenshots empty)
├── evaluation/
│   ├── ground_truth_labels.csv  # Frame-level binary labels for Stage 1 F1 eval
│   ├── dialogue_scores.csv      # LLM-as-judge scores for all evaluated runs
│   ├── dialogue_scores.json     # Same data as structured JSON
│   └── human_scoring_template.csv  # Template for human rater study
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

1. Python 3.10+ recommended.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Do not commit `.env`.

Optionally set `GEMINI_MODEL` in `.env` to override the default model (currently `gemini-2.5-flash`).

---

## Running the pipeline

From the **repository root**:

```bash
python src/main_pipeline.py
```

Default behavior: reads from `data/test_screenshots/` if it contains images, otherwise falls back to `test_images/`. Outputs are timestamped (`YYYYMMDD_HHMMSS`) under `data/extracted_state/` and `data/generated_stories/`.

**Common options:**

```bash
# Use a custom screenshots folder
python src/main_pipeline.py --images-dir path/to/screenshots

# Skip the LLM scene-graph merge (cheaper — one VLM call per frame + one dialogue call)
python src/main_pipeline.py --simple-consolidate
<<<<<<< HEAD
python src/main_pipeline.py --model gemini-3-flash-preview --max-frames 15 --max-image-side 1024
```

`--simple-consolidate` skips the LLM scene-graph merge (one VLM call per frame plus one dialogue LLM call). Use `--max-frames`, `--sleep`, and `--max-image-side` to limit cost and ease API rate limits on large folders.

If VLM finished but merge/dialogue failed, resume without redoing vision calls:

```bash
python src/main_pipeline.py --from-frames-json data/extracted_state/<run_id>_frames.json --model gemini-3-flash-preview
```
=======

# Override model and limit cost on large folders
python src/main_pipeline.py --model gemini-3-flash-preview --max-frames 15 --max-image-side 1024

# Resume after a failed LLM step without re-running VLM extraction
python src/main_pipeline.py --from-frames-json data/extracted_state/<run_id>_frames.json
```

Use `--max-frames`, `--sleep`, and `--max-image-side` to manage API rate limits and costs.
>>>>>>> 9b2823caeea4531b83eb1eaacc4d6e8cebe1f7ba

---

<<<<<<< HEAD
`evaluation/ground_truth_labels.csv` lists one row per **frame** (`Image_ID`, e.g. `frame_0001.png`) and **condition** (binary checks such as `location_detectable`, `action_described`). Replace the sample **Ground_Truth** and **VLM_Prediction** (0/1) with labels from your rubric and pipeline outputs, then:
=======
## Evaluation

The project has two complementary evaluation tracks.

### Stage 1 — VLM extraction accuracy (F1)

`evaluation/ground_truth_labels.csv` lists one row per frame and condition (binary checks such as `location_detectable`, `action_described`). Fill in the `Ground_Truth` and `VLM_Prediction` columns (0/1) from your rubric and pipeline outputs, then run:
>>>>>>> 9b2823caeea4531b83eb1eaacc4d6e8cebe1f7ba

```bash
python src/evaluator.py
```

<<<<<<< HEAD
Human study ratings can be recorded from `evaluation/human_scoring_template.csv`.

## EDA / figures from saved runs

After you have `data/extracted_state/*_frames.json` (and matching `*_scene_graph.json`), generate plots and summary stats:

```bash
pip install -r requirements.txt
=======
Outputs Precision, Recall, and F1 per condition plus a macro-average F1.

> **Note:** This measures Stage 1 only — whether the VLM correctly reads the screenshots. It does not evaluate the quality or accuracy of the generated dialogue.

### Stage 3 — Dialogue quality (LLM-as-judge)

`src/dialogue_evaluator.py` uses a Gemini model as an automated judge. The scoring dimensions are derived from Ernest Adams, *Fundamentals of Game Design* (3rd ed., Pearson, 2014), which defines a good game story as credible, coherent, free of undue repetition, free of arbitrary content, and dramatically meaningful.

For each run the judge loads the scene graph and the generated dialogue and scores **7 dimensions** on a 1–5 scale:

| Dimension | What it measures |
|---|---|
| **Grounding** | All concrete claims traceable to the scene graph; no invented entities or events |
| **Coherence** | Internally consistent, logically ordered, single narrative voice |
| **Credibility** | Believable within the game universe; fits what an NPC would plausibly say |
| **Repetition** | Free from undue repetition of ideas or phrases (5 = none) |
| **Arbitrary Content** | No game-UI elements such as health bars, menus, or HUD text (5 = none) |
| **Emotional Richness** | Genuine emotion and human warmth, not a dry event log |
| **Engagement** | Dramatically meaningful and immersive as a campfire story |

It also flags **hallucinated entities** — proper nouns in the dialogue absent from the scene graph.

```bash
# Score all discovered runs automatically
python src/dialogue_evaluator.py

# Score a specific run
python src/dialogue_evaluator.py --run-id 20260408_182648

# Use a different judge model
python src/dialogue_evaluator.py --model gemini-3-flash-preview
```

Results are saved to `evaluation/dialogue_scores.csv` and `evaluation/dialogue_scores.json`.

> **Note:** The scores below were produced with the previous 3-dimension judge. Re-run `dialogue_evaluator.py` to get updated scores across all 7 dimensions.

### Human evaluation

`evaluation/human_scoring_template.csv` is pre-populated with the 3 run IDs and columns for all 7 dimensions. Provide raters with the corresponding `.txt` files from `data/generated_stories/` and ask them to score each dialogue using the same 1–5 scale. Recommended: 3–5 raters scoring all 3 samples — this allows inter-rater agreement to be computed and compared against the automated scores.

---

## EDA (Exploratory Data Analysis) and figures

After pipeline runs have produced `data/extracted_state/*_frames.json` and matching `*_scene_graph.json`:

```bash
>>>>>>> 9b2823caeea4531b83eb1eaacc4d6e8cebe1f7ba
python src/eda_visualize.py
python src/eda_visualize.py --run-id 20260408_182648
```

<<<<<<< HEAD
PNGs are written to `data/analysis/figures/` (run overview, per-frame metrics, entity types/spans, event timeline, evaluation bars when the CSV is filled).
=======
PNGs are written to `data/analysis/figures/`: run overview, per-frame metrics, entity types and lifespans, event timeline, and evaluation bar charts (when the CSV is filled).
>>>>>>> 9b2823caeea4531b83eb1eaacc4d6e8cebe1f7ba
