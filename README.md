# Dynamic NPC Narrative Generator

**Georgia State University - Master's Project (Non-Thesis)** — *Automated multimodal storytelling using Vision-Language Models (VLMs) and Large Language Models (LLMs).*

See also: https://arxiv.org/pdf/2305.14314

---

## Project overview

The pipeline asks whether an LLM can produce engaging, grounded NPC dialogue from **sequential** gameplay screenshots. Screenshots are processed in order: each frame is summarized by a VLM, observations are merged into a single structured **scene graph** (JSON), then an LLM generates campfire-style dialogue under explicit **grounding constraints** — no entities, locations, or plot beats may be invented beyond what appears in the scene graph.

---

## Architecture

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
The same model role-plays an NPC companion speaking at a campfire. The prompt explicitly forbids inventing characters, locations, or quests not present in the scene graph, directing the model to use vague language when the visual data is thin.

---

## Repository layout

```text
talking_agent/
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

# Override model and limit cost on large folders
python src/main_pipeline.py --model gemini-3-flash-preview --max-frames 15 --max-image-side 1024

# Resume after a failed LLM step without re-running VLM extraction
python src/main_pipeline.py --from-frames-json data/extracted_state/<run_id>_frames.json
```

Use `--max-frames`, `--sleep`, and `--max-image-side` to manage API rate limits and costs.

---

## Evaluation

The project has two complementary evaluation tracks.

### Stage 1 — VLM extraction accuracy (F1)

`evaluation/ground_truth_labels.csv` lists one row per frame and condition (binary checks such as `location_detectable`, `action_described`). Fill in the `Ground_Truth` and `VLM_Prediction` columns (0/1) from your rubric and pipeline outputs, then run:

```bash
python src/evaluator.py
```

Outputs Precision, Recall, and F1 per condition plus a macro-average F1.

> **Note:** This measures Stage 1 only — whether the VLM correctly reads the screenshots. It does not evaluate the quality or accuracy of the generated dialogue.

### Stage 3 — Dialogue quality (LLM-as-judge)

`src/dialogue_evaluator.py` uses a Gemini model as an automated judge. For each run it loads the scene graph and the generated dialogue, then scores three dimensions:

| Dimension | Description |
|---|---|
| **Grounding** (1–5) | Are all concrete claims traceable to the scene graph? Penalises invented entities, places, or events. |
| **Coherence** (1–5) | Is the narrative internally consistent and logically ordered? |
| **Engagement** (1–5) | Does it feel like a believable, immersive NPC campfire story? |

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

**Current results (3 runs, judge: gemini-2.5-flash):**

| Run ID | Grounding | Coherence | Engagement | Hallucinations |
|---|---|---|---|---|
| 20260408_182648 | 4/5 | 5/5 | 5/5 | 0 |
| 20260408_185849 | 5/5 | 5/5 | 5/5 | 0 |
| 20260408_193459 | 5/5 | 4/5 | 5/5 | 0 |
| **Average** | **4.67** | **4.67** | **5.00** | **0** |

The Grounding deduction in run `182648` is the word "flames" at the end of the dialogue — a campfire detail inferred rather than present in the scene graph. The Coherence deduction in run `193459` is a minor event ordering inconsistency. Zero hallucinated entities across all runs.

### Human evaluation

`evaluation/human_scoring_template.csv` is pre-populated with the 3 run IDs. Provide raters with the corresponding `.txt` files from `data/generated_stories/` and ask them to score each on Coherence, Grounding, and Engagement (1–5 scale). Recommended: 3–5 raters, 3+ dialogue samples each.

---

## EDA and figures

After pipeline runs have produced `data/extracted_state/*_frames.json` and matching `*_scene_graph.json`:

```bash
python src/eda_visualize.py
python src/eda_visualize.py --run-id 20260408_182648
```

PNGs are written to `data/analysis/figures/`: run overview, per-frame metrics, entity types and lifespans, event timeline, and evaluation bar charts (when the CSV is filled).
