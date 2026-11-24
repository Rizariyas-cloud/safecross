# SafeCross

**Giving time to the people the camera misses, and explaining why the light changed.**

SafeCross is a university research prototype for smart pedestrian signals. It is built for people who use **wheelchairs, walkers (rollators) and canes**, who cross more slowly than other pedestrians and need more time on the crossing.

Smart pedestrian signals, like those Dubai's RTA has installed at 27 locations, detect people and adjust the timing. But a smart signal is only as good as its camera. SafeCross asks three questions:

1. **How often does the camera miss mobility-aid users** in darkness, in blur, or at a camera position it was never trained on?
2. **What happens to the signal when the camera misses someone** who is still on the road, and how can we prevent it?
3. **Can the signal explain its decisions honestly**, citing what it recorded and saying "cannot determine" instead of guessing?

> University of Wollongong in Dubai: MAAI901 (Computer Vision) and MAAI902 (Natural Language Processing).
> Team: Risa Fathima, Minha Idris, Zeang Gao, Nahan Salam, Syed.
> Not an RTA product. RTA's public announcements are used only to describe what exists today.

---

## How it works

```
 COMPUTER VISION (MAAI901)        SMART SIGNAL (the bridge)            EXPLANATION (MAAI902)
 ─────────────────────────        ─────────────────────────            ──────────────────────
 PMMA real video                  Simulated crossing & junction        Knowledge graph of facts
   → YOLO11s detector               → SafeCross controller               → Retrieval (GraphRAG)
   → ByteTrack tracker              → Ghost filter                       → Fine-tuned explainer
   → measured camera failures  ──►  → Event log of every decision  ──►  → Citation checker
                                                                         → Answer with evidence,
                                                                           or "cannot determine"
```

- **The camera measurements are real.** They come from video of real mobility-aid users.
- **The crossing, the signal and the people walking are simulated**, because we cannot test a red light on a real wheelchair user. The simulated camera misses people at the rates we measured on the real video.

---

## Key results

| What we measured | Result |
|---|---|
| Mobility-aid users found at a **new camera position** | **75%** (seed 0) and **74%** (seed 1), against 88% at the training position |
| Wheelchair users found at the new position | **47% → 80%** (off-the-shelf YOLO11s → our fine-tuned model) |
| Found in strong darkness / strong blur (clean-trained model) | 13% / 5%. Robust training raises darkness to 35% but lowers clean new-site detection to 57% |
| Tracking on 19 minutes of real video | 123 real people received 2,288 tracker IDs |
| Aid users still on the road when the light turned red *(simulated)* | **16.1% → 0.7%** (standard controller → SafeCross), for about 1.7 s of extra green |
| Aid users meeting a camera-gated turning car at a junction *(simulated)* | **30.5% → 0%** when the camera found 30% of people, for about 0.6 s extra per car |
| What-if questions answered correctly | **100%** for our 1.5B explainer, against 64% and 85% for 7B models |
| Declines when the facts are missing | **100%** for our explainer; a 7B model never declined |
| Planted errors caught in incident reports (MiniCheck) | **91.5%** |
| Right page in the top 3, Dubai Universal Design guide search | **57% → 93%** (keyword → hybrid search + re-ranker) |

The full numbers, confidence intervals and statistical tests are in the two course reports.

---

## The three parts

### 1. Computer vision (MAAI901)
- **Data:** the PMMA dataset (Polytechnique Montréal Mobility Aids). It has 3 videos, 29,637 labelled images and 452,788 boxes. We mapped its 9 original classes to 4 groups: pedestrian, wheelchair, rollator and cane.
- **Fair split:** we train on video 1, test on video 2 (same camera position) and test on video 3 (a camera position the model never saw).
- **Model:** YOLO11s fine-tuned at 1280 px, trained twice with different seeds. There is a "clean" version and a "robust" version (trained on darkened and blurred copies).
- **Tests:** darkness, motion blur and occlusion, each at three strengths. The main measure is "found as a person" (IoU ≥ 0.5), with block-bootstrap confidence intervals.
- **Tracking:** ByteTrack, scored with MOTA and IDF1. We separately measured misses, ID changes and ghost (phantom) tracks.

### 2. Smart signal (the bridge)
- **The standard controller** treats the crossing as empty after 1 second without a detection.
- **SafeCross** keeps a person lost mid-crossing as "uncertain" for distance left ÷ 0.5 m/s × 1.3, capped at 20 s, with a 40 s maximum phase.
- **The ghost filter** ignores tracks seen for less than 0.5 s or moving less than 0.3 m. Without it, false tracks would hold the light about 19 s too long.
- Every decision is written to an **event log** with its time, reason and cause.

### 3. Explanation (MAAI902)
- **Provenance knowledge graph:** the event log becomes numbered facts (G1, G2, …) that record who kept the light green and when.
- **Retrieval:** only the facts the question needs are retrieved (GraphRAG-style subgraph retrieval).
- **Explainer:** Qwen2.5-1.5B-Instruct fine-tuned with LoRA (0.34% of the weights) on 7,812 examples. It answers in JSON, cites the facts it used, or says "cannot determine".
- **Citation checker:** a rule-based check that withholds any answer the cited facts do not support.
- **Also:**
  - MiniCheck checks longer incident reports sentence by sentence.
  - A hybrid search (BM25 + dense BGE + cross-encoder re-ranker) searches the Dubai Universal Design guide.
- **Tests:**
  - a 272-question benchmark;
  - out-of-distribution sets: 282 reworded questions and 149 questions about an unseen controller;
  - McNemar significance tests.

---

## The demo dashboard

A browser dashboard with five pages:

| Page | What it shows |
|---|---|
| **Crossing Overview** | A PMMA test frame with our model's detections (faces blurred), a detection summary, and a small assistant |
| **Smart Signal** | A simulated road with cars, lights and pedestrians; the SafeCross light holds for a person the camera loses |
| **Ask SafeCross** | Ask why the light changed: answers show citation chips and a "verified by checker" badge, or "cannot determine" |
| **Reports & Insights** | Our measured results and the session's incidents (CSV export) |
| **Settings** | The controller rules, models and privacy settings, shown for reference |

In the browser demo, the assistant follows the same rules as our explainer (cite, verify, decline) but runs as simple rules. The fine-tuned model itself runs in the Python pipeline.

---

## Running it

### Requirements
- Python 3.10 or later; developed on an Apple M4 Pro (Apple-silicon GPU / MPS).
- `pip install -r requirements.txt`
- For the 7B baselines and MiniCheck: [Ollama](https://ollama.com) with `qwen2.5vl:7b`, `llava:7b` and `bespoke-minicheck`.
- The explainer is trained and run with Apple MLX (`mlx-lm`).

### Setup
```bash
git clone https://github.com/Rizariyas-cloud/safecross.git
cd safecross
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data and weights (not in this repository)
- **PMMA dataset:** download from Borealis, doi:[10.5683/SP3/XJPQUG](https://doi.org/10.5683/SP3/XJPQUG), and place it under `data/pmma/`.
- **YOLO11s base weights** are downloaded automatically by Ultralytics.
- **Trained weights** (our YOLO checkpoints and the LoRA explainer) are too large for GitHub and are kept by the team; ask us for them.

### Run the demo
```bash
python scripts/28_demo_camera_frame.py   # optional: puts a real PMMA test frame on the camera card
python app/run_demo.py                   # opens http://127.0.0.1:8765/index.html
```
Without trained weights and PMMA, the demo still runs and shows an illustrative camera image.

### Reproduce the experiments
The scripts in `scripts/` are numbered in the order they were run. Settings are in `configs/config.yaml` (the PMMA class mapping is `pmma_class_map`). Results are written to `results/`, with a summary in `results/RESULTS.md`.

---

## Repository layout

```
safecross/       core package: vision (data, degradation), controllers, simulator, knowledge graph, NLP
scripts/         numbered experiment scripts (data prep → training → experiments → NLP → demo)
configs/         settings and dataset configs
app/             demo dashboard (app/web/index.html) and launcher (app/run_demo.py)
tests/           unit tests, including the knowledge-graph what-if rule (tests/test_kg.py)
results/         summary tables, charts and logs from our runs
```

---

## Limitations, stated honestly

- PMMA was filmed in a **parking lot**, on one day, in daylight, from two camera positions close together. "New position" means a new camera position, not a new city or a real road crossing.
- The crossing, the signal and the pedestrians are **simulated**. Real-road validation would need a city partner.
- **Not yet run:**
  - A DPO preference-training step (the script is implemented: `scripts/27_dpo_explainer.py`).
  - A blinded human evaluation of the explanations (the evaluation setup is prepared but has not been conducted).
- A "what if the camera had not lost this person" question **failed for every model**, including ours, which remained constrained by its trained yes/no response format. **DPO is a proposed future improvement** to explore whether preference-based training can improve this type of counterfactual reasoning.
- **English only.** Arabic explanations are future work.
---

## Ethics and privacy

- PMMA frames are used only within the team. Faces are blurred in anything shown, and **no images of people are stored in this repository**.
- We filmed no one in the UAE.
- The system design stores anonymous tracks and decision logs, not video.

---

## Main references

- Liu, Saunier & Bilodeau (2026). PMMA: Polytechnique Montréal Mobility Aids dataset. arXiv:2602.10259; doi:10.5683/SP3/XJPQUG.
- Mohr et al. (2023). Mobility-aid detection dataset and baselines. BMVC.
- Ni, Kühnel & Jiang (2026). Thermal detection that extends the green for walking-impaired pedestrians.
- Zhang et al. (2022). ByteTrack.
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
- Lewis et al. (2020). Retrieval-Augmented Generation. Edge et al. (2024). GraphRAG.
- Gao et al. (2023). Enabling LLMs to Generate Text with Citations (ALCE).
- Tang et al. (2024). MiniCheck.
- Dubai Media Office (2023, 2025). RTA smart pedestrian signals announcements.
