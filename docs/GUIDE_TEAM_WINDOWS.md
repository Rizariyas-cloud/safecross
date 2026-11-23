# SafeCross: Step-by-Step Guide

**SafeCross** studies what happens to mobility-aid users (wheelchair, walker, crutches, cane) when a smart crossing's camera is wrong, and whether a vision-language model can explain the signal's decisions honestly.

This guide says **where** to run each step (which computer) and **how** (exact commands). Follow the parts in order.

---

## Part 0 · What is in this folder

```
safecross/
├─ app/app.py                  ← the demo app (Dashboard, Crossing Check, Smart Signal, Ask SafeCross)
├─ configs/config.yaml         ← ALL settings (speeds, thresholds, failure levels, model names)
├─ configs/regulation_questions.yaml  ← questions about the Dubai code (you fill in)
├─ safecross/                  ← the project's code (you import it, you rarely edit it)
│   ├─ sim.py                  simulated crossing and walking people
│   ├─ perception.py           the 4 camera failures (false negative, misclassification, track loss, false positive)
│   ├─ controller.py           baseline controller + uncertainty-aware controller (+ event logs)
│   ├─ metrics.py              cut-off rate, extra green time, bootstrap confidence intervals, McNemar test
│   ├─ render.py               crossing pictures, animations, timelines
│   ├─ vision/                 image degradation, YOLO / YOLO-World, detection scoring
│   └─ explain/                questions, RAG over the Dubai code, AI model backends, scoring, chatbot
├─ scripts/01 … 09             ← the experiments, run them in order
├─ tests/                      ← quick checks that the logic works
├─ data/                       ← you put the downloaded datasets here
├─ weights/                    ← trained models are saved here
└─ results/                    ← every experiment writes its tables and figures here
```

### The three computers you will use

| Name in this guide | What it is | Used for |
|---|---|---|
| **Any laptop** | every member's own laptop (Windows or Mac) | setup, app, Experiment B, RAG, questions, writing |
| **GPU laptop** | your friend's laptop with the NVIDIA GPU | YOLO training, Experiment A, the vision-language model |
| **Kaggle** (backup) | free online GPU at kaggle.com | YOLO training if the GPU laptop is busy |

### Who does what

| Member | Owns | Scripts | Where |
|---|---|---|---|
| 1 | YOLO detection under corruption | 01, 02, 03 | GPU laptop |
| 2 | Tracking and trajectories (PMMA) | 08, plus the tracking analysis | Any laptop / GPU laptop |
| 3 | Simulated crossing and controllers | 04, the Smart Signal tab | Any laptop |
| 4 | VLM explanations, visual grounding, counterfactuals | 06, 06b, 07 | GPU laptop |
| 5 | RAG, evaluation framework, integration, the app | 05, 09, app | Any laptop |

Everyone helps with the final evaluation and writes their own section of both papers.

---

## Part 1 · Setup (EVERY member, ONCE, on their own laptop)

**Time needed:** about 30 minutes.

### 1.1 Install the tools

1. **Python 3.11**: download it from https://www.python.org/downloads/
   * **Windows:** on the first installer screen, **tick "Add python.exe to PATH"**.
2. **VS Code**: download it from https://code.visualstudio.com, then install the "Python" extension inside VS Code.
3. **Git**: download it from https://git-scm.com/downloads.

### 1.2 Get the project

**Option A (the person who received the zip):**
1. Unzip `safecross.zip` into your Documents folder.
2. In VS Code: **File → Open Folder → safecross**.

**Option B (the other members, after Part 2):**
1. In VS Code, open **Terminal → New Terminal**.
2. Run:
   ```
   cd Documents
   git clone https://github.com/<your-team-account>/safecross.git
   ```
3. Open the folder in VS Code.

### 1.3 Create the project's Python environment

In VS Code, open **Terminal → New Terminal**. Make sure the prompt ends in `safecross`.

**Windows (PowerShell):**
```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell says "running scripts is disabled", run this once and then try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Mac (Terminal):**
```
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

You'll see `(.venv)` at the start of the prompt. **Every time you open a new terminal, activate the environment again** (the second line above).

### 1.4 Check that everything works

```
python -m pytest -q
```
You should see `4 passed`.

### 1.5 Open the app

```
python app/app.py
```
Open http://127.0.0.1:7860 in your browser. Before any data is downloaded, the **Smart Signal** and **Ask SafeCross** (Offline mode) tabs already work. Stop the app with **Ctrl + C** in the terminal.

---

## Part 2 · Share the code on GitHub (Member 5, once)

1. Create a GitHub repository called `safecross` (private is fine) and add the 4 teammates as collaborators (**Settings → Collaborators**).
2. In the project folder, run:
   ```
   git init
   git add .
   git commit -m "SafeCross starting code"
   git branch -M main
   git remote add origin https://github.com/<your-account>/safecross.git
   git push -u origin main
   ```
3. Daily routine for everyone:
   * `git pull` before you start.
   * After your work:
     ```
     git add .
     git commit -m "what you did"
     git push
     ```

Each member commits their own work. This commit history is your evidence of individual contribution.

Datasets, weights and large images are **not** uploaded; `.gitignore` takes care of that.

---

## Part 3 · Download the data (Member 1 and Member 2)

Put everything inside the project's `data/` folder.

| What | Where to get it | Put it in | Who needs it |
|---|---|---|---|
| **ICG Crossroad Camera Dataset** | https://repository.tugraz.at/records/2gat1-pev27, download `mobility.zip` (1.8 GB) | unzip into `data/icg/` | GPU laptop |
| **PMMA** | https://doi.org/10.5683/SP3/XJPQUG (read the licence on that page) | unzip into `data/pmma/` | Member 2 |
| **Dubai Universal Design Code** | Dubai Municipality website, search "Dubai Universal Design Code" | the PDF goes into `data/rag_docs/` | Member 5 |

Write down the dataset versions and licences you used for the report.

---

## Part 4 · Run the experiments, in order

Before every step, open a terminal in the project folder and activate the environment.

### STEP 1: Check the ICG dataset
**Where:** GPU laptop (the one with the data).

```
python scripts/01_check_dataset.py
```
This prints the class names and how many images and boxes are in each split, and writes `configs/icg_dataset.yaml`.

If it prints a **WARNING** about prompts, add the missing class to `vision.world_prompts` in `configs/config.yaml`.

### STEP 2: Train YOLO (Member 1)
**Where:** GPU laptop (or Kaggle, see Part 7).

1. **Install PyTorch with GPU support first.** `pip install ultralytics` alone may install a CPU-only version on Windows.
   * Go to https://pytorch.org/get-started/locally and choose: Stable / your OS / Pip / Python / the CUDA version your driver supports (run `nvidia-smi` to see it).
   * Copy the command it shows and run it.
2. Check the GPU is visible:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   It must print `True`.
3. Do a quick test run first (a few minutes):
   ```
   python scripts/02_train_yolo.py --epochs 3 --fraction 0.1 --device 0
   ```
4. Do the real run. Start with one seed, and add the other two seeds later for the paper:
   ```
   python scripts/02_train_yolo.py --epochs 50 --device 0 --seed 0
   python scripts/02_train_yolo.py --epochs 50 --device 0 --seed 1 --seed 2
   ```
   * If you see "CUDA out of memory", add `--batch 8` (or `--batch 4`).

**Result:** `weights/icg_yolo11s_seed0.pt`. Share this file with the team (Google Drive, not GitHub).

### STEP 3: Experiment A, visual robustness (Member 1)
**Where:** GPU laptop.

```
python scripts/03_experiment_a.py --yolo weights/icg_yolo11s_seed0.pt --world --limit 300 --device 0
```
* `--world` adds zero-shot YOLO-World. The first run downloads its weights.
* Start with `--limit 50` to check the timing, then increase it.

**Results:** `results/experiment_a/summary.csv` (recall per class, with 95% confidence intervals) and `examples/` (pictures for the report).

### STEP 4: Experiment B, decision sensitivity (Member 3)
**Where:** any laptop (no GPU needed; takes about a minute).

```
python scripts/04_experiment_b.py
```
**Results:**
* `results/experiment_b/summary.csv`: cut-off rate, extra green time and false extensions for each failure type and level, for both controllers.
* `cutoff_vs_level.png`: the main figure for RQ2.

To use the real PMMA walking speeds after STEP 8, run:
```
python scripts/04_experiment_b.py --trajectories data/pmma_trajectories.json
```

### STEP 5: Build the Dubai code search index (Member 5)
**Where:** any laptop.

```
python scripts/05_build_rag_index.py --ask "pedestrian crossing requirements"
```
Check that the printed passages make sense.

### STEP 6: Build the explanation benchmark (Member 4)
**Where:** any laptop for 6; GPU laptop for 6b.

```
python scripts/06_make_questions.py
python scripts/06b_make_real_visual_questions.py --weights weights/icg_yolo11s_seed0.pt --limit 100 --device 0
```
* **6** creates decision, counterfactual and visual questions from simulated runs.
* **6b** adds visual questions on **real** ICG images: did the detector miss a mobility-aid user? The truth comes from the dataset labels.

### STEP 7: Ask the vision-language model (Member 4)
**Where:** GPU laptop.

1. Install **Ollama** from https://ollama.com/download and open it.
2. In a terminal, download a model:
   ```
   ollama pull qwen2.5vl:7b
   ```
   * If the name is not found, search "qwen2.5vl" on https://ollama.com/library and use the exact name shown.
   * A second model for comparison: `ollama pull llava:7b`
3. Test the pipeline without AI first:
   ```
   python scripts/07_run_explanations.py --backend offline
   ```
4. Do a timing test with the real model:
   ```
   python scripts/07_run_explanations.py --backend ollama --model qwen2.5vl:7b --limit 30
   ```
5. Do the full run. If it stops, run the same command again and it continues from where it stopped:
   ```
   python scripts/07_run_explanations.py --backend ollama --model qwen2.5vl:7b
   python scripts/07_run_explanations.py --backend ollama --model llava:7b
   ```

**Results:** `results/explanations/summary_<model>.csv` gives, for each question type and each input condition (picture + log, log only, picture only):
* factual correctness
* controller faithfulness
* visual grounding
* abstentions
* invented citations

It also prints a **McNemar test** (picture + log vs log only), which answers "does the picture make a difference?".

### STEP 8: Real walking speeds from PMMA (Member 2, optional but recommended)
**Where:** any laptop.

1. Look at the annotation format first:
   ```
   python scripts/08_pmma_to_trajectories.py --inspect
   ```
2. Measure how many pixels make 1 metre in the PMMA video (for example, from the spacing of the traffic cones), then run:
   ```
   python scripts/08_pmma_to_trajectories.py --pixels-per-metre 60
   ```
   Replace 60 with your own measured value.

**Method (put this in the report):** PMMA was recorded in a parking lot, so we keep only each person's **walking-speed profile** and replay it along the virtual crosswalk. The crossing is simulated; the speeds are real. This is documented at the top of the script.

If `--inspect` shows the track id stored under a different field name, tell Member 5 so the loader can be adjusted.

### STEP 9: Regulation questions (Member 5)
**Where:** any laptop.

1. Read the Dubai code.
2. Write 15–20 questions with the correct page numbers in `configs/regulation_questions.yaml`.
3. Run:
   ```
   python scripts/09_eval_regulation.py
   python scripts/09_eval_regulation.py --backend ollama --model qwen2.5vl:7b
   ```

---

## Part 5 · The app (for the demo and the presentation)

**Where:** any laptop for Offline mode; the GPU laptop when you want to use the AI model.

```
python app/app.py
```
Open http://127.0.0.1:7860.

| Tab | What it shows | Needs |
|---|---|---|
| **Dashboard** | status of each part, results tables and figures, buttons to the other tabs | nothing (fills in as you run the steps) |
| **Crossing Check** | detector on a crossing image, then the same image darkened / blurred / occluded; missed people in dashed red | trained weights (STEP 2) or YOLO-World |
| **Smart Signal** | one simulated walk phase with camera failures; both controllers side by side, timeline, event logs | nothing |
| **Ask SafeCross** | chat about the last run: why the phase ended, what-if questions, missed people, Dubai code | Offline mode works at once; AI mode needs Ollama (STEP 7) |

### A 3-minute demo script for the presentation

1. **Crossing Check:** load an ICG image with a wheelchair user and choose *darkness*, severity 3. The wheelchair user disappears from the detections.
2. **Smart Signal:** use the default scenario (61). The baseline ends the phase while a mobility-aid user is still crossing (red x on the timeline). The uncertainty-aware controller holds the phase.
3. **Ask SafeCross:** switch to *AI model (Ollama)* and ask "Why was the pedestrian phase extended?". Then untick *Include event log* and ask again, which shows whether the model really uses the evidence.

To show the app on another device on the same Wi-Fi, set `SAFECROSS_HOST=0.0.0.0` before starting it and open `http://<laptop-IP>:7860`.

---

## Part 6 · Which output goes into which paper

| Paper section | File |
|---|---|
| MAAI901 RQ1, robustness table | `results/experiment_a/summary.csv` (rows `ALL_MOBILITY_AIDS` and per class) |
| MAAI901 figure, degraded examples | `results/experiment_a/examples/*.jpg` |
| MAAI901 RQ2, safety vs efficiency | `results/experiment_b/summary.csv` and `cutoff_vs_level.png` |
| MAAI901 example episode figure | screenshot of the Smart Signal timeline |
| MAAI902 RQ3, main table | `results/explanations/summary_<model>.csv` |
| MAAI902, does the picture matter | the McNemar lines printed by STEP 7 |
| MAAI902, RAG | `results/regulation_eval.csv` |

All settings (speeds, thresholds, seeds, failure levels) are in `configs/config.yaml`. Copy that file into your appendix so the results can be reproduced.

---

## Part 7 · Kaggle (if the GPU laptop is not available)

1. Upload `mobility.zip` as a Kaggle **Dataset** (Datasets → New Dataset).
2. Create a Notebook. In the right panel:
   * **Accelerator:** GPU T4
   * **Add Data:** your dataset
   * **Internet:** On
3. In the first cells, run:
   ```
   !git clone https://github.com/<your-account>/safecross.git
   %cd safecross
   !pip install -q -r requirements.txt
   !mkdir -p data/icg && cp -r /kaggle/input/<your-dataset-name>/* data/icg/
   !python scripts/01_check_dataset.py
   !python scripts/02_train_yolo.py --epochs 50 --device 0 --seed 0
   ```
4. Download `weights/*.pt` from the notebook's output panel.

If the dataset is still zipped inside the input folder, unzip it into `data/icg` instead of copying.

---

## Part 8 · Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | the environment is not active; run the activate command from 1.3 |
| `torch.cuda.is_available()` prints False | install the CUDA build of PyTorch from pytorch.org (STEP 2.1) and update the NVIDIA driver |
| CUDA out of memory | lower `--batch` to 8 or 4; use `yolo11n.pt` with `--model yolo11n.pt` |
| YOLO-World error about `clip` | `pip install git+https://github.com/ultralytics/CLIP.git` |
| Ask SafeCross says it cannot reach the AI model | open the Ollama app, check `ollama list` shows the model, and that the model name in the app matches |
| Dashboard shows "Not yet" | that step has not been run; press **Refresh** after running it |
| Port 7860 already in use | close the other app, or set `SAFECROSS_PORT=7861` |

---

## Part 9 · Be honest in the reports (this protects your marks)

* All signal-safety results are **simulated**: the crossing is virtual, and the failures are injected at controlled rates. Say so every time.
* Walking speeds and thresholds in `configs/config.yaml` are **assumptions**. State them, cite sources where you can, and replace the synthetic speeds with PMMA speeds (STEP 8) if possible.
* **Offline mode is not an AI model.** Never report its numbers as model results; it exists to test the pipeline.
* Visual questions in STEP 6 use **schematic** pictures from the simulation. Questions on **real** images come from STEP 6b. Report them separately (`visual` vs `visual_real`).
* State in both reports that AI assistance was used for coding and language editing, as allowed by the course.
