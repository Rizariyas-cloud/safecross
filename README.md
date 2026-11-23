# SafeCross on your MacBook (M4 Pro): Complete Guide

This guide is for doing the whole project **alone on your MacBook**. Every step says:

* **WHERE**: which app or website
* **WHAT**: what to download or type
* **HOW**: exactly what to click or copy

Follow it from top to bottom. When a box shows a command, copy the whole box, paste it into the Terminal and press **Return**.

---

## 0 · Before you start (5 minutes)

* **Check your Mac:** Apple menu  → **About This Mac**. It should say **Chip: Apple M4 Pro**. Note the **Memory** (for example 24 GB); anything 16 GB or more is fine.
* **Free space:** Apple menu  → **System Settings → General → Storage**. You need about **30 GB free**.
* **Folder:** everything lives in **Documents → safecross**.

### The everyday routine (you will do this many times)

1. Open **VS Code**.
2. Choose **File → Open Folder… → Documents → safecross**.
3. Open a terminal with **Terminal → New Terminal** (a panel opens at the bottom).
4. Turn on the project environment:
   ```
   source .venv/bin/activate
   ```
   You'll see **(.venv)** at the start of the line. If you don't see it, the commands below will not work.

---

## 1 · Install the tools (once, about 30 minutes)

### 1.1 Python 3.11

1. **WHERE:** Safari → https://www.python.org/downloads/macos/
2. **WHAT:** under **Python 3.11.x**, click **"Download macOS 64-bit universal2 installer"**.
3. **HOW:**
   1. Open the downloaded `.pkg` and click **Continue → Agree → Install**.
   2. When it finishes, a Finder window opens. Double-click **"Install Certificates.command"**.

   Don't skip the certificates step. Without it, model downloads fail with an "SSL" error.

### 1.2 Git

1. **WHERE:** open the **Terminal** app (press **Cmd + Space**, type `Terminal`, press Return).
2. **WHAT:** type `git --version` and press Return.
3. **HOW:** if a box asks to install "command line developer tools", click **Install** and wait for it to finish.

### 1.3 VS Code

1. **WHERE:** https://code.visualstudio.com
2. **WHAT:** click **Download for macOS**.
3. **HOW:**
   1. Unzip it and drag **Visual Studio Code** into **Applications**.
   2. Open it. Click the **Extensions** icon (four squares on the left), search **Python** and install the one by Microsoft.

### 1.4 Ollama (runs the AI model on your Mac)

1. **WHERE:** https://ollama.com/download
2. **WHAT:** click **Download for macOS**.
3. **HOW:**
   1. Unzip it, drag **Ollama** into **Applications** and open it.
   2. A small llama icon appears in the menu bar at the top of the screen. Keep it running whenever you use the AI model.

---

## 2 · Set up the project (once, about 20 minutes)

1. **WHERE:** Finder → Downloads.
   **HOW:** double-click `safecross.zip` to unzip it, then drag the `safecross` folder into **Documents**.
2. **WHERE:** VS Code → **File → Open Folder… → Documents → safecross → Open**. If it asks "Do you trust the authors?", click **Yes**.
3. **WHERE:** VS Code → **Terminal → New Terminal**.
   **WHAT:** paste these lines one by one:
   ```
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install git+https://github.com/ultralytics/CLIP.git
   ```
   This takes 5–10 minutes. Warnings in yellow are normal; red **ERROR** lines are not.
4. **Check that the Mac GPU is ready:**
   ```
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```
   It must print **True**. The project then uses your Mac's GPU automatically ("mps").
5. **Check the project works:**
   ```
   python -m pytest -q
   ```
   It must print **4 passed**.
6. **Open the app:**
   ```
   python app/app.py
   ```
   Open **Safari** and go to **http://127.0.0.1:7860**. You'll see the dashboard with a replay already running.
   To stop the app, click in the Terminal and press **Ctrl + C**.

The first time the app starts, it downloads a small YOLO model (about 5 MB), so you need internet.

---

## 3 · Download the data (once, about 30 minutes plus download time)

The `data` folder inside `safecross` already has three empty folders: `icg`, `pmma`, `rag_docs`.

### 3.1 ICG Crossroad Camera Dataset (main dataset, 1.8 GB)

1. **WHERE:** Safari → https://repository.tugraz.at/records/2gat1-pev27
2. **WHAT:** in the **Files** section, click **Download** next to **mobility.zip**.
3. **HOW:**
   1. In Finder → Downloads, double-click `mobility.zip` to unzip it.
   2. Drag the unzipped folder into **Documents → safecross → data → icg**.
4. **Write down** for your report: dataset version 1.0, licence CC BY-SA 4.0, DOI 10.3217/2gat1-pev27.

### 3.2 PMMA dataset (real walking speeds; optional but recommended)

1. **WHERE:** Safari → https://doi.org/10.5683/SP3/XJPQUG (it opens the Borealis data page).
2. **WHAT:** read the **Terms / Licence** tab first, then download the annotation files (the COCO `.json` files). You don't need all the images for this step.
3. **HOW:** put the downloaded files into **Documents → safecross → data → pmma**.

### 3.3 Dubai Universal Design Code (for the Ask SafeCross rule answers)

1. **WHERE:** Safari → search **"Dubai Universal Design Code Dubai Municipality"** and open the dm.gov.ae page.
2. **WHAT:** download the code as a PDF.
3. **HOW:** the easiest way is in the app: **Settings → Upload the PDF → Build index**. Or put the PDF in `data/rag_docs` and run step 5 below.

---

### 3.4 Backup: PMMA images (use this if the ICG download keeps failing)

The ICG server cuts downloads after about 40 seconds. PMMA (on Borealis) can resume, so it works on slow Wi-Fi.

1. **Annotations:** `train.json`, `val.json`, `test.json` and `README.md` go in `data/pmma` (done already).
2. **Images (one zip is enough to start, 2.9 GB):** in a Terminal tab run
   ```
   mkdir -p ~/Documents/safecross/data/pmma/zips
   cd ~/Documents/safecross/data/pmma/zips
   until curl -L -C - -o video_2_1.zip "https://borealisdata.ca/api/access/datafile/1008954"; do echo "Retrying in 5 s..."; sleep 5; done
   ```
   It continues by itself after every drop. Keep the Mac awake with `caffeinate -dis` in another tab.
3. **Check the zip is complete:** `unzip -l video_2_1.zip | tail -1` must show about **1010 files** and no error.
4. **Make the YOLO dataset:**
   ```
   cd ~/Documents/safecross
   python scripts/11_pmma_to_yolo.py --every 3 --train-share 0.6
   ```
   * `--every 3` keeps every 3rd frame (frames 1/15 s apart look almost the same).
   * `--train-share 0.6` = first 60% of the frames (in time) for training, 10% validation, the rest test, with a gap between them.
   * It switches `configs/config.yaml` to the PMMA dataset, so steps 2, 3 and 6 below use it automatically.
5. **Open `results/pmma_check_first.jpg` and `pmma_check_last.jpg`:** the boxes must sit on the people. If they don't, stop and tell me.
6. Continue with section 4. Step 1 (check ICG dataset) is not needed for PMMA.

**What to write in the paper:** PMMA is filmed in an outdoor parking area (not at a road crossing), from a 4 m pole.
Training and testing frames come from the same video, separated in time, so results are an optimistic,
same-scene estimate. Cite: Liu, Saunier & Bilodeau (2025), PMMA dataset, Borealis, doi:10.5683/SP3/XJPQUG.
Check the licence on the Borealis Terms tab.

**Two recall numbers in Experiment A:** `recall` = found **and** given the right class (wheelchair, rollator...);
`found_as_person` = found as any person, which is what a signal controller needs. A normal COCO model
(`--yolo yolo11s.pt`, it only knows "person") is a useful extra baseline: compare its `found_as_person` with the fine-tuned model.

---

## 4 · Run the experiments (in this order)

For every step:
1. Open VS Code.
2. Open a Terminal.
3. Run `source .venv/bin/activate`.
4. Paste the command(s).

Each step tells you which files it creates in the `results` folder.

### Step 1: Check the dataset (2 minutes; ICG only, skip for PMMA)
```
python scripts/01_check_dataset.py
```
It prints the class names (for example wheelchair, rollator, crutches, cane) and the number of images.

If you see a **WARNING** about prompts:
1. Open `configs/config.yaml` in VS Code.
2. Under `world_prompts`, add a line for the missing class.

### Step 2: Train YOLO on your Mac (quick test 10 minutes; full run a few hours)

1. **Quick test first.** It prints the time per epoch, so you can estimate the full run:
   ```
   python scripts/02_train_yolo.py --epochs 3 --fraction 0.1
   ```
2. **Full training.** Start it in the evening and let it run:
   ```
   python scripts/02_train_yolo.py --epochs 50 --seed 0
   ```
3. **Two more seeds for the paper.** Run these on another night:
   ```
   python scripts/02_train_yolo.py --epochs 50 --seed 1 --seed 2
   ```

Things to know:
* The first line it prints is **Using device: mps**. That means your Mac GPU is being used.
* **Plug in your charger** and stop the Mac from sleeping during long runs by putting `caffeinate -i` in front of the command:
  ```
  caffeinate -i python scripts/02_train_yolo.py --epochs 50 --seed 0
  ```
* If it fails with an "MPS" error, add `--device cpu` (slower) or `--model yolo11n.pt` (a smaller model).
* **Result:** `weights/icg_yolo11s_seed0.pt`

### Step 3: Experiment A, does the camera fail? (about 30–60 minutes)
```
python scripts/03_experiment_a.py --yolo weights/icg_yolo11s_seed0.pt --world --limit 300
```
Start with `--limit 50` to see how long it takes.

**Results:**
* `results/experiment_a/summary.csv` gives recall per class with 95% confidence intervals.
* `results/experiment_a/examples/` has pictures for the report.

### Step 4: Experiment B, do failures cause unsafe decisions? (about 1 minute)
```
python scripts/04_experiment_b.py
```
**Results:** `results/experiment_b/summary.csv` and the figure `cutoff_vs_level.png`.

### Step 4b: Junction study, what does the decision mean for cars? (follow-up, about 1 minute)
```
python scripts/13_junction_experiment.py
python scripts/12_collect_results.py
```
The same walk phases as Experiment B are placed inside a simulated 4-way junction (`configs/junction.yaml`):
stage A = parallel road green + our walk phase (cars may turn over the crossing), all-red, stage B = cross-road
cars drive over the crossing, all-red. It measures, for mobility-aid users:
* **cross-road conflict**: still on the crossing when the first cross-road car reaches it;
* **turning conflict**: a turning car reaches the crossing while they are on it, with *permissive* turns
  (cars must give way) and *gated* turns (turn arrow green only while the controller says "clear");
* the cost for other traffic: extra delay per cross-road car, cross-road green lost, turning-car delay.

It also compares a third controller, **uncertainty_aware_filtered**: our controller plus a ghost filter
(a track only counts after it has been seen for 0.5 s and moved 0.3 m), which removes phantom detections.
**Results:** `results/junction/summary.csv`, `junction_safety.png`, `junction_cost.png`, and a new section in
`results/RESULTS.md`. Report it as a **follow-up simulation**; the junction timings and traffic are assumptions.

### Step 5: Index the Dubai code (1 minute)

You can skip this if you already used **Settings → Build index** in the app.
```
python scripts/05_build_rag_index.py --ask "pedestrian crossing requirements"
```

### Step 6: Build the explanation questions (5 minutes)
```
python scripts/06_make_questions.py
python scripts/06b_make_real_visual_questions.py --weights weights/icg_yolo11s_seed0.pt --limit 100
```

### Step 7: Ask the AI model (one evening; leave it running)

1. **Download the model once** (about 6 GB). Make sure the Ollama llama icon is in the menu bar, then run:
   ```
   ollama pull qwen2.5vl:7b
   ```
   If it says the model is not found, open https://ollama.com/library, search **qwen2.5vl** and use the exact name shown there.
2. **Test the pipeline** (no AI, a few seconds):
   ```
   python scripts/07_run_explanations.py --backend offline
   ```
3. **Timing test** with 30 questions:
   ```
   python scripts/07_run_explanations.py --backend ollama --model qwen2.5vl:7b --limit 30
   ```
4. **Full run.** If it stops, run the same command again and it continues where it left off:
   ```
   python scripts/07_run_explanations.py --backend ollama --model qwen2.5vl:7b
   ```
5. **Second model** for comparison:
   ```
   ollama pull llava:7b
   python scripts/07_run_explanations.py --backend ollama --model llava:7b
   ```

**Results:** `results/explanations/summary_<model>.csv` and the McNemar test printed at the end ("does the picture make a difference?").

### Step 7b: Knowledge-graph explanations + citation checker (NLP-RQ1, follow-up; one evening)
The flat event log says *what* the signal did. The **decision knowledge graph** also stores *why*, as facts
built automatically from the controller's own trace: decisions → caused_by → camera tracks, which track kept the
crossing occupied (and when it was the only one), when each track was lost, and the timing rules. Every fact has
an id like `[G12]`, so answers can be checked.
```bash
python scripts/15_build_kg_questions.py      # adds a graph to every decision / "what if" question (checks they match)
python scripts/16_run_kg_explanations.py --backend ollama --model qwen2.5vl:7b
python scripts/16_run_kg_explanations.py --backend ollama --model llava:7b
```
Script 16 compares, on the same questions: the flat log (earlier run) vs the whole graph vs the question-focused
subgraph vs the subgraph + citation checker (unsupported answers are withheld). In the app: **Settings → Explain
decisions from → Knowledge graph**; the graph of the last simulation is drawn under Ask SafeCross, with the facts
the answer cited in gold.

### Step 7c: Our own fine-tuned explainer (NLP step 2; about 1-2 hours on the Mac)
A small Qwen model (1.5B) is fine-tuned with LoRA to answer from knowledge-graph facts, cite them, and say
"cannot_determine" when the facts are missing. Training walk phases use a different random seed and are
checked against the benchmark (no overlap).
```bash
pip install "mlx-lm==0.32.2"
python scripts/17_make_explainer_data.py                 # ~8,000 training examples, leakage check
python scripts/18_train_explainer.py --iters 30          # 3-minute test (downloads Qwen2.5-1.5B the first time)
python scripts/18_train_explainer.py                     # full training
python scripts/16_run_kg_explanations.py --backend mlx-base   # same small model WITHOUT training (control)
python scripts/16_run_kg_explanations.py --backend mlx        # our fine-tuned explainer
```

### Step 8: Real walking speeds from PMMA (optional, 30 minutes)

1. Look at the annotation format:
   ```
   python scripts/08_pmma_to_trajectories.py --inspect
   ```
2. Measure roughly how many pixels make 1 metre in a PMMA frame. For example, open a PMMA frame in Preview, drag a selection box from one traffic cone to the next (Preview shows the selection size in pixels while you drag) and divide by the real distance in metres. Then run (replace 60 with your value):
   ```
   python scripts/08_pmma_to_trajectories.py --pixels-per-metre 60
   python scripts/04_experiment_b.py --trajectories data/pmma_trajectories.json
   ```

If `--inspect` shows something unexpected, copy what it prints and send it to me.

### Step 9: Questions about the Dubai code (1 hour, mostly reading)

1. Open `configs/regulation_questions.yaml`.
2. Write 15–20 questions and the page numbers where the answers are.
3. Run:
   ```
   python scripts/09_eval_regulation.py
   python scripts/09_eval_regulation.py --backend ollama --model qwen2.5vl:7b
   ```

### Step 11: Tracking on real video + calibrated Experiment B (CV; about 1 hour on the Mac)

Experiment A scores single pictures; the signal sees video through a tracker. This measures, on video 3
(the new camera position), how often the tracker misses each person, for how long, and how often their ID
changes - the things Experiment B had to assume - and then re-runs Experiment B with the measured values.
```
python scripts/20_eval_tracking.py --max-frames 300 --save-video 300      # 2-minute test; watch tracking_sample.mp4
python scripts/20_eval_tracking.py                                         # all of video 3 (clean model)
python scripts/20_eval_tracking.py --weights weights/pmma_v2_aug_yolo11s_seed0.pt
python scripts/04_experiment_b.py --calibration results/tracking/pmma_v2_yolo11s_seed0_v3/calibration.json --only-calibrated --out experiment_b_calibrated
```
Outputs in `results/tracking/<model>_v3/`: summary.csv (MOTA, IDF1, ID switches), per_person.csv,
miss_bursts.csv, id_changes.csv, calibration.json. The script prints where the "who is who" labels came
from (COCO track ids, the CVAT XML, or box-overlap linking) - write that sentence in the report.

### Step 10: A real video for the dashboard (5 minutes)

This makes a short video from ICG images, so Live Monitoring shows real mobility-aid users:
```
python scripts/10_images_to_video.py data/icg/mobility/images/test --fps 10 --max 300
```
The folder name may differ. Open `data/icg` in Finder to find the folder that holds the test images, and use that path.

Restart the app. Because `results/demo_video.mp4` now exists, the dashboard's **Live Crossing Feed** tracks it
automatically at start-up (about 15–30 seconds) and shows the real footage with green boxes, the pedestrian count
and the location label. The **Pedestrian & Vehicle Trends** chart then shows the real counts from that video.
You can also upload any other video on **Live Monitoring → Track people in this video**.

---

## 5 · The app

**Start it** in the Terminal:
```
python app/app.py
```
Then open Safari → **http://127.0.0.1:7860**.

| Menu item | What you do there |
|---|---|
| **Dashboard** | the overview: camera feed replay, signal status, wait-time cards, alerts, trends, quick actions. The floating **AI Assistant** button (bottom right) opens a small chat |
| **Live Monitoring** | upload a crossing video → people tracked with IDs → the dashboard replays it |
| **Crossing Check** | upload an image (+ its label `.txt` from ICG) → choose darkness / blur / occlusion → see who is missed. Below: **Scan the image for barriers** (YOLO-World, zero-shot) + matching Dubai code passages |
| **Smart Signal** | change the camera failure sliders → **Run simulation** → baseline vs uncertainty-aware |
| **Ask SafeCross** | ask *"Why was the pedestrian phase extended?"* and other questions |
| **Reports** | all experiment tables and figures, plus downloads |
| **Settings** | switch answers from **Offline** to **AI model (Ollama)**; location label + cycle time for the dashboard; upload the Dubai code PDF; project status |

### Finish everything in one go (after training seed 0)

```
cd ~/Documents/safecross
bash scripts/finish_all.sh
```
About 5-6 hours; start it in the evening (Ollama open, charger in, lid open). It runs, in order:
1. balanced picture questions on real PMMA frames (scripts/06b): each marks ONE mobility-aid user with a yellow
   box and asks whether the detector reported a person there; half the true answers are "yes", half "no";
2. the "what if" questions again WITH the controller rules (scripts/06c, a follow-up: report it as one);
3. the explanation test for qwen2.5vl:7b (only the new questions are asked);
4. the same test for a second model, llava:7b;
5. two more training seeds;
6. Experiment A with all three seeds and the ordinary YOLO;
7. `scripts/12_collect_results.py`, which writes **results/RESULTS.md**: every table for the papers in one file.

If it stops, run it again: finished parts are kept or skipped.

### What the dashboard numbers are (say this in the viva)

* **Live Crossing Feed:** a *replay* — the simulation, or a recorded video run through YOLO + ByteTrack. Not a live camera, so the badge says "Replay".
* **Signal Status:** the simulated controller. "Remaining" is the normal phase time left *if the crossing is clear*; the controller may extend it. The cycle time is a setting (`dashboard:` in `configs/config.yaml`).
* **Avg. Pedestrian / Vehicle Wait:** *estimates*, using the standard uniform-arrival formula (wait = red² ÷ (2 × cycle)) with the simulated walk phase. The arrows compare with the baseline controller: pedestrians wait less, vehicles a bit more. That trade-off is honest and is worth saying.
* **Safety Events:** people still on the crossing when the walk phase ended, summed over this session's simulated runs.
* **Alerts:** only things the app actually computed (a simulation run, a crossing check, a barrier scan).
* **Barrier scan:** zero-shot text prompts, not a trained model — results must be checked by eye. It needs YOLO-World; if it says it cannot load, run `pip install git+https://github.com/ultralytics/CLIP.git` and try again (the first run downloads the model).

### Your 3-minute presentation demo

1. **Crossing Check:** an ICG image with a wheelchair user → **darkness, severity 3** → the wheelchair user is missed.
2. **Smart Signal:** scenario **61** → the baseline ends the phase while a mobility-aid user is still crossing (red x); the uncertainty-aware controller holds it. Then scenario **3** and scroll to **Inside the junction**: the baseline's camera misses two wheelchair users, reports "clear", the turn arrow goes green and a turning car meets them (red ▼); ours keeps the arrow red.
3. **Ask SafeCross**, with **Settings → AI model (Ollama)** turned on:
   1. Ask *"Why was the pedestrian phase extended?"*
   2. In Settings, turn off **"Give the model the event log"**.
   3. Ask again, to show whether the model really uses the evidence.

**Record a demo video:** press **Cmd + Shift + 5 → Record Selected Portion**, then click the stop button in the menu bar when you're done.

---

## 6 · Put the code on GitHub (for submission)

1. **WHERE:** https://github.com → sign in → **+ → New repository**.
2. **WHAT:** name it `safecross`, choose Private or Public, and don't add a README.
3. **HOW:** in the VS Code Terminal, run (put in your username):
   ```
   git init
   git add .
   git commit -m "SafeCross project"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/safecross.git
   git push -u origin main
   ```
   When it asks for a password, use a **personal access token** (GitHub → Settings → Developer settings → Personal access tokens), not your account password.
4. After each work session, run:
   ```
   git add .
   git commit -m "what I did"
   git push
   ```

Datasets and model weights are not uploaded; the `.gitignore` file handles that.

---

## 7 · Two-week plan (one person)

| Day | Do this | Leave running overnight |
|---|---|---|
| 1 | Parts 0–3: install, set up, download data | ICG download |
| 2 | Step 1, quick training test, explore the app | **Step 2 full training (seed 0)** |
| 3 | Step 4 (Experiment B), Step 8 (PMMA speeds), study the Smart Signal results | |
| 4 | Step 3 (Experiment A), look at the example images | |
| 5 | Dubai code: index it, write the regulation questions (Step 9) | |
| 6 | Step 6 questions, Ollama setup, timing test | **Step 7 full run (model 1)** |
| 7 | Check the results, Step 6b real-image questions | **Step 7 again (model 2), seeds 1 and 2** |
| 8 | Step 10 demo video, the full app demo, screenshots for the papers | |
| 9–12 | Write: MAAI901 report + paper, MAAI902 report + paper (send me drafts to paraphrase) | |
| 13 | Record the demo video, GitHub, final checks | |
| 14 | Submit | |

---

## 8 · Which result goes where in the papers

| Paper part | Use this |
|---|---|
| MAAI901 RQ1 table | `results/experiment_a/summary.csv` (rows `ALL_MOBILITY_AIDS` + per class) |
| MAAI901 figure | `results/experiment_a/examples/*.jpg` |
| MAAI901 RQ2 table and figure | `results/experiment_b/summary.csv` and `cutoff_vs_level.png` |
| MAAI901 example episode | screenshot of the Smart Signal timeline |
| MAAI902 RQ3 table | `results/explanations/summary_<model>.csv` |
| MAAI902 "does the picture matter" | the McNemar lines printed by Step 7 |
| MAAI902 RAG | `results/regulation_eval.csv` |
| Appendix (reproducibility) | `configs/config.yaml` (all settings and seeds) |

---

## 9 · If something goes wrong

| What you see | What to do |
|---|---|
| `command not found: python3.11` | Python is not installed; redo 1.1 |
| `ModuleNotFoundError` | you forgot `source .venv/bin/activate` |
| `SSL: CERTIFICATE_VERIFY_FAILED` | run **Applications → Python 3.11 → Install Certificates.command** |
| `mps` error during training | add `--device cpu`, or use the smaller `--model yolo11n.pt` |
| YOLO-World error about `clip` | `pip install git+https://github.com/ultralytics/CLIP.git` |
| Ask SafeCross cannot reach the AI model | open the Ollama app (llama icon), run `ollama list`, and check the name matches **Settings → Ollama model name** |
| `Address already in use` | an app is already running; close that Terminal or run `SAFECROSS_PORT=7861 python app/app.py` |
| The Mac gets hot or slow | normal during training; keep it on the charger with good airflow |

For anything else, copy the **red error text** and send it to me.

---

## 10 · Be honest in the reports (this protects your marks)

* All signal-safety results are **simulated**: a virtual crossing with camera failures added on purpose. Say so every time.
* The walking speeds and thresholds in `configs/config.yaml` are **assumptions**. Say so, and use PMMA speeds (Step 8) where you can.
* **Offline mode is not an AI model.** Never report its numbers as model results.
* The dashboard's **"Live Monitoring"** is a **replay** (the simulation or a recorded video), not a live camera.
* State that AI assistance was used for coding and language editing.

The team version of this guide (Windows and NVIDIA) is in `docs/GUIDE_TEAM_WINDOWS.md`.
