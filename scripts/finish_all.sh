#!/bin/bash
# Runs every remaining step in order. Start it in the evening and leave it (about 5-6 hours):
#
#     cd ~/Documents/safecross
#     bash scripts/finish_all.sh
#
# Needs: Ollama app open, charger in, lid open. Safe to run again if it stops:
# finished steps are skipped or resumed (answers and weights are kept).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
W0=weights/pmma_yolo11s_seed0.pt
[ -f "$W0" ] || { echo "Missing $W0 - train first (README section 3.4)."; exit 1; }
caffeinate -dis -w $$ &

Q=results/questions/questions.json
echo; echo "=== 1/7 Balanced picture questions (real PMMA frames) ==="
if grep -q '"realbox_' "$Q" 2>/dev/null; then echo "already made - kept (so answers stay valid)"; else
python scripts/06b_make_real_visual_questions.py --weights "$W0" --per-answer 60; fi

echo; echo "=== 2/7 'What if' questions with the controller rules (follow-up) ==="
if grep -q '"counterfactual_rules"' "$Q" 2>/dev/null; then echo "already made - kept"; else
python scripts/06c_add_counterfactual_with_rules.py; fi

echo; echo "=== 3/7 Explanations: qwen2.5vl:7b (only new questions are asked) ==="
python scripts/07_run_explanations.py --backend ollama --model qwen2.5vl:7b

echo; echo "=== 4/7 Explanations: second model llava:7b ==="
ollama pull llava:7b
python scripts/07_run_explanations.py --backend ollama --model llava:7b

echo; echo "=== 5/7 Two more training seeds (for mean +/- sd) ==="
# free the GPU memory the language models are using, otherwise training runs out of memory
ollama stop qwen2.5vl:7b 2>/dev/null || true
ollama stop llava:7b 2>/dev/null || true
sleep 5
for S in 1 2; do
  if [ ! -f "weights/pmma_yolo11s_seed$S.pt" ]; then
    python scripts/02_train_yolo.py --epochs 60 --imgsz 1280 --batch 8 --seed $S
  else
    echo "seed $S already trained"
  fi
done

echo; echo "=== 6/7 Experiment A with all three seeds + the ordinary YOLO ==="
python scripts/03_experiment_a.py --yolo weights/pmma_yolo11s_seed0.pt --yolo weights/pmma_yolo11s_seed1.pt \
       --yolo weights/pmma_yolo11s_seed2.pt --yolo yolo11s.pt

echo; echo "=== 7/7 Collect every result into results/RESULTS.md ==="
python scripts/12_collect_results.py
echo; echo "ALL DONE. Open results/RESULTS.md"
