"""STEP 18 (NLP step 2) - fine-tune a small Qwen model with LoRA on the Mac (Apple MLX).

Install once (in the project environment):
    pip install "mlx-lm==0.32.2"
Quick test (2-3 minutes, checks everything works):
    python scripts/18_train_explainer.py --iters 30
Full training (about 30-60 minutes on an M4 Pro):
    python scripts/18_train_explainer.py

LoRA = only small extra matrices are trained (about 0.5% of the weights); the original model stays frozen.
--mask-prompt = the model is only trained on the ANSWER, not on copying the question and facts.
Output: weights/explainer_lora/ (the adapter) + results/explanations/explainer_training_log.txt
v2 (also trained on whole graphs):
    python scripts/18_train_explainer.py --data data/explainer_sft_v2 --adapter weights/explainer_lora_v2 --iters 600 --max-seq-length 4096
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="downloaded from Hugging Face the first time")
    ap.add_argument("--data", default="data/explainer_sft")
    ap.add_argument("--adapter", default="weights/explainer_lora")
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq-length", type=int, default=2048,
                    help="v2 data holds whole graphs (up to ~3,200 tokens): use 4096 so nothing is cut off")
    args = ap.parse_args()
    data = resolve(args.data)
    if not (data / "train.jsonl").exists():
        sys.exit(f"{data}/train.jsonl not found - run scripts/17_make_explainer_data.py first.")
    adapter = resolve(args.adapter)
    adapter.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_lm.lora", "--model", args.model, "--train", "--data", str(data),
           "--fine-tune-type", "lora", "--num-layers", str(args.num_layers), "--batch-size", str(args.batch_size),
           "--iters", str(args.iters), "--learning-rate", str(args.lr), "--mask-prompt",
           "--steps-per-report", "20", "--steps-per-eval", "100", "--val-batches", "25",
           "--save-every", "200", "--max-seq-length", str(args.max_seq_length), "--grad-checkpoint",
           "--adapter-path", str(adapter), "--seed", str(args.seed)]
    print("Running:\n  " + " ".join(cmd) + "\n")
    log = resolve("results/explanations")
    log.mkdir(parents=True, exist_ok=True)
    log_name = ("explainer_training_log.txt" if adapter.name == "explainer_lora"
                else f"explainer_training_log_{adapter.name.replace('explainer_', '')}.txt")
    with open(log / log_name, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="")
            f.write(line)
        p.wait()
    if p.returncode:
        sys.exit(f"Training stopped with an error (code {p.returncode}) - send the output above to Claude.")
    extra = "" if adapter.name == "explainer_lora" else f" --adapter {args.adapter}"
    print(f"\nDone. Adapter saved in {adapter}. Next: scripts/16_run_kg_explanations.py --backend mlx{extra}")


if __name__ == "__main__":
    main()
