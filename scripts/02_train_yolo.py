"""STEP 2 - Fine-tune YOLO on the ICG dataset (run on the GPU laptop or Kaggle).

Examples:
    python scripts/02_train_yolo.py --seed 0                 # one model
    python scripts/02_train_yolo.py --seed 0 --seed 1 --seed 2   # three seeds (for the paper)
    python scripts/02_train_yolo.py --epochs 3 --fraction 0.1    # quick test run first!

The best weights are copied to weights/icg_<model>_seed<k>.pt
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve, ROOT


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg["vision"]["base_model"], help="e.g. yolo11n.pt (fast) or yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=cfg["vision"]["imgsz"])
    ap.add_argument("--batch", type=int, default=16, help="lower this (8 or 4) if you get 'CUDA out of memory'")
    ap.add_argument("--seed", type=int, action="append", help="repeat for several seeds")
    ap.add_argument("--fraction", type=float, default=1.0, help="use part of the training data (for quick tests)")
    ap.add_argument("--data", default=None, help="dataset .yaml (default: vision.dataset_yaml in config.yaml)")
    ap.add_argument("--device", default=None, help="auto (default), mps (Mac GPU), 0 (NVIDIA GPU) or cpu")
    args = ap.parse_args()
    from safecross.vision.detect import pick_device
    args.device = pick_device(args.device or cfg["vision"].get("device", "auto"))
    print(f"Using device: {args.device}")

    from safecross.vision.dataset import dataset_yaml_path
    data = dataset_yaml_path(cfg, args.data)
    if not data.exists():
        sys.exit(f"{data} not found. Run scripts/01_check_dataset.py (ICG) or scripts/11_pmma_to_yolo.py (PMMA) first.")
    print(f"Dataset: {data.name}")
    prefix = data.stem.replace("_dataset", "")
    from ultralytics import YOLO

    wdir = resolve(cfg["paths"]["weights_dir"])
    wdir.mkdir(parents=True, exist_ok=True)
    for seed in args.seed or [0]:
        model = YOLO(args.model)
        name = f"{prefix}_{Path(args.model).stem}_seed{seed}"
        model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=seed,
                    deterministic=True, fraction=args.fraction, device=args.device,
                    project=str(ROOT / "runs"), name=name, exist_ok=True)
        best = ROOT / "runs" / name / "weights" / "best.pt"
        shutil.copy(best, wdir / f"{name}.pt")
        print(f"Saved {wdir / f'{name}.pt'}")


if __name__ == "__main__":
    main()
