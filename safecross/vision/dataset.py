"""Helpers for YOLO-format datasets such as the ICG Crossroad Camera Dataset."""
from __future__ import annotations

from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def find_dataset_yaml(root: str | Path) -> Path | None:
    """Find the dataset .yaml inside the unzipped dataset folder."""
    root = Path(root)
    cands = [p for p in root.rglob("*.yaml") if "names" in p.read_text(errors="ignore")]
    return sorted(cands, key=lambda p: len(p.parts))[0] if cands else None


def dataset_yaml_path(cfg: dict, override: str | None = None) -> Path:
    """The dataset .yaml to use: --data if given, else vision.dataset_yaml in config.yaml."""
    from safecross.config import ROOT
    p = Path(override or cfg["vision"].get("dataset_yaml") or "configs/icg_dataset.yaml")
    return p if p.is_absolute() else ROOT / p


def load_dataset_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    names = d.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    d["names"] = names
    return d


def _split_dirs(d: dict, yaml_path: Path, split: str) -> list[Path]:
    base = Path(d.get("path") or yaml_path.parent)
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    entry = d.get(split)
    if entry is None:
        return []
    entries = entry if isinstance(entry, list) else [entry]
    out = []
    for e in entries:
        p = Path(e)
        out.append(p if p.is_absolute() else base / p)
    return out


def list_images(d: dict, yaml_path: str | Path, split: str = "test") -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs for a split. Label path follows the YOLO
    convention: .../images/xxx.jpg  ->  .../labels/xxx.txt"""
    yaml_path = Path(yaml_path)
    pairs = []
    for src in _split_dirs(d, yaml_path, split):
        if src.is_file() and src.suffix == ".txt":           # a list of image paths
            imgs = [Path(l.strip()) for l in src.read_text().splitlines() if l.strip()]
            imgs = [p if p.is_absolute() else (src.parent / p) for p in imgs]
        else:
            imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT]
        for im in sorted(imgs):
            parts = list(im.parts)
            if "images" in parts:
                i = len(parts) - 1 - parts[::-1].index("images")
                parts[i] = "labels"
            lab = Path(*parts).with_suffix(".txt")
            pairs.append((im, lab))
    return pairs
