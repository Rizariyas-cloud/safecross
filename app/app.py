"""SafeCross app: Intelligent Pedestrian Safety & Accessibility System.

Run from the project folder (with the environment activated):
    python app/app.py
then open http://127.0.0.1:7860 in Safari or Chrome.

Pages (left menu):
  Dashboard        - crossing feed replay, signal status, safety metrics, alerts, trends, quick actions
  Live Monitoring  - replay the simulation, or track people in your own video (YOLO + ByteTrack)
  Crossing Check   - detector on an image, then darkness / blur / occlusion: who gets missed?
  Smart Signal     - simulate a walk phase with camera failures; baseline vs uncertainty-aware controller
  Ask SafeCross    - ask why the signal did what it did (offline rules or an AI model in Ollama)
  Reports          - experiment results, figures and downloadable tables
  Settings         - AI model, detector, Dubai code index, project status
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safecross.config import load_config, resolve
from safecross.controller import run_controller, base_phase_length
from safecross.explain.assistant import offline_answer, ai_answer
from safecross.explain.backends import OllamaBackend, OfflineBackend, parse_json
from safecross.explain.kg import build_kg, lines as kg_lines, subgraph
from safecross.explain.kg_draw import draw_kg
from safecross.explain.prompts import SYSTEM_KG, build_kg_prompt
from safecross.explain.evidence import format_log
from safecross.explain.rag import build_index, load_retriever
from safecross.metrics import episode_metrics
from safecross.junction import load_junction_config, junction_metrics, junction_figure
from safecross.perception import perceive
from safecross.render import draw_frame, make_gif, timeline_figure, fig_to_image
from safecross.sim import generate_episode, load_trajectory_bank

CFG = load_config()
RESULTS = resolve(CFG["paths"]["results_dir"])
WEIGHTS = resolve(CFG["paths"]["weights_dir"])
RAG_DIR = resolve(CFG["paths"]["rag_docs_dir"])
TMP = Path(tempfile.mkdtemp(prefix="safecross_"))
STATE = {"retriever": load_retriever(RESULTS / "rag_index.json")}
_DETECTORS: dict = {}
DEFAULT_SEED = 61
DCFG = {"location_label": "Sample crossing", "cycle_time_s": 90, "vehicle_red_extra_s": 20,
        "vehicle_arrival_per_s": 0.25, "vehicle_discharge_per_s": 0.5, "pedestrian_arrival_per_s": 0.06,
        "demo_video": "results/demo_video.mp4", "demo_video_seconds": 15}
DCFG.update(CFG.get("dashboard") or {})

# =============================================================================== look and feel
CSS = """
:root { --mar:#8C1D2C; --mar2:#6E1522; --ink:#262626; --mute:#6B6B6B; --cream:#FFFFFF; --card:#FFFFFF;
        --line:#E6E6E6; --green:#2F8F4E; --green2:#EAF4EC; --rose:#FBEDEE; --soft:#F6F6F6; }
body, .gradio-container { background: var(--cream) !important; }
.gradio-container { max-width: 1480px !important; padding: 0 !important; }
footer { display:none !important; }
#sc-side { background:#FBF8F5; border-right:1px solid var(--line); min-height: 100vh; padding: 22px 14px !important; }
#sc-side button.sc-nav { justify-content:flex-start !important; text-align:left !important; font-size:15px !important;
   border-radius:12px !important; padding:12px 18px !important; margin:1px 0 !important; box-shadow:none !important; }
#sc-side button.sc-nav.secondary { background:transparent !important; color:var(--ink) !important; border:none !important; font-weight:500; }
#sc-side button.sc-nav.secondary:hover { background:#F1EAE3 !important; }
#sc-side button.sc-nav.primary { background:var(--mar) !important; color:#fff !important; border:none !important; font-weight:600; }
.sc-card { gap:6px !important; }
#sc-main { padding: 0 22px 22px !important; }
.sc-top { display:flex; justify-content:space-between; align-items:center; padding:18px 4px 14px; border-bottom:1px solid var(--line); }
.sc-top h1 { font-size:23px; margin:0; color:var(--ink); font-weight:700; }
.sc-top .meta { display:flex; gap:16px; align-items:center; color:var(--ink); font-size:14px; }
.sc-dot { width:9px; height:9px; border-radius:50%; background:#36B35A; display:inline-block; margin-right:6px; }
.sc-avatar { width:38px; height:38px; border-radius:50%; background:#E7E1DA; display:inline-flex; align-items:center; justify-content:center; }
.sc-card { background:var(--card) !important; border:1px solid var(--line) !important; border-radius:16px !important;
   padding:14px !important; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.sc-card-h { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
.sc-card-h .t { display:flex; align-items:center; gap:10px; font-size:18px; font-weight:700; color:var(--ink); }
.sc-pill { background:var(--green2); color:var(--green); border-radius:999px; padding:4px 11px; font-size:13px; font-weight:600; }
.sc-pill.grey { background:#F1EDE8; color:var(--mute); }
.sc-chip { background:rgba(30,26,24,.78); color:#fff; border-radius:10px; padding:6px 12px; font-size:14px; font-weight:600; }
.sc-issue { background:var(--rose); border:1px solid #F2C9C9; color:var(--mar2); border-radius:10px; padding:10px 12px; font-weight:600; }
.sc-ok { background:var(--green2); border:1px solid #CBE6D3; color:#1F6B3A; border-radius:10px; padding:10px 12px; font-weight:600; }
.sc-tile { border:1px solid var(--line); border-radius:12px; padding:12px; background:#fff; }
.sc-tile .k { color:var(--mute); font-size:13px; } .sc-tile .v { font-size:26px; font-weight:700; color:var(--ink); }
.sc-bar { height:10px; border-radius:6px; background:#EEE8E2; overflow:hidden; } .sc-bar i { display:block; height:100%; }
.sc-small { font-size:12px; color:var(--mute); }
button.primary { background: var(--mar) !important; border:none !important; color:#fff !important; }
button.secondary { background:#F1EAE3 !important; border:none !important; color:var(--ink) !important; }
button.sc-btn-outline { background:#fff !important; border:1.5px solid var(--mar) !important; color:var(--mar) !important; font-weight:600; }
#sc-side button.sc-nav img { width:20px !important; height:20px !important; margin-right:10px; }
#sc-side button.sc-nav.primary img { filter: brightness(0) invert(1); }
.sc-hero { position:relative; display:flex; justify-content:space-between; align-items:flex-start; padding:22px 4px 18px;
   border-bottom:1px solid var(--line); overflow:hidden; min-height:118px; }
.sc-hero .hi { font-size:22px; color:var(--ink); } .sc-hero .name { font-size:36px; font-weight:800; color:var(--ink); line-height:1.15; }
.sc-hero .sub { color:var(--mute); font-size:15px; margin-top:4px; }
.sc-hero .meta { position:relative; z-index:2; display:flex; gap:14px; align-items:center; font-size:14px; color:var(--ink);
   background:rgba(246,242,237,.85); border-radius:12px; padding:6px 8px; }
.sc-hero .sky { position:absolute; right:0; bottom:0; z-index:1; opacity:.55; }
.sc-av { width:40px; height:40px; border-radius:50%; background:var(--mar); color:#fff; display:inline-flex; align-items:center;
   justify-content:center; font-weight:700; }
.sc-metric { background:#fff; border:1px solid var(--line); border-radius:16px; padding:14px 16px; display:grid;
   grid-template-columns: minmax(0,1fr) 118px; overflow:hidden; align-items:center; gap:4px; margin-bottom:10px; }
.sc-metric .ttl { display:flex; gap:8px; align-items:center; font-size:14.5px; white-space:nowrap; font-weight:600; color:var(--ink); }
.sc-metric .val { font-size:30px; font-weight:800; color:var(--ink); }
.sc-delta { display:inline-block; background:var(--green2); color:var(--green); border-radius:8px; padding:2px 8px; font-size:13px; font-weight:700; margin-left:8px; vertical-align:middle; }
.sc-delta.bad { background:var(--rose); color:var(--mar2); }
.sc-alert { display:grid; grid-template-columns:92px 1fr; gap:12px; border-radius:12px; padding:8px; margin-bottom:8px; background:#FCF4F4; }
.sc-alert.med { background:#FBF6EE; } .sc-alert.ok { background:var(--green2); grid-template-columns:1fr; }
.sc-alert img { width:92px; height:64px; object-fit:cover; border-radius:8px; }
.sc-alert .a1 { font-weight:700; font-size:14.5px; } .sc-alert .a2 { font-size:13px; color:var(--ink); } .sc-alert .a3 { font-size:12.5px; color:var(--mute); }
button.sc-qa { flex-direction:column !important; gap:8px !important; min-height:118px !important; background:#FBF8F5 !important;
   border:1px solid var(--line) !important; border-radius:14px !important; color:var(--ink) !important; font-size:15px !important; }
button.sc-qa img { width:30px !important; height:30px !important; }
#sc-fab { position:fixed; right:26px; bottom:22px; z-index:50; width:auto !important; border-radius:999px !important;
   background:var(--mar) !important; color:#fff !important; padding:12px 20px !important; box-shadow:0 6px 18px rgba(140,29,44,.35);
   font-weight:600; }
.sc-card { justify-content:flex-start !important; }
.sc-card > * { flex-grow:0 !important; }
#sc-live { flex-grow:0 !important; }
#sc-live, #sc-live .image-container, #sc-live img { background:#FFFFFF !important; }
#sc-live img { object-fit:contain; }
.sc-qa-row { flex-wrap:nowrap !important; gap:10px !important; }
.sc-qa-row button.sc-qa { min-width:0 !important; }
#sc-fab img { width:22px !important; height:22px !important; }
/* ---------- white municipal theme (overrides) ---------- */
body, .gradio-container, #sc-main { background:#FFFFFF !important; }
#sc-side { background:#FFFFFF !important; border-right:1px solid var(--line) !important; position:sticky; top:0;
   height:100vh; align-self:flex-start; overflow-y:auto; }
#sc-side button.sc-nav.secondary:hover { background:var(--soft) !important; }
.sc-card { border:1px solid var(--line) !important; box-shadow:0 1px 3px rgba(0,0,0,.05) !important; }
.sc-metric { border:1px solid var(--line); box-shadow:0 1px 3px rgba(0,0,0,.05); }
button.secondary { background:var(--soft) !important; border:1px solid var(--line) !important; }
.sc-pill.grey { background:var(--soft); color:var(--mute); }
.sc-hero { border-bottom:1px solid var(--line); min-height:128px; align-items:center; }
.sc-hero .title { font-size:32px; font-weight:800; color:var(--ink); letter-spacing:-.4px; line-height:1.15; }
.sc-hero .meta { background:rgba(255,255,255,.92); align-self:flex-start; }
.sc-hero .sky { opacity:1; }
.sc-alert { background:#FCF3F4; border:1px solid #F3E0E2; } .sc-alert.med { background:#FDF8F1; border-color:#F3E9D9; }
.sc-alert .loc { font-size:12.5px; color:var(--ink); }
.sc-link { color:var(--mar) !important; font-weight:600; font-size:13px; cursor:pointer; text-decoration:none; }
.sc-link:hover { text-decoration:underline; }
button.sc-qa { background:#FFFFFF !important; box-shadow:0 1px 3px rgba(0,0,0,.05); }
button.sc-qa:hover { border-color:var(--mar) !important; }
button.sc-linkbtn { background:none !important; border:none !important; box-shadow:none !important; color:var(--mar) !important;
   font-weight:600 !important; padding:2px 4px !important; min-width:0 !important; width:auto !important; flex-grow:0 !important; }
.sc-hidden { display:none !important; }
#sc-fab { background:#FFFFFF !important; color:var(--ink) !important; border:1px solid var(--line) !important;
   flex-direction:row-reverse !important; gap:12px !important; padding:6px 6px 6px 18px !important; white-space:pre-line !important;
   text-align:left !important; line-height:1.25 !important; font-size:13px !important; box-shadow:0 6px 20px rgba(0,0,0,.12) !important; }
#sc-fab img { width:46px !important; height:46px !important; background:var(--mar); border-radius:50%; padding:10px; }
#sc-pop { position:fixed !important; right:24px; bottom:22px; z-index:60; width:370px !important; max-width:calc(100vw - 32px);
   background:#fff !important; border:1px solid var(--line) !important; border-radius:18px !important;
   box-shadow:0 12px 36px rgba(0,0,0,.18) !important; padding:12px !important; gap:8px !important; }
#sc-pop-x { min-width:36px !important; max-width:40px; background:var(--soft) !important; }
.sc-pop-h { display:flex; gap:10px; align-items:center; font-weight:700; color:var(--ink); font-size:16px; }
.sc-pop-h i { width:34px; height:34px; border-radius:50%; background:var(--mar); display:inline-flex; align-items:center; justify-content:center; }
@media (max-width: 1000px) {
  #sc-side { position:static; height:auto; min-height:0; border-right:none; border-bottom:1px solid var(--line); }
  #sc-main { padding:0 12px 90px !important; }
  .sc-hero .sky { display:none; }
  .sc-hero { flex-direction:column; align-items:flex-start; gap:10px; }
}
"""
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red, neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(body_background_fill="#FFFFFF", block_background_fill="#FFFFFF", block_border_color="#E6E6E6",
      block_radius="14px", input_background_fill="#FFFFFF")

ICON = {  # simple line icons (inline SVG), no external files
    "cam": '<svg width="26" height="20" viewBox="0 0 26 20"><rect x="1" y="3" width="16" height="14" rx="3" fill="#2A2523"/><path d="M18 8l7-4v12l-7-4z" fill="#2A2523"/></svg>',
    "light": '<svg width="18" height="28" viewBox="0 0 18 28"><rect x="2" y="1" width="14" height="26" rx="5" fill="#2A2523"/><circle cx="9" cy="7" r="3" fill="#D9473B"/><circle cx="9" cy="14" r="3" fill="#E7A93A"/><circle cx="9" cy="21" r="3" fill="#3FB061"/></svg>',
    "photo": '<svg width="26" height="22" viewBox="0 0 26 22"><rect x="1" y="4" width="24" height="17" rx="4" fill="#2A2523"/><rect x="8" y="1" width="10" height="5" rx="2" fill="#2A2523"/><circle cx="13" cy="12.5" r="5" fill="#fff"/><circle cx="13" cy="12.5" r="3" fill="#2A2523"/></svg>',
    "chat": '<svg width="28" height="24" viewBox="0 0 28 24"><path d="M3 2h22a2 2 0 012 2v12a2 2 0 01-2 2H11l-6 5v-5H3a2 2 0 01-2-2V4a2 2 0 012-2z" fill="#2A2523"/><circle cx="9" cy="10" r="1.8" fill="#fff"/><circle cx="14" cy="10" r="1.8" fill="#fff"/><circle cx="19" cy="10" r="1.8" fill="#fff"/></svg>',
    "bell": '<svg width="22" height="24" viewBox="0 0 22 24"><path d="M11 2a7 7 0 017 7v5l2 4H2l2-4V9a7 7 0 017-7z" fill="#2A2523"/><circle cx="11" cy="21" r="2.5" fill="#2A2523"/></svg>',
    "trend": '<svg width="26" height="22" viewBox="0 0 26 22"><path d="M2 18l7-7 5 4 9-11" stroke="#2A2523" stroke-width="2.6" fill="none"/><path d="M2 21h22" stroke="#2A2523" stroke-width="2"/></svg>',
    "bolt": '<svg width="20" height="26" viewBox="0 0 20 26"><path d="M12 1L2 15h7l-2 10 11-15h-7z" fill="#2A2523"/></svg>',
    "bars": '<svg width="24" height="22" viewBox="0 0 24 22"><rect x="1" y="10" width="5" height="11" rx="1.5" fill="#2A2523"/><rect x="9" y="3" width="5" height="18" rx="1.5" fill="#2A2523"/><rect x="17" y="7" width="5" height="14" rx="1.5" fill="#2A2523"/></svg>',
}

BRAND = """
<div style="text-align:center;padding:4px 0 18px;border-bottom:1px solid #E8E1D9;margin-bottom:14px">
  <svg width="78" height="62" viewBox="0 0 78 62">
    <circle cx="40" cy="7" r="5.5" fill="#2A2523"/>
    <path d="M38 14l-7 14 6 3 3-6 2 9-6 13h6l5-11 5 11h6l-7-16-2-10 6 5 7-2-1-4-5 1-8-8z" fill="#2A2523"/>
    <path d="M4 58l6-9h7l-6 9zM18 58l5-9h7l-4 9zM33 58l3-9h7l-2 9zM47 58l1-9h7l1 9zM61 58l-1-9h7l3 9z" fill="#2A2523"/>
  </svg>
  <div style="font-size:30px;font-weight:800;color:#2A2523;letter-spacing:-.5px;margin-top:6px">SafeCross</div>
  <div style="font-size:12px;color:#2A2523;white-space:nowrap">Safer. Smarter. More Accessible.</div>
</div>
"""
SIDE_FOOT = """
<div style="margin-top:40px;text-align:center;opacity:.85">
  <svg width="190" height="90" viewBox="0 0 190 90" fill="none" stroke="#8C847E" stroke-width="1.2">
    <path d="M0 88 Q95 70 190 88"/>
    <path d="M14 84V58h8v26M26 84V48h7v36M37 84V62h9v22M50 84V40h6v44M60 84V55h8v29"/>
    <path d="M92 84V30l3-28 3 28v54M86 84V50h4M100 84V50h4"/>
    <path d="M122 84V52h8v32M134 84V44h7v40M145 84V58h9v26M158 84V48h6v36M168 84V60h8v24"/>
  </svg>
  <div style="font-size:13px;color:#2A2523;margin-top:4px;font-weight:600">MAAI901 × MAAI902</div>
  <div style="font-size:11.5px;color:#6F6762">University of Wollongong in Dubai</div>
</div>
"""


def _skyline(width=760, height=128) -> str:
    """A generic city skyline drawn with simple grey shapes (no real logos or photos)."""
    rng = np.random.default_rng(7)
    parts, x = [], 0
    while x < width:
        w = int(rng.integers(12, 30))
        h = int(rng.integers(16, 66))
        tone = ["#EDEDED", "#E4E4E4", "#DADADA"][int(rng.integers(0, 3))]
        parts.append(f'<rect x="{x}" y="{height - h}" width="{w - 2}" height="{h}" fill="{tone}"/>')
        for wy in range(height - h + 6, height - 4, 7):
            parts.append(f'<line x1="{x + 3}" y1="{wy}" x2="{x + w - 5}" y2="{wy}" stroke="#FFFFFF" stroke-width=".8"/>')
        x += w
    tx = int(width * 0.42)                                              # one tall needle tower
    parts.append(f'<path d="M{tx} {height} L{tx + 5} 44 L{tx + 9} 2 L{tx + 13} 44 L{tx + 18} {height} Z" fill="#D2D2D2"/>')
    return f'<svg class="sky" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(parts)}</svg>'


SKY = _skyline(width=560)


def topbar_html() -> str:
    now = dt.datetime.now()
    when = now.strftime("%a, %b %d, %Y  •  %I:%M %p").replace(" 0", " ")
    cal = ('<svg width="18" height="18" viewBox="0 0 18 18"><rect x="2" y="3" width="14" height="13" rx="2" fill="none" '
           'stroke="#262626" stroke-width="1.6"/><path d="M2 7h14M6 1v4M12 1v4" stroke="#262626" stroke-width="1.6"/></svg>')
    user = ('<span class="sc-av"><svg width="20" height="20" viewBox="0 0 20 20"><circle cx="10" cy="7" r="4" fill="#fff"/>'
            '<path d="M2 19c0-5 4-7 8-7s8 2 8 7z" fill="#fff"/></svg></span>')
    return f"""<div class="sc-hero">
      <div style="position:relative;z-index:2"><div class="title">Dubai Traffic Control</div>
        <div class="sub">Real-time insights for safer and more accessible pedestrian crossings.</div></div>
      <div class="meta"><span title="The feed is a replay of a simulation or of a recorded video, not a live camera">
        <span class="sc-dot"></span>Replay</span><span style="color:#DDDDDD">|</span>
        <span style="display:flex;gap:6px;align-items:center">{cal}{when}</span>{user}</div>
      {SKY}</div>"""


def card_head(icon: str, title: str, right: str = "") -> str:
    return f'<div class="sc-card-h"><div class="t">{ICON[icon]}<span>{title}</span></div><div>{right}</div></div>'


# =============================================================================== helpers
PREFERRED_WEIGHTS = "pmma_v2_yolo11s_seed0.pt"   # main detector: trained on PMMA video 1 (scripts/14 + 02)


def _weights_list() -> list[str]:
    w = sorted(glob.glob(str(WEIGHTS / "*.pt")))
    return sorted(w, key=lambda x: Path(x).name != PREFERRED_WEIGHTS)   # the main detector first (the default)


def model_choices() -> list[str]:
    return [Path(w).name for w in _weights_list()] + ["yolo11n.pt (pre-trained, people only)",
                                                      "YOLO-World (zero-shot, text prompts)"]


def _get_detector(choice: str):
    from safecross.vision.detect import Detector
    v = CFG["vision"]
    if choice in _DETECTORS:
        return _DETECTORS[choice]
    dev = v.get("device", "auto")
    if choice.startswith("YOLO-World"):
        det = Detector("yolo_world", v["yolo_world_model"], prompts=v["world_prompts"], conf=v["conf"], imgsz=v["imgsz"], device=dev)
    elif choice.startswith("yolo11n.pt"):
        det = Detector("yolo", "yolo11n.pt", conf=v["conf"], imgsz=v["imgsz"], device=dev)
    else:
        det = Detector("yolo", str(WEIGHTS / choice), conf=v["conf"], imgsz=1280 if "pmma" in choice.lower() else v["imgsz"], device=dev)
    _DETECTORS[choice] = det
    return det


def _example_pair():
    """(image, label file) for the Crossing Check: a test frame from your own dataset (ICG or PMMA) with
    the most mobility-aid users, so the demo uses real labels. Falls back to the Ultralytics sample photo."""
    try:
        from safecross.vision.dataset import dataset_yaml_path, list_images
        y = dataset_yaml_path(CFG)
        if y.exists():
            from safecross.vision.dataset import load_dataset_yaml
            d = load_dataset_yaml(y)
            aid_ids = {i for i, n in enumerate(d["names"]) if n in set(CFG["vision"]["mobility_aid_classes"])}
            best, best_n = None, -1
            for img, lab in list_images(d, y, "test")[:200]:
                if lab.exists():
                    n = sum(1 for l in lab.read_text().splitlines() if l.split() and int(l.split()[0]) in aid_ids)
                    if n > best_n:
                        best, best_n = (str(img), str(lab)), n
            if best:
                return best
    except Exception:
        pass
    return _example_image(), None


def _example_image():
    try:
        import ultralytics
        p = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        return str(p) if p.exists() else None
    except Exception:
        return None


def _people(dets):
    keep = {"pedestrian"} | set(CFG["vision"]["mobility_aid_classes"])
    return [d for d in dets if d["cls"] in keep]


# =============================================================================== crossing check
def crossing_check(image, label_file, model_choice, kind, severity):
    if image is None:
        raise gr.Error("Upload a crossing image first.")
    import cv2
    from safecross.vision import degrade
    from safecross.vision.detect import load_yolo_labels, draw_detections
    from safecross.vision.evaluate import match
    try:
        det = _get_detector(model_choice)
    except Exception as e:
        raise gr.Error(f"Could not load the model: {e}")
    if isinstance(image, str):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    gts = []
    if label_file is not None:
        names = ["pedestrian", "wheelchair", "rollator", "crutches", "cane"]
        from safecross.vision.dataset import dataset_yaml_path
        y = dataset_yaml_path(CFG)
        if y.exists():
            from safecross.vision.dataset import load_dataset_yaml
            names = load_dataset_yaml(y)["names"]
        gts = load_yolo_labels(label_file if isinstance(label_file, str) else label_file.name, w, h, names)
    sev = int(severity) if kind != "clean" else 0
    deg = degrade.apply(bgr, kind, sev, seed=0, boxes=[g["box"] for g in gts] or None)
    d_clean, d_deg = _people(det.predict(bgr)), _people(det.predict(deg))
    aid = set(CFG["vision"]["mobility_aid_classes"])
    if gts:
        _, miss_c, _ = match(d_clean, gts, 0.5, class_aware=False)
        _, miss_d, _ = match(d_deg, gts, 0.5, class_aware=False)
        ref = "the ground-truth labels"
    else:
        miss_c = []
        _, miss_d, _ = match(d_deg, d_clean, 0.5, class_aware=False)
        ref = "the detections on the clean image (no labels uploaded)"
    img_c = cv2.cvtColor(draw_detections(bgr, d_clean, miss_c), cv2.COLOR_BGR2RGB)
    img_d = cv2.cvtColor(draw_detections(deg, d_deg, miss_d), cv2.COLOR_BGR2RGB)
    n_aid = sum(1 for m in miss_d if m["cls"] in aid)
    md = (f"### Result\n| | clean image | {kind} (severity {sev}) |\n|---|---|---|\n"
          f"| people detected | {len(d_clean)} | {len(d_deg)} |\n"
          f"| of which mobility-aid users | {sum(1 for d in d_clean if d['cls'] in aid)} | "
          f"{sum(1 for d in d_deg if d['cls'] in aid)} |\n\n"
          f"Compared with {ref}: **{len(miss_d)} people missed after degradation**, of which "
          f"**{n_aid} mobility-aid users** (dashed red boxes).")
    summary = {"kind": kind, "sev": sev, "missed": len(miss_d), "missed_aid": n_aid, "model": model_choice,
               "image": img_d, "time": dt.datetime.now().strftime("%I:%M %p").lstrip("0")}
    return img_c, img_d, md, summary


def barrier_scan(image):
    """Zero-shot scan for possible accessibility barriers with YOLO-World, then look up the matching
    passages in the Dubai Universal Design Code index (if it has been built in Settings)."""
    if image is None:
        raise gr.Error("Upload a crossing image first.")
    import cv2
    from safecross.vision.detect import Detector, draw_detections
    v = CFG["vision"]
    try:
        if "_barrier" not in _DETECTORS:
            _DETECTORS["_barrier"] = Detector("yolo_world", v["yolo_world_model"], prompts=v["barrier_prompts"],
                                              conf=v["conf"], imgsz=v["imgsz"], device=v.get("device", "auto"))
        det = _DETECTORS["_barrier"]
    except Exception as e:
        raise gr.Error(f"Could not load YOLO-World ({e}). See the README (CLIP install).")
    if isinstance(image, str):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    dets = det.predict(bgr)
    img = cv2.cvtColor(draw_detections(bgr, dets, mobility_classes=tuple(v["barrier_prompts"])), cv2.COLOR_BGR2RGB)
    found = [d["cls"] for d in dets if d["cls"] != "curb_ramp"]
    md = "### Barrier scan (YOLO-World, zero-shot)\n"
    if dets:
        md += "| object | confidence |\n|---|---|\n" + "".join(
            f'| {d["cls"].replace("_", " ")} | {d.get("conf", 0):.2f} |\n' for d in sorted(dets, key=lambda d: -d.get("conf", 0)))
    else:
        md += "No prompted objects found.\n"
    md += ("\n_These are zero-shot model outputs from text prompts. They have not been validated - check the image "
           "yourself before reporting an issue._\n")
    ret = STATE["retriever"]
    if ret is None:
        md += "\n**Guidelines:** build the Dubai Universal Design Code index in **Settings** to show relevant passages here."
    else:
        md += "\n### Relevant guideline passages (retrieved from the Dubai Universal Design Code)\n"
        for cls in sorted(set(d["cls"] for d in dets)) or ["pedestrian crossing"]:
            for hit in ret.search(f"{cls.replace('_', ' ')} pedestrian crossing accessible route", k=1):
                where = f'{hit["source"]}, p. {hit["page"]}' if hit.get("page") else hit["source"]
                md += f'\n**{cls.replace("_", " ")}** ({where}):\n> {hit["text"][:420].strip()}…\n'
    summary = {"found": found, "image": img, "time": dt.datetime.now().strftime("%I:%M %p").lstrip("0")}
    return img, md, summary


def crossing_card_html(summary) -> str:
    if not summary:
        return '<div class="sc-small">Run a check on the Crossing Check page.</div>'
    if summary["missed_aid"]:
        issue = (f'<div class="sc-issue">&#9888;&nbsp; Potential issue: {summary["missed_aid"]} mobility-aid '
                 f'user(s) missed under {summary["kind"]} (severity {summary["sev"]})</div>')
    elif summary["missed"]:
        issue = (f'<div class="sc-issue">&#9888;&nbsp; Potential issue: {summary["missed"]} person(s) missed '
                 f'under {summary["kind"]} (severity {summary["sev"]})</div>')
    else:
        issue = f'<div class="sc-ok">&#10003;&nbsp; No one missed under {summary["kind"]} (severity {summary["sev"]})</div>'
    ret = STATE["retriever"]
    if ret:
        hits = ret.search("pedestrian crossing accessible signal", 1)
        rule = (f'{hits[0]["source"]}, page {hits[0]["page"]}<br><span class="sc-small">'
                f'{hits[0]["text"][:120]}…</span>') if hits else "No matching passage found."
    else:
        rule = '<span class="sc-small">Add the Dubai Universal Design Code PDF in Settings to show the relevant rule.</span>'
    return (f'{issue}<div style="margin-top:10px;font-weight:700">Relevant rule</div>'
            f'<div style="font-size:14px">{rule}</div>'
            f'<div class="sc-small" style="margin-top:6px">Detector: {summary["model"]}</div>')


# =============================================================================== simulation + replay
REASON_TEXT = {
    "confirmed_occupancy": "Waiting for the last pedestrian to clear the crossing…",
    "mobility_aid_on_crossing": "Mobility-aid user on the crossing: walk phase extended…",
    "uncertain_occupancy_lost_track": "Camera lost a person on the crossing: holding the walk phase (uncertain occupancy)…",
    "crossing_clear": "Crossing clear: walk phase ended.",
    "max_phase_reached": "Maximum phase length reached.",
}


def build_replay(bundle, controller="uncertainty_aware", every_s=0.5):
    ep, obs, res = bundle["episode"], bundle["observations"], bundle["results"][controller]
    stop = min(len(ep.times) - 1, int((res.end_time + 3.0) / ep.dt))
    step = max(1, int(every_s / ep.dt))
    frames, infos = [], []
    base = base_phase_length(bundle["cfg"])
    for k in range(0, stop + 1, step):
        t = float(ep.times[k])
        frames.append(np.asarray(draw_frame(ep, k, obs, res, title=""))[30:412])   # drop the legend strip
        evts = [e for e in res.log if e["t"] <= t + 1e-9]
        last = evts[-1] if evts else res.log[0]
        walk = t < res.end_time
        msg = REASON_TEXT.get(last["reason"], f"Walk phase running (normal length {base:.1f} s)…")
        if last["event"] == "PHASE_START":
            msg = f"Walk phase running normally (normal length {base:.1f} s)."
        infos.append({"t": t, "walk": walk, "extended": walk and t >= base, "end": res.end_time, "base": base,
                      "max": bundle["cfg"]["controller"]["max_phase_s"], "msg": msg, "conf": last.get("confidence"),
                      "people": len([o for o in obs[k] if o.gt_pid >= 0]),
                      "state": res.occupancy[k] if k < len(res.occupancy) else "ended"})
    return {"source": "Simulation", "frames": frames, "infos": infos, "i": 0,
            "label": f"Simulated crossing · scenario {bundle.get('seed', '')}"}


def signal_html(info, source_note: str = "") -> str:
    """Signal card. Times come from the simulated controller; the cycle time is a configured assumption."""
    if not info:
        return ""
    walk, t, base, C = info["walk"], info["t"], info["base"], float(DCFG["cycle_time_s"])
    col = "#2F8F4E" if walk else "#B3343A"
    figure = ('<path d="M50 22a7 7 0 110 .1M46 34l-9 17 6 3 4-8 3 11-8 17h7l6-13 7 13h7l-9-19-2-12 7 6 8-2-1-5-6 1-9-9z" fill="#fff"/>'
              if walk else '<path d="M40 30h20v12l4-6h4v14c0 9-6 16-14 16s-12-5-14-10l-4-10 4-2 4 6V30z" fill="#fff"/>')
    if walk and t < base:
        label, big, small = "Walk Phase", f"{base - t:.0f}s remaining", "if the crossing is clear"
        pct, bar = 100 * t / base, col
    elif walk:
        label, big, small = "Walk Phase · Extended", "Held for people crossing", f"up to {info['max']:.0f} s maximum"
        pct, bar = 100 * t / info["max"], "#C98A1B"
    else:
        rest = max(0.0, C - t)
        label, big, small = "Don't Walk", f"{rest:.0f}s to next walk", f"walk ended at {info['end']:.1f} s"
        pct, bar = 100 * (t - info["end"]) / max(1.0, C - info["end"]), col
    nxt, nxt2 = ("Red", "(Vehicles go)") if walk else ("Walk", "(next cycle)")
    car = ('<svg width="30" height="22" viewBox="0 0 30 22"><path d="M4 12l3-7h16l3 7v7h-3v-2H7v2H4z" fill="#8C1D2C"/>'
           '<circle cx="9" cy="14" r="2" fill="#fff"/><circle cx="21" cy="14" r="2" fill="#fff"/></svg>')
    clock = ('<svg width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="#262626" '
             'stroke-width="2"/><path d="M12 6v6l4 3" stroke="#262626" stroke-width="2" fill="none"/></svg>')
    return f"""
    <div style="text-align:center">
      <svg width="132" height="132" viewBox="0 0 100 100"><circle cx="50" cy="50" r="48" fill="{col}" opacity=".14"/>
        <circle cx="50" cy="50" r="40" fill="{col}"/>{figure}</svg>
      <div style="font-size:24px;font-weight:800;color:#262626;margin-top:2px">{label}</div>
      <div style="font-size:16px;color:#262626">{big}</div>
      <div class="sc-small">{small}</div>
    </div>
    <div class="sc-bar" style="margin:12px 6px 4px"><i style="width:{min(100, pct):.0f}%;background:{bar}"></i></div>
    <div class="sc-small" style="text-align:center;margin-bottom:10px">{info['msg']}</div>
    {f'<div class="sc-small" style="text-align:center;margin:-4px 0 10px;padding:6px 8px;background:#F6F6F6;border-radius:8px">{source_note}</div>' if source_note else ''}
    <div style="display:grid;grid-template-columns:1fr 1px 1fr;gap:10px;align-items:center;border-top:1px solid #EEEEEE;padding-top:10px">
      <div style="display:flex;gap:8px;align-items:center">{car}<div><div class="sc-small">Next Phase</div>
        <div style="font-size:14px"><b style="color:#B3343A">{nxt}</b> <span class="sc-small">{nxt2}</span></div></div></div>
      <div style="background:#EEEEEE;height:34px"></div>
      <div style="display:flex;gap:8px;align-items:center">{clock}<div><div class="sc-small">Cycle Time</div>
        <div style="font-weight:700;font-size:15px">{C:.0f}s <span class="sc-small" style="font-weight:400">(configured)</span></div></div></div>
    </div>"""


def live_head_html(replay) -> str:
    if not replay or not replay["infos"]:
        return card_head("cam", "Live Crossing Feed", '<span class="sc-pill grey">No replay</span>')
    info = replay["infos"][replay["i"] % len(replay["infos"])]
    src = "Replay · simulation" if replay["source"] == "Simulation" else "Replay · recorded video"
    return card_head("cam", "Live Crossing Feed", f'<span class="sc-pill">&#9679; {src}</span>')


def run_signal(seed, n_people, aid_share, recall, misclass, track_loss, fp_rate, use_pmma, history):
    cfg = json.loads(json.dumps(CFG))
    cfg["scenario"]["n_pedestrians"] = [int(n_people), int(n_people)]
    cfg["scenario"]["mobility_aid_share"] = float(aid_share)
    bank = None
    pm = resolve(cfg["paths"]["data_dir"]) / "pmma_trajectories.json"
    if use_pmma and pm.exists():
        bank = load_trajectory_bank(pm)
    ep = generate_episode(int(seed), cfg, np.random.default_rng(int(seed)), bank)
    f = dict(cfg["failures"])
    f.update({"recall": float(recall), "misclassification_prob": float(misclass),
              "track_loss_prob": float(track_loss), "false_positive_rate": float(fp_rate)})
    obs = perceive(ep, f, np.random.default_rng(int(seed) + 1))
    results = {c: run_controller(c, cfg, ep, obs) for c in ("baseline", "uncertainty_aware")}
    gifs = {c: make_gif(ep, obs, r, str(TMP / f"{c}_{seed}_{os.getpid()}.gif"),
                        "Baseline controller" if c == "baseline" else "Uncertainty-aware controller")
            for c, r in results.items()}
    timeline = fig_to_image(timeline_figure(ep, list(results.values())))
    rows, run_rec = [], {"seed": int(seed)}
    for c, r in results.items():
        m = episode_metrics(ep, r, cfg)
        run_rec[c] = m
        rows.append({"controller": c, "phase ended at (s)": r.end_time, "time needed (s)": m["needed_time"],
                     "mobility-aid users cut off": f'{m["aid_cutoffs"]} of {m["aid_crossings"]}',
                     "anyone cut off": f'{m["all_cutoffs"]} of {m["all_crossings"]}',
                     "extra green time (s)": m["added_time"]})
    people = pd.DataFrame([{"person": f"P{p.pid}", "type": p.cls, "starts (s)": round(p.t_start, 1),
                            "reaches other side (s)": round(ep.exit_time(i), 1)} for i, p in enumerate(ep.pedestrians)])
    logs = {c: "\n".join(format_log(r)) for c, r in results.items()}
    bundle = {"episode": ep, "observations": obs, "results": results, "cfg": cfg, "controller": "uncertainty_aware",
              "seed": int(seed), "time": dt.datetime.now().strftime("%I:%M %p").lstrip("0")}
    jfig, jtab = junction_view(ep, obs, cfg, results, int(seed))
    cut_b = run_rec["baseline"]["aid_cutoffs"] > 0
    headline = ("**The baseline cut off a mobility-aid user (red x on the timeline).**" if cut_b
                else "No mobility-aid user was cut off by the baseline in this run. Try lower recall or more track loss.")
    history = (history or quick_history()) + [run_rec]
    replay = build_replay(bundle)
    return (gifs["baseline"], gifs["uncertainty_aware"], timeline, pd.DataFrame(rows), people,
            logs["baseline"], logs["uncertainty_aware"], headline, bundle, history, replay, replay,
            metric_cards_html(history), trends_html(bundle), jfig, jtab)


def junction_view(ep, obs, cfg, results, seed):
    """The same walk phase placed inside the simulated junction (configs/junction.yaml)."""
    try:
        j = load_junction_config()
    except Exception:
        return None, pd.DataFrame({"info": ["configs/junction.yaml is missing"]})
    c2 = json.loads(json.dumps(cfg))
    c2["controller"].update(j.get("ghost_filter", {}))
    allres = dict(results)
    allres["uncertainty_aware_filtered"] = run_controller("uncertainty_aware_filtered", c2, ep, obs)
    rows = []
    for c, r in allres.items():
        m = junction_metrics(ep, r, j, seed)
        rows.append({"controller": c, "walk ends (s)": r.end_time,
                     "cross-road cars arrive (s)": round(m["cross_green_start"] + j["startup_s"], 1),
                     "mobility-aid users still on crossing then": f'{m["aid_cross_conflicts"]} of {m["aid_crossings"]}',
                     "meet a turning car (permissive)": f'{m["permissive_aid_turn_conflicts"]} of {m["aid_crossings"]}',
                     "meet a turning car (gated)": f'{m["gated_aid_turn_conflicts"]} of {m["aid_crossings"]}',
                     "cross-road green lost (s)": round(m["cross_green_lost_s"], 1),
                     "extra delay per cross-road car (s)": round(m["cross_extra_delay_s"], 1)})
    fig = fig_to_image(junction_figure(ep, results, j, seed))
    return fig, pd.DataFrame(rows)


# =============================================================================== metric cards, alerts, trends
def _spark(values, colour, fill, w=118, h=50) -> str:
    v = [float(x) for x in values] or [0.0]
    if len(v) == 1:
        v = v * 2
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = [(i * (w - 4) / (len(v) - 1) + 2, h - 6 - (x - lo) / span * (h - 14)) for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"2,{h} " + line + f" {w - 2},{h}"
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8" fill="{colour}"/>' for x, y in pts[-6:])
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polygon points="{area}" fill="{fill}"/>'
            f'<polyline points="{line}" fill="none" stroke="{colour}" stroke-width="2"/>{dots}</svg>')


def estimated_waits(walk_end: float) -> tuple[float, float]:
    """Average waiting time per arrival with uniform arrivals (the first term of Webster's delay formula):
    wait = red^2 / (2 * cycle). Pedestrians wait while it is Don't Walk; vehicles wait during the walk
    phase plus the other phases set in config (vehicle_red_extra_s). These are ESTIMATES from the
    simulated walk phase and configured assumptions, not measurements."""
    C = float(DCFG["cycle_time_s"])
    g = min(float(walk_end), C)
    ped_red = C - g
    veh_red = min(C, g + float(DCFG["vehicle_red_extra_s"]))
    return ped_red ** 2 / (2 * C), veh_red ** 2 / (2 * C)


MI = {
    "walker": '<svg width="22" height="26" viewBox="0 0 22 26"><circle cx="12" cy="3.5" r="3" fill="#8C1D2C"/><path d="M11 8l-5 9 3 1 3-5 1 5-4 7h3l4-6 3 6h3l-5-10-1-6 4 3 4-1-1-2-3 1z" fill="#8C1D2C"/></svg>',
    "car": '<svg width="28" height="22" viewBox="0 0 30 22"><path d="M4 12l3-7h16l3 7v7h-3v-2H7v2H4z" fill="#8C1D2C"/><circle cx="9" cy="14" r="2" fill="#fff"/><circle cx="21" cy="14" r="2" fill="#fff"/></svg>',
    "shield": '<svg width="24" height="26" viewBox="0 0 24 26"><path d="M12 1l10 4v7c0 6-4 11-10 13C6 23 2 18 2 12V5z" fill="#2F8F4E"/><path d="M7 13l4 4 7-8" stroke="#fff" stroke-width="2.4" fill="none"/></svg>',
}


def metric_cards_html(history) -> str:
    h = history or []
    if not h:
        return '<div class="sc-small">Run a simulation to see metrics.</div>'
    ua_w = [estimated_waits(r["uncertainty_aware"]["end_time"]) for r in h]
    b_w = [estimated_waits(r["baseline"]["end_time"]) for r in h]
    ped, veh = [w[0] for w in ua_w], [w[1] for w in ua_w]
    d_ped = np.mean(ped) - np.mean([w[0] for w in b_w])
    d_veh = np.mean(veh) - np.mean([w[1] for w in b_w])
    ev_u = [r["uncertainty_aware"]["all_cutoffs"] for r in h]
    tb, tu = sum(r["baseline"]["all_cutoffs"] for r in h), sum(ev_u)
    ev_delta = (f"&darr; {100 * (tb - tu) / tb:.0f}%", False) if tb else ("no change", False)
    if tb and tu > tb:
        ev_delta = (f"&uarr; {100 * (tu - tb) / tb:.0f}%", True)

    def fmt(d):
        return (f"&darr; {abs(d):.1f}s" if d < 0 else f"&uarr; {d:.1f}s"), d > 0.05

    def card(icon, title, value, delta, spark):
        txt, bad = delta
        return (f'<div class="sc-metric"><div><div class="ttl">{MI[icon]}{title}</div><div style="margin-top:6px">'
                f'<span class="val">{value}</span><span class="sc-delta{" bad" if bad else ""}">{txt}</span></div></div>'
                f'<div>{spark}</div></div>')
    return (card("walker", "Avg. Pedestrian Wait", f"{np.mean(ped):.0f}s", fmt(d_ped),
                 _spark(ped, "#2F8F4E", "rgba(47,143,78,.12)"))
            + card("car", "Avg. Vehicle Wait", f"{np.mean(veh):.0f}s", fmt(d_veh),
                   _spark(veh, "#8A8A8A", "rgba(138,138,138,.14)"))
            + card("shield", "Safety Events", f"{tu}", ev_delta, _spark(ev_u, "#B3343A", "rgba(179,52,58,.12)"))
            + f'<div class="sc-small" style="text-align:right">Simulated · {len(h)} runs · arrows compare with the baseline '
              f'controller · waits estimated for a {DCFG["cycle_time_s"]:.0f} s cycle</div>')


def _thumb(img_rgb, w=220) -> str:
    import base64
    import cv2
    im = np.ascontiguousarray(img_rgb)
    h0, w0 = im.shape[:2]
    im = cv2.resize(im, (w, int(h0 * w / w0)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def alerts_html(check, bundle, barrier=None) -> str:
    """Alerts come only from things the app actually computed: the simulation run, the crossing check
    (detector on a degraded image) and the barrier scan (zero-shot detector)."""
    loc = DCFG["location_label"]
    items = []   # (level, thumbnail, title, description, where, when, page)
    if barrier and barrier.get("found"):
        items.append(("high", _thumb(barrier["image"]), "Accessibility issue detected",
                      "Possible " + ", ".join(sorted(set(barrier["found"]))).replace("_", " ") + " (zero-shot, please verify)",
                      "Crossing Check image", barrier["time"], "check"))
    if bundle:
        ep, obs = bundle["episode"], bundle["observations"]
        rb, ru = bundle["results"]["baseline"], bundle["results"]["uncertainty_aware"]
        stamp = bundle.get("time", "")
        for i, p in enumerate(ep.pedestrians):
            if p.is_mobility_aid and p.t_start < rb.end_time and ep.exit_time(i) > rb.end_time:
                k = min(len(ep.times) - 1, int(rb.end_time / ep.dt))
                img = np.asarray(draw_frame(ep, k, obs, rb, title=""))
                items.append(("high", _thumb(img), "Unsafe signal decision prevented",
                              f"Baseline would end the walk phase with P{p.pid} ({p.cls}) still crossing",
                              "Simulated crossing", stamp, "signal"))
                break
        holds = [e for e in ru.log if e["reason"] == "uncertain_occupancy_lost_track"]
        if holds:
            e = holds[0]
            k = min(len(ep.times) - 1, int(e["t"] / ep.dt))
            img = np.asarray(draw_frame(ep, k, obs, ru, title=""))
            items.append(("med", _thumb(img), "Lost track on the crossing",
                          "Camera lost a person mid-crossing; walk phase held", "Simulated crossing", stamp, "signal"))
    if check and check.get("missed"):
        lvl = "high" if check["missed_aid"] else "med"
        who = f'{check["missed_aid"]} mobility-aid user(s)' if check["missed_aid"] else f'{check["missed"]} person(s)'
        items.append((lvl, _thumb(check["image"]), "Detection failure",
                      f'{who} missed under {check["kind"]} (severity {check["sev"]})', "Crossing Check image",
                      check.get("time", ""), "check"))
    if not items:
        return '<div class="sc-alert ok"><div class="a1">&#10004; No alerts in the latest runs</div></div>'
    out = ""
    for lvl, img, a1, a2, where, when, page in items[:3]:
        dot = "#C2382F" if lvl == "high" else "#E3A13A"
        pr = "High Priority" if lvl == "high" else "Medium Priority"
        out += (f'<div class="sc-alert {"" if lvl == "high" else "med"}"><img src="{img}"/><div>'
                f'<div class="a1"><span style="color:{dot}">&#9679;</span> {a1}</div><div class="a2">{a2}</div>'
                f'<div class="loc">&#128205; {where}</div><div class="a3">{when} · {pr}</div>'
                f'<div style="text-align:right"><a class="sc-link" data-go="{page}">View details &rarr;</a></div></div></div>')
    return out


def _cycle_series(bundle):
    """Pedestrians and vehicles over one signal cycle of the latest simulation.
    Pedestrians: people really on the crossing during the walk phase (simulation), then people arriving to wait.
    Vehicles: a simple queue (arrive at a fixed rate during red, leave at the discharge rate on green).
    Detected: what the damaged camera output saw during the walk phase."""
    ep, obs = bundle["episode"], bundle["observations"]
    end = bundle["results"]["uncertainty_aware"].end_time
    C = float(DCFG["cycle_time_s"])
    lam_p, lam_v, mu = (float(DCFG["pedestrian_arrival_per_s"]), float(DCFG["vehicle_arrival_per_s"]),
                        float(DCFG["vehicle_discharge_per_s"]))
    veh_red = min(C, end + float(DCFG["vehicle_red_extra_s"]))
    xs, ped, det, veh = [], [], [], []
    for t in np.arange(0, C + 0.01, 1.0):
        k = int(t / ep.dt)
        if t < end and k < len(ep.times):
            ped.append(sum(ep.on_crossing(i, k) for i in range(len(ep.pedestrians))))
            det.append(sum(1 for o in obs[k] if 0 <= o.y < ep.length))
        else:
            ped.append(round(lam_p * (t - end), 2))
            det.append(None)
        q = lam_v * t if t <= veh_red else max(0.0, lam_v * veh_red - (mu - lam_v) * (t - veh_red))
        veh.append(round(q, 2))
        xs.append(float(t))
    return xs, ped, det, veh, end


def trends_html(bundle=None, video=None) -> str:
    if video:
        xs = [i["t"] for i in video["infos"]]
        series = [("Pedestrians detected", "#8C1D2C", [i["people"] for i in video["infos"]], False),
                  ("Vehicles detected", "#9A9A9A", [i.get("vehicles", 0) for i in video["infos"]], False)]
        note = "Counts from the detector on the recorded video (model output)"
        end = None
    elif bundle:
        xs, ped, det, veh, end = _cycle_series(bundle)
        series = [("Pedestrians", "#8C1D2C", ped, False), ("Vehicles", "#9A9A9A", veh, False),
                  ("Pedestrians detected", "#C98A8F", det, True)]
        note = ("Simulated signal cycle · people on the crossing come from the simulation; waiting pedestrians and "
                "the vehicle queue use the arrival rates in config")
    else:
        return '<div class="sc-small">Run a simulation to see trends.</div>'
    W, H, L, B, T, R = 480, 230, 36, 32, 18, 10
    xmax = max(xs) or 1
    ymax = max(max(v for v in vals if v is not None) for _, _, vals, _ in series) * 1.15 + 1
    X = lambda x: L + (W - L - R) * x / xmax
    Y = lambda y: H - B - (H - B - T) * y / ymax
    g = ""
    for yv in np.linspace(0, ymax, 5):
        g += (f'<line x1="{L}" y1="{Y(yv):.1f}" x2="{W - R}" y2="{Y(yv):.1f}" stroke="#EEEEEE"/>'
              f'<text x="{L - 8}" y="{Y(yv) + 4:.1f}" font-size="11" text-anchor="end" fill="#6B6B6B">{yv:.0f}</text>')
    for xv in np.linspace(0, xmax, 7):
        g += f'<text x="{X(xv):.1f}" y="{H - B + 16}" font-size="11" text-anchor="middle" fill="#6B6B6B">{xv:.0f}s</text>'
    if end is not None:
        g += (f'<rect x="{L}" y="{T}" width="{X(end) - L:.1f}" height="{H - B - T}" fill="rgba(47,143,78,.06)"/>'
              f'<text x="{L + 6}" y="{T + 12}" font-size="11" fill="#2F8F4E">walk</text>'
              f'<text x="{X(end) + 6:.1f}" y="{T + 12}" font-size="11" fill="#B3343A">don&#39;t walk</text>')
    lines, legend = "", ""
    for j, (name, col, vals, dashed) in enumerate(series):
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, vals) if y is not None)
        if j == 0:
            lines += f'<polygon points="{X(xs[0]):.1f},{H - B} {pts} {X(xs[-1]):.1f},{H - B}" fill="rgba(140,29,44,.07)"/>'
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        lines += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{1.6 if dashed else 2.2}"{dash}/>'
        if not dashed:
            step = max(1, len(xs) // 10)
            lines += "".join(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.6" fill="{col}"/>'
                             for x, y in list(zip(xs, vals))[::step] if y is not None)
        legend += f'<span style="margin-left:14px"><span style="color:{col}">&#9679;</span> {name}</span>'
    return (f'<div style="text-align:right;font-size:13px;color:#262626">{legend}</div>'
            f'<svg width="100%" viewBox="0 0 {W} {H}">{g}{lines}</svg>'
            f'<div class="sc-small" style="text-align:center">{note}</div>')


def quick_history(n=12) -> list:
    """Run a few scenarios quickly (no animations) so the metric cards have a trend from the start."""
    out = []
    for seed in range(1, n + 1):
        cfg = json.loads(json.dumps(CFG))
        cfg["scenario"]["n_pedestrians"] = [4, 4]
        cfg["scenario"]["mobility_aid_share"] = 0.5
        ep = generate_episode(seed, cfg, np.random.default_rng(seed))
        f = dict(cfg["failures"])
        f.update({"recall": 0.85, "misclassification_prob": 0.1, "track_loss_prob": 0.3, "false_positive_rate": 0.05})
        obs = perceive(ep, f, np.random.default_rng(seed + 1))
        rec = {"seed": seed}
        for c in ("baseline", "uncertainty_aware"):
            rec[c] = episode_metrics(ep, run_controller(c, cfg, ep, obs), cfg)
        out.append(rec)
    return out


# =============================================================================== ask safecross
def _kg_for(bundle, controller):
    ep, res = bundle["episode"], bundle["results"][controller]
    return build_kg(ep, res, bundle["cfg"], STATE["retriever"])


def show_kg(bundle, controller, cited=None):
    """Picture of the decision knowledge graph for the last simulation (Ask SafeCross page)."""
    if bundle is None:
        return None
    path = str(TMP / f"kg_{controller}_{abs(hash(tuple(sorted(cited or []))))}.png")
    return draw_kg(_kg_for(bundle, controller), path, set(cited or []),
                   title=f"Decision knowledge graph - {controller} controller (last simulation)")


def ask_kg(message, bundle, controller, backend_kind, model_name, include_img, include_log, img_path):
    """Answer from the knowledge graph. Returns (answer, evidence markdown, cited fact ids) or None to fall back."""
    facts = _kg_for(bundle, controller)
    sub = subgraph(facts, message)
    use = sub if sum(1 for f in sub if "rule" not in f.o) >= 6 else facts
    given = kg_lines(use) if include_log else []
    ids = {f.gid for f in use} if include_log else set()
    if backend_kind.startswith("Offline"):
        raw = OfflineBackend._from_graph(build_kg_prompt(message, kg_lines(facts), {"x": "y"}))
        j = parse_json(raw)
        val = j.get("answer", j.get("reason_code"))
        if not val or val == "cannot_determine":
            return None                                   # offline graph reader only knows why/what-if questions
        cited = [str(c) for c in j.get("evidence", [])]
        ans = (f"**{str(val).replace('_', ' ')}**: {j.get('explanation', '')} "
               + " ".join(f"[{c}]" for c in cited)
               + "\n\n_Offline mode: read from the decision knowledge graph with fixed rules, no AI model._")
        used = [l for l in kg_lines(facts) if l.split("]")[0][1:] in cited]
        return ans, "**Facts used**\n\n```\n" + "\n".join(used) + "\n```", cited
    backend = OllamaBackend(model_name or CFG["explain"]["ollama_model"], CFG["explain"]["ollama_url"])
    user = build_kg_prompt(message, given, None)
    reply = backend.generate(SYSTEM_KG, user, img_path if include_img else None)
    cited = sorted(set(re.findall(r"\bG\d+\b", reply)), key=lambda g: int(g[1:]))
    bad = [c for c in cited if c not in ids]
    ok = [c for c in cited if c in ids]
    reply += (f"\n\n_Model: {backend.name} · knowledge graph {'on' if include_log else 'off'} "
              f"({len(given)} of {len(facts)} facts) · picture {'on' if include_img else 'off'}_")
    if bad:
        reply += (f"\n\n⚠️ **Citation check failed:** the answer cites {', '.join(bad)}, which "
                  f"{'was' if len(bad) == 1 else 'were'} not given to the model. Treat this answer as unsupported.")
    elif not ok and include_log:
        reply += "\n\n⚠️ **Citation check:** the answer cites no fact, so it cannot be checked."
    else:
        reply += f"\n\n✅ Citation check: every cited fact exists ({', '.join(ok)})." if ok else ""
    evidence = (f"**Facts given to the model** ({'question-focused subgraph' if use is sub else 'whole graph'}, "
                f"{len(given)} of {len(facts)})\n\n```\n" + "\n".join(given or ["(none: evidence switched off)"]) + "\n```")
    return reply, evidence, ok


def ask(message, history, bundle, controller, backend_kind, model_name, include_img, include_log,
        source="Event log (text)"):
    history = history or []
    if not message or not message.strip():
        return history, history, "", gr.update(), gr.update()
    if bundle is None:
        raise gr.Error("Run a simulation first (Smart Signal page).")
    bundle = dict(bundle)
    bundle["controller"] = controller
    ep, res = bundle["episode"], bundle["results"][controller]
    m = re.search(r"(\d+(?:\.\d+)?)\s*s", message)
    t = float(m.group(1)) if m else res.end_time
    k = max(0, min(len(ep.times) - 1, int(round(t / ep.dt))))
    img_path = str(TMP / f"ask_{controller}_{k}.png")
    draw_frame(ep, k, None, res, title=f"{controller} - camera view (schematic)").save(img_path)
    cited = []
    try:
        from safecross.explain.assistant import is_regulation_question
        kg_out = None
        if source.startswith("Knowledge") and not is_regulation_question(message):
            kg_out = ask_kg(message, bundle, controller, backend_kind, model_name, include_img, include_log, img_path)
        if kg_out is not None:
            answer, evidence, cited = kg_out
        elif backend_kind.startswith("Offline"):
            answer, ev = offline_answer(message, bundle, STATE["retriever"])
            evidence = "**Evidence used**\n\n```\n" + "\n".join(ev) + "\n```" if ev else "_No evidence lines._"
        else:
            backend = OllamaBackend(model_name or CFG["explain"]["ollama_model"], CFG["explain"]["ollama_url"])
            answer, prompt = ai_answer(message, bundle, backend, include_img, include_log, img_path, STATE["retriever"])
            answer += (f"\n\n_Model: {backend.name} · picture {'on' if include_img else 'off'} · "
                       f"log {'on' if include_log else 'off'}_")
            evidence = "**Exact prompt sent to the model**\n\n```\n" + prompt + "\n```"
    except Exception as e:
        answer = (f"Could not reach the AI model ({e}). Is the Ollama app open, and is the model downloaded "
                  f"(`ollama list`)? You can switch to Offline mode in Settings.")
        evidence = ""
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    try:
        graph = show_kg(bundle, controller, cited)
    except Exception:
        graph = None
    return history, history, "", gr.update(value=evidence), graph


# =============================================================================== reports + settings
def status_html() -> str:
    from safecross.vision.detect import pick_device
    w = _weights_list()
    ollama = OllamaBackend(CFG["explain"]["ollama_model"], CFG["explain"]["ollama_url"]).available()
    rag = STATE["retriever"]
    items = [
        ("Compute device", True, pick_device(CFG["vision"].get("device", "auto"))),
        ("Trained detector", bool(w), f"{len(w)} weight file(s)" if w else "Step 2 in the guide"),
        ("Dubai code index", rag is not None, f"{len(rag.chunks)} passages" if rag else "upload the PDF below"),
        ("AI model (Ollama)", ollama, CFG["explain"]["ollama_model"] + (" ready" if ollama else " not found")),
        ("Experiment A", (RESULTS / "experiment_a" / "summary.csv").exists(), "visual robustness"),
        ("Experiment B", (RESULTS / "experiment_b" / "summary.csv").exists(), "decision sensitivity"),
        ("Explanations", bool(glob.glob(str(RESULTS / "explanations" / "summary_*.csv"))), "faithfulness"),
    ]
    cards = "".join(
        f'<div class="sc-tile"><div class="k">{name}</div><div style="font-weight:700;color:'
        f'{"#2F8F4E" if ok else "#8C1D2C"}">{"Ready" if ok else "Not yet"}</div><div class="sc-small">{note}</div></div>'
        for name, ok, note in items)
    return f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px">{cards}</div>'


def results_views():
    empty = pd.DataFrame({"info": ["No results yet - run the steps in the guide"]})
    a = RESULTS / "experiment_a" / "summary.csv"
    df_a = empty
    if a.exists():
        d = pd.read_csv(a)
        d = d[d["class"] == "ALL_MOBILITY_AIDS"]
        if "found_as_person" in d.columns:
            # the measure used in the report: found as ANY person (IoU >= 0.5); class-aware recall kept for reference
            df_a = d[["model", "condition", "severity", "found_as_person", "found_as_person_lo", "found_as_person_hi",
                      "recall"]].rename(columns={"found_as_person": "found as person", "found_as_person_lo": "95% CI low",
                                                 "found_as_person_hi": "95% CI high",
                                                 "recall": "right group label too"}).round(3)
        else:
            df_a = d[["model", "condition", "severity", "recall", "recall_lo", "recall_hi", "precision"]].round(3)
    b = RESULTS / "experiment_b" / "summary.csv"
    df_b = empty
    if b.exists():
        df_b = pd.read_csv(b).pivot_table(index=["failure", "level"], columns="controller",
                                          values="cutoff_rate").reset_index().round(3)
    ex = sorted(glob.glob(str(RESULTS / "explanations" / "summary_*.csv")))
    df_e = empty
    if ex:
        parts = []
        for f in ex:
            d = pd.read_csv(f)
            d.insert(0, "model", Path(f).stem.replace("summary_", ""))
            parts.append(d)
        df_e = pd.concat(parts).round(3)
    b_png = RESULTS / "experiment_b" / "cutoff_vs_level.png"
    gallery = sorted(glob.glob(str(RESULTS / "experiment_a" / "examples" / "*.jpg")))[:12]
    files = [f for f in [str(a), str(b), *ex, str(RESULTS / "regulation_eval.csv")] if Path(f).exists()]
    return df_a, (str(b_png) if b_png.exists() else None), df_b, df_e, gallery, files or None


def junction_report():
    f = RESULTS / "junction" / "summary.csv"
    png = RESULTS / "junction" / "junction_safety.png"
    if not f.exists():
        return None, pd.DataFrame({"info": ["Not run yet: python scripts/13_junction_experiment.py"]})
    d = pd.read_csv(f)
    d = d[d.failure == "recall"][["level", "controller", "cutoff_rate", "cross_conflict_rate",
                                  "permissive_turn_conflict_rate", "gated_turn_conflict_rate",
                                  "cross_extra_delay_s", "gated_turn_delay_s"]].round(3)
    d.columns = ["recall", "controller", "cut off", "on crossing when cross-road cars arrive",
                 "meets turning car (permissive)", "meets turning car (gated)",
                 "extra delay per cross-road car (s)", "turning-car delay, gated (s)"]
    return (str(png) if png.exists() else None), d


def upload_code_pdf(files):
    if not files:
        raise gr.Error("Choose the Dubai Universal Design Code PDF first.")
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    for f in files if isinstance(files, list) else [files]:
        src = f if isinstance(f, str) else f.name
        shutil.copy(src, RAG_DIR / Path(src).name)
    n = build_index(RAG_DIR, RESULTS / "rag_index.json", CFG["explain"]["rag_chunk_chars"], CFG["explain"]["rag_chunk_overlap"])
    STATE["retriever"] = load_retriever(RESULTS / "rag_index.json")
    return f"Indexed {n} passages from {len(list(RAG_DIR.glob('*.pdf')))} PDF file(s).", status_html()


def process_video(video, model_choice, seconds):
    if not video:
        raise gr.Error("Upload a video first.")
    from safecross.vision.track import track_video
    if model_choice.startswith("YOLO-World"):
        raise gr.Error("Tracking uses a YOLO model. Choose your trained weights or yolo11n.pt.")
    weights = "yolo11n.pt" if model_choice.startswith("yolo11n.pt") else str(WEIGHTS / model_choice)
    frames, infos, rows = track_video(video, weights, CFG["vision"].get("device", "auto"),
                                      CFG["vision"]["conf"], float(seconds))
    if not frames:
        raise gr.Error("No frames could be read from this video.")
    replay = {"source": "Video", "frames": frames, "infos": infos, "i": 0, "label": f"Your video · {Path(video).name[:28]}"}
    table = pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["no people tracked"]})
    return replay, table, f"Tracked {len(rows)} people over {infos[-1]['t']:.0f} s. The dashboard now replays your video."


def _with_chip(frame, text, corner="top", pin=False):
    """Dark rounded label on the frame, like a camera overlay (top-left or bottom-left)."""
    import cv2
    img = np.ascontiguousarray(frame).copy()
    k = max(1.0, max(img.shape[:2]) / 720)          # keep the label readable on large frames
    fs, thick = 0.6 * k, max(2, int(round(2 * k)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    x, y = int(12 * k), int(12 * k) if corner == "top" else img.shape[0] - th - int(30 * k)
    pad = int(22 * k) if pin else 0
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + tw + int(24 * k) + pad, y + th + int(18 * k)), (30, 30, 30), -1)
    img = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
    if pin:
        cx, cy = x + int(16 * k), y + (th + int(18 * k)) // 2 - int(3 * k)
        cv2.circle(img, (cx, cy), int(6 * k), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([[cx - int(5 * k), cy + int(3 * k)], [cx + int(5 * k), cy + int(3 * k)], [cx, cy + int(11 * k)]])],
                     (255, 255, 255))
        cv2.circle(img, (cx, cy), max(2, int(2 * k)), (30, 30, 30), -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + int(12 * k) + pad, y + th + int(8 * k)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def tick(live, sim):
    """Called every half second: advance the replays and refresh the live cards."""
    out_img, head, sig = gr.update(), live_head_html(live), gr.update()
    if live and live["frames"]:
        live = dict(live)
        live["i"] = (live["i"] + 1) % len(live["frames"])
        info = live["infos"][live["i"]]
        n = info["people"]
        out_img = _with_chip(live["frames"][live["i"]], f'{n} {"pedestrian" if n == 1 else "pedestrians"} detected')
        where = (live.get("label") or "Simulated crossing") if live["source"] == "Simulation" \
            else DCFG["location_label"] + " · recorded video"
        out_img = _with_chip(out_img, where, corner="bottom", pin=True)
        head = live_head_html(live)
    if sim and sim["infos"]:
        sim = dict(sim)
        sim["i"] = (sim["i"] + 1) % len(sim["infos"])
        sim_label = sim.get("label", "Simulated crossing")
        if live and live.get("source") != "Simulation":
            note = f"Signal comes from the {sim_label.lower()}. It is not driven by the video on the left."
        else:
            note = f"{sim_label} (the same scene as the feed)."
        sig = signal_html(sim["infos"][sim["i"]], note)
    return live, sim, out_img, head, sig, topbar_html()


# =============================================================================== layout
# "View details" links inside HTML cards click a hidden Gradio button, so every link really navigates.
HEAD = """<script>
document.addEventListener('click', function (e) {
  var a = e.target.closest('[data-go]');
  if (!a) return;
  e.preventDefault();
  var b = document.getElementById('sc-go-' + a.getAttribute('data-go'));
  if (b) { (b.tagName === 'BUTTON' ? b : b.querySelector('button') || b).click(); }
});
</script>"""
ICON_DIR = Path(__file__).resolve().parent / "icons"
NAV_ICON = {"Dashboard": "home", "Live Monitoring": "video", "Crossing Check": "image", "Smart Signal": "light",
            "Ask SafeCross": "chat", "Reports": "bars", "Settings": "gear"}
PAGES = ["Dashboard", "Live Monitoring", "Crossing Check", "Smart Signal", "Ask SafeCross", "Reports", "Settings"]
EXAMPLES = ["Why was the pedestrian phase extended?", "Why did the pedestrian phase end when it did?",
            "Would the phase have ended earlier without person P0?", "Did the camera miss anyone at 12 s?",
            "What does the Dubai Universal Design Code say about pedestrian crossings?"]


def build():
    with gr.Blocks(title="SafeCross") as demo:
        sim_bundle = gr.State(None)
        runs = gr.State([])
        live = gr.State(None)
        simrep = gr.State(None)
        check_sum = gr.State(None)
        chat_hist = gr.State([])

        with gr.Row(equal_height=False):
            # ------------------------------------------------------------ sidebar
            with gr.Column(scale=0, min_width=240, elem_id="sc-side"):
                gr.HTML(BRAND)
                nav = [gr.Button(p, variant="primary" if p == "Dashboard" else "secondary", elem_classes="sc-nav",
                                 icon=str(ICON_DIR / f"{NAV_ICON[p]}.svg")) for p in PAGES]
                gr.HTML(SIDE_FOOT)

            # ------------------------------------------------------------ main
            with gr.Column(scale=1, elem_id="sc-main"):
                top = gr.HTML(topbar_html())
                pages = {}

                # ======== Dashboard
                with gr.Column(visible=True) as pages["Dashboard"]:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=5, min_width=360, elem_classes="sc-card"):
                            live_head = gr.HTML(card_head("cam", "Live Crossing Feed"))
                            live_img = gr.Image(show_label=False, height=290, interactive=False, buttons=[], elem_id="sc-live")
                        with gr.Column(scale=3, min_width=280, elem_classes="sc-card"):
                            gr.HTML(card_head("light", "Signal Status", '<span class="sc-pill grey">Simulated</span>'))
                            sig_html = gr.HTML()
                        with gr.Column(scale=4, min_width=300):
                            metrics = gr.HTML(metric_cards_html([]))
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=4, min_width=320, elem_classes="sc-card"):
                            gr.HTML(card_head("bell", "Recent Alerts", '<a class="sc-link" data-go="signal">View All &rarr;</a>'))
                            alerts = gr.HTML()
                        with gr.Column(scale=5, min_width=360, elem_classes="sc-card"):
                            gr.HTML(card_head("trend", "Pedestrian &amp; Vehicle Trends"))
                            trends = gr.HTML(trends_html())
                        with gr.Column(scale=4, min_width=300, elem_classes="sc-card"):
                            gr.HTML(card_head("bolt", "Quick Actions"))
                            with gr.Row(elem_classes="sc-qa-row"):
                                qa_cc = gr.Button("Crossing Check", icon=str(ICON_DIR / "image.svg"), elem_classes="sc-qa", min_width=60)
                                qa_ask = gr.Button("Ask SafeCross", icon=str(ICON_DIR / "chat.svg"), elem_classes="sc-qa", min_width=60)
                                qa_rep = gr.Button("Reports", icon=str(ICON_DIR / "report.svg"), elem_classes="sc-qa", min_width=60)
                    # hidden targets for the "View details" links inside the alert cards
                    go_signal = gr.Button("signal", elem_id="sc-go-signal", elem_classes="sc-hidden")
                    go_check = gr.Button("check", elem_id="sc-go-check", elem_classes="sc-hidden")

                # ======== Live Monitoring
                with gr.Column(visible=False) as pages["Live Monitoring"]:
                    gr.Markdown("## Live Monitoring\nThe dashboard replays either the **simulation** or **your own "
                                "video**. For a video, people are detected with YOLO and followed with ByteTrack "
                                "(each person keeps the same #id).")
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes="sc-card"):
                            lv_video = gr.Video(label="Crossing video (mp4 / mov)")
                            lv_model = gr.Dropdown(model_choices(), value=model_choices()[0], label="Detector")
                            lv_secs = gr.Slider(5, 60, value=20, step=5, label="Seconds to process")
                            lv_run = gr.Button("Track people in this video", variant="primary")
                            lv_sim = gr.Button("Switch back to the simulation replay", variant="secondary")
                            lv_msg = gr.Markdown()
                        with gr.Column(scale=1, elem_classes="sc-card"):
                            lv_table = gr.Dataframe(value=pd.DataFrame({"info": ["Upload a video and press Track"]}), label="Tracked people", interactive=False)

                # ======== Crossing Check
                with gr.Column(visible=False) as pages["Crossing Check"]:
                    gr.Markdown("## Crossing Check\nRun a detector on a crossing image, then make the image darker, "
                                "blurrier or partly covered, and see who the camera misses. Upload the image's YOLO "
                                "label file (from the PMMA test set) to compare with the real people.")
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes="sc-card"):
                            cc_img = gr.Image(label="Crossing image", type="numpy", value=_example_pair()[0])
                            cc_lab = gr.File(label="Optional: YOLO label file (.txt)", file_types=[".txt"], value=_example_pair()[1])
                            cc_model = gr.Dropdown(model_choices(), value=model_choices()[0], label="Detector")
                            cc_kind = gr.Radio(["clean", "darkness", "blur", "occlusion"], value="darkness", label="Condition")
                            cc_sev = gr.Slider(1, 3, value=2, step=1, label="Severity")
                            cc_run = gr.Button("Run check", variant="primary")
                        with gr.Column(scale=2, elem_classes="sc-card"):
                            with gr.Row():
                                cc_out1 = gr.Image(label="Clean image + detections")
                                cc_out2 = gr.Image(label="Degraded image + detections")
                            cc_md = gr.Markdown()
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes="sc-card"):
                            gr.Markdown("### Accessibility barrier scan\nLooks for possible barriers (steps, bollards, "
                                        "poles, parked cars, construction barriers, potholes) and curb ramps in the "
                                        "crossing image above, using YOLO-World text prompts (zero-shot). Then shows "
                                        "the matching passages from the Dubai Universal Design Code.")
                            br_run = gr.Button("Scan the image for barriers", variant="primary")
                        with gr.Column(scale=2, elem_classes="sc-card"):
                            with gr.Row():
                                br_img = gr.Image(label="Possible barriers (zero-shot)")
                                br_md = gr.Markdown()

                # ======== Smart Signal
                with gr.Column(visible=False) as pages["Smart Signal"]:
                    gr.Markdown("## Smart Signal\nOne simulated walk phase. Real people cross; the camera output is "
                                "damaged on purpose; each controller decides from the camera output alone.")
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes="sc-card"):
                            s_seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Scenario number")
                            s_n = gr.Slider(1, 8, value=4, step=1, label="People in this phase")
                            s_aid = gr.Slider(0, 1, value=0.5, step=0.05, label="Share using mobility aids")
                            s_pmma = gr.Checkbox(label="Use real PMMA walking speeds (after step 8)", value=False)
                            s_rec = gr.Slider(0.5, 1.0, value=0.85, step=0.05, label="Detection recall (1 = never misses)")
                            s_mis = gr.Slider(0, 0.5, value=0.1, step=0.05, label="Misclassification probability")
                            s_tl = gr.Slider(0, 1, value=0.3, step=0.05, label="Track-loss probability")
                            s_fp = gr.Slider(0, 0.5, value=0.05, step=0.05, label="False positives per second")
                            s_run = gr.Button("Run simulation", variant="primary")
                        with gr.Column(scale=2, elem_classes="sc-card"):
                            s_head = gr.Markdown()
                            with gr.Row():
                                s_g1 = gr.Image(label="Baseline controller")
                                s_g2 = gr.Image(label="Uncertainty-aware controller")
                            gr.Markdown("**Timeline:** bars = people on the crossing; lines = where each controller "
                                        "ended the walk phase; x = person cut off.")
                            s_tim = gr.Image(show_label=False)
                            s_tab = gr.Dataframe(label="Outcome", interactive=False)
                            s_people = gr.Dataframe(label="People (ground truth, hidden from the controllers)", interactive=False)
                            with gr.Row():
                                s_log1 = gr.Code(label="Baseline event log", language=None)
                                s_log2 = gr.Code(label="Uncertainty-aware event log", language=None)
                    with gr.Column(elem_classes="sc-card"):
                        gr.Markdown("### Inside the junction (simulated)\nThe same walk phase placed in a 4-way junction "
                                    "(`configs/junction.yaml`): green = walk, grey = parallel-road green continues, red = "
                                    "all-red clearance, blue = cross-road cars go (dashed line = first car reaches the "
                                    "crossing). ▼ = a turning car drives over the crossing while the turn arrow is gated by "
                                    "the controller; red ▼ = it meets a mobility-aid user.")
                        s_jimg = gr.Image(show_label=False)
                        s_jtab = gr.Dataframe(label="Junction outcome for this walk phase", interactive=False)

                # ======== Ask SafeCross
                with gr.Column(visible=False) as pages["Ask SafeCross"]:
                    gr.Markdown("## Ask SafeCross\nQuestions are about the last simulation. Rule questions use the "
                                "Dubai Universal Design Code index. Change the answer source in **Settings**.")
                    with gr.Row():
                        with gr.Column(scale=2, elem_classes="sc-card"):
                            achat = gr.Chatbot(height=430, show_label=False)
                            amsg = gr.Textbox(placeholder="e.g. Why was the pedestrian phase extended?", show_label=False)
                            with gr.Row():
                                asend = gr.Button("Ask", variant="primary")
                                aclear = gr.Button("Clear chat", variant="secondary")
                            gr.Examples(EXAMPLES, inputs=amsg)
                        with gr.Column(scale=1, elem_classes="sc-card"):
                            a_ctrl = gr.Radio(["uncertainty_aware", "baseline"], value="uncertainty_aware",
                                              label="Explain which controller?")
                            a_ev = gr.Markdown()
                    with gr.Column(elem_classes="sc-card"):
                        a_kg = gr.Image(label="Decision knowledge graph of the last simulation (gold = facts the answer cited)",
                                        interactive=False, show_label=True)

                # ======== Reports
                with gr.Column(visible=False) as pages["Reports"]:
                    gr.Markdown("## Reports\nResults appear here after you run the experiment steps in the guide.")
                    with gr.Column(elem_classes="sc-card"):
                        r_a = gr.Dataframe(label="Experiment A · mobility-aid users found as a person (IoU ≥ 0.5, 95% block-bootstrap CI); last column = also given the right group label", interactive=False)
                        r_gal = gr.Gallery(label="Experiment A · example images", columns=4, height=260)
                    with gr.Column(elem_classes="sc-card"):
                        r_bimg = gr.Image(label="Experiment B · cut-off rate vs failure level (simulated)")
                        r_b = gr.Dataframe(label="Experiment B · cut-off rate by controller", interactive=False)
                    with gr.Column(elem_classes="sc-card"):
                        r_jimg = gr.Image(label="Junction study · conflicts with cars (simulated)")
                        r_j = gr.Dataframe(label="Junction study · by detection recall", interactive=False)
                    with gr.Column(elem_classes="sc-card"):
                        r_e = gr.Dataframe(label="Explanations · scores by question type and input condition", interactive=False)
                        r_files = gr.File(label="Download result tables", file_count="multiple")
                    r_refresh = gr.Button("Refresh reports", variant="secondary")

                # ======== Settings
                with gr.Column(visible=False) as pages["Settings"]:
                    gr.Markdown("## Settings")
                    with gr.Row():
                        with gr.Column(elem_classes="sc-card"):
                            gr.Markdown("**Ask SafeCross answers with**")
                            set_backend = gr.Radio(["Offline (rules, no AI)", "AI model (Ollama)"],
                                                   value="Offline (rules, no AI)", show_label=False)
                            set_model = gr.Textbox(value=CFG["explain"]["ollama_model"], label="Ollama model name")
                            set_img = gr.Checkbox(value=True, label="Give the model the picture")
                            set_log = gr.Checkbox(value=True, label="Give the model the evidence (event log or knowledge graph)")
                            set_src = gr.Radio(["Knowledge graph", "Event log (text)"], value="Knowledge graph",
                                               label="Explain decisions from")
                        with gr.Column(elem_classes="sc-card"):
                            gr.Markdown("**Dashboard**")
                            set_loc = gr.Textbox(value=DCFG["location_label"], label="Location label on the feed and alerts")
                            set_cycle = gr.Number(value=DCFG["cycle_time_s"], precision=0, label="Signal cycle time (s)")
                            set_dash = gr.Button("Apply", variant="primary")
                            set_dash_msg = gr.Markdown()
                        with gr.Column(elem_classes="sc-card"):
                            gr.Markdown("**Dubai Universal Design Code**")
                            set_pdf = gr.File(label="Upload the PDF", file_types=[".pdf"], file_count="multiple")
                            set_pdf_btn = gr.Button("Build index", variant="primary")
                            set_pdf_msg = gr.Markdown()
                    with gr.Column(elem_classes="sc-card"):
                        gr.Markdown("**Project status**")
                        set_status = gr.HTML(status_html())
                        set_refresh = gr.Button("Refresh status", variant="secondary")

        # ------------------------------------------------------------ floating assistant (launcher + compact chat)
        fab = gr.Button("AI Assistant\nAsk SafeCross", icon=str(ICON_DIR / "bot.svg"), elem_id="sc-fab")
        with gr.Column(visible=False, elem_id="sc-pop") as pop:
            with gr.Row(equal_height=True):
                gr.HTML('<div class="sc-pop-h"><i><img src="/gradio_api/file=' + str(ICON_DIR / "bot.svg") +
                        '" width="20"/></i>Ask SafeCross</div>')
                pop_x = gr.Button("✕", elem_id="sc-pop-x", scale=0, min_width=36, size="sm")
            pchat = gr.Chatbot(height=300, show_label=False)
            with gr.Row():
                pmsg = gr.Textbox(placeholder="Ask about the last signal decision…", show_label=False, scale=6)
                psend = gr.Button("Send", variant="primary", scale=1, min_width=70)
            gr.HTML('<div class="sc-small">Answer source is set in Settings · the full chat is on the Ask SafeCross page</div>')
        barrier_sum = gr.State(None)
        timer = gr.Timer(0.5, active=True)

        # ------------------------------------------------------------ wiring
        page_cols = [pages[p] for p in PAGES]

        def go(name):
            return [gr.Column(visible=(p == name)) for p in PAGES] + \
                   [gr.Button(variant="primary" if p == name else "secondary") for p in PAGES]
        for p, b in zip(PAGES, nav):
            b.click(lambda p=p: go(p), None, page_cols + nav)
        for btn, target in [(qa_cc, "Crossing Check"), (qa_ask, "Ask SafeCross"), (qa_rep, "Reports"),
                            (go_signal, "Smart Signal"), (go_check, "Crossing Check")]:
            btn.click(lambda t=target: go(t), None, page_cols + nav)
        fab.click(lambda: (gr.Column(visible=True), gr.Button(visible=False)), None, [pop, fab])
        pop_x.click(lambda: (gr.Column(visible=False), gr.Button(visible=True)), None, [pop, fab])

        sig_inputs = [s_seed, s_n, s_aid, s_rec, s_mis, s_tl, s_fp, s_pmma, runs]
        sig_outputs = [s_g1, s_g2, s_tim, s_tab, s_people, s_log1, s_log2, s_head, sim_bundle, runs, live, simrep, metrics, trends, s_jimg, s_jtab]
        s_run.click(run_signal, sig_inputs, sig_outputs).then(alerts_html, [check_sum, sim_bundle, barrier_sum], alerts) \
            .then(lambda b, c: show_kg(b, c), [sim_bundle, a_ctrl], a_kg)

        def _check(img, lab, model, kind, sev):
            a, b_, md, summ = crossing_check(img, lab, model, kind, sev)
            return a, b_, md, summ
        cc_outputs = [cc_out1, cc_out2, cc_md, check_sum]
        cc_run.click(_check, [cc_img, cc_lab, cc_model, cc_kind, cc_sev], cc_outputs).then(
            alerts_html, [check_sum, sim_bundle, barrier_sum], alerts)

        def _startup(*args):
            out = list(run_signal(*args))
            try:
                ex_img, ex_lab = _example_pair()
                c = _check(ex_img, ex_lab, model_choices()[0], "darkness", 2)
            except Exception as e:
                c = [None, None, f"Crossing check not available yet: {e}", None]
            return out + list(c)
        demo.load(_startup, sig_inputs, sig_outputs + cc_outputs).then(alerts_html, [check_sum, sim_bundle, barrier_sum], alerts)

        ask_outputs = [achat, pchat, chat_hist, a_ev, a_kg]

        def _ask(msg, hist, bundle, ctrl, be, mdl, im, lg, src):
            h, _, _, ev, g = ask(msg, hist, bundle, ctrl, be, mdl, im, lg, src)
            return h, h, h, ev, g, ""
        ask_inputs_p = [pmsg, chat_hist, sim_bundle, gr.State("uncertainty_aware"), set_backend, set_model, set_img, set_log, set_src]
        psend.click(_ask, ask_inputs_p, ask_outputs + [pmsg])
        pmsg.submit(_ask, ask_inputs_p, ask_outputs + [pmsg])
        ask_inputs_a = [amsg, chat_hist, sim_bundle, a_ctrl, set_backend, set_model, set_img, set_log, set_src]
        asend.click(_ask, ask_inputs_a, ask_outputs + [amsg])
        amsg.submit(_ask, ask_inputs_a, ask_outputs + [amsg])
        aclear.click(lambda: ([], [], [], ""), None, [achat, pchat, chat_hist, a_ev])
        a_ctrl.change(lambda b, c: show_kg(b, c), [sim_bundle, a_ctrl], a_kg)

        timer.tick(tick, [live, simrep], [live, simrep, live_img, live_head, sig_html, top])

        lv_run.click(process_video, [lv_video, lv_model, lv_secs], [live, lv_table, lv_msg]).then(
            lambda l: trends_html(video=l), [live], trends)
        lv_sim.click(lambda s: (dict(s) if s else None, "Dashboard replays the simulation again."), [simrep], [live, lv_msg]).then(
            lambda b: trends_html(b), [sim_bundle], trends)

        def _reports():
            a, bimg, b, e, gal, files = results_views()
            jimg, jdf = junction_report()
            return a, gal, bimg, b, e, files, jimg, jdf
        r_outs = [r_a, r_gal, r_bimg, r_b, r_e, r_files, r_jimg, r_j]
        r_refresh.click(_reports, None, r_outs)
        demo.load(_reports, None, r_outs)

        set_pdf_btn.click(upload_code_pdf, [set_pdf], [set_pdf_msg, set_status])

        br_run.click(barrier_scan, [cc_img], [br_img, br_md, barrier_sum]).then(
            alerts_html, [check_sum, sim_bundle, barrier_sum], alerts)

        def _apply_dash(loc, cycle, bundle, history, check, barrier):
            DCFG["location_label"] = (loc or "Sample crossing").strip()
            DCFG["cycle_time_s"] = max(30, int(cycle or 90))
            return (f"Saved for this session (location: {DCFG['location_label']}, cycle {DCFG['cycle_time_s']} s). "
                    f"To keep it, change `dashboard:` in configs/config.yaml.",
                    metric_cards_html(history), trends_html(bundle), alerts_html(check, bundle, barrier))
        set_dash.click(_apply_dash, [set_loc, set_cycle, sim_bundle, runs, check_sum, barrier_sum],
                       [set_dash_msg, metrics, trends, alerts])

        def _auto_video():
            """If a demo video exists (scripts/10), track it at start-up so the feed shows real footage."""
            p = resolve(DCFG["demo_video"])
            if not p.exists():
                return gr.update(), gr.update()
            try:
                replay, _, _ = process_video(str(p), model_choices()[0], DCFG["demo_video_seconds"])
                return replay, trends_html(video=replay)
            except Exception as e:
                print("Demo video not loaded:", e)
                return gr.update(), gr.update()
        demo.load(_auto_video, None, [live, trends])

        def _refresh_status():
            ch = model_choices()
            return status_html(), gr.Dropdown(choices=ch), gr.Dropdown(choices=ch)
        set_refresh.click(_refresh_status, None, [set_status, cc_model, lv_model])
    return demo


if __name__ == "__main__":
    build().launch(theme=THEME, css=CSS, head=HEAD, allowed_paths=[str(ICON_DIR)], server_name=os.environ.get("SAFECROSS_HOST", "127.0.0.1"),
                   server_port=int(os.environ.get("SAFECROSS_PORT", "7860")))
