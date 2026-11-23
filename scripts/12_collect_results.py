"""STEP 12 - collect every result into ONE report for the papers: results/RESULTS.md

    python scripts/12_collect_results.py

It only reads files that the earlier steps wrote; nothing is re-run. Sections that have no
results yet are listed as "not run yet". Numbers are copied exactly from the CSV files.
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.metrics import mcnemar


def table(df: pd.DataFrame, digits: int = 3) -> str:
    """Markdown table without extra packages."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def ci(v, lo, hi):
    if pd.isna(v):
        return ""
    return f"{v:.2f} [{lo:.2f}, {hi:.2f}]" if pd.notna(lo) and pd.notna(hi) else f"{v:.2f}"


def section_a(res: Path, out: list):
    f = res / "experiment_a" / "summary.csv"
    out.append("## Experiment A - does the camera find mobility-aid users? (MAAI901)\n")
    if not f.exists():
        out.append("_Not run yet (scripts/03)._\n")
        return
    d = pd.read_csv(f)
    models = sorted(d["model"].unique())
    seeds = [m for m in models if "_seed" in m]
    out.append(f"Models: {', '.join(models)}. Values are **found_as_person** (detected as any person, "
               f"IoU >= 0.5) with 95% block-bootstrap intervals; `recall` (right class too) is in the CSV.\n")

    # per class, clean
    cl = d[d["severity"] == 0]
    piv = cl.pivot_table(index="class", columns="model", values="found_as_person")
    if len(seeds) > 1:
        piv["fine-tuned mean"] = piv[seeds].mean(axis=1)
        piv["fine-tuned sd"] = piv[seeds].std(axis=1)
    out.append("### Clean images, per group\n")
    out.append(table(piv.reset_index()) + "\n")

    # degradations, all mobility aids
    aid = d[d["class"] == "ALL_MOBILITY_AIDS"].copy()
    aid["value"] = [ci(a, b, c) for a, b, c in zip(aid.found_as_person, aid.found_as_person_lo, aid.found_as_person_hi)]
    aid["condition"] = aid["condition"] + " " + aid["severity"].astype(str)
    piv2 = aid.pivot_table(index="condition", columns="model", values="value", aggfunc="first")
    out.append("### All mobility-aid users, clean and degraded\n")
    out.append(table(piv2.reset_index()) + "\n")
    if len(seeds) > 1:
        m = aid[aid.model.isin(seeds)].groupby("condition")["found_as_person"].agg(["mean", "std"]).reset_index()
        out.append(f"Fine-tuned model over {len(seeds)} training seeds (mean, sd):\n")
        out.append(table(m) + "\n")

    f3 = res / "experiment_a_iou03" / "summary.csv"
    if f3.exists():
        d3 = pd.read_csv(f3)
        p3 = d3[d3.severity == 0].pivot_table(index="class", columns="model", values="found_as_person")
        out.append("### Sensitivity check: looser box matching (IoU >= 0.3), clean images\n")
        out.append(table(p3.reset_index()) + "\n")


def section_b(res: Path, out: list):
    f = res / "experiment_b" / "summary.csv"
    out.append("## Experiment B - do camera failures cause unsafe signal decisions? (simulated)\n")
    if not f.exists():
        out.append("_Not run yet (scripts/04)._\n")
        return
    d = pd.read_csv(f)
    d["cut-off rate"] = [ci(a, b, c) for a, b, c in zip(d.cutoff_rate, d.ci_low, d.ci_high)]
    keep = d[["failure", "level", "controller", "cut-off rate", "mean_added_time_s", "false_extension_rate"]]
    out.append(f"{int(d.aid_crossings.iloc[0])} mobility-aid crossings per condition. Cut-off rate = share of "
               "mobility-aid users still on the crossing when the walk phase ended.\n")
    out.append(table(keep) + "\n")
    out.append("![Experiment B](experiment_b/cutoff_vs_level.png)\n")


def section_junction(res: Path, out: list):
    f = res / "junction" / "summary.csv"
    out.append("## Junction study - what the decision means for cars at the junction (simulated follow-up)\n")
    if not f.exists():
        out.append("_Not run yet (scripts/13)._\n")
        return
    d = pd.read_csv(f)
    out.append(f"{int(d.aid_crossings.iloc[0])} mobility-aid crossings per condition, two-stage junction from "
               "configs/junction.yaml. Rates are shares of mobility-aid users with 95% episode-bootstrap intervals. "
               "'gated' = the turn arrow is green only while the controller reports the crossing as clear.\n")
    for col in ("cutoff_rate", "cross_conflict_rate", "permissive_turn_conflict_rate", "gated_turn_conflict_rate"):
        d[col.replace("_rate", "")] = [ci(a, b, c) for a, b, c in zip(d[col], d[col + "_lo"], d[col + "_hi"])]
    keep = ["failure", "level", "controller", "cutoff", "cross_conflict", "permissive_turn_conflict",
            "gated_turn_conflict", "cross_extra_delay_s", "cross_green_lost_s", "longer_cycle_share",
            "permissive_turn_delay_s", "gated_turn_delay_s"]
    out.append(table(d[keep]) + "\n")
    out.append("![Junction safety](junction/junction_safety.png)\n\n![Junction cost](junction/junction_cost.png)\n")


def section_c(res: Path, cfg: dict, out: list):
    out.append("## Explanations - are the vision-language model's explanations faithful? (MAAI902)\n")
    files = sorted(glob.glob(str(res / "explanations" / "scores_*.csv")))
    qfile = res / "questions" / "questions.json"
    if not files or not qfile.exists():
        out.append("_Not run yet (scripts/06, 06b, 06c, 07)._\n")
        return
    items = {q["qid"]: q for q in json.loads(qfile.read_text())}
    bal = Counter((q["qtype"], q["truth"].get("answer", q["truth"].get("reason_code"))) for q in items.values())
    out.append("Question set (true answers): " + ", ".join(f"{k[0]} {k[1]}: {v}" for k, v in sorted(bal.items())) + "\n")
    from safecross.explain.backends import parse_json
    for f in files:
        tag = Path(f).stem.replace("scores_", "")
        df = pd.read_csv(f)
        df = df[df.qid.isin(items)]
        label = tag + (" (rule-based reference, NOT an AI model)" if "offline" in tag else "")
        out.append(f"### {label}\n")
        summ = (df.groupby(["qtype", "condition"])[["factual_correct", "controller_faithful", "visually_grounded",
                                                    "abstained", "hallucinated_citation"]]
                  .mean(numeric_only=False).astype(float).round(3).reset_index())
        out.append(table(summ) + "\n")
        lines = []
        for q in ("decision", "visual", "visual_real"):
            a = df[(df.qtype == q) & (df.condition == "image_log")].set_index("qid")["factual_correct"]
            b = df[(df.qtype == q) & (df.condition == "log_only")].set_index("qid")["factual_correct"]
            c = a.index.intersection(b.index)
            if len(c):
                t = mcnemar(list(a[c]), list(b[c]))
                lines.append(f"- {q}: picture+log {a[c].mean():.3f} vs log only {b[c].mean():.3f} "
                             f"(n={len(c)}, McNemar p={t['p_value']:.3f})")
        for cond in ("image_log", "log_only"):
            a = df[(df.qtype == "counterfactual_rules") & (df.condition == cond)].copy()
            b = df[(df.qtype == "counterfactual") & (df.condition == cond)].set_index("qid")["factual_correct"]
            if len(a):
                a["base"] = a["qid"].str.replace("_rules$", "", regex=True)
                a = a.set_index("base")["factual_correct"]
                c = a.index.intersection(b.index)
                t = mcnemar(list(a[c]), list(b[c]))
                lines.append(f"- what-if ({cond}), follow-up: with controller rules {a[c].mean():.3f} vs without "
                             f"{b[c].mean():.3f} (n={len(c)}, McNemar p={t['p_value']:.3f})")
        ans_file = res / "explanations" / f"answers_{tag}.jsonl"
        if ans_file.exists():
            given = {}
            for l in ans_file.read_text().splitlines():
                r = json.loads(l)
                if r["qid"] in items and r["condition"] == "image_log":
                    j = parse_json(r["raw"])
                    given.setdefault(items[r["qid"]]["qtype"], Counter())[str(j.get("answer", j.get("reason_code")))] += 1
            for q in ("counterfactual", "counterfactual_rules", "visual_real"):
                if q in given:
                    truth = Counter(i["truth"]["answer"] for i in items.values() if i["qtype"] == q)
                    lines.append(f"- {q} answers (picture+log): model {dict(given[q])} vs truth {dict(truth)}")
        out.append("\n".join(lines) + "\n")


def main():
    cfg = load_config()
    res = resolve(cfg["paths"]["results_dir"])
    out = ["# SafeCross - results summary\n",
           f"Dataset for vision: `{cfg['vision'].get('dataset_yaml')}` | image size {cfg['vision']['imgsz']} | "
           f"IoU match {cfg['vision']['iou_match']}\n",
           "Everything below is copied from the CSV files in `results/`. Simulated results are labelled as simulated.\n"]
    section_a(res, out)
    section_b(res, out)
    section_junction(res, out)
    section_c(res, cfg, out)
    out.append("## Notes for the papers\n"
               "- PMMA frames come from one outdoor parking-area video; training and test frames are separated in time "
               "(an optimistic, same-scene estimate). Cite Liu, Saunier & Bilodeau (2025), doi:10.5683/SP3/XJPQUG.\n"
               "- Confidence intervals in Experiment A resample 2-second blocks of frames, not single images.\n"
               "- Experiment B is a simulation with configurable assumptions (configs/config.yaml).\n"
               "- The junction study is a follow-up simulation (configs/junction.yaml): two signal stages, Poisson car "
               "arrivals, no real traffic data. Phantom detections in the simulation do not move, so the ghost filter "
               "removes them completely; real false positives can move, so its real benefit will be smaller.\n"
               "- The 'what if with rules' and balanced picture questions were added after the first explanation run; "
               "report them as follow-up analyses.\n")
    path = res / "RESULTS.md"
    path.write_text("\n".join(out))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
