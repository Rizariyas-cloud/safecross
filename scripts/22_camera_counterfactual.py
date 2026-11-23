"""STEP 22 (NLP) - "If the camera had not lost track N, would the walk phase have ended earlier, later or at
the same time?"  A question type NEVER seen in training, linked to the real tracking failures of script 20.

Truth = re-running the controller with the camera repaired for that person (safecross/explain/camera_cf.py).
Balanced: every 'earlier' / 'later' case plus the same number of 'same' cases per controller.

    python scripts/22_camera_counterfactual.py --backend mlx                              # our explainer (v1)
    python scripts/22_camera_counterfactual.py --backend mlx-base                         # same model, untrained
    python scripts/22_camera_counterfactual.py --backend ollama --model qwen2.5vl:7b
    python scripts/22_camera_counterfactual.py --backend ollama --model llava:7b
    python scripts/22_camera_counterfactual.py --summary                                  # table of all runs
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.controller import run_controller
from safecross.explain import camera_cf as C
from safecross.explain.backends import make_backend, parse_json
from safecross.explain.kg import build_kg, lines, subgraph
from safecross.explain.prompts import SYSTEM_KG, build_kg_prompt
from safecross.perception import perceive
from safecross.sim import generate_episode


def build_questions(cfg, n_episodes, seed):
    rng = np.random.default_rng(seed)
    fail = dict(cfg["failures"])
    fail.update({"recall": 0.85, "track_loss_prob": 0.5, "misclassification_prob": 0.1, "false_positive_rate": 0.0})
    items = []
    for e in range(n_episodes):
        ep = generate_episode(95000 + e, cfg, rng)
        obs = perceive(ep, fail, np.random.default_rng(int(rng.integers(1e9))))
        for c in ("baseline", "uncertainty_aware"):
            res = run_controller(c, cfg, ep, obs)
            facts = build_kg(ep, res, cfg)
            for tid in C.lost_tracks(res, ep):
                t = C.truth(cfg, c, ep, obs, res, tid)
                if t is None:
                    continue
                q = C.CAM_Q.format(n=tid, t=res.end_time)
                items.append({"qid": f"cam_{95000 + e}_{c}_t{tid}", "qtype": "camera_cf", "controller": c,
                              "question": q, "fields": C.FIELDS,
                              "truth": {"answer": t[0], "end_time_repaired": round(t[1], 1),
                                        "end_time": round(res.end_time, 1)},
                              "kg_sub": lines(subgraph(facts, q)), "kg_full": lines(facts)})
    keep = []
    for c in ("baseline", "uncertainty_aware"):                 # balance: all changes + as many 'same'
        mine = [it for it in items if it["controller"] == c]
        changed = [it for it in mine if it["truth"]["answer"] != "same"]
        same = [it for it in mine if it["truth"]["answer"] == "same"]
        pick = rng.permutation(len(same))[:len(changed)]
        keep += changed + [same[i] for i in sorted(pick)]
    return keep


def score(it, raw, facts):
    a = parse_json(raw)
    given = str(a.get("answer", "")).strip().lower().replace(" ", "_")
    ids = {l.split("]")[0][1:] for l in facts}
    cited = [str(c).strip("[] ").upper() for c in a.get("evidence", [])] if isinstance(a.get("evidence"), list) else []
    track = it["question"].split("track ")[1].split(",")[0]
    about = {l.split("]")[0][1:] for l in facts if f"(track {track})" in l}
    return {"qid": it["qid"], "controller": it["controller"], "truth": it["truth"]["answer"], "given": given,
            "parsed": given != "", "abstained": given in ("cannot_determine", "unknown"),
            "correct": given == it["truth"]["answer"],
            "hallucinated_citation": any(c not in ids for c in cited),
            "cites_the_track": any(c in about for c in cited)}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "mlx-base", "ollama", "offline"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=777002)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    out = resolve(cfg["paths"]["results_dir"]) / "explanations"
    out.mkdir(parents=True, exist_ok=True)
    qpath = resolve(cfg["paths"]["results_dir"]) / "questions_camera_cf" / "questions.json"
    if not qpath.exists():
        qpath.parent.mkdir(parents=True, exist_ok=True)
        items = build_questions(cfg, args.episodes, args.seed)
        qpath.write_text(json.dumps(items, indent=1, default=float))
    items = json.loads(qpath.read_text())
    counts = pd.Series([f"{it['controller']}:{it['truth']['answer']}" for it in items]).value_counts().to_dict()
    print(f"{len(items)} camera what-if questions {counts}")

    if args.summary or not args.backend:
        files = sorted(out.glob("scores_camera_cf_*.csv"))
        if not files:
            sys.exit("No runs yet - run with --backend first.")
        rows = []
        for f in files:
            d = pd.read_csv(f)
            for cond, g in d.groupby("condition"):
                ans = g[g.parsed & ~g.abstained]
                rows.append({"model": f.stem.replace("scores_camera_cf_", ""), "condition": cond,
                             "correct": g.correct.mean(), "abstained": g.abstained.mean(),
                             "accuracy_when_answered": ans.correct.mean() if len(ans) else float("nan"),
                             "cites_the_track": g.cites_the_track.mean(),
                             "hallucinated_citation": g.hallucinated_citation.mean(), "n": len(g)})
        s = pd.DataFrame(rows)
        print(s.round(3).to_string(index=False))
        s.to_csv(out / "summary_camera_cf.csv", index=False)
        return

    backend = make_backend(cfg, args.backend, args.model, args.adapter)
    tag = backend.name.replace(":", "_").replace("/", "_")
    ans_path = out / f"answers_camera_cf_{tag}.jsonl"
    done = {(json.loads(l)["qid"], json.loads(l)["condition"]) for l in ans_path.read_text().splitlines()} \
        if ans_path.exists() else set()
    with open(ans_path, "a") as f:
        for n, it in enumerate(items, 1):
            for cond in ("kg_sub", "kg_full"):
                if (it["qid"], cond) in done:
                    continue
                raw = backend.generate(SYSTEM_KG, build_kg_prompt(it["question"], it[cond], it["fields"]), None,
                                       json_mode=True)
                f.write(json.dumps({"qid": it["qid"], "condition": cond, "raw": raw}) + "\n")
                f.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(items)} questions")
    by = {it["qid"]: it for it in items}
    rows = []
    for l in ans_path.read_text().splitlines():
        r = json.loads(l)
        s = score(by[r["qid"]], r["raw"], by[r["qid"]][r["condition"]])
        s["condition"] = r["condition"]
        rows.append(s)
    d = pd.DataFrame(rows)
    d.to_csv(out / f"scores_camera_cf_{tag}.csv", index=False)
    print(f"\n{backend.name}: share correct / declined, by controller and condition")
    print(d.groupby(["condition", "controller"])[["correct", "abstained", "cites_the_track", "hallucinated_citation"]]
          .mean().round(3).to_string())
    print("\nWhat it answered vs the truth (kg_sub):")
    print(pd.crosstab(d[d.condition == "kg_sub"].truth, d[d.condition == "kg_sub"].given).to_string())


if __name__ == "__main__":
    main()
