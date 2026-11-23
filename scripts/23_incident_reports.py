"""STEP 23 (NLP) - incident reports, sentence-level fact-checking with MiniCheck, and a planted-error test.

    ollama pull bespoke-minicheck                                  # once (about 4.7 GB)
    python scripts/23_incident_reports.py                          # planted-error test + reports by Qwen-7B
    python scripts/23_incident_reports.py --generator none         # planted-error test only

Part 1 (planted errors): true template sentences from the knowledge graph + one corrupted copy per error type.
   Checkers: MiniCheck and a rule baseline (all numbers must appear in the facts).
   Scores: share of TRUE sentences accepted, share of FALSE sentences caught, per error type.
Part 2 (generated reports): a language model writes a 3-5 sentence incident report from the facts; every
   sentence is checked by MiniCheck. Score: share of sentences supported, share of fully supported reports.
Results: results/explanations/reports/
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.controller import run_controller
from safecross.explain import reports as RP
from safecross.explain.backends import make_backend
from safecross.explain.kg import build_kg, lines
from safecross.perception import perceive
from safecross.sim import generate_episode

REPORT_SYSTEM = ("You write short incident reports about a pedestrian traffic signal. Use ONLY the facts given. "
                 "Never invent times, tracks, people or rules. Cite the fact ids you used after each sentence, "
                 "for example [G12].")
REPORT_USER = ("FACTS:\n{facts}\n\nWrite a 3 to 5 sentence incident report of this walk phase for an investigator: "
               "when and why the walk phase was extended or ended, and any person the camera lost on the crossing.")


def phases(cfg, n, seed):
    rng = np.random.default_rng(seed)
    fail = dict(cfg["failures"])
    fail.update({"recall": 0.85, "track_loss_prob": 0.4, "misclassification_prob": 0.1, "false_positive_rate": 0.02})
    for e in range(n):
        ep = generate_episode(97000 + e, cfg, rng)
        obs = perceive(ep, fail, np.random.default_rng(int(rng.integers(1e9))))
        for c in ("baseline", "uncertainty_aware"):
            yield f"{97000 + e}_{c}", build_kg(ep, run_controller(c, cfg, ep, obs), cfg)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=777003)
    ap.add_argument("--generator", default="ollama:qwen2.5vl:7b",
                    help="ollama:<model>, mlx-base, or none (skip part 2)")
    ap.add_argument("--checker-model", default="bespoke-minicheck")
    args = ap.parse_args()
    out = resolve(cfg["paths"]["results_dir"]) / "explanations" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    mc = RP.MiniCheck(cfg["explain"]["ollama_url"], args.checker_model)
    try:
        mc.supported("The light is red.", "The light is red.")
    except Exception as e:
        sys.exit(f"MiniCheck is not reachable in Ollama ({e}). Open Ollama and run: ollama pull {args.checker_model}")

    data = list(phases(cfg, args.episodes, args.seed))
    # ---------------- part 1: planted errors
    rows, t0 = [], time.time()
    for k, (pid, facts) in enumerate(data, 1):
        doc = RP.fact_document(facts)
        tracks = sorted({f.s for f in facts if re.match(r"track \d+$", f.s)})
        for s in RP.true_sentences(facts):
            for etype, text in [("none (true)", s["text"])] + RP.corruptions(s, tracks):
                rows.append({"phase": pid, "kind": s["kind"], "error": etype, "is_true": etype == "none (true)",
                             "sentence": text, "minicheck_ok": mc.supported(doc, text),
                             "numbers_ok": RP.numbers_check(text, doc)})
        if k % 10 == 0:
            print(f"  planted-error test: {k}/{len(data)} walk phases, {len(rows)} sentences, "
                  f"{(time.time() - t0) / len(rows):.1f}s each")
    d = pd.DataFrame(rows)
    d.to_csv(out / "planted_errors.csv", index=False)
    print("\nPart 1 - planted-error test: share ACCEPTED as supported (true sentences should be near 1, "
          "false ones near 0)")
    tab = d.groupby("error")[["minicheck_ok", "numbers_ok"]].mean().round(3)
    tab["n"] = d.groupby("error").size()
    print(tab.to_string())
    true, false = d[d.is_true], d[~d.is_true]
    summ = []
    for ch in ("minicheck_ok", "numbers_ok"):
        tpr, tnr = true[ch].mean(), 1 - false[ch].mean()
        summ.append({"checker": ch.replace("_ok", ""), "true_sentences_accepted": tpr, "false_sentences_caught": tnr,
                     "balanced_accuracy": (tpr + tnr) / 2})
    summ = pd.DataFrame(summ)
    print("\n" + summ.round(3).to_string(index=False))
    summ.to_csv(out / "planted_errors_summary.csv", index=False)

    # ---------------- part 2: generated reports
    if args.generator == "none":
        return
    if args.generator.startswith("ollama:"):
        gen = make_backend(cfg, "ollama", args.generator.split(":", 1)[1])
    else:
        gen = make_backend(cfg, args.generator)
    rep_rows, sent_rows = [], []
    for k, (pid, facts) in enumerate(data, 1):
        doc = RP.fact_document(facts)
        text = gen.generate(REPORT_SYSTEM, REPORT_USER.format(facts="\n".join(lines(facts))))
        sents = RP.split_sentences(text)
        ok = [mc.supported(doc, s) for s in sents]
        ids = {f.gid for f in facts}
        cited = set(re.findall(r"G\d+", text))
        rep_rows.append({"phase": pid, "generator": gen.name, "report": text, "sentences": len(sents),
                         "supported": sum(ok), "all_supported": bool(sents) and all(ok),
                         "invented_citations": len(cited - ids)})
        sent_rows += [{"phase": pid, "sentence": s, "minicheck_ok": o} for s, o in zip(sents, ok)]
        if k % 10 == 0:
            print(f"  reports: {k}/{len(data)}")
    r = pd.DataFrame(rep_rows)
    tag = gen.name.replace(":", "_").replace("/", "_")
    r.to_csv(out / f"reports_{tag}.csv", index=False)
    pd.DataFrame(sent_rows).to_csv(out / f"report_sentences_{tag}.csv", index=False)
    with open(out / f"reports_{tag}.jsonl", "w") as f:
        for x in rep_rows:
            f.write(json.dumps(x) + "\n")
    print(f"\nPart 2 - {len(r)} reports by {gen.name}: {r.sentences.sum()} sentences, "
          f"{r.supported.sum() / max(r.sentences.sum(), 1):.1%} supported by MiniCheck; "
          f"{r.all_supported.mean():.1%} of reports fully supported; "
          f"{(r.invented_citations > 0).mean():.1%} of reports cite a fact id that does not exist")
    print(f"Example:\n{r.report.iloc[0]}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
