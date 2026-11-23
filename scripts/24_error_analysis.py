"""STEP 24 (NLP) - error analysis and risk-coverage, from the answers already saved by scripts 16 and 22.
No model is run: this only reads results/explanations/answers_kg_*.jsonl and the question files.

  Error analysis: every answer sorted into one type -
     decision questions:  correct | wrong_reason | false_decline (declined although the facts were there) | unreadable
     what-if questions:   correct | false_yes | false_no | false_decline | unreadable
     facts removed:       declined (correct) | guessed (answered although the facts were missing) | unreadable
  Risk-coverage: each model, with and without the citation checker, on a pool of answerable questions (subgraph)
     plus the same questions with the key facts removed:
        coverage = share of ANSWERABLE questions it answered
        risk     = share of ALL its answers that are wrong (a wrong answer, or any answer when facts were missing)
     A safe system has low risk; a useful one has high coverage.

    python scripts/24_error_analysis.py
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOTDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOTDIR))
from safecross.config import load_config, resolve
from safecross.explain.backends import parse_json
from safecross.explain.verify import verify

_spec = importlib.util.spec_from_file_location("s16", ROOTDIR / "scripts" / "16_run_kg_explanations.py")
S16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S16)
TESTS = {"": ("main", "questions"), "_ood_wording": ("reworded", "questions_ood_wording"),
         "_ood_controller": ("unseen controller", "questions_ood_controller")}
NICE = {"mlx_Qwen2.5-1.5B-Instruct+lora": "Ours 1.5B (v1)", "mlx_Qwen2.5-1.5B-Instruct+lora_v2": "Ours 1.5B (v2)",
        "mlx_Qwen2.5-1.5B-Instruct+base": "Qwen-1.5B untrained", "ollama_qwen2.5vl_7b": "Qwen2.5-VL-7B",
        "ollama_llava_7b": "LLaVA-7B"}


def given_of(item, raw):
    a = parse_json(raw)
    key = "reason_code" if item["qtype"] == "decision" else "answer"
    return str(a.get(key, "")).strip().lower().replace(" ", "_") if "_unparsed" not in a else ""


def error_type(item, given, missing):
    if not given:
        return "unreadable"
    declined = given in ("cannot_determine", "unknown")
    if missing:
        return "declined (correct)" if declined else "guessed"
    if declined:
        return "false_decline"
    truth = str(item["truth"]["reason_code" if item["qtype"] == "decision" else "answer"]).lower()
    if given == truth:
        return "correct"
    if item["qtype"] == "decision":
        return "wrong_reason"
    return "false_yes" if given == "yes" else ("false_no" if given == "no" else "other_wrong")


def main():
    cfg = load_config()
    res = resolve(cfg["paths"]["results_dir"])
    exp = res / "explanations"
    out = exp / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(exp.glob("answers_kg_*.jsonl")):
        tag = f.stem.replace("answers_kg_", "")
        suffix = next((s for s in ("_ood_wording", "_ood_controller") if tag.endswith(s)), "")
        model = tag[: len(tag) - len(suffix)] if suffix else tag
        test, qdir = TESTS[suffix]
        qd = res / qdir
        if not (qd / "questions.json").exists():
            continue
        items = {it["qid"]: it for it in json.loads((qd / "questions.json").read_text())}
        kg = json.loads((qd / "kg.json").read_text())
        for line in f.read_text().splitlines():
            r = json.loads(line)
            it = items.get(r["qid"])
            if it is None:
                continue
            missing = r["condition"] == "kg_sub_missing"
            facts = S16.missing_evidence(it, kg[r["qid"]]["kg_sub"]) if missing else kg[r["qid"]][r["condition"]]
            g = given_of(it, r["raw"])
            row = {"model": NICE.get(model, model), "test": test, "qtype": it["qtype"], "qid": r["qid"],
                   "condition": r["condition"], "error": error_type(it, g, missing), "given": g}
            if r["condition"] in ("kg_sub", "kg_sub_missing"):
                raw2, kept, why = verify(it, r["raw"], facts)
                row["checker_kept"] = kept
                row["error_after_checker"] = error_type(it, given_of(it, raw2), missing)
            rows.append(row)
    if not rows:
        sys.exit("No answers_kg_*.jsonl found - run scripts/16_run_kg_explanations.py first.")
    d = pd.DataFrame(rows)
    d.to_csv(out / "all_answers_typed.csv", index=False)

    # ---------------- error analysis table
    tab = (d.groupby(["test", "model", "qtype", "condition"])["error"].value_counts(normalize=True)
           .unstack(fill_value=0).round(3))
    tab["n"] = d.groupby(["test", "model", "qtype", "condition"]).size()
    tab.to_csv(out / "error_types.csv")
    main_view = tab.loc["main"] if "main" in tab.index.get_level_values(0) else tab
    pd.set_option("display.width", 220)
    print("ERROR TYPES on the main test (share of answers; 1.0 = 100%):")
    print(main_view.to_string())

    # ---------------- risk-coverage
    rc = []
    for (test, model), g in d[d.condition.isin(["kg_sub", "kg_sub_missing"])].groupby(["test", "model"]):
        for checker in (False, True):
            col = "error_after_checker" if checker else "error"
            ans = g[g.condition == "kg_sub"]
            mis = g[g.condition == "kg_sub_missing"]
            answered_ok = ans[col].isin(["correct", "wrong_reason", "false_yes", "false_no", "other_wrong"])
            wrong = ans[col].isin(["wrong_reason", "false_yes", "false_no", "other_wrong"]).sum()
            guessed = (mis[col] == "guessed").sum()
            n_answers = answered_ok.sum() + guessed
            rc.append({"test": test, "model": model, "checker": checker, "coverage": answered_ok.mean(),
                       "risk": (wrong + guessed) / n_answers if n_answers else 0.0,
                       "wrong_answers": int(wrong), "guesses_without_facts": int(guessed), "n": len(ans)})
    rc = pd.DataFrame(rc)
    rc.to_csv(out / "risk_coverage.csv", index=False)
    print("\nRISK-COVERAGE (pool = answerable questions + the same questions with facts removed):")
    print(rc.round(3).to_string(index=False))

    # one fixed colour per model in every panel (same legend everywhere)
    palette = {"Ours 1.5B (v1)": "#2F8F4E", "Ours 1.5B (v2)": "#8FBF8F", "Qwen-1.5B untrained": "#7A7A7A",
               "Qwen2.5-VL-7B": "#8C1D2C", "LLaVA-7B": "#C9A227"}
    extra = ["#5B6C8F", "#B5555F", "#3A3A3A"]
    for m in sorted(rc.model.unique()):
        if m not in palette:
            palette[m] = extra[len([k for k in palette if k not in NICE.values()]) % len(extra)]
    tests = [t for t in ("main", "reworded", "unseen controller") if t in set(rc.test)]
    fig, axes = plt.subplots(1, len(tests), figsize=(5.4 * len(tests), 4.8), squeeze=False)
    for ax, test in zip(axes[0], tests):
        g = rc[rc.test == test]
        for model, m in g.groupby("model"):
            a, b = m[~m.checker].iloc[0], m[m.checker].iloc[0]
            c = palette[model]
            ax.annotate("", xy=(b.coverage, b.risk), xytext=(a.coverage, a.risk),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
            ax.scatter([a.coverage], [a.risk], color=c, s=70, zorder=3, edgecolor="white")
            ax.scatter([b.coverage], [b.risk], color=c, s=70, marker="s", zorder=3, edgecolor="white")
            dy = {"Ours 1.5B (v1)": 10, "Ours 1.5B (v2)": -16}.get(model, 6)          # keep overlapping labels apart
            ax.annotate(model, (a.coverage, a.risk), xytext=(-8, dy), textcoords="offset points",
                        ha="right", fontsize=7.5, color=c, fontweight="bold")
        ax.set_title(f"{test} test", fontsize=10)
        ax.set_xlabel("coverage: share of answerable questions answered")
        ax.set_ylabel("risk: share of answers that are wrong")
        ax.set_xlim(0.6, 1.04); ax.set_ylim(-0.03, max(0.65, rc.risk.max() + 0.05)); ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=m) for m, c in palette.items() if m in set(rc.model)]
    handles += [plt.Line2D([], [], marker="o", ls="", color="black", label="model alone"),
                plt.Line2D([], [], marker="s", ls="", color="black", label="with citation checker")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle("Risk vs coverage: answerable questions + the same questions with the key facts removed "
                 "(best = bottom right)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out / "risk_coverage.png", dpi=160)
    print(f"\nSaved to {out} (error_types.csv, risk_coverage.csv, risk_coverage.png)")


if __name__ == "__main__":
    main()
