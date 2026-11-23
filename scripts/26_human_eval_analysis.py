"""STEP 26 (NLP) - analyse the blinded human ratings.

Input: results/human_eval/ratings.json (downloaded from the rating page's owner panel, or the answers raters
sent by message, one JSON object per rater in a list) + results/human_eval/key.json (from step 25).

Reports, per system (ours v1 vs Qwen2.5-VL-7B):
  * mean faithfulness and clarity (1-5), overall and for normal vs facts-removed questions
  * which answer raters trusted more (A/B/equal, un-blinded with the key)
  * Wilcoxon signed-rank test on paired faithfulness (same rater, same question)
  * agreement between raters: Krippendorff's alpha (interval) on faithfulness
  * how human faithfulness relates to automatic correctness

    python scripts/26_human_eval_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve


def krippendorff_alpha_interval(units: dict) -> float:
    """units: {unit_id: [ratings from different raters]} -> alpha (interval metric)."""
    vals = [v for r in units.values() if len(r) >= 2 for v in r]
    if len(vals) < 2:
        return float("nan")
    n = len(vals)
    do = sum(sum((a - b) ** 2 for i, a in enumerate(r) for j, b in enumerate(r) if i != j) / (len(r) - 1)
             for r in units.values() if len(r) >= 2) / n
    de = sum((a - b) ** 2 for i, a in enumerate(vals) for j, b in enumerate(vals) if i != j) / (n * (n - 1))
    return 1 - do / de if de else float("nan")


def main():
    cfg = load_config()
    d = resolve(cfg["paths"]["results_dir"]) / "human_eval"
    key = json.loads((d / "key.json").read_text())
    raw = json.loads((d / "ratings.json").read_text())
    raters = raw if isinstance(raw, list) else list(raw.values())
    rows, prefs = [], []
    for r_i, r in enumerate(raters):
        rid = r.get("rater_code") or f"R{r_i + 1}"
        team = r.get("team_member")
        for iid, a in (r.get("answers") or {}).items():
            k = key.get(iid)
            if not k:
                continue
            for side in ("A", "B"):
                if a.get(f"{side}_faithful") is None:
                    continue
                rows.append({"rater": rid, "team_member": team, "item": iid, "condition": k["condition"],
                             "qtype": k["qtype"], "system": k[side], "faithful": a[f"{side}_faithful"],
                             "clear": a.get(f"{side}_clear"), "auto_correct": k[f"{side}_correct"]})
            if a.get("trust"):
                t = a["trust"]
                prefs.append({"rater": rid, "item": iid, "condition": k["condition"],
                              "preferred": "equal" if t == "equal" else k[t]})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No ratings found in ratings.json")
    df.to_csv(d / "ratings_long.csv", index=False)
    print(f"{df.rater.nunique()} raters ({int(df.drop_duplicates('rater').team_member.fillna(False).sum())} team "
          f"members), {df.item.nunique()} questions, {len(df)} answer ratings\n")
    tab = df.groupby(["condition", "system"])[["faithful", "clear"]].mean().round(2)
    tab["n"] = df.groupby(["condition", "system"]).size()
    print("Mean ratings (1 = poor, 5 = excellent):")
    print(tab.to_string())
    print("\nOverall:")
    print(df.groupby("system")[["faithful", "clear"]].mean().round(2).to_string())
    p = pd.DataFrame(prefs)
    if len(p):
        print("\nWhich answer raters trusted more (share of judgements):")
        print(pd.crosstab(p.condition, p.preferred, normalize="index").round(2).to_string())
    wide = df.pivot_table(index=["rater", "item"], columns="system", values="faithful").dropna()
    if len(wide) >= 5 and wide.shape[1] == 2:
        from scipy.stats import wilcoxon
        diff = wide["ours_v1"] - wide["qwen7b"]
        stat = wilcoxon(wide["ours_v1"], wide["qwen7b"], zero_method="zsplit")
        print(f"\nPaired faithfulness, ours minus Qwen-7B: mean {diff.mean():+.2f} on {len(wide)} pairs; "
              f"Wilcoxon p = {stat.pvalue:.4f}")
    units = df.groupby(["item", "system"])["faithful"].apply(list).to_dict()
    alpha = krippendorff_alpha_interval(units)
    print(f"Agreement between raters on faithfulness: Krippendorff's alpha (interval) = {alpha:.2f} "
          f"(1 = perfect, 0 = chance; 0.67+ is usually called acceptable)")
    corr = df.groupby("auto_correct")["faithful"].mean().round(2).to_dict()
    print(f"Human faithfulness when the automatic score said wrong / right: {corr}")
    pd.DataFrame([{"alpha_faithful": alpha, "raters": df.rater.nunique()}]).to_csv(d / "agreement.csv", index=False)
    tab.to_csv(d / "summary.csv")
    print(f"\nSaved to {d}")


if __name__ == "__main__":
    main()
