"""Score model answers with three SEPARATE measures.

factual_correct        - the answer matches the ground truth (log event, controller replay or label).
controller_faithful    - decision/counterfactual questions: the answer matches what the controller
                         actually did AND every cited log line exists and supports it.
visually_grounded      - visual questions: the answer is correct when BOTH the picture and the
                         detection list are given; with only one of them the question cannot be
                         answered, so the only grounded answer is "cannot_determine".
Also recorded:
abstained              - the model said "cannot_determine" (a readable answer).
parsed                 - the answer was readable JSON with the requested key; unreadable answers count as
                         wrong, never as abstentions.
hallucinated_citation  - the model cited a log line that does not exist.
"""
from __future__ import annotations

import re

from .backends import parse_json


def _norm(v) -> str:
    return str(v).strip().lower().replace(" ", "_") if v is not None else ""


def score(item: dict, condition: str, raw: str, evidence_lines: list[str] | None = None) -> dict:
    """evidence_lines: for the knowledge-graph conditions (kg_*), the fact lines that were given ([G1] ...).
    For the original conditions it is None and the log lines [L1] ... are used, exactly as before."""
    ans = parse_json(raw)
    t = item["truth"]
    q = item["qtype"]
    out = {"qid": item["qid"], "qtype": q, "condition": condition, "parsed": "_unparsed" not in ans}

    key = "reason_code" if q == "decision" else "answer"
    given = _norm(ans.get(key))
    # An answer that could not be read (no JSON / missing key) is NOT an abstention: it is a format failure.
    out["parsed"] = out["parsed"] and given != ""
    out["abstained"] = out["parsed"] and given in ("cannot_determine", "unknown")
    out["factual_correct"] = out["parsed"] and given == _norm(t[key])

    # citations
    cited = [str(c) for c in ans.get("evidence", []) if isinstance(c, (str, int))] if isinstance(ans.get("evidence"), list) else []
    cited_ids = [re.sub(r"[\[\]\s]", "", c).upper() for c in cited]
    if evidence_lines is None:
        ids = {f"L{i}" for i in range(1, len(item["log"]) + 1)}
        out["hallucinated_citation"] = any(c.startswith("L") and c not in ids for c in cited_ids)
        line_of = {f"L{i}": l for i, l in enumerate(item["log"], 1)}
    else:
        line_of = {m.group(1): l for l in evidence_lines if (m := re.match(r"\[(\w+)\]", l))}
        out["hallucinated_citation"] = any(c[:1] in ("L", "G") and c not in line_of for c in cited_ids)

    if q in ("decision", "counterfactual", "counterfactual_rules"):
        has_log = condition in ("image_log", "log_only") or condition.startswith("kg")
        if has_log:
            supported = True
            if q == "decision" and cited_ids:
                supported = any(given and given in line_of.get(c, "").lower() for c in cited_ids)
            out["controller_faithful"] = out["factual_correct"] and supported and not out["hallucinated_citation"]
        else:
            # without the log the decision cannot be known; the faithful answer is to abstain
            out["controller_faithful"] = out["abstained"]
        out["visually_grounded"] = None
    else:  # visual: needs BOTH the picture (who is there) and the detections (who was seen)
        has_both = condition == "image_log"
        if has_both:
            n_ok = True
            if "people_on_crossing" in ans and ans["people_on_crossing"] not in (None, ""):
                try:
                    n_ok = int(ans["people_on_crossing"]) == int(t["people_on_crossing"])
                except (TypeError, ValueError):
                    n_ok = False
            out["visually_grounded"] = out["factual_correct"] and n_ok
        else:
            out["visually_grounded"] = out["abstained"]
        out["controller_faithful"] = None
    return out
