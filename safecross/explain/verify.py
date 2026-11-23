"""Citation checker for knowledge-graph answers (follow-up, NLP-RQ1).

A model answer is KEPT only if the facts it cites exist and actually support it. Otherwise the answer
is replaced by "cannot_determine" (the system declines instead of showing an unsupported explanation).
No model is called: the check is symbolic, so it cannot itself hallucinate.

Rules
  * every cited id must be one of the facts that were given;  at least one fact must be cited
  * decision      : a cited fact must contain the given reason code
  * counterfactual: a cited fact must be about the track in the question;
                    "yes" needs a cited  (track N) --only_occupant--> fact,
                    "no"  is rejected if the given facts contain such an only_occupant fact for that track
  * "cannot_determine" always passes (declining is never unsupported)
"""
from __future__ import annotations

import re

from .backends import parse_json


def _ids(lines: list[str]) -> dict[str, str]:
    return {m.group(1): l for l in lines if (m := re.match(r"\[(\w+)\]", l))}


def verify(item: dict, raw: str, facts: list[str]) -> tuple[str, bool, str]:
    """Return (raw answer to score, kept?, reason). If not kept, the returned raw is an abstention."""
    ans = parse_json(raw)
    key = "reason_code" if item["qtype"] == "decision" else "answer"
    given = str(ans.get(key, "")).strip().lower().replace(" ", "_")
    if given in ("cannot_determine", "unknown"):
        return raw, True, "abstained"
    line_of = _ids(facts)
    cited = ans.get("evidence") if isinstance(ans.get("evidence"), list) else []
    cited = [re.sub(r"[\[\]\s]", "", str(c)).upper() for c in cited]

    def reject(why):
        return ('{"%s": "cannot_determine", "explanation": "withheld by the citation checker: %s", "evidence": []}'
                % (key, why)), False, why

    if not given:
        return raw, True, "unparsed (scored as wrong, not changed)"
    if not cited:
        return reject("no fact cited")
    missing = [c for c in cited if c not in line_of]
    if missing:
        return reject(f"cited facts that do not exist: {missing}")
    text = [line_of[c].lower() for c in cited]
    if item["qtype"] == "decision":
        if not any(given in t for t in text):
            return reject(f"no cited fact contains '{given}'")
        return raw, True, "supported"
    m = re.search(r"track\s+(\d+)", item["question"].lower())
    tid = m.group(1) if m else None
    about = [t for t in text if tid and f"(track {tid})" in t]
    if not about:
        return reject(f"no cited fact is about track {tid}")
    only_given = any(f"(track {tid}) --only_occupant-->" in l.lower() for l in facts)
    if given == "yes" and not any("--only_occupant-->" in t for t in about):
        return reject("'yes' without a cited only_occupant fact for that track")
    if given == "no" and only_given:
        return reject("'no' contradicts an only_occupant fact for that track")
    return raw, True, "supported"
