"""Incident reports + sentence-level fact-checking (MiniCheck) + a planted-error test.

Investigators need short written reports, not only one-line JSON answers. Free text is harder to check, so
every sentence is checked against the walk phase's knowledge-graph facts (the grounding document):

  * MiniCheck (Bespoke-MiniCheck-7B, run in Ollama): a small model trained to decide whether a claim is
    supported by a document. Prompt "Document: ...  Claim: ..." -> "Yes" / "No".
  * a rule baseline ("numbers check"): every number in the sentence must appear in the facts.

Planted-error test: TRUE sentences are written from the graph by templates (true by construction); each is
then deliberately corrupted in one way (wrong time, wrong track, wrong reason, wrong outcome, invented rule).
A good checker accepts the true sentences and rejects the corrupted ones.
"""
from __future__ import annotations

import re

from .evidence import REASON_CODES


def fact_document(facts) -> str:
    """Facts as plain sentences (an NLI-style checker reads prose better than id/arrow notation)."""
    return "\n".join(f"{f.s} {f.p.replace('_', ' ')} {f.o}." for f in facts)


def _o(facts, s, p):
    return next((f.o for f in facts if f.s == s and f.p == p), None)


def true_sentences(facts, max_tracks: int = 2) -> list[dict]:
    """Template sentences that are true by construction (each built from one or two facts)."""
    out = []
    ended = next((f.o for f in facts if f.p == "ended_at"), None)
    if ended:
        t, reason = re.match(r"([\d.]+) s because (\w+)", ended).groups()
        out.append({"kind": "end", "text": f"The walk phase ended at {t} s because {REASON_CODES[reason]}.",
                    "t": t, "reason": reason})
    ext = sorted({f.s for f in facts if re.match(r"L\d+ EXTEND", f.s)}, key=lambda s: int(s.split()[0][1:]))
    if ext:
        t = _o(facts, ext[0], "at_time").split()[0]
        reason = _o(facts, ext[0], "reason").split()[0]
        out.append({"kind": "extend", "text": f"The walk phase was extended at {t} s because {REASON_CODES[reason]}.",
                    "t": t, "reason": reason})
    tracks = sorted({f.s for f in facts if re.match(r"track \d+$", f.s)}, key=lambda s: int(s.split()[1]))
    lost = [t for t in tracks if _o(facts, t, "lost_by_camera")][:max_tracks]
    for tr in lost:
        seen = _o(facts, tr, "last_seen_at")
        t, y = re.match(r"([\d.]+) s at y=([\d.]+) m", seen).groups()
        label = _o(facts, tr, "labelled_as").split(" (")[0]
        out.append({"kind": "lost", "text": f"{tr.capitalize()}, labelled as a {label}, was last seen at {t} s, "
                                            f"{y} m into the crossing, and did not reach the exit.", "t": t, "track": tr})
    only = [t for t in tracks if _o(facts, t, "only_occupant")][:max_tracks]
    for tr in only:
        a, b = re.match(r"([\d.]+)-([\d.]+) s", _o(facts, tr, "only_occupant")).groups()
        out.append({"kind": "only", "text": f"Between {a} and {b} s, {tr} was the only track keeping the crossing "
                                            f"occupied.", "t": a, "track": tr})
    exited = [t for t in tracks if _o(facts, t, "reached_exit_at")][:1]
    for tr in exited:
        t = _o(facts, tr, "reached_exit_at").split()[0]
        out.append({"kind": "exit", "text": f"{tr.capitalize()} reached the exit at {t} s.", "t": t, "track": tr})
    return out


def corruptions(sent: dict, all_tracks: list[str]) -> list[tuple[str, str]]:
    """One sentence -> [(error_type, false sentence)]."""
    out, txt = [], sent["text"]
    if sent.get("t"):
        t2 = f"{float(sent['t']) + 3.0:.1f}"
        out.append(("wrong_time", txt.replace(f"{sent['t']} s", f"{t2} s", 1)))
    if sent.get("track"):
        n = int(sent["track"].split()[1])
        others = [t for t in all_tracks if t != sent["track"]]
        new = others[0] if others else f"track {n + 7}"
        out.append(("wrong_track", txt.replace(sent["track"], new).replace(sent["track"].capitalize(), new.capitalize())))
    if sent.get("reason"):
        alt = next(r for r in ("max_phase_reached", "crossing_clear", "mobility_aid_on_crossing")
                   if r != sent["reason"])
        out.append(("wrong_reason", txt.replace(REASON_CODES[sent["reason"]], REASON_CODES[alt])))
    if sent["kind"] == "lost":
        out.append(("wrong_outcome", txt.replace("and did not reach the exit", "and then reached the exit")))
    if sent["kind"] == "only":
        out.append(("wrong_outcome", txt.replace("was the only track", "was one of several tracks")))
    out.append(("invented_rule", txt[:-1] + ", as required by the Dubai Universal Design guide."))
    return [(k, s) for k, s in out if s != txt]


def numbers_check(sentence: str, document: str) -> bool:
    """Rule baseline: every number in the sentence appears in the document."""
    doc_nums = set(re.findall(r"\d+(?:\.\d+)?", document))
    return all(n in doc_nums for n in re.findall(r"\d+(?:\.\d+)?", sentence))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s*\[(?:G\d+(?:,\s*)?)+\]", "", text)          # citations out before checking
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", " ".join(text.split()))
    return [p.strip() for p in parts if len(p.strip()) > 12]


class MiniCheck:
    """Bespoke-MiniCheck-7B through Ollama:  ollama pull bespoke-minicheck"""

    def __init__(self, url: str = "http://localhost:11434", model: str = "bespoke-minicheck"):
        self.url, self.model = url.rstrip("/"), model

    def supported(self, document: str, claim: str) -> bool:
        import requests
        r = requests.post(f"{self.url}/api/generate", timeout=300, json={
            "model": self.model, "prompt": f"Document: {document}\nClaim: {claim}", "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2}})
        r.raise_for_status()
        return r.json()["response"].strip().lower().startswith("yes")
