"""Ask SafeCross: the chatbot used in the app.

It decides which evidence to give the model:
  * questions about rules / requirements / accessibility -> retrieved Dubai code passages
  * everything else -> the controller event log (+ detections + picture if chosen)
In offline mode (no AI model installed) it answers from the log with simple rules
and says clearly that no AI model was used.
"""
from __future__ import annotations

import re

from ..controller import run_controller
from ..perception import remove_person, remove_track
from .evidence import format_log, format_detections, REASON_CODES
from .prompts import SYSTEM, build_user_prompt

RULE_WORDS = ("rule", "regulation", "requirement", "code", "standard", "accessib", "law", "guideline", "dubai")


def is_regulation_question(q: str) -> bool:
    q = q.lower()
    return any(w in q for w in RULE_WORDS)


def _time_in(q: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*s", q)
    return float(m.group(1)) if m else None


def offline_answer(q: str, bundle: dict, retriever=None) -> tuple[str, list[str]]:
    """Rule-based answers straight from the evidence. Returns (answer, evidence lines)."""
    ql = q.lower()
    cname = bundle["controller"]
    res = bundle["results"][cname]
    ep, obs, cfg = bundle["episode"], bundle["observations"], bundle["cfg"]
    log = format_log(res)
    note = "\n\n_Offline mode: answered with simple rules from the event log, no AI model._"

    if is_regulation_question(q):
        note = "\n\n_Offline mode: passages found by keyword search, no AI model. Read them before relying on them._"
        if retriever is None:
            return ("No regulation documents are indexed yet. Put the Dubai Universal Design Code PDF in "
                    "data/rag_docs and run scripts/05_build_rag_index.py." + note, [])
        hits = retriever.search(q, 3)
        if not hits:
            return "I could not find a relevant passage in the indexed documents." + note, []
        lines = [f"[R{i}] {h['source']} p.{h['page']}: {h['text'][:350]}…" for i, h in enumerate(hits, 1)]
        return "Most relevant passages (read them to confirm the exact requirement):\n\n" + "\n\n".join(lines) + note, lines

    m = re.search(r"(?:track|p|person)\s*#?\s*(\d+)", ql)
    if m and any(w in ql for w in ("would", "if", "without", "had not", "hadn't")):
        n = int(m.group(1))
        by_person = "person" in ql or re.search(r"\bp\s*\d", ql)
        new_obs = remove_person(obs, n) if by_person else remove_track(obs, n)
        new = run_controller(cname, cfg, ep, new_obs)
        what = f"person P{n}" if by_person else f"track {n}"
        if new.end_time < res.end_time - 0.05:
            ans = (f"Yes. I re-ran the {cname} controller without {what}: the phase would have ended at "
                   f"{new.end_time:.1f} s instead of {res.end_time:.1f} s.")
        else:
            ans = (f"No. Re-running the {cname} controller without {what} gives the same end time "
                   f"({new.end_time:.1f} s).")
        return ans + note, log

    if any(w in ql for w in ("miss", "see", "camera", "detect")):
        t = _time_in(ql)
        k = int(round(t / ep.dt)) if t is not None else min(len(ep.times) - 1, int(round(res.end_time / ep.dt)))
        k = max(0, min(k, len(ep.times) - 1))
        dets = format_detections(obs, k, ep.length)
        return ("At t = %.1f s the camera reported:\n- %s\n\nThe offline reader cannot look at the picture, so it "
                "cannot say whether someone was missed. Choose an AI model and tick 'include picture' to check."
                % (ep.times[k], "\n- ".join(dets))) + note, dets

    target = "EXTEND" if ("extend" in ql or "longer" in ql or "hold" in ql) else "PHASE_END"
    for line, e in zip(log, res.log):
        if e["event"] == target or (target == "EXTEND" and e["event"] == "HOLD"):
            conf = f" The logged confidence was {e['confidence']:.2f}." if e.get("confidence") is not None else ""
            return (f"At {e['t']:.1f} s the {cname} controller recorded {e['event']} because "
                    f"{REASON_CODES.get(e['reason'], e['reason'])} ({e['details']}).{conf} {line.split(']')[0]}]"
                    + note), [line]
    if target == "EXTEND":
        return (f"The {cname} controller did not extend the phase; it ended at {res.end_time:.1f} s."
                + note), log
    return "I could not find that in the event log." + note, log


def ai_answer(q: str, bundle: dict, backend, include_image: bool, include_log: bool,
              image_path: str | None, retriever=None) -> tuple[str, str]:
    """Ask the AI model. Returns (answer, the prompt that was sent) so users can see the evidence."""
    cname = bundle["controller"]
    res = bundle["results"][cname]
    ep, obs = bundle["episode"], bundle["observations"]
    passages = retriever.search(q, 3) if (retriever and is_regulation_question(q)) else None
    t = _time_in(q.lower())
    k = int(round((t if t is not None else res.end_time) / ep.dt))
    k = max(0, min(k, len(ep.times) - 1))
    log = format_log(res) if include_log else None
    dets = format_detections(obs, k, ep.length) if include_log else None
    user = build_user_prompt(q, log, dets, passages, None, has_image=include_image and bool(image_path))
    reply = backend.generate(SYSTEM, user, image_path if include_image else None)
    return reply, user
