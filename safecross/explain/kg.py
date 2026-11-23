"""SafeCross decision knowledge graph (provenance graph of one walk phase).

The flat event log ([L1], [L2] ...) says WHAT the controller did. The knowledge graph also stores WHY,
as facts (subject, relation, object) built automatically from the controller's own trace:

    walk phase   --normal_length-->         17.0 s                      (rule)
    L3 (HOLD)    --caused_by-->             track 4
    track 4      --labelled_as-->           mobility-aid user (80% of frames)
    track 4      --lost_by_camera_at-->     t=17.6 s, y=5.2 m
    track 4      --kept_crossing_occupied-> 17.0-19.5 s
    track 4      --only_occupant-->         17.0-19.5 s
    walk phase   --ended_at-->              19.6 s because crossing_clear

Every fact gets an id [G1], [G2] ... so a model can cite it and a checker can verify the citation.
Nothing here changes a decision: the graph is read from EpisodeResult.trace / .tracks.

Three entity types are linked:  camera tracks (from computer vision)  ->  controller decisions (event log)
->  timing rules (controller settings, optionally linked to Dubai code passages via the RAG index).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..controller import base_phase_length
from .evidence import REASON_CODES


@dataclass
class Fact:
    gid: str
    s: str
    p: str
    o: str

    def line(self) -> str:
        return f"[{self.gid}] ({self.s}) --{self.p}--> ({self.o})"


def _intervals(times: list[float], flags: list[bool], dt: float) -> list[tuple[float, float]]:
    out, start = [], None
    for t, f in zip(times, flags):
        if f and start is None:
            start = t
        if not f and start is not None:
            out.append((start, round(t - dt, 1)))
            start = None
    if start is not None:
        out.append((start, times[-1]))
    return out


def _fmt(iv: list[tuple[float, float]]) -> str:
    return ", ".join(f"{a:.1f}-{b:.1f} s" for a, b in iv)


def build_kg(ep, res, cfg: dict, retriever=None) -> list[Fact]:
    c = cfg["controller"]
    base = base_phase_length(cfg)
    facts: list[Fact] = []

    def add(s, p, o):
        facts.append(Fact(f"G{len(facts) + 1}", s, p, str(o)))

    ctrl = res.controller
    phase = "walk phase"
    # ---- rules (the controller's settings)
    add(phase, "controlled_by", f"{ctrl} controller")
    add(phase, "normal_length", f"{base:.1f} s (rule: walk interval + crossing length / design speed)")
    add(phase, "maximum_length", f"{c['max_phase_s']:.0f} s (rule)")
    add(phase, "ends_when", f"the crossing is not occupied at some moment after {base:.1f} s (rule)")
    if ctrl == "baseline":
        add(f"{ctrl} controller", "counts_crossing_occupied_if",
            f"a track was detected on the crossing in the last {c['persistence_s']:.1f} s (rule)")
    else:
        add(f"{ctrl} controller", "counts_crossing_occupied_if",
            "a track is detected on the crossing, OR a track was lost on the crossing and is being held (rule)")
        add(f"{ctrl} controller", "holds_lost_track_until",
            f"it could have finished crossing at its speed, or {c['assumed_slow_speed_mps']} m/s if it is a "
            f"mobility-aid user or its speed is unknown, x{c['remaining_time_margin']}, at most "
            f"{c['safety_timeout_s']:.0f} s (rule)")
        if ctrl.endswith("filtered"):
            add(f"{ctrl} controller", "ignores_tracks_that",
                f"were seen for less than {c.get('confirm_min_s', 0.5)} s or moved less than "
                f"{c.get('confirm_min_move_m', 0.3)} m (ghost filter rule)")
    # NOTE: no link from the timing rules to the Dubai guide. The guide (Dubai Universal Design audit guide)
    # contains no signal-timing or walking-speed requirement, so such a link would cite an unrelated page.
    # The timing values are engineering assumptions from configs/config.yaml.
    add("timing rules", "source", "SafeCross settings in configs/config.yaml (engineering assumptions; "
        "the Dubai Universal Design guide sets no signal timing)")

    # ---- decisions (the event log) and their causes
    for i, e in enumerate(res.log, start=1):
        node = f"L{i} {e['event']}"
        add(node, "at_time", f"{e['t']:.1f} s")
        add(node, "reason", f"{e['reason']} ({REASON_CODES.get(e['reason'], '')})")
        if e.get("confidence") is not None:
            add(node, "confidence", f"{e['confidence']:.2f}")
        for tid in re.findall(r"track[s]?\s*\[?([\d,\s]+)\]?", e.get("details", "")):
            for n in re.findall(r"\d+", tid):
                add(node, "caused_by", f"track {n}")
        if e["event"] == "PHASE_END":
            add(phase, "ended_at", f"{e['t']:.1f} s because {e['reason']} (see L{i})")

    # ---- camera tracks (from the detector + tracker)
    times = [t for t, _, _ in res.trace]
    after = [t >= base - 1e-9 for t in times]
    contributors = [set(a) | set(b) for _, a, b in res.trace]
    seen_tracks = sorted({tid for s in contributors for tid in s} |
                         {tid for tid, d in res.tracks.items() if 0 <= d["last_y"] < ep.length})
    for tid in seen_tracks:
        d = res.tracks.get(tid)
        if d is None:
            continue
        name = f"track {tid}"
        share = d["aid_share"]
        add(name, "labelled_as", f"mobility-aid user ({share:.0%} of frames)" if share >= c["mobility_aid_threshold"]
            else f"pedestrian ({share:.0%} of frames as mobility aid)")
        add(name, "first_seen_at", f"{d['first_t']:.1f} s")
        if d.get("exited"):
            add(name, "reached_exit_at", f"{d['last_t']:.1f} s")
        else:
            add(name, "last_seen_at", f"{d['last_t']:.1f} s at y={d['last_y']:.1f} m of {ep.length:.0f} m")
            if d["last_t"] < res.end_time - 1e-6:
                add(name, "lost_by_camera", f"not seen after {d['last_t']:.1f} s and did not reach the exit")
        if d.get("speed") is not None:
            add(name, "estimated_speed", f"{d['speed']:.2f} m/s")
        if ctrl != "baseline" and not d.get("exited") and d["last_t"] < res.end_time - 1e-6:
            add(name, "held_until", f"{d['hold_until']:.1f} s")
        if ctrl.endswith("filtered") and not d.get("counts", True):
            add(name, "ignored_by_ghost_filter", "yes")
        # occupancy provenance after the normal phase length
        keeps = [a and (tid in s) for a, s in zip(after, contributors)]
        iv = _intervals(times, keeps, ep.dt)
        if iv:
            add(name, "kept_crossing_occupied", _fmt(iv) + f" (after the normal length {base:.1f} s)")
            add(phase, "extended_by", name)
        only = [a and s == {tid} for a, s in zip(after, contributors)]
        iv1 = _intervals(times, only, ep.dt)
        if iv1:
            add(name, "only_occupant", _fmt(iv1) + " (no other track kept the crossing occupied then)")
    if not any(a and s for a, s in zip(after, contributors)):
        add(phase, "extended_by", "nobody (crossing clear at the normal length)")
    return facts


def lines(facts: list[Fact]) -> list[str]:
    return [f.line() for f in facts]


# ------------------------------------------------------------------ question-focused subgraph (GraphRAG)
def subgraph(facts: list[Fact], question: str) -> list[Fact]:
    """Question-focused subgraph (a simple GraphRAG step): link the entities named in the question
    (track numbers, the extension / phase end, a time), follow caused_by edges from decisions to tracks,
    and keep every fact about those entities + the phase summary + the rules.
    'walk phase' is a hub, so we never expand THROUGH it (that would return the whole graph)."""
    q = question.lower()
    keep = {f"track {n}" for n in re.findall(r"track\s+(\d+)", q)}
    # simple keyword entity linking (a rule-based linker; synonyms let other wordings find the same decision)
    if re.search(r"extend|longer|stay(ed)? green|kept .* going|beyond its normal|past its normal", q):
        keep |= {f.s for f in facts if " EXTEND" in f.s}
    if re.search(r"\bend|\bstop|turn(ed)? (red|off)|finish|switch(ed)? off", q):
        keep |= {f.s for f in facts if "PHASE_END" in f.s}
    for t in re.findall(r"(\d+(?:\.\d+)?)\s*s\b", q):
        keep |= {f.s for f in facts if f.p == "at_time" and f.o.startswith(f"{float(t):.1f} ")}
    keep |= {f.o for f in facts if f.s in keep and f.p == "caused_by"}      # decision -> its tracks
    return [f for f in facts if f.s in keep or f.s == "walk phase" or "(rule" in f.o or "rule)" in f.o]


# ------------------------------------------------------------------ checks used by the verifier and tests
def phase_ends_earlier_without(res, cfg: dict, tid: int) -> bool:
    """Read from the provenance: would the phase have ended earlier if `tid` had never been detected?
    True iff, at some moment after the normal length and before the actual end, `tid` was the ONLY
    track keeping the crossing occupied."""
    base = base_phase_length(cfg)
    for t, a, b in res.trace:
        if base - 1e-9 <= t < res.end_time - 1e-9:
            s = set(a) | set(b)
            if s == {tid}:
                return True
    return False
