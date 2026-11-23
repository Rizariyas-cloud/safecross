"""Training data for the SafeCross explainer (supervised fine-tuning, NLP step 2).

Each example = (question + knowledge-graph facts)  ->  correct JSON answer that CITES the supporting facts.
Answers are computed exactly from the controller, never written by hand:

  decision        "Why did the walk phase end / get extended at t?"  -> reason code + the fact that states it
  counterfactual  "If track N had never been detected, would it have ended earlier?"
                  -> yes/no from the provenance (checked against re-running the controller in tests)
  abstain         the same questions with the key facts REMOVED  -> "cannot_determine", no citation
                  (teaches the model to decline instead of guessing - the failure we measured in Qwen-7B)

Questions are phrased in several ways (the benchmark wording is one of them) so the model learns the task,
not one sentence. Episodes come from a different random seed than the benchmark (checked in script 17).
"""
from __future__ import annotations

import json
import re

import numpy as np

from ..controller import run_controller
from ..perception import perceive
from ..sim import generate_episode
from .evidence import REASON_CODES
from .kg import build_kg, lines, subgraph, phase_ends_earlier_without
from .prompts import SYSTEM_KG, build_kg_prompt

REASON_LIST = [r for r in REASON_CODES if r != "walk_requested"]
END_Q = ["Why did the pedestrian phase end at {t:.1f} s?", "Why did the walk phase end at {t:.1f} s?",
         "What made the pedestrian phase end at {t:.1f} s?", "Explain why the walk signal ended at {t:.1f} s."]
EXT_Q = ["Why was the pedestrian phase extended at {t:.1f} s?", "Why was the walk phase extended at {t:.1f} s?",
         "What caused the extension of the pedestrian phase at {t:.1f} s?"]
CF_Q = ["If track {n} had never been detected, would the pedestrian phase have ended earlier than {t:.1f} s?",
        "Had the camera never detected track {n}, would the walk phase have ended before {t:.1f} s?",
        "Would the pedestrian phase have ended earlier than {t:.1f} s without track {n}?"]


def _gid(line: str) -> str:
    return line.split("]")[0][1:]


def _answer(key, value, explanation, evidence):
    return json.dumps({key: value, "explanation": explanation, "evidence": evidence})


def episode_examples(ep, res, cfg, rng, full_share: float = 0.0) -> list[dict]:
    """All training examples for one walk phase of one controller.
    full_share > 0: that share of the questions shows the WHOLE graph instead of the question-focused subgraph
    (v2 data: teaches the model to find the relevant facts itself; the cited fact ids are identical, because
    the subgraph keeps the ids of the whole graph)."""
    facts = build_kg(ep, res, cfg)
    out = []

    def add(question, fields, facts_given, target, kind, graph):
        out.append({"kind": kind, "question": question, "fields": fields,
                    "facts": lines(facts_given), "target": target, "graph": graph})

    def view(q):
        """(facts shown, 'full' or 'sub') for one question."""
        if full_share > 0 and rng.random() < full_share:
            return facts, "full"
        return subgraph(facts, q), "sub"

    # ---- decisions
    for i, e in enumerate(res.log, start=1):
        if e["event"] not in ("PHASE_END", "EXTEND"):
            continue
        if e["event"] == "EXTEND" and any(x["event"] == "EXTEND" for x in res.log[:i - 1]):
            continue                                            # benchmark asks about the first extension
        node = f"L{i} {e['event']}"
        q = (rng.choice(END_Q) if e["event"] == "PHASE_END" else rng.choice(EXT_Q)).format(t=e["t"])
        sub, graph = view(q)
        reason_fact = next(f for f in sub if f.s == node and f.p == "reason")
        expl = (f"At {e['t']:.1f} s the controller logged {e['event']} with reason {e['reason']} "
                f"({REASON_CODES.get(e['reason'], '')}).")
        add(q, {"reason_code": REASON_LIST}, sub,
            _answer("reason_code", e["reason"], expl, [reason_fact.gid]), "decision", graph)
        # abstain version: the decision's own facts (and the phase-end summary) are missing
        gone = [f for f in sub if not (f.s == node or (f.p == "ended_at" and node.startswith(f"L{i} PHASE_END")))]
        add(q, {"reason_code": REASON_LIST}, gone,
            _answer("reason_code", "cannot_determine",
                    "The facts given do not include this decision, so its reason cannot be determined.", []),
            "abstain_decision", graph)

    # ---- counterfactuals (tracks seen on the crossing)
    tracks = sorted({int(n) for f in facts for n in re.findall(r"^track (\d+)$", f.s)})
    yes = [t for t in tracks if phase_ends_earlier_without(res, cfg, t)]
    no = [t for t in tracks if t not in yes]
    chosen = ([int(rng.choice(yes))] if yes else []) + ([int(rng.choice(no))] if no else [])   # balanced yes / no
    for tid in chosen:
        q = rng.choice(CF_Q).format(n=tid, t=res.end_time)
        sub, graph = view(q)
        about = [f for f in sub if f.s == f"track {tid}"]
        if not about:
            continue
        only = [f for f in about if f.p == "only_occupant"]
        rule = next((f for f in sub if f.p == "ends_when"), None)
        if phase_ends_earlier_without(res, cfg, tid):
            ev = [only[0].gid] + ([rule.gid] if rule else [])
            expl = (f"Track {tid} was the only track keeping the crossing occupied ({only[0].o.split(' (')[0]}), "
                    f"so without it the phase would have ended earlier.")
            target = _answer("answer", "yes", expl, ev)
        else:
            keep = next((f for p in ("kept_crossing_occupied", "last_seen_at", "reached_exit_at")
                         for f in about if f.p == p), about[0])
            ev = [keep.gid] + ([rule.gid] if rule else [])
            expl = (f"Track {tid} was never the only track keeping the crossing occupied after the normal length, "
                    f"so removing it would not have ended the phase earlier.")
            target = _answer("answer", "no", expl, ev)
        add(q, {"answer": ["yes", "no"]}, sub, target, "counterfactual", graph)
        gone = [f for f in sub if f.s != f"track {tid}" and f.o != f"track {tid}"]
        add(q, {"answer": ["yes", "no"]}, gone,
            _answer("answer", "cannot_determine",
                    f"The facts given contain nothing about track {tid}, so this cannot be determined.", []),
            "abstain_counterfactual", graph)
    return out


def to_chat(ex: dict) -> dict:
    """mlx-lm 'chat' format: one JSON line with system / user / assistant messages."""
    return {"messages": [{"role": "system", "content": SYSTEM_KG},
                         {"role": "user", "content": build_kg_prompt(ex["question"], ex["facts"], ex["fields"])},
                         {"role": "assistant", "content": ex["target"]}]}


def fingerprint(ep) -> str:
    """Identifies a simulated walk phase by its people (to prove train and test never share one)."""
    return "|".join(f"{p.cls}:{p.t_start:.2f}:{p.x:.3f}:{float(p.speeds[0]):.4f}" for p in ep.pedestrians)


def build_examples(cfg: dict, n_episodes: int, seed: int, abstain_share: float = 0.2, full_share: float = 0.0):
    rng = np.random.default_rng(seed)
    examples, prints = [], []
    for e in range(n_episodes):
        ep = generate_episode(50000 + e, cfg, rng)
        prints.append(fingerprint(ep))
        fail = dict(cfg["failures"])
        fail.update(recall=float(rng.uniform(0.5, 1.0)), track_loss_prob=float(rng.uniform(0.0, 0.5)),
                    misclassification_prob=float(rng.uniform(0.0, 0.2)),
                    false_positive_rate=float(rng.uniform(0.0, 0.1)))
        obs = perceive(ep, fail, np.random.default_rng(int(rng.integers(1e9))))
        for c in ("baseline", "uncertainty_aware"):
            res = run_controller(c, cfg, ep, obs)
            for ex in episode_examples(ep, res, cfg, rng, full_share):
                if ex["kind"].startswith("abstain") and rng.random() > abstain_share / (1 - abstain_share):
                    continue                                    # keep about abstain_share of the data as abstain
                ex["episode"] = 50000 + e
                ex["controller"] = c
                examples.append(ex)
    cf_yes = [x for x in examples if x["kind"] == "counterfactual" and '"answer": "yes"' in x["target"]]
    cf_no = [x for x in examples if x["kind"] == "counterfactual" and '"answer": "no"' in x["target"]]
    keep_no = set(id(x) for x in list(rng.permutation(np.array(cf_no, dtype=object)))[:int(len(cf_yes) * 1.3)])
    examples = [x for x in examples if not (x in cf_no and id(x) not in keep_no)]
    return examples, prints
