"""Prompts used for both the chatbot and the evaluation."""
from __future__ import annotations

SYSTEM = (
    "You are SafeCross, an assistant that explains the decisions of a pedestrian traffic signal. "
    "Use ONLY the evidence you are given: the picture (if provided), the controller event log (lines "
    "marked [L1], [L2], ...), the camera detection list and any retrieved regulation passages "
    "(marked [R1], [R2], ...). If the evidence does not contain the answer, say that you cannot determine it. "
    "Never invent traffic-signal rules, legal requirements, people or detections. "
    "Do not sound more certain than the evidence allows: if the log gives a confidence value, mention it."
)

CONDITIONS = {
    "image_log": "picture + event log + detections",
    "log_only": "event log + detections (no picture)",
    "image_only": "picture only (no event log, no detections)",
}


def build_user_prompt(question: str, log: list[str] | None, detections: list[str] | None,
                      passages: list[dict] | None = None, fields: dict | None = None,
                      has_image: bool = False) -> str:
    parts = []
    if has_image:
        parts.append("A picture of the crossing is attached. Filled circles are real people.")
    if log:
        parts.append("CONTROLLER EVENT LOG:\n" + "\n".join(log))
    if detections:
        parts.append("CAMERA DETECTIONS AT THIS MOMENT:\n" + "\n".join(f"- {d}" for d in detections))
    if passages:
        parts.append("RETRIEVED REGULATION PASSAGES:\n" + "\n".join(
            f"[R{i}] ({p['source']}, page {p['page']}): {p['text']}" for i, p in enumerate(passages, 1)))
    if not (log or detections or passages or has_image):
        parts.append("(No evidence was provided.)")
    parts.append("QUESTION: " + question)
    if fields:
        spec = ", ".join(f'"{k}": {("one of " + str(v)) if isinstance(v, list) else v}' for k, v in fields.items())
        parts.append("Reply with ONE JSON object with these keys: " + spec +
                     ', "explanation": a short sentence, "evidence": a list of the line ids you used '
                     '(for example ["L3"]). Use "cannot_determine" if the evidence is not enough.')
    else:
        parts.append("Answer in 2-4 sentences and cite the lines you used, e.g. [L3] or [R1].")
    return "\n\n".join(parts)


def evidence_for_condition(item: dict, condition: str) -> tuple[list[str] | None, list[str] | None, str | None]:
    """Return (log, detections, image_path) for one of the three input conditions."""
    if condition == "image_log":
        return item["log"], item["detections"], item["image"]
    if condition == "log_only":
        return item["log"], item["detections"], None
    if condition == "image_only":
        return None, None, item["image"]
    raise ValueError(condition)


# ------------------------------------------------------------------ knowledge-graph conditions (follow-up, NLP-RQ1)
SYSTEM_KG = (
    "You are SafeCross, an assistant that explains the decisions of a pedestrian traffic signal. "
    "You are given facts from the signal's decision knowledge graph. Each fact has an id like [G12] and the "
    "form (subject) --relation--> (object). Use ONLY these facts. Cite the ids of the facts you used. "
    "If the facts do not contain the answer, answer \"cannot_determine\". "
    "Never invent facts, tracks, rules or times."
)

KG_CONDITIONS = {
    "kg_full": "the whole decision knowledge graph of the walk phase (no log, no picture)",
    "kg_sub": "only the facts linked to the question (question-focused subgraph, no log, no picture)",
}


def build_kg_prompt(question: str, facts: list[str], fields: dict | None = None) -> str:
    parts = ["DECISION KNOWLEDGE GRAPH FACTS:\n" + ("\n".join(facts) if facts else "(no facts)"),
             "QUESTION: " + question]
    if fields:
        spec = ", ".join(f'"{k}": {("one of " + str(v)) if isinstance(v, list) else v}' for k, v in fields.items())
        parts.append("Reply with ONE JSON object with these keys: " + spec +
                     ', "explanation": a short sentence, "evidence": a list of the fact ids you used '
                     '(for example ["G12", "G30"]). Use "cannot_determine" if the facts are not enough.')
    else:
        parts.append("Answer in 2-4 sentences and cite the facts you used, e.g. [G12].")
    return "\n\n".join(parts)
