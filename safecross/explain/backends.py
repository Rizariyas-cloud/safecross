"""Ways to answer a question.

offline : NOT an AI model. Reads the evidence with simple rules. Used so the app
          and the evaluation pipeline work before any model is installed, and as
          a sanity check. Never report it as a model result.
ollama  : a vision-language model running locally in Ollama (easiest on a GPU laptop).
hf      : a Hugging Face model (Qwen2.5-VL) loaded with transformers (needs a GPU).

All backends take: a system prompt, a user prompt and an optional image path,
and return the model's raw text.
"""
from __future__ import annotations

from pathlib import Path

import base64
import json
import re


class OllamaBackend:
    def __init__(self, model: str, url: str = "http://localhost:11434", temperature: float = 0.0, timeout: int = 300):
        self.model, self.url, self.temperature, self.timeout = model, url.rstrip("/"), temperature, timeout
        self.name = f"ollama:{model}"

    def available(self) -> bool:
        import requests
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            return r.ok and any(m.get("name", "").startswith(self.model.split(":")[0])
                                for m in r.json().get("models", []))
        except Exception:
            return False

    def generate(self, system: str, user: str, image_path: str | None = None, json_mode: bool = False) -> str:
        import requests
        msg = {"role": "user", "content": user}
        if image_path:
            with open(image_path, "rb") as f:
                msg["images"] = [base64.b64encode(f.read()).decode()]
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "system", "content": system}, msg],
                   "options": {"temperature": self.temperature, "seed": 0}}
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]


class HFBackend:
    """Qwen2.5-VL through transformers. Needs a GPU with enough memory (about 16 GB in
    half precision for 7B; use the 3B model or 4-bit loading on smaller GPUs)."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", temperature: float = 0.0, load_4bit: bool = False):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.name = f"hf:{model_id}"

    def generate(self, system: str, user: str, image_path: str | None = None, json_mode: bool = False) -> str:
        from PIL import Image
        content = []
        images = None
        if image_path:
            content.append({"type": "image"})
            images = [Image.open(image_path).convert("RGB")]
        content.append({"type": "text", "text": user + ("\nReply with JSON only." if json_mode else "")})
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0]


class OfflineBackend:
    """Rule-based reader of the evidence (no AI). It can read the log text, but it cannot
    look at images, so for visual questions it answers 'cannot_determine'."""
    name = "offline_rules"

    def generate(self, system: str, user: str, image_path: str | None = None, json_mode: bool = False) -> str:
        if "DECISION KNOWLEDGE GRAPH FACTS" in user:
            return self._from_graph(user)
        log = "\n".join(l for l in user.splitlines() if l.startswith("[L"))
        q = user.split("QUESTION:")[-1].lower()
        whatif = re.search(r"track \d+", q) and re.search(r"\b(would|if|without)\b", q) and \
            re.search(r"earlier|sooner|end", q)
        if whatif:
            m = re.search(r"track (\d+)", q)
            tid = m.group(1) if m else None
            used = [l for l in log.splitlines() if tid and re.search(rf"\b{tid}\b", l.split("(")[-1])]
            return json.dumps({"answer": "yes" if used else "no",
                               "explanation": "Guess from whether the track appears in the log.",
                               "evidence": [l.split("]")[0] + "]" for l in used]})
        if "miss" in q or '"answer": one of' in user:     # picture questions (it cannot look at images)
            return json.dumps({"answer": "cannot_determine", "people_on_crossing": None,
                               "explanation": "The offline reader cannot inspect images.", "evidence": []})
        target = "EXTEND" if "extend" in q else "PHASE_END"
        for l in log.splitlines():
            if target in l:
                code = re.search(r"reason=(\w+)", l).group(1)
                return json.dumps({"reason_code": code, "explanation": l, "evidence": [l.split("]")[0] + "]"]})
        return json.dumps({"reason_code": "cannot_determine", "explanation": "Not found in evidence.", "evidence": []})

    @staticmethod
    def _from_graph(user: str) -> str:
        """Read the knowledge-graph facts ([G..] lines) with fixed rules (symbolic reference, no AI)."""
        facts = [l for l in user.splitlines() if l.startswith("[G")]
        q = user.split("QUESTION:")[-1].lower()
        gid = lambda l: l.split("]")[0][1:]
        whatif = re.search(r"track \d+", q) and re.search(r"\b(would|if|without)\b", q) and \
            re.search(r"earlier|sooner|end", q)
        if whatif:
            m = re.search(r"track (\d+)", q)
            tid = m.group(1) if m else None
            only = [l for l in facts if f"(track {tid}) --only_occupant-->" in l]
            about = [l for l in facts if f"(track {tid})" in l]
            if only:
                return json.dumps({"answer": "yes", "explanation": only[0], "evidence": [gid(only[0])]})
            if about:
                return json.dumps({"answer": "no", "explanation": f"track {tid} was never the only occupant",
                                   "evidence": [gid(about[0])]})
            return json.dumps({"answer": "cannot_determine", "explanation": "track not in the graph", "evidence": []})
        if re.search(r"extend|longer|stay green|stayed green", q):
            target = "EXTEND"
        elif re.search(r"\bend|\bstop|turn red|turned red", q):
            target = "PHASE_END"
        else:
            return json.dumps({"reason_code": "cannot_determine", "explanation": "question type not handled",
                               "evidence": []})
        for l in facts:
            if f" {target}) --reason-->" in l:
                code = re.search(r"--reason--> \((\w+)", l).group(1)
                return json.dumps({"reason_code": code, "explanation": l, "evidence": [gid(l)]})
        return json.dumps({"reason_code": "cannot_determine", "explanation": "Not found in the graph.", "evidence": []})


class MLXBackend:
    """Small Qwen model run with Apple MLX, optionally with our LoRA adapter (the fine-tuned explainer).
    Greedy decoding (deterministic). Text only: the explainer reads knowledge-graph facts, not pictures."""

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", adapter: str | None = "weights/explainer_lora",
                 max_tokens: int = 400):
        from mlx_lm import load
        from ..config import resolve
        ad = str(resolve(adapter)) if adapter else None
        self.model, self.tok = load(model, adapter_path=ad)
        self.max_tokens = max_tokens
        tagname = Path(adapter).name.replace("explainer_", "") if adapter else "base"   # lora, lora_v2, ...
        self.name = f"mlx:{model.split('/')[-1]}+{tagname}"

    def generate(self, system: str, user: str, image_path: str | None = None, json_mode: bool = False) -> str:
        from mlx_lm import generate
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tok, prompt=prompt, max_tokens=self.max_tokens, verbose=False)


def make_backend(cfg: dict, kind: str | None = None, model: str | None = None, adapter: str | None = None):
    e = cfg["explain"]
    kind = kind or e["backend"]
    if kind == "mlx":
        return MLXBackend(model or "Qwen/Qwen2.5-1.5B-Instruct",
                          adapter or e.get("explainer_adapter", "weights/explainer_lora"))
    if kind == "mlx-base":
        return MLXBackend(model or "Qwen/Qwen2.5-1.5B-Instruct", None)
    if kind == "ollama":
        return OllamaBackend(model or e["ollama_model"], e["ollama_url"], e["temperature"])
    if kind == "hf":
        return HFBackend(model or e["hf_model"], e["temperature"])
    return OfflineBackend()


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (models sometimes add extra words)."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"_unparsed": text}
