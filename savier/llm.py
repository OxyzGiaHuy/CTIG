"""Shared MLLM backbone for the three SAVIER agents (Mistral-Small-3.1-24B-Instruct, local transformers).

All three agents run on ONE model and differ only by system prompt; the paper must describe them as
"role-specialized agents sharing the same MLLM backbone", never as three independent models.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

MAX_SIDE = 768   # Pixtral tiles images into 16x16 patches; a 1024px image costs ~4k tokens.


class JSONExtractError(ValueError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in `text`, with a few gentle repairs."""
    if not text or not text.strip():
        raise JSONExtractError("empty output")
    cands = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S) + [_first_balanced(text), text.strip()]
    for c in cands:
        if not c:
            continue
        for fix in (lambda s: s, lambda s: re.sub(r",\s*([}\]])", r"\1", s), lambda s: s.replace("'", '"')):
            try:
                o = json.loads(fix(c))
                if isinstance(o, dict):
                    return o
            except json.JSONDecodeError:
                continue
    raise JSONExtractError(f"no JSON object in: {text[:200]!r}")


def _first_balanced(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def schema_hint(schema: dict) -> str:
    def sk(node):
        t = node.get("type")
        if t == "object": return {k: sk(v) for k, v in node.get("properties", {}).items()}
        if t == "array": return [sk(node.get("items", {"type": "string"}))]
        if "enum" in node: return " | ".join(map(str, node["enum"]))
        return {"string": "...", "number": 0.0, "integer": 0, "boolean": True}.get(t, "...")
    return json.dumps(sk(schema), ensure_ascii=False, indent=1)


class MistralVL:
    """Mistral-Small-3.1-24B: text or text+image -> JSON. Greedy decoding by default (temperature 0)."""

    def __init__(self, model: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503", device: str = "cuda:0",
                 max_new_tokens: int = 900, temperature: float = 0.0, json_retries: int = 2):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.device, self.model_id = torch, device, model
        self.max_new_tokens, self.temperature, self.json_retries = max_new_tokens, temperature, json_retries
        dt = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=dt, device_map=device)
        try:
            self.processor = AutoProcessor.from_pretrained(model, fix_mistral_regex=True)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(model)
        self.calls, self.seconds = 0, 0.0

    def _messages(self, system: str, user: str, images: list[str] | None):
        from PIL import Image
        content = []
        for p in images or []:
            im = Image.open(p).convert("RGB"); im.thumbnail((MAX_SIDE, MAX_SIDE))
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": user})
        return [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]

    def chat(self, system: str, user: str, images: list[str] | None = None, max_new_tokens: int | None = None) -> str:
        inputs = self.processor.apply_chat_template(self._messages(system, user, images), add_generation_prompt=True,
                                                    tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
        gen = dict(max_new_tokens=max_new_tokens or self.max_new_tokens)
        gen.update(dict(do_sample=True, temperature=self.temperature, top_p=0.9) if self.temperature > 0
                   else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
        t0 = time.time()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        self.calls += 1; self.seconds += time.time() - t0
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def complete_json(self, system: str, user: str, schema: dict, images: list[str] | None = None,
                      max_new_tokens: int | None = None) -> dict:
        sys_full = system + "\n\nReturn ONE valid JSON object and nothing else, no markdown. Required structure:\n" + schema_hint(schema)
        prompt, err = user, None
        for _ in range(self.json_retries + 1):
            text = self.chat(sys_full, prompt, images, max_new_tokens)
            try:
                return extract_json(text)
            except JSONExtractError as e:
                err = e
                prompt = user + "\n\nYour previous answer was not valid JSON. Return exactly one JSON object starting with '{' and ending with '}'."
        raise RuntimeError(f"no valid JSON after {self.json_retries + 1} attempts: {err}")
