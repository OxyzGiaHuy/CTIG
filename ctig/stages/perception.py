"""
Khối TRI GIÁC - mũi tên thiếu trong sơ đồ draft.

Bộ sinh thật chỉ trả pixel. Trước khi review được, cần biến pixel thành thứ đọc
được. v1 dùng hai tín hiệu độc lập:

  VLM   - liệt kê những gì nhìn thấy, hỏi MỞ ("thấy gì"), không hỏi dẫn.
  CLIP  - với mỗi thực thể: P(ảnh giống tên mục tiêu) so với P(giống từng confusable).
          Không hiểu chi tiết nhưng không bị prompt dẫn dắt.

Hai tín hiệu này bất đồng là chuyện thường, và stage review ghi lại bất đồng đó.
"""

from __future__ import annotations

import json

from ..llm.json_utils import extract_json, schema_to_hint
from ..llm.shared import confusable_labels
from ..schema import CulturalSpec, GenOutput, Perception, VisualElement


class CLIPProbe:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda:0"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    def probs(self, image_path: str, labels: list[str]) -> list[float]:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        texts = [f"a photo of {l}" for l in labels]
        inputs = self.proc(text=texts, images=img, return_tensors="pt", padding=True).to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        return out.logits_per_image.softmax(dim=-1)[0].tolist()

    def entity_probs(self, image_path: str, spec: CulturalSpec) -> dict[str, dict[str, float]]:
        """entity_id -> {'__target__': p, confusable: p, ...}."""
        result = {}
        for se in spec.entities:
            target = se.name_en.split("(")[0].strip()
            labels = [target]
            for cf in se.confusables:
                labels.extend(confusable_labels(cf.get("name", "")))
            labels = list(dict.fromkeys(l for l in labels if l))
            if len(labels) < 2:
                labels.append("something else")
            p = self.probs(image_path, labels)
            result[se.entity_id] = {"__target__": p[0], **{l: p[i] for i, l in enumerate(labels) if i}}
        return result

    def image_matches(self, image_path: str, target: str, confusables: list[str]) -> float:
        labels = [target] + [x for c in confusables for x in confusable_labels(c)] + ["unrelated photo"]
        return self.probs(image_path, list(dict.fromkeys(labels)))[0]


class VLMClipPerceiver:
    name = "vlm_clip"

    def __init__(self, llm, clip: CLIPProbe | None):
        self.llm = llm
        self.clip = clip

    def perceive(self, gen: GenOutput, spec: CulturalSpec) -> Perception:
        path = gen.image_path
        checklist = "\n".join(f"- {e.name_en.split('(')[0].strip()} ({e.name_vi}): "
                              + "; ".join(e.required_attrs[:3]) for e in spec.entities)
        system = (
            "Bạn mô tả một ảnh cho hệ thống kiểm tra. Liệt kê các đối tượng chính bạn THẤY. "
            "Với mỗi đối tượng: label là tên bạn thực sự nhận ra (nếu là kimono thì ghi 'kimono', "
            "nếu là áo dài thì ghi 'áo dài'; nếu không chắc ghi 'trang phục dài không rõ loại'), "
            "category (trang_phuc / am_thuc / kien_truc / nhac_cu / le_hoi / sinh_hoat / canh_quan / other), "
            "attrs là các đặc điểm thị giác quan sát được, confidence 0..1.\n"
            "KHÔNG suy đoán theo ý định của prompt. Chỉ ghi cái thấy. "
            "Dưới đây là danh sách điểm cần chú ý soi, nhưng đừng để nó dẫn bạn tới kết luận:\n" + checklist
        )
        schema = {"type": "object", "properties": {
            "caption": {"type": "string"},
            "elements": {"type": "array", "items": {"type": "object", "properties": {
                "label": {"type": "string"}, "category": {"type": "string"},
                "attrs": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number"},
            }, "required": ["label", "category", "attrs", "confidence"], "additionalProperties": False}},
        }, "required": ["caption", "elements"], "additionalProperties": False}
        d = self.llm.complete_json(system, "Mô tả ảnh này.", schema, images=[path])
        elements = []
        for e in d.get("elements", []):
            try:
                elements.append(VisualElement(str(e["label"]), str(e.get("category", "other")),
                                              [str(a) for a in e.get("attrs", [])],
                                              float(e.get("confidence", 0.7))))
            except (KeyError, ValueError, TypeError):
                continue
        clip_probs = self.clip.entity_probs(path, spec) if self.clip else {}
        return Perception(path, elements, str(d.get("caption", "")), clip_probs, self.name)


class StubPerceiver:
    """Đọc thẳng nhãn mà StubT2I đã vẽ. CHỈ dùng cho test offline."""

    name = "stub"

    def perceive(self, gen: GenOutput, spec: CulturalSpec) -> Perception:
        from ..kb import KnowledgeBase  # noqa: F401  (giữ import nhẹ)

        elements = []
        for se in spec.entities:
            drawn = (gen.oracle or {}).get(se.entity_id)
            if not drawn or drawn == "<không vẽ>":
                continue
            if drawn == se.name_vi:
                n = max(1, round(len(se.required_attrs) * 0.7))
                elements.append(VisualElement(se.name_vi, "object", list(se.required_attrs[:n]), 0.8))
            else:
                elements.append(VisualElement(drawn, "object", list(se.forbidden_attrs[:2]), 0.8))
        clip_probs = {}
        for se in spec.entities:
            drawn = (gen.oracle or {}).get(se.entity_id)
            labels = [x for c in se.confusables for x in confusable_labels(c.get("name", ""))] or ["other"]
            if drawn == se.name_vi:
                clip_probs[se.entity_id] = {"__target__": 0.8, **{l: 0.2 / len(labels) for l in labels}}
            elif drawn and drawn != "<không vẽ>":
                clip_probs[se.entity_id] = {"__target__": 0.15, **{l: (0.85 if l in drawn else 0.0) for l in labels}}
        return Perception(gen.image_path, elements, "stub", clip_probs, self.name)


def get_perceiver(cfg, llm=None):
    if cfg.backend == "stub":
        return StubPerceiver(), None
    clip = CLIPProbe(cfg.clip_model, cfg.device)
    if llm is None:
        raise ValueError("Perceiver vlm_clip cần một LLM backend có đọc ảnh")
    return VLMClipPerceiver(llm, clip), clip
