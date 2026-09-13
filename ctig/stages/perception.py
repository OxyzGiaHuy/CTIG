"""
Khối TRI GIÁC - mũi tên thiếu trong sơ đồ draft. Ba tín hiệu, mỗi cái một việc:

  CLIP       danh tính: P(ảnh giống nhãn MÔ TẢ của mục tiêu) so với nhãn mô tả từng confusable.
             Skill clip: "descriptive labels; not for fine-grained tasks; spatial/counting weak"
             -> CLIP chỉ chấm danh tính và chọn ứng viên, không chấm thuộc tính, không chấm sự kiện.
  VLM mô tả  liệt kê thứ nhìn thấy (hỏi mở) - dùng cho caption/báo cáo, không dùng để chấm điểm.
  VLM checklist  câu hỏi ĐÓNG cho từng thực thể: đây là X, là một confusable, hay không có?
             từng must_have có / không / không rõ? từng must_not có xuất hiện?
             Điểm và findings suy ra bằng luật ở llm/shared.checklist_critique.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm.shared import confusable_clip_label, confusable_labels
from ..schema import CulturalSpec, GenOutput, Perception, VisualElement

ANSWERS = ("yes", "no", "unsure")


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
        # Nhãn đã là câu mô tả đầy đủ; không bọc "a photo of" lần nữa nếu đã có mạo từ.
        texts = [l if l[:2].lower() in ("a ", "an", "th") else f"a photo of {l}" for l in labels]
        inputs = self.proc(text=texts, images=img, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        return out.logits_per_image.softmax(dim=-1)[0].tolist()

    def similarity(self, image_path: str, texts: list[str]) -> list[float]:
        """Cosine(ảnh, câu) trong [-1, 1], KHÔNG softmax -> so được giữa các ảnh khác nhau.

        probs() so các nhãn với nhau trên MỘT ảnh; similarity() so MỘT câu trên nhiều ảnh.
        Dùng để xếp ảnh search và ảnh sinh theo độ khớp với prompt_en.
        """
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        inputs = self.proc(text=texts, images=img, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
            scale = self.model.logit_scale.exp()
            sims = (out.logits_per_image / scale)[0]
        return [float(x) for x in sims.tolist()]

    def image_embed(self, image):
        """Vector CLIP chuẩn hoá của một ảnh (đường dẫn hoặc PIL)."""
        from PIL import Image

        img = Image.open(image).convert("RGB") if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__") else image
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            f = self.model.get_image_features(**inputs)
        return (f / f.norm(dim=-1, keepdim=True))[0]

    def image_similarity(self, image_a, image_b) -> float:
        """Cosine ảnh-ảnh (v1.4.2): đo ảnh sinh 'chép' ảnh tham chiếu đến đâu. Ảnh khác nhau cùng chủ đề ~0,6-0,8; gần chép > 0,9."""
        a, b = self.image_embed(image_a), self.image_embed(image_b)
        return float((a * b).sum())

    def similarity_image(self, image, texts: list[str]) -> list[float]:
        """Như similarity() nhưng nhận PIL image (dùng cho các ô cắt khi tìm vùng thực thể)."""
        inputs = self.proc(text=texts, images=image, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
            sims = (out.logits_per_image / self.model.logit_scale.exp())[0]
        return [float(x) for x in sims.tolist()]

    def entity_probs(self, image_path: str, spec: CulturalSpec) -> dict[str, dict[str, float]]:
        """entity_id -> {'__target__': p, <confusable name>: p, ...}. Chỉ thực thể kind == object."""
        result = {}
        for se in spec.entities:
            if se.kind != "object":
                continue
            target = se.clip_label or f"a photo of Vietnamese {se.name_en.split('(')[0].strip()}"
            names, labels = ["__target__"], [target]
            for cf in se.confusables:
                names.append(confusable_labels(cf.get("name", ""))[0] if cf.get("name") else cf.get("name_en", "?"))
                labels.append(confusable_clip_label(cf))
            if len(labels) < 2:
                names.append("something else"); labels.append("a photo of something unrelated")
            p = self.probs(image_path, labels)
            result[se.entity_id] = {n: p[i] for i, n in enumerate(names)}
        return result

    def image_matches(self, image_path: str, target_label: str, confusable_labels_en: list[str]) -> float:
        labels = [target_label] + [c for c in confusable_labels_en if c] + ["a photo of something unrelated"]
        return self.probs(image_path, list(dict.fromkeys(labels)))[0]


class VLMClipPerceiver:
    name = "vlm_clip"

    def __init__(self, llm, clip: CLIPProbe | None):
        self.llm = llm
        self.clip = clip

    # ------------------------------------------------------------------
    def perceive(self, gen: GenOutput, spec: CulturalSpec) -> Perception:
        path = gen.image_path
        caption, elements = self._describe(path, spec)
        checklist = {se.entity_id: self._checklist(path, se) for se in spec.entities}
        clip_probs = self.clip.entity_probs(path, spec) if self.clip else {}
        return Perception(path, elements, caption, clip_probs, checklist, self.name)

    def _describe(self, path: str, spec: CulturalSpec) -> tuple[str, list[VisualElement]]:
        system = (
            "Mô tả ảnh cho hệ thống kiểm tra. Liệt kê các đối tượng chính bạn THẤY, mỗi đối tượng gồm label "
            "(tên bạn thực sự nhận ra; nếu là kimono thì ghi 'kimono', không chắc thì ghi 'trang phục dài không rõ'), "
            "category, attrs (đặc điểm quan sát được), confidence 0..1. KHÔNG suy đoán theo ý định của prompt. "
            "Trả lời bằng tiếng Việt."
        )
        schema = {"type": "object", "properties": {
            "caption": {"type": "string"},
            "elements": {"type": "array", "items": {"type": "object", "properties": {
                "label": {"type": "string"}, "category": {"type": "string"},
                "attrs": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number"},
            }, "required": ["label", "category", "attrs", "confidence"], "additionalProperties": False}},
        }, "required": ["caption", "elements"], "additionalProperties": False}
        try:
            d = self.llm.complete_json(system, "Mô tả ảnh này.", schema, images=[path])
        except RuntimeError:
            return "", []
        elements = []
        for e in d.get("elements", []) or []:
            try:
                elements.append(VisualElement(str(e["label"]), str(e.get("category", "other")),
                                              [str(a) for a in e.get("attrs", [])], float(e.get("confidence", 0.7))))
            except (KeyError, ValueError, TypeError):
                continue
        return str(d.get("caption", "")), elements

    def _checklist(self, path: str, se) -> dict[str, Any]:
        """Câu hỏi đóng cho một thực thể. VLM chỉ chọn đáp án, không viết tự do."""
        cfs = [confusable_labels(c.get("name", ""))[0] if c.get("name") else c.get("name_en", "") for c in se.confusables]
        cfs = [c for c in cfs if c]
        identity_opts = ["target", "absent", "unsure"] + [f"confusable:{c}" for c in cfs]
        req = list(se.required_attrs)
        forb = list(se.forbidden_attrs)
        q_req = "\n".join(f"  R{i}. {a}" for i, a in enumerate(req)) or "  (không có)"
        q_forb = "\n".join(f"  F{i}. {a}" for i, a in enumerate(forb)) or "  (không có)"
        cf_txt = ", ".join(f"'{c}'" for c in cfs) or "(không có)"
        system = (
            "Bạn nhìn ảnh và trả lời các câu hỏi ĐÓNG. Chỉ chọn đáp án cho sẵn, không giải thích dài.\n"
            "Nếu không nhìn rõ hoặc không chắc, chọn 'unsure' - đừng đoán theo ý định của prompt."
        )
        user = (
            f"Thực thể cần kiểm: {se.name_vi} ({se.name_en}).\n\n"
            f"Câu 1 - identity: trong ảnh, đối tượng tương ứng là gì?\n"
            f"  'target' = đúng là {se.name_vi}; 'absent' = không có thứ gì như vậy; 'unsure' = không chắc;\n"
            f"  hoặc 'confusable:<tên>' nếu thực ra là một trong: {cf_txt}.\n\n"
            f"Câu 2 - attrs: với TỪNG đặc điểm sau, ảnh có thể hiện không? Trả 'yes' / 'no' / 'unsure', "
            f"đúng thứ tự, đủ {len(req)} phần tử:\n{q_req}\n\n"
            f"Câu 3 - forbidden: với TỪNG chi tiết sau, ảnh có xuất hiện không? 'yes' / 'no' / 'unsure', "
            f"đủ {len(forb)} phần tử:\n{q_forb}"
        )
        schema = {"type": "object", "properties": {
            "identity": {"type": "string", "enum": identity_opts},
            "attrs": {"type": "array", "items": {"type": "string", "enum": list(ANSWERS)}},
            "forbidden": {"type": "array", "items": {"type": "string", "enum": list(ANSWERS)}},
            "note": {"type": "string"},
        }, "required": ["identity", "attrs", "forbidden", "note"], "additionalProperties": False}
        try:
            d = self.llm.complete_json(system, user, schema, images=[path])
        except RuntimeError as exc:
            return {"identity": "unsure", "attrs": ["unsure"] * len(req), "forbidden": ["unsure"] * len(forb),
                    "note": f"lỗi: {exc}"}
        ident = str(d.get("identity", "unsure"))
        if ident not in identity_opts:
            # model có thể viết 'confusable:kimono có obi' -> chuẩn hoá về confusable gần nhất
            low = ident.lower()
            match = next((c for c in cfs if c.lower() in low or low in c.lower()), None)
            ident = f"confusable:{match}" if match else ("target" if "target" in low or se.name_vi.lower() in low else "unsure")

        def fix(ans, n):
            ans = [a if a in ANSWERS else "unsure" for a in (ans or [])][:n]
            return ans + ["unsure"] * (n - len(ans))

        return {"identity": ident, "attrs": fix(d.get("attrs"), len(req)),
                "forbidden": fix(d.get("forbidden"), len(forb)), "note": str(d.get("note", ""))[:200]}


class StubPerceiver:
    """Đọc thẳng nhãn mà StubT2I đã vẽ. CHỈ dùng cho test offline. Sinh cả checklist và CLIP giả."""

    name = "stub"

    def perceive(self, gen: GenOutput, spec: CulturalSpec) -> Perception:
        elements, clip_probs, checklist = [], {}, {}
        for se in spec.entities:
            drawn = (gen.oracle or {}).get(se.entity_id)
            cf_names = [confusable_labels(c.get("name", ""))[0] for c in se.confusables if c.get("name")] or ["other"]
            n_req, n_forb = len(se.required_attrs), len(se.forbidden_attrs)
            if drawn == se.name_vi:
                k = max(1, round(n_req * 0.7))
                elements.append(VisualElement(se.name_vi, "object", list(se.required_attrs[:k]), 0.8))
                checklist[se.entity_id] = {"identity": "target", "attrs": ["yes"] * k + ["no"] * (n_req - k),
                                           "forbidden": ["no"] * n_forb, "note": "stub"}
                if se.kind == "object":
                    clip_probs[se.entity_id] = {"__target__": 0.8, **{c: 0.2 / len(cf_names) for c in cf_names}}
            elif drawn and drawn != "<không vẽ>":
                elements.append(VisualElement(drawn, "object", list(se.forbidden_attrs[:2]), 0.8))
                hit = next((c for c in cf_names if c in drawn), cf_names[0])
                checklist[se.entity_id] = {"identity": f"confusable:{hit}", "attrs": ["no"] * n_req,
                                           "forbidden": ["yes"] * min(1, n_forb) + ["no"] * max(0, n_forb - 1), "note": "stub"}
                if se.kind == "object":
                    clip_probs[se.entity_id] = {"__target__": 0.15, **{c: (0.85 if c == hit else 0.0) for c in cf_names}}
            else:
                checklist[se.entity_id] = {"identity": "absent", "attrs": ["no"] * n_req, "forbidden": ["no"] * n_forb, "note": "stub"}
        return Perception(gen.image_path, elements, "stub", clip_probs, checklist, self.name)


def get_perceiver(cfg, llm=None):
    if cfg.backend == "stub":
        return StubPerceiver(), None
    clip = CLIPProbe(cfg.clip_model, cfg.device)
    if llm is None:
        raise ValueError("Perceiver vlm_clip cần một LLM backend có đọc ảnh")
    return VLMClipPerceiver(llm, clip), clip
