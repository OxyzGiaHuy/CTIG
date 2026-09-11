"""
Agent dùng LLM/VLM thật (Qwen local hoặc Claude) qua LLMBackend.

Nguyên tắc prompt:
  1. Mỗi nhiệm vụ một lần gọi, có JSON schema riêng.
  2. Model phải DẪN BẰNG CHỨNG được cung cấp, không dùng tri thức riêng về văn hoá
     Việt Nam - tri thức đó chính là thứ đang bị nghi ngờ.
  3. Với ảnh: hỏi mở ("thấy gì"), không hỏi dẫn ("có phải áo dài không").
  4. Phần cần nhất quán (lọc, xếp hạng, hoà giải, lập bản sửa) là luật, không phải LLM.
"""

from __future__ import annotations

import json
from typing import Any

from ..kb import KnowledgeBase
from ..schema import (
    Adjudication, AnalysisResult, Critique, CulturalSpec, Finding, GenSpec, Keyword,
    NewEntity, Perception, Prompt, RevisionPlan, SearchResult,
)
from . import shared
from .base import LLMBackend
from .rule_agent import RuleAgent

SEVERITIES = ["critical", "major", "minor"]


def _s(**props) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _arr(item) -> dict[str, Any]:
    return {"type": "array", "items": item}


STR = {"type": "string"}
NUM = {"type": "number"}

FINDING_SCHEMA = _s(entity_id=STR, severity={"type": "string", "enum": SEVERITIES},
                    observed=STR, expected=STR, message=STR)


class PromptAgent:
    name = "prompt"

    def __init__(self, backend: LLMBackend):
        self.llm = backend
        self._rule = RuleAgent()
        self.name = f"prompt/{backend.name}"

    # ------------------------------------------------------------ stage 1
    def analyze(self, prompt: Prompt, kb: KnowledgeBase) -> AnalysisResult:
        catalog = "\n".join(f"- {e.id} | {e.name_vi} | {e.name_en} | {e.category} | {e.region}"
                            for e in kb.all())
        system = (
            "Bạn là agent phân tích prompt cho hệ sinh ảnh văn hoá Việt Nam.\n"
            "Tách hai loại keyword:\n"
            "  surface  = thực thể hoặc bối cảnh được NÊU TÊN thẳng trong prompt.\n"
            "  expanded = thực thể văn hoá bạn SUY RA từ bối cảnh, ví dụ 'cụ bà chở dừa ra chợ "
            "lúc trời chưa sáng ở miền Tây' suy ra chợ nổi, áo bà ba, nón lá.\n"
            "candidate_entity_ids CHỈ được chứa id có trong danh mục, ưu tiên độ chính xác hơn "
            "độ phủ: đừng thêm thực thể chỉ vì 'có thể có'.\n"
            "region_hint là một trong: bac_bo, trung_bo, nam_bo, tay_bac, tay_nguyen, hoặc chuỗi rỗng.\n"
            "prompt_en: viết lại prompt thành một câu tiếng Anh tả cảnh cho model sinh ảnh, giữ "
            "nguyên tên riêng Việt (ao dai, non la, banh chung...), không thêm thực thể chưa suy ra.\n"
            "new_entities: thực thể văn hoá Việt Nam RÕ RÀNG cần cho ảnh nhưng KHÔNG có trong danh mục "
            "(ví dụ 'gốm Bát Tràng', 'khèn H'Mông'). Ghi name_vi, name_en, category, region. "
            "Để rỗng nếu danh mục đã đủ. Không đưa thứ chung chung như 'con người', 'cây'."
        )
        user = (f"Prompt tiếng Việt: {prompt.text_vi}\nPrompt tiếng Anh gốc: {prompt.text_en}\n\n"
                f"Danh mục (id | tên | tên EN | loại | vùng):\n{catalog}")
        schema = _s(
            keywords=_arr(_s(term=STR, kind={"type": "string", "enum": ["entity", "scene", "attribute", "style", "region"]},
                             source={"type": "string", "enum": ["surface", "expanded"]},
                             confidence=NUM, rationale=STR)),
            candidate_entity_ids=_arr(STR), region_hint=STR, prompt_en=STR, notes=STR,
            new_entities=_arr(_s(name_vi=STR, name_en=STR, category=STR, region=STR, rationale=STR)),
        )
        d = self.llm.complete_json(system, user, schema)
        new_entities = []
        for ne in d.get("new_entities", []) or []:
            if isinstance(ne, dict) and ne.get("name_vi") and ne.get("name_en"):
                new_entities.append(NewEntity(str(ne["name_vi"]), str(ne["name_en"]), str(ne.get("category") or "other"),
                                              str(ne.get("region") or "toan_quoc"), ne.get("rationale")))
        kws = []
        for k in d.get("keywords", []):
            try:
                kws.append(Keyword(str(k["term"]), k.get("kind", "entity"), k.get("source", "surface"),
                                   float(k.get("confidence", 0.7)), k.get("rationale")))
            except (KeyError, ValueError, TypeError):
                continue
        ids = [i for i in d.get("candidate_entity_ids", []) if isinstance(i, str)]
        region = d.get("region_hint") or None
        if region not in ("bac_bo", "trung_bo", "nam_bo", "tay_bac", "tay_nguyen"):
            region = None
        return AnalysisResult(prompt.id, kws, ids, region, new_entities=new_entities,
                              prompt_en=d.get("prompt_en") or prompt.text_en, notes=d.get("notes") or None)

    # ------------------------------------------------------------ stage 2b
    def extract_evidence(self, ent, texts: list[dict]) -> dict:
        """Văn bản -> thuộc tính thị giác kiểm chứng được, mỗi thuộc tính kèm trích đoạn gốc."""
        numbered = "\n\n".join(f"[{i}] {t.get('title', '')}\n{t.get('text', '')[:2500]}" for i, t in enumerate(texts))
        system = (
            f"Bạn đọc các đoạn văn bản về '{ent.name_vi}' ({ent.name_en}) và rút ra bằng chứng cho hệ "
            "kiểm tra ảnh sinh bởi AI.\n"
            "must_have: 3-6 đặc điểm THỊ GIÁC (hình dạng, chất liệu, màu, cách mặc/bày) mà một ảnh PHẢI có "
            "để được coi là đúng thực thể này. Mỗi mục gồm attr (cụm tiếng Việt ngắn, kiểm được bằng mắt) "
            "và quote (CHÉP NGUYÊN VĂN câu trong văn bản làm căn cứ, kèm chỉ số nguồn như [0]). "
            "KHÔNG lấy lịch sử, nguồn gốc, ý nghĩa.\n"
            "must_not: 2-4 đặc điểm mà nếu xuất hiện là ảnh đã sai, cùng cấu trúc attr + quote.\n"
            "confusable_with: 1-3 thứ của văn hoá KHÁC dễ bị nhầm sang: name (tiếng Việt), name_en "
            "(cụm tiếng Anh mô tả, ví dụ 'a Japanese kimono with wide obi sash'), culture, why.\n"
            "Quy tắc quan trọng nhất: chỉ rút từ văn bản, không dùng kiến thức riêng. Không có câu gốc thì "
            "KHÔNG đưa mục đó vào. Văn bản không đủ thì trả về ít, thậm chí rỗng."
        )
        item = _s(attr=STR, quote=STR)
        schema = _s(must_have=_arr(item), must_not=_arr(item),
                    confusable_with=_arr(_s(name=STR, name_en=STR, culture=STR, why=STR)))
        d = self.llm.complete_json(system, f"Văn bản:\n\n{numbered}", schema)

        from ..stages.extraction import quote_in_texts

        raw_texts = [t.get("text", "") for t in texts]
        mh, mn, srcs, dropped = [], [], {}, []
        for key, out in (("must_have", mh), ("must_not", mn)):
            for it in d.get(key, []) or []:
                if not isinstance(it, dict) or not it.get("attr"):
                    continue
                attr, quote = str(it["attr"]).strip(), str(it.get("quote") or "").strip()
                if quote and quote_in_texts(quote, raw_texts):
                    out.append(attr)
                    srcs[attr] = quote
                else:
                    dropped.append(attr)
        cf = [{"name": c.get("name") or c.get("name_en"), "name_en": c.get("name_en") or c.get("name"),
               "culture": c.get("culture", ""), "why": c.get("why", "")}
              for c in d.get("confusable_with", []) or [] if isinstance(c, dict) and (c.get("name") or c.get("name_en"))]
        return {"must_have": mh, "must_not": mn, "confusable_with": cf, "attr_sources": srcs, "dropped_unsourced": dropped}

    # ------------------------------------------------------------ stage 3
    def build_spec(self, prompt, analysis, search, kb, max_entities, min_score) -> CulturalSpec:
        # Gộp / lọc / xếp hạng bằng luật để nhất quán; LLM chỉ dịch thuộc tính.
        spec = self._rule.build_spec(prompt, analysis, search, kb, max_entities, min_score)
        if not spec.entities:
            return spec
        # Lần chạy đầu: một lần gọi dịch cả bắt buộc lẫn cấm, model 3B trộn "wide obi at back"
        # (đặc điểm kimono, nằm trong CẤM) vào bản dịch của BẮT BUỘC -> prompt tích cực kéo ảnh
        # về kimono. Nay: hai lần gọi riêng, và bản dịch bắt buộc chứa tên confusable thì bị loại.
        # v1.2: KB có bản tiếng Anh viết tay (must_have_en) nên RuleAgent đã điền sẵn phần lớn; chỉ dịch
        # các cụm còn "" (thuộc tính rút thêm từ web, thực thể ad-hoc). Dịch thất bại -> cụm đó bị bỏ khỏi
        # prompt (giữ "" để không đưa tiếng Việt vào SDXL), có ghi chú.
        for field_vi, field_en, label in (("required_attrs", "required_attrs_en", "bắt buộc phải có"),
                                          ("forbidden_attrs", "forbidden_attrs_en", "KHÔNG được có")):
            payload = {}
            for e in spec.entities:
                vi = getattr(e, field_vi)
                en = list(getattr(e, field_en) or [])
                en += [""] * (len(vi) - len(en))
                setattr(e, field_en, en)
                todo = [a for a, b in zip(vi, en) if not b]
                if todo:
                    payload[e.entity_id] = {"name_en": e.name_en, "attrs": todo}
            if not payload:
                continue
            system = (
                f"Dịch các đặc điểm thị giác ({label}) sang cụm tiếng Anh NGẮN, 3-8 từ mỗi cụm, để ghép "
                "vào prompt cho model sinh ảnh. Giữ tên riêng Việt dạng không dấu (ao dai, non la, banh chung). "
                "Dịch sát nghĩa từng cụm, KHÔNG thêm ý, KHÔNG nhắc tới thứ của văn hoá khác, giữ đúng số lượng "
                "và thứ tự cụm."
            )
            schema = _s(entities=_arr(_s(entity_id=STR, attrs_en=_arr(STR))))
            try:
                d = self.llm.complete_json(system, json.dumps(payload, ensure_ascii=False, indent=1), schema)
            except RuntimeError as exc:
                spec.dropped.append(["-", f"dịch {label} thất bại: {exc}"])
                continue
            rows = {r.get("entity_id"): r for r in d.get("entities", []) if isinstance(r, dict)}
            if not rows:
                spec.dropped.append(["-", f"dịch {label}: model trả về 0 thực thể, giữ cụm chưa dịch ngoài prompt"])
            for eid, want in payload.items():
                se = spec.entity(eid)
                row = rows.get(eid)
                if se is None or row is None:
                    if se is not None:
                        spec.dropped.append(["-", f"dịch {label} của {se.name_vi}: không có kết quả"])
                    continue
                out = [str(x).strip() for x in row.get("attrs_en", [])]
                if len(out) != len(want["attrs"]):
                    spec.dropped.append(["-", f"dịch {label} của {se.name_vi}: số cụm lệch ({len(out)} vs {len(want['attrs'])}), bỏ"])
                    continue
                if field_en == "required_attrs_en":
                    bad = {t.lower() for c in se.confusables for t in shared.confusable_labels(c.get("name", ""))}
                    bad |= {"obi", "kimono", "qipao", "cheongsam", "hanbok", "hanfu", "sari", "sombrero"}
                    poisoned = [a for a in out if any(b and b in a.lower() for b in bad)]
                    if poisoned:
                        spec.dropped.append(["-", f"dịch {se.name_vi}: loại cụm nhiễm confusable {poisoned}"])
                        out = ["" if a in poisoned else a for a in out]
                # Điền vào đúng vị trí còn trống.
                en = list(getattr(se, field_en))
                vi = getattr(se, field_vi)
                it = iter(out)
                for i, (a, b) in enumerate(zip(vi, en)):
                    if not b and a in want["attrs"]:
                        en[i] = next(it, "")
                setattr(se, field_en, en)
        return spec

    # ------------------------------------------------------------ stage 5
    def critique(self, prompt: Prompt, spec: CulturalSpec, perception: Perception) -> Critique:
        # v1.1: điểm và findings suy ra bằng luật từ checklist mà perceiver đã hỏi VLM.
        # VLM không tự viết findings, không tự chấm điểm - hai việc nó làm kém nhất.
        det = shared.checklist_critique(spec, perception)
        if det is not None:
            return det
        return self._critique_freeform(prompt, spec, perception)

    def _critique_freeform(self, prompt: Prompt, spec: CulturalSpec, perception: Perception) -> Critique:
        """Đường cũ (v1): VLM viết findings tự do. Chỉ dùng khi không có checklist."""
        spec_json = json.dumps([{
            "entity_id": e.entity_id, "name_vi": e.name_vi, "name_en": e.name_en, "weight": e.weight,
            "must_have": e.required_attrs, "must_not": e.forbidden_attrs,
            "confusable_with": [c.get("name") for c in e.confusables],
        } for e in spec.entities], ensure_ascii=False, indent=1)
        obs = json.dumps([{"label": e.label, "attrs": e.attrs, "confidence": e.confidence}
                          for e in perception.elements], ensure_ascii=False, indent=1)
        system = (
            "Bạn là chuyên gia văn hoá vật chất Việt Nam, đang kiểm một ảnh do AI sinh.\n"
            "Nhìn ẢNH đính kèm. Với MỖI thực thể trong hợp đồng, xác định:\n"
            "  - Ảnh có vẽ đúng thực thể đó, hay vẽ một thứ tương tự của văn hoá khác "
            "(xem confusable_with), hay không vẽ?\n"
            "  - Các must_have nào có mặt, must_not nào xuất hiện?\n"
            "Phán quyết DỰA TRÊN must_have / must_not được cung cấp, không dựa vào cảm nhận riêng.\n"
            "Mức độ: critical = thay bằng thực thể văn hoá khác hoặc có must_not; "
            "major = vắng mặt hoặc thiếu quá nửa must_have; minor = thiếu lẻ tẻ.\n"
            "observed: ghi cái bạn THẤY (vd 'kimono có obi'), dùng đúng chữ 'không thấy' nếu vắng mặt, "
            "'thiếu' nếu chỉ thiếu thuộc tính. expected: chép nguyên thuộc tính hoặc tên thực thể.\n"
            "score 0..1 là mức đúng văn hoá tổng thể, có trọng số theo weight. "
            "verdict = pass chỉ khi không có critical và score >= 0.8.\n"
            "Một danh sách quan sát tự động được đính kèm để tham khảo; nếu nó mâu thuẫn với "
            "ảnh, tin vào ảnh."
        )
        user = (f"Prompt gốc: {prompt.text_vi}\n\nHợp đồng văn hoá:\n{spec_json}\n\n"
                f"Quan sát tự động (tham khảo):\n{obs}")
        schema = _s(findings=_arr(FINDING_SCHEMA), score=NUM,
                    verdict={"type": "string", "enum": ["pass", "revise"]}, reasoning=STR)
        d = self.llm.complete_json(system, user, schema, images=[perception.image_path])
        valid = {e.entity_id for e in spec.entities}
        findings = []
        for f in d.get("findings", []):
            if f.get("entity_id") not in valid or f.get("severity") not in SEVERITIES:
                continue
            findings.append(Finding(f["entity_id"], f["severity"], str(f.get("observed", "")),
                                    str(f.get("expected", "")), str(f.get("message", ""))))
        score = min(1.0, max(0.0, float(d.get("score", 0.0))))
        crit = any(f.severity == "critical" for f in findings)
        verdict = "revise" if (crit or score < 0.8) else "pass"
        return Critique(shared.PERSONA, findings, score, verdict, str(d.get("reasoning", "")))

    def adjudicate(self, critique, perception, spec, threshold, clip_weight, drift_margin) -> Adjudication:
        return shared.adjudicate(critique, perception, spec, threshold, clip_weight, drift_margin)

    def plan_revision(self, adjudication, spec, gen_spec, kb, lora_available, reference_available) -> RevisionPlan:
        return shared.plan_revision(adjudication, spec, gen_spec, kb, lora_available, reference_available)

    # ------------------------------------------------------------ stage 6
    def judge(self, prompt: Prompt, spec: CulturalSpec, perception: Perception) -> tuple[float, str]:
        if not spec.entities:
            return 0.0, "Không có thực thể để đánh giá."
        system = (
            "Bạn là trọng tài đánh giá độc lập. Nhìn ảnh và chấm ba trục, mỗi trục 0..1:\n"
            "  identity     - các thực thể yêu cầu có đúng là thứ được vẽ\n"
            "  completeness - các thuộc tính bắt buộc có đủ\n"
            "  purity       - ảnh KHÔNG lẫn dấu hiệu của văn hoá khác\n"
            "score = trung bình ba trục. reasoning ghi điểm từng trục và căn cứ ngắn gọn. "
            "Chỉ dựa vào yêu cầu được cung cấp."
        )
        user = f"Prompt: {prompt.text_vi}\n\nYêu cầu:\n" + json.dumps([{
            "name": f"{e.name_vi} / {e.name_en}", "must_have": e.required_attrs,
            "must_not": e.forbidden_attrs, "confusable_with": [c.get("name") for c in e.confusables],
        } for e in spec.entities], ensure_ascii=False, indent=1)
        schema = _s(identity=NUM, completeness=NUM, purity=NUM, score=NUM, reasoning=STR)
        d = self.llm.complete_json(system, user, schema, images=[perception.image_path])
        try:
            score = (float(d["identity"]) + float(d["completeness"]) + float(d["purity"])) / 3
        except (KeyError, ValueError, TypeError):
            score = float(d.get("score", 0.0))
        return min(1.0, max(0.0, score)), str(d.get("reasoning", ""))
