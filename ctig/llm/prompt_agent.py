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


_GENERIC_EN = {"traditional", "beautiful", "famous", "dress", "clothing", "clothes", "costume", "vietnamese", "vietnam", "item",
               "culture", "cultural", "style", "typical", "popular", "common", "national", "outfit", "garment"}
_COLORS_EN = {"white", "red", "black", "blue", "green", "yellow", "pink", "purple", "brown", "gold", "golden", "silver", "orange", "grey", "gray"}
#: từ cho thấy thuộc tính KHÔNG phải đặc điểm nhìn thấy (v1.8 áo dài: "one of the few Vietnamese words that appear in English dictionaries")
_NONVISUAL_EN = {"dung", "resin", "tar", "varnish", "coating", "coated", "sealed", "sealant", "glue", "treated", "waterproofed",
                 "word", "words", "dictionary", "dictionaries", "language", "history", "historical", "century", "centuries", "origin",
                 "originated", "named", "called", "name", "symbol", "symbolizes", "meaning", "means", "popular", "popularity", "famous",
                 "price", "cost", "festival-goers", "believed", "considered", "known", "unesco", "heritage", "year", "years", "dynasty",
                 "emperor", "king", "designer", "designed", "introduced", "invented"}
_EXAMPLE_STRINGS = {"round conical shape with pointed tip", "silk chin strap under the chin", "wide flat brim", "very wide flat brim with no point",
                    "open lattice bamboo weave", "smooth pale palm-leaf surface over bamboo rings", "silk or cloth chin strap"}


_VAGUE_EN = ("either", " or ", "may be", "can be", "sometimes", "usually", "often", "various", "different", "depending")


def _attr_ok_en(attr_en: str) -> bool:
    """Thuộc tính tiếng Anh phải cụ thể: >= 3 từ, không chỉ là màu, không toàn từ chung ('Is white', 'Vietnamese traditional dress')."""
    ws = [w.strip(".,;:()").lower() for w in attr_en.split() if w.strip(".,;:()")]
    if len(ws) < 3:
        return False
    content = [w for w in ws if w not in {"is", "has", "with", "a", "an", "the", "of", "and", "or", "in", "on"}]
    if not content or all(w in _COLORS_EN for w in content):
        return False
    if all(w in _GENERIC_EN or w in _COLORS_EN for w in content):
        return False
    if any(w in _NONVISUAL_EN for w in content):
        return False
    if attr_en.strip().lower() in _EXAMPLE_STRINGS:
        return False  # model chép ví dụ định dạng
    low2 = " " + attr_en.lower() + " "
    if any(v in low2 for v in _VAGUE_EN):
        return False  # "either loose or reaching past the wrist" -> VQA trả lời không nhất quán
    return True


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
    def draft_kb_entry(self, ent, texts: list[dict], ref_images: list[str] | None = None) -> dict:  # noqa: C901
        """v1.8.3 KB tự sinh — làm ĐÚNG cách người viết KB tay, thay vì trích dẫn Wikipedia:
          1. đọc nguồn, chép ra các câu MÔ TẢ HÌNH DÁNG (tư liệu, không phải để trích dẫn);
          2. NHÌN 2-3 ẢNH THẬT của thực thể + tư liệu -> VIẾT cụm ngắn 3-6 từ như bản tay ("high stand-up mandarin collar"),
             hai cụm đầu là đặc điểm ĐỊNH DANH phân biệt với vật gần giống nhất, must_not viết TỪ vật gần giống đó;
          3. (ở extraction) kiểm lại bằng chính ảnh thật -> thay cho ràng buộc "phải có câu gốc" vốn ép model chép câu dài.
        Không có ảnh thì vẫn chạy bằng tư liệu."""
        from ..stages.extraction import quote_in_texts

        name = f"'{ent.name_vi}' ({ent.name_en})"
        short = ent.name_en.split("(")[0].strip()
        # ---- bước 1: tư liệu hình dáng ----
        sentences: list[str] = []
        for i, tx in enumerate(texts[:4]):
            body = (tx.get("text") or "")[:3000]
            if len(body) < 80:
                continue
            sys1 = (f"You read a text about the Vietnamese cultural item {name}. COPY VERBATIM up to 6 sentences that describe how it "
                    "LOOKS or is WORN/USED/PLACED: shape, structure, parts, material, pattern, size, how it differs from similar items. "
                    "Each at least 8 words. Skip history, origin, meaning, prices, names of people. Return {\"sentences\": [...]}.")
            try:
                d1 = self._complete(sys1, f"TEXT [{i}] {tx.get('title', '')}:\n{body}", _s(sentences=_arr(STR)), max_new_tokens=700)
            except Exception:  # noqa: BLE001
                continue
            for s_ in d1.get("sentences", []) or []:
                s_ = str(s_).strip()
                if len(s_.split()) >= 8 and quote_in_texts(s_, [body], min_overlap=0.8) and s_ not in sentences:
                    sentences.append(s_)
        out = {"must_have": [], "must_have_en": [], "must_not": [], "must_not_en": [], "attr_sources": {}, "dropped_unsourced": [],
               "confusable_with": [], "tags_en": [], "neg_tags_en": [], "clip_label": "", "kind": "object", "prior_strength": 0.2,
               "n_sentences": len(sentences), "saw_images": len(ref_images or [])}
        imgs = list(ref_images or [])[:3]
        if not sentences and not imgs:
            return out
        # ---- bước 2: viết bản ghi như người viết KB tay (nhìn ảnh + tư liệu) ----
        src_txt = ("REFERENCE SENTENCES:\n" + "\n".join(f"- {s_}" for s_ in sentences[:12])) if sentences else "(no text sources)"
        sys2 = (
            f"You are writing a compact visual knowledge record about {name} for a system that generates and checks images. "
            + ("You are shown REAL PHOTOGRAPHS of this item; describe what you actually see in them, using the sentences only as "
               "background. " if imgs else "Use the sentences below. ")
            + "Write like a domain expert filling a checklist, NOT like an encyclopedia.\n"
            "must_have: 4-6 items {attr_vi, attr_en, salience}. attr_en = a SHORT noun phrase of 3-6 words naming one visible feature "
            "(shape, part, structure, material, how it is worn or placed), e.g. 'high stand-up mandarin collar', 'round basket-shaped "
            "bamboo hull', 'wide flat brim with silk tassels'. No full sentences, no clauses with 'that/which', no 'either/or', no "
            "history, no colors alone. The FIRST TWO must be IDENTIFYING: what separates it from the most similar item of another "
            "culture or region. salience = 1-5 (5 = visible at a glance from a few meters).\n"
            "confusable_with: the 1-3 items most likely mistaken for it {name, name_en, culture, why}.\n"
            "must_not: 2-4 items {attr_vi, attr_en}: a visible feature OF THAT CONFUSABLE ITEM whose presence means the image is wrong "
            "(e.g. for a Vietnamese ao dai: 'diagonal Y-shaped crossed collar' from the Chinese qipao). Must not repeat must_have words.\n"
            "tags_en: 3-5 prompt tags (2-4 words), identifying tag first. neg_tags_en: 2-4 short negative tags without must_have nouns.\n"
            f"analogy_en: one phrase (5-12 words) comparing {short} to a familiar object an image model knows, stating the key difference.\n"
            "clip_label: 'a photo of …'. kind: 'object' or 'context'. prior_strength: 0-1 (how well a generic text-to-image model already "
            "draws it; ao dai ~0.55, coracle boat ~0.1)."
        )
        item = _s(attr_vi=STR, attr_en=STR, salience={"type": "integer"})
        schema = _s(must_have=_arr(item), must_not=_arr(_s(attr_vi=STR, attr_en=STR)),
                    confusable_with=_arr(_s(name=STR, name_en=STR, culture=STR, why=STR)),
                    tags_en=_arr(STR), neg_tags_en=_arr(STR), clip_label=STR, analogy_en=STR,
                    kind={"type": "string", "enum": ["object", "context"]}, prior_strength=NUM)
        try:
            d = self.llm.complete_json(sys2, src_txt, schema, images=imgs or None, max_new_tokens=1100)
        except TypeError:
            d = self.llm.complete_json(sys2, src_txt, schema, images=imgs or None)
        for key in ("must_have", "must_not"):
            seen: set[str] = set()
            items = [it for it in (d.get(key, []) or []) if isinstance(it, dict) and it.get("attr_en")]
            if key == "must_have":
                def _sal(it):
                    try:
                        return int(it.get("salience", 3))
                    except (TypeError, ValueError):
                        return 3
                items.sort(key=lambda it: -_sal(it))
                good = [it for it in items if _sal(it) >= 3]
                items = good if len(good) >= 2 else items
            for it in items:
                en = " ".join(str(it["attr_en"]).split()).strip(" .")
                vi = " ".join(str(it.get("attr_vi") or en).split()).strip(" .")
                if en.lower() in seen or not _attr_ok_en(en) or len(en.split()) > 8:
                    continue  # bản tay không bao giờ dài quá 8 từ
                if key == "must_not" and any(w in {x.lower() for x in " ".join(out["must_have_en"]).split()} for w in en.lower().split() if len(w) > 4):
                    continue
                seen.add(en.lower())
                out[key].append(vi); out[key + "_en"].append(en)
        cfs = []
        for c in d.get("confusable_with") or []:
            if isinstance(c, dict) and (c.get("name") or c.get("name_en")):
                cfs.append({"name": str(c.get("name") or c.get("name_en")), "name_en": str(c.get("name_en") or c.get("name")),
                            "culture": str(c.get("culture") or ""), "why": str(c.get("why") or "")})
        out["confusable_with"] = cfs

        def _tags(xs, drop_example=False):
            flat = []
            for x in xs or []:
                flat += [y.strip(" .;") for y in str(x).split(",")]
            outp = []
            for y in flat:
                if not y or len(y.split()) > 5 or y.lower().startswith("no "):
                    continue
                if all(w.lower() in _GENERIC_EN for w in y.split()):
                    continue
                if drop_example and (y.lower() in _EXAMPLE_STRINGS or ("conical" in y.lower() and "nón" not in ent.name_vi.lower())):
                    continue
                if y.lower() not in {z.lower() for z in outp}:
                    outp.append(y)
            return outp
        out["tags_en"] = _tags(d.get("tags_en"))[:5]
        out["neg_tags_en"] = _tags(d.get("neg_tags_en"), drop_example=True)[:4]
        out["clip_label"] = str(d.get("clip_label") or "").strip()
        an = " ".join(str(d.get("analogy_en") or "").split()).strip(" .")
        out["analogy_en"] = an if 4 <= len(an.split()) <= 14 and not any(w in _NONVISUAL_EN for w in an.lower().split()) else ""
        kind = "context" if str(d.get("kind", "")).strip().lower() == "context" else "object"
        low = ent.name_vi.lower()
        if any(low.startswith(w) for w in ("tết", "lễ", "hội", "chợ", "múa", "hát", "đám")) or "festival" in ent.name_en.lower():
            kind = "context"
        out["kind"] = kind
        try:
            out["prior_strength"] = max(0.0, min(1.0, float(d.get("prior_strength", 0.2))))
        except (TypeError, ValueError):
            out["prior_strength"] = 0.2
        return out

    def _complete(self, system: str, user: str, schema: dict, max_new_tokens: int | None = None) -> dict:
        try:
            return self.llm.complete_json(system, user, schema, max_new_tokens=max_new_tokens)
        except TypeError:  # backend không nhận max_new_tokens
            return self.llm.complete_json(system, user, schema)

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
            # Qwen 3B hay trả {"ao_dai": {"attrs_en": [...]}} thay vì {"entities": [{"entity_id": ..}]}; nhận cả hai.
            rows = {r.get("entity_id"): r for r in d.get("entities", []) if isinstance(r, dict) and r.get("entity_id")}
            if not rows:
                rows = {k: v for k, v in d.items() if isinstance(v, dict) and "attrs_en" in v and k in payload}
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
    # ------------------------------------------------------------ v1.4 agents: Summary / Filter / Rank
    def summarize(self, se, ent, texts: list[dict]) -> dict:
        """Summary agent: tư liệu -> facts thị giác VI/EN, khác gì với confusable, một câu 'vẽ thế nào'."""
        corpus = "\n\n".join(f"[{i}] {t['title']}\n{t['text'][:2500]}" for i, t in enumerate(texts)) or "(không có văn bản)"
        kb_hint = ""
        if ent is not None:
            kb_hint = ("\nTri thức viết tay (chỉ để đối chiếu, KHÔNG chép lại nếu văn bản không nói): "
                       + "; ".join(ent.must_have[:4]))
        system = (
            "Bạn là người tóm tắt tư liệu văn hoá cho hệ thống sinh ảnh. Chỉ dùng thông tin CÓ TRONG văn bản; "
            "không dùng tri thức riêng. Chỉ ghi đặc điểm THỊ GIÁC nhìn thấy được trong ảnh (hình dạng, chất liệu, "
            "cách mặc/bày, màu, bối cảnh), bỏ lịch sử và ý nghĩa. Mỗi fact một câu ngắn.\n"
            "facts_vi: 3-6 câu tiếng Việt, càng gần chữ trong văn bản càng tốt. facts_en: dịch tương ứng.\n"
            "confusions_en: 1-3 câu tiếng Anh nói thực thể này KHÁC gì so với thứ dễ nhầm (kimono, qipao, hanbok, zongzi...) "
            "nếu văn bản có nói; không thì để rỗng.\n"
            "depiction_en: MỘT câu tiếng Anh <= 18 từ mô tả cách vẽ đúng thực thể, dùng từ trong facts_en.\n"
            "dimensions: xếp các facts_en vào 4 chiều (CULTIVate): attire (trang phục trên người), objects (vật thể, món ăn, "
            "nhạc cụ), background (bối cảnh, kiến trúc, cảnh quan), interactions (hành động, cách dùng). Chiều nào không có thì rỗng."
        )
        user = f"Thực thể: {se.name_vi} / {se.name_en}{kb_hint}\n\nVăn bản:\n{corpus}"
        schema = _s(facts_vi=_arr(STR), facts_en=_arr(STR), confusions_en=_arr(STR), depiction_en=STR,
                    dimensions=_s(attire=_arr(STR), objects=_arr(STR), background=_arr(STR), interactions=_arr(STR)))
        return self.llm.complete_json(system, user, schema)

    def describe_image(self, path: str) -> dict:
        """Filter agent tầng 1: MÔ TẢ ảnh có cấu trúc, tiếng Anh, không phán đoán văn hoá."""
        system = (
            "Describe the attached image for an automatic checker. Do NOT name any culture, country or garment tradition; "
            "describe only what is visible.\n"
            "people_count: number of people visible (0 if none).\n"
            "subjects: main subjects, e.g. 'young woman standing', 'wooden boat'.\n"
            "garments: one string per garment or outfit worn by the main person, each covering: type (dress/tunic/shirt/robe), "
            "fit (fitted/loose), length (knee/ankle/floor), collar (stand-up/round/crossed/v-neck/none), sleeves, "
            "lower body (trousers/skirt/bare legs/not visible), any sash or belt, slits, patterns, color.\n"
            "objects: notable objects (hat, boat, food, instrument...) with shape and material.\n"
            "background: one sentence.\n"
            "watermark_or_text: true if the image contains visible text, logo or watermark."
        )
        schema = _s(people_count=NUM, subjects=_arr(STR), garments=_arr(STR), objects=_arr(STR), background=STR,
                    watermark_or_text={"type": "boolean"})
        # v1.6 vast: 5 mô tả bị cắt ở 700 token (garments dict dài) -> cho riêng bước này 1400 token, và yêu cầu ngắn.
        return self.llm.complete_json(system, "Describe the image. Keep every string short (<= 12 words); at most 3 garments and 4 objects.",
                                      schema, images=[path], max_new_tokens=1400)

    def match_descriptors(self, description: str, must_have_en: list[str], must_not_en: list[str]) -> dict:
        """Filter agent tầng 2 (văn bản): thuộc tính nào được mô tả nói rõ là có. Cụm trích phải nằm trong mô tả."""
        from ..stages.extraction import quote_in_texts

        system = (
            "You compare an IMAGE DESCRIPTION with a list of visual attributes. For each attribute decide: "
            "'present' only if the description explicitly states it (paraphrase allowed), 'absent' if the description "
            "states something contradicting it, otherwise 'unsure'. Never infer from culture knowledge; use the description only.\n"
            "OUTPUT RULES: attr = copy the attribute text EXACTLY as listed (do not rewrite, do not copy description text into attr); "
            "quote = at most 10 words copied from the description, empty string when not 'present'. One item per listed attribute, "
            "no extra items. Keep the whole answer short."
        )
        user = ("DESCRIPTION:\n" + description + "\n\nMUST_HAVE attributes:\n" + "\n".join(f"- {a}" for a in must_have_en)
                + "\n\nMUST_NOT attributes:\n" + "\n".join(f"- {a}" for a in must_not_en))
        item = _s(attr=STR, status={"type": "string", "enum": ["present", "absent", "unsure"]}, quote=STR)
        schema = _s(must_have=_arr(item), must_not=_arr(item))
        d = self.llm.complete_json(system, user, schema)
        out = {"present_must_have": [], "present_must_not": [], "unsure": []}
        for key, pool, dst in (("must_have", must_have_en, "present_must_have"), ("must_not", must_not_en, "present_must_not")):
            for it in d.get(key, []) or []:
                if not isinstance(it, dict):
                    continue
                attr = self._closest(str(it.get("attr", "")), pool)
                if attr is None:
                    continue
                if it.get("status") == "present" and quote_in_texts(str(it.get("quote", "")), [description]):
                    out[dst].append(attr)
                elif it.get("status") == "present":
                    out["unsure"].append(attr)  # nói 'present' nhưng câu trích không có trong mô tả -> không tin
        return out

    @staticmethod
    def _closest(name: str, pool: list[str]) -> str | None:
        from ..kb import tokens

        if name in pool:
            return name
        tn = tokens(name)
        best, best_s = None, 0.0
        for a in pool:
            ta = tokens(a)
            s = len(tn & ta) / max(1, len(tn | ta))
            if s > best_s:
                best, best_s = a, s
        return best if best_s >= 0.5 else None

    def rank_candidates(self, prompt_en: str, brief_txt: str, items: list[dict]) -> dict:
        """Rank agent (văn bản): xếp các ứng viên từ mô tả + must_have/must_not đã khớp + brief."""
        system = (
            "You rank AI-generated image candidates for cultural correctness against a reference brief. You only see text "
            "descriptions. Rank by: (1) no must_not_seen, (2) more must_have_seen, (3) description matches the prompt "
            "(number of people, scene), (4) metric_score as tie-breaker. Return order = list of ids best first, and one short "
            "reason per id."
        )
        user = f"PROMPT: {prompt_en}\n\nREFERENCE BRIEF:\n{brief_txt}\n\nCANDIDATES:\n" + json.dumps(items, ensure_ascii=False, indent=1)
        schema = _s(order=_arr(STR), reasons={"type": "object"})
        return self.llm.complete_json(system, user, schema)

    def vqa_yes(self, question: str, image: str) -> float | None:
        """P(Yes) cho một câu hỏi có/không trên ảnh (VQAScore). None nếu backend không hỗ trợ (API text-only, RuleAgent)."""
        fn = getattr(self.llm, "yes_prob", None)
        if fn is None:
            return None
        try:
            return float(fn(question, [image]))
        except Exception:  # noqa: BLE001
            return None

    def rewrite_prompt(self, image: str, prompt_en: str, name_en: str, missing: list[str], wrong: list[str], facts: list[str]) -> str:
        """Reflector kiểu Idea2Img: nhìn ảnh lỗi, biết thiếu gì / sai gì, viết lại câu prompt chính (<= 60 từ) giữ cảnh gốc,
        mô tả đúng chỗ sai bằng từ ngữ mà model sinh ảnh hiểu (hình dáng, vị trí, chất liệu), không dùng từ phủ định."""
        system = (
            "You improve a text-to-image prompt after seeing the image it produced. Keep the original scene, subject and setting. "
            "Rewrite ONE English prompt (max 60 words, one paragraph, no lists, no negations like 'no' or 'without') that makes the "
            "image model draw the missing features and avoid the wrong ones by describing the correct appearance concretely "
            "(shape, position on the body/object, material, how it is worn or placed). Do not add new objects. "
            "Return {\"prompt\": \"...\"}."
        )
        user = (f"ORIGINAL PROMPT: {prompt_en}\nITEM: {name_en}\nREFERENCE FACTS: " + "; ".join(facts[:4]) +
                f"\nMISSING IN IMAGE: " + "; ".join(missing[:3]) + f"\nWRONG IN IMAGE: " + "; ".join(wrong[:2]))
        d = self.llm.complete_json(system, user, _s(prompt=STR), images=[image])
        s_ = " ".join(str(d.get("prompt", "")).split())
        return s_ if 8 <= len(s_.split()) <= 80 else ""

    def write_retrieval_captions(self, name_en: str, missing: list[str]) -> list[str]:
        """Reflector (ImageRAG): một caption ảnh độc lập cho MỖI thuộc tính thiếu, để truy hồi ảnh minh hoạ thuộc tính đó."""
        system = (
            "You write short English image-search captions. For each missing visual attribute of a cultural item, write ONE "
            "caption (8-16 words) describing a real photograph that clearly shows that attribute on that item: close-up, "
            "plain background, no people unless the attribute is about wearing. No adjectives like beautiful. Return captions "
            "in the same order as the attributes."
        )
        user = f"ITEM: {name_en}\nMISSING ATTRIBUTES:\n" + "\n".join(f"- {a}" for a in missing)
        d = self.llm.complete_json(system, user, _s(captions=_arr(STR)))
        return [str(c) for c in d.get("captions", [])]

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
