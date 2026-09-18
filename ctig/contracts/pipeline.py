"""Evidence-grounded prompt -> candidate Visual Contract pipeline.

The output is intentionally a *candidate*. It is never written into the runtime
contract database without a separate human approval step.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from .wikipedia import WikipediaPassage


STR = {"type": "string"}


def _obj(**properties) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _arr(item) -> dict:
    return {"type": "array", "items": item}


ENTITY_SCHEMA = _obj(
    name_vi=STR,
    name_en=STR,
    entity_type={"type": "string", "enum": ["object", "person", "place", "food", "event", "scene"]},
    prompt_specific_vi=_arr(STR),
    prompt_specific_en=_arr(STR),
    search_queries_vi=_arr(STR),
    search_queries_en=_arr(STR),
)

CITATION_SCHEMA = _obj(source_id=STR, quote=STR)
ATTRIBUTE_SCHEMA = _obj(
    id=STR,
    label_vi=STR,
    label_en=STR,
    description_vi=STR,
    description_en=STR,
    visual_evidence_vi=STR,
    visual_evidence_en=STR,
    part_vi=STR,
    part_en=STR,
    requirement_type={"type": "string", "enum": ["identity", "canonical_cue", "prompt_specific"]},
    visibility={"type": "string", "enum": ["must_be_visible", "check_if_visible", "optional"]},
    importance={"type": "integer"},
    citations=_arr(CITATION_SCHEMA),
)
CONFUSABLE_SCHEMA = _obj(
    id=STR,
    name_vi=STR,
    name_en=STR,
    culture=STR,
    difference_vi=STR,
    difference_en=STR,
    citations=_arr(CITATION_SCHEMA),
)
EXTRACTION_SCHEMA = _obj(required=_arr(ATTRIBUTE_SCHEMA), confusables=_arr(CONFUSABLE_SCHEMA))

OBS_ITEM_SCHEMA = _obj(
    attribute_id=STR,
    verdict={"type": "string", "enum": ["visible", "conditional", "not_visual"]},
    rationale_vi=STR,
    rationale_en=STR,
)
OBSERVABILITY_SCHEMA = _obj(items=_arr(OBS_ITEM_SCHEMA))

CHECK_ITEM_SCHEMA = _obj(
    attribute_id=STR,
    verdict={"type": "string", "enum": ["accept", "revise", "reject"]},
    comment_vi=STR,
    comment_en=STR,
)
CROSSCHECK_SCHEMA = _obj(
    overall={"type": "string", "enum": ["accept", "revise", "reject"]},
    items=_arr(CHECK_ITEM_SCHEMA),
    missing_visual_cues_vi=_arr(STR),
    missing_visual_cues_en=_arr(STR),
)

_NONVISUAL_VI = (
    "lịch sử", "nguồn gốc", "ý nghĩa", "biểu tượng", "được công nhận", "di sản",
    "nổi tiếng", "hương vị", "mùi vị", "niềm tin", "tượng trưng",
)
_NONVISUAL_EN = (
    "history", "origin", "meaning", "symbolizes", "heritage", "unesco",
    "famous", "flavor", "taste", "smell", "belief",
)


class ContractExtractionPipeline:
    def __init__(self, extractor, retriever, verifier=None, require_vi_evidence: bool = True):
        self.extractor = extractor
        self.retriever = retriever
        self.verifier = verifier
        self.require_vi_evidence = require_vi_evidence

    def run(self, prompt_id: str, prompt_vi: str, prompt_en: str) -> dict[str, Any]:
        entity = self._link_entity(prompt_vi, prompt_en)
        sources = self.retriever.retrieve(entity)
        raw = self._extract(prompt_vi, prompt_en, entity, sources)
        observable = self._classify_observability(raw.get("required", []))
        required, dropped = self._ground_and_filter(
            raw.get("required", []), observable, sources, prompt_vi, prompt_en
        )
        confusables, dropped_conf = self._ground_confusables(raw.get("confusables", []), sources)
        candidate = {
            "contract_version": "auto-1.0",
            "generator": {
                "entity_linker": getattr(self.extractor, "model_id", getattr(self.extractor, "name", "?")),
                "contract_extractor": getattr(self.extractor, "model_id", getattr(self.extractor, "name", "?")),
                "observability_filter": getattr(self.extractor, "model_id", getattr(self.extractor, "name", "?")),
            },
            "prompt_id": prompt_id,
            "prompt_text_vi": prompt_vi,
            "prompt_text_en": prompt_en,
            "main_entity": entity,
            "sources": [x.to_dict() for x in sources],
            "source_language_summary": {
                "vi": sum(x.lang == "vi" for x in sources),
                "en": sum(x.lang == "en" for x in sources),
                "vietnamese_evidence_required": self.require_vi_evidence,
            },
            "required": required,
            "confusables": confusables,
            "dropped": dropped + dropped_conf,
            "review": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        candidate["review"] = self._cross_check(candidate) if self.verifier else self._human_review(candidate)
        candidate["status"] = self._status(candidate)
        return candidate

    def _link_entity(self, prompt_vi: str, prompt_en: str) -> dict:
        system = (
            "Bạn là Entity Linker cho prompt sinh ảnh văn hóa Việt Nam. Chỉ xác định MỘT thực thể văn hóa chính. "
            "Không trích đặc điểm và không bổ sung kiến thức văn hóa. Giữ tên tiếng Việt có dấu. "
            "prompt_specific chỉ chứa màu sắc, người, hành động, số lượng hoặc bối cảnh được NÊU RÕ trong prompt. "
            "Tạo 1-2 truy vấn Wikipedia ngắn cho mỗi ngôn ngữ."
        )
        user = f"PROMPT VI:\n{prompt_vi}\n\nPROMPT EN:\n{prompt_en}"
        linked = self.extractor.complete_json(system, user, ENTITY_SCHEMA)
        linked["name_vi"] = " ".join(str(linked.get("name_vi") or "").split())
        linked["name_en"] = " ".join(str(linked.get("name_en") or "").split())
        if not linked["name_vi"] or not linked["name_en"]:
            raise ValueError("Entity linker did not return bilingual entity names")
        return linked

    def _source_block(self, sources: list[WikipediaPassage]) -> str:
        chunks = []
        for source in sources:
            language = "TIẾNG VIỆT" if source.lang == "vi" else "ENGLISH"
            chunks.append(
                f"[{source.source_id}] {language} | {source.title} | {source.url}\n{source.text[:7000]}"
            )
        return "\n\n".join(chunks)

    def _extract(self, prompt_vi: str, prompt_en: str, entity: dict,
                 sources: list[WikipediaPassage]) -> dict:
        system = (
            "Bạn là chuyên gia dựng Visual Contract có căn cứ. Đầu ra phải song ngữ Việt-Anh để chuyên gia Việt Nam kiểm tra.\n"
            "Chỉ dùng PROMPT và các nguồn được cung cấp; không dùng trí nhớ riêng. Mỗi identity/canonical_cue phải có ít nhất "
            "một citation chép NGUYÊN VĂN từ nguồn. Ưu tiên [VI*]; tiếng Anh chỉ bổ sung. prompt_specific có thể citation "
            "source_id=PROMPT và quote nguyên cụm trong prompt.\n"
            "required chỉ chứa thứ có thể nhìn thấy trong một ảnh: hình dạng, cấu tạo, màu, bộ phận, quan hệ không gian, "
            "cách mặc hoặc hành động. Không lấy lịch sử, ý nghĩa, hương vị, tên gọi, công dụng hay địa vị UNESCO.\n"
            "identity = cấu trúc quyết định thực thể; canonical_cue = đặc điểm nguyên mẫu hữu ích nhưng có biến thể; "
            "prompt_specific = chi tiết nêu thẳng trong prompt.\n"
            "description_vi/en mô tả chuẩn; visual_evidence_vi/en nói đúng thứ camera phải thấy. part_vi/en chỉ tên vùng trung tính "
            "để Observer nhìn vào, không được tiết lộ đáp án. importance là 1-3.\n"
            "Confusable phải là thứ thực sự dễ nhầm bằng mắt và difference phải nêu khác biệt quan sát được."
        )
        user = (
            f"ENTITY LINK:\n{json.dumps(entity, ensure_ascii=False)}\n\n"
            f"PROMPT VI:\n{prompt_vi}\nPROMPT EN:\n{prompt_en}\n\n"
            f"SOURCES (VI first):\n{self._source_block(sources)}"
        )
        return self.extractor.complete_json(system, user, EXTRACTION_SCHEMA)

    def _classify_observability(self, attributes: list[dict]) -> dict[str, dict]:
        if not attributes:
            return {}
        system = (
            "Bạn là bộ lọc khả năng quan sát. Với từng thuộc tính, chỉ hỏi: một người có thể kiểm chứng nó từ MỘT ảnh tĩnh không? "
            "visible = nhìn trực tiếp; conditional = chỉ kiểm được nếu bộ phận được lộ ra/đủ gần; not_visual = cần nếm, ngửi, "
            "đọc lịch sử, biết địa điểm, suy luận quan hệ hoặc nhìn xuyên vật kín. Không đánh giá tính đúng văn hóa. "
            "Giải thích bằng cả tiếng Việt và tiếng Anh."
        )
        compact = [{k: x.get(k) for k in (
            "id", "description_vi", "description_en", "visual_evidence_vi", "visual_evidence_en", "part_vi", "part_en"
        )} for x in attributes]
        result = self.extractor.complete_json(
            system, json.dumps(compact, ensure_ascii=False), OBSERVABILITY_SCHEMA
        )
        return {_slug(x.get("attribute_id")): x for x in result.get("items", []) if x.get("attribute_id")}

    def _ground_and_filter(self, attributes: list[dict], observations: dict[str, dict],
                           sources: list[WikipediaPassage], prompt_vi: str,
                           prompt_en: str) -> tuple[list[dict], list[dict]]:
        source_map = {x.source_id: x for x in sources}
        kept, dropped = [], []
        seen = set()
        for raw in attributes:
            item = dict(raw)
            item["id"] = _slug(item.get("id") or item.get("label_en") or item.get("label_vi"))
            if not item["id"] or item["id"] in seen:
                dropped.append({"id": item.get("id", ""), "reason": "empty_or_duplicate_id"})
                continue
            mandatory = ("label_vi", "label_en", "description_vi", "description_en",
                         "visual_evidence_vi", "visual_evidence_en", "part_vi", "part_en")
            missing = [key for key in mandatory if not str(item.get(key) or "").strip()]
            if missing:
                dropped.append({"id": item["id"], "reason": "missing_bilingual_fields", "fields": missing})
                continue
            seen.add(item["id"])
            citations, has_vi, bad = [], False, []
            for citation in item.get("citations") or []:
                sid, quote = str(citation.get("source_id") or ""), str(citation.get("quote") or "")
                if sid == "PROMPT":
                    if _quote_in_prompt(quote, prompt_vi) or _quote_in_prompt(quote, prompt_en):
                        citations.append({"source_id": sid, "quote": quote})
                    else:
                        bad.append(sid)
                    continue
                source = source_map.get(sid)
                if source and _quote_in(quote, source.text):
                    citations.append({"source_id": sid, "quote": quote})
                    has_vi |= source.lang == "vi"
                else:
                    bad.append(sid or "(empty)")
            req_type = item.get("requirement_type")
            if req_type in {"identity", "canonical_cue"} and not citations:
                dropped.append({"id": item["id"], "reason": "no_verbatim_source_quote", "bad_sources": bad})
                continue
            if self.require_vi_evidence and req_type in {"identity", "canonical_cue"} and not has_vi:
                dropped.append({"id": item["id"], "reason": "no_vietnamese_source_evidence"})
                continue
            evidence_vi = str(item.get("visual_evidence_vi") or "")
            evidence_en = str(item.get("visual_evidence_en") or "")
            obs = observations.get(item["id"], {})
            verdict = obs.get("verdict") or _rule_observability(evidence_vi, evidence_en)
            # Deterministic veto: an LLM cannot rescue explicitly non-visual wording.
            rule_verdict = _rule_observability(evidence_vi, evidence_en)
            if rule_verdict == "not_visual":
                verdict = "not_visual"
            if verdict == "not_visual":
                dropped.append({"id": item["id"], "reason": "not_observable_in_single_image",
                                "rationale_vi": obs.get("rationale_vi", ""),
                                "rationale_en": obs.get("rationale_en", "")})
                continue
            visibility = "check_if_visible" if verdict == "conditional" else str(item.get("visibility") or "must_be_visible")
            if visibility not in {"must_be_visible", "check_if_visible", "optional"}:
                visibility = "must_be_visible"
            item.update({
                "citations": citations,
                "visibility": visibility,
                "scoring": {"must_be_visible": "required", "check_if_visible": "conditional",
                            "optional": "excluded"}[visibility],
                "importance": _importance(item.get("importance")),
                "repairable": visibility != "optional",
                "observability": {
                    "verdict": verdict,
                    "rationale_vi": obs.get("rationale_vi", ""),
                    "rationale_en": obs.get("rationale_en", ""),
                },
            })
            kept.append(item)
        return kept, dropped

    def _ground_confusables(self, rows: list[dict], sources: list[WikipediaPassage]) -> tuple[list[dict], list[dict]]:
        source_map = {x.source_id: x for x in sources}
        kept, dropped = [], []
        for raw in rows:
            item = dict(raw)
            item["id"] = _slug(item.get("id") or item.get("name_en") or item.get("name_vi"))
            mandatory = ("name_vi", "name_en", "difference_vi", "difference_en")
            missing = [key for key in mandatory if not str(item.get(key) or "").strip()]
            if not item["id"] or missing:
                dropped.append({"id": item.get("id", ""), "reason": "missing_bilingual_fields",
                                "fields": missing})
                continue
            valid = []
            for citation in item.get("citations") or []:
                sid, quote = str(citation.get("source_id") or ""), str(citation.get("quote") or "")
                source = source_map.get(sid)
                if source and _quote_in(quote, source.text):
                    valid.append({"source_id": sid, "quote": quote})
            if not item["id"] or not valid:
                dropped.append({"id": item.get("id", ""), "reason": "confusable_without_source_quote"})
                continue
            item["citations"] = valid
            kept.append(item)
        return kept, dropped

    def _cross_check(self, candidate: dict) -> dict:
        # PhoGPT receives Vietnamese text first and is not allowed to silently edit.
        source_vi = [x for x in candidate["sources"] if x["lang"] == "vi"]
        system = (
            "Bạn là người kiểm tra hợp đồng thị giác bằng tiếng Việt. Chỉ đối chiếu candidate với prompt và nguồn tiếng Việt. "
            "Không thêm tri thức riêng. Đánh dấu accept/revise/reject; mọi gợi ý thiếu chỉ là đề nghị để con người xem, không tự merge."
        )
        payload = {"prompt_vi": candidate["prompt_text_vi"], "entity": candidate["main_entity"],
                   "required": candidate["required"], "confusables": candidate["confusables"],
                   "vietnamese_sources": source_vi}
        result = self.verifier.complete_json(system, json.dumps(payload, ensure_ascii=False), CROSSCHECK_SCHEMA)
        return {"mode": "model_cross_check", "model": getattr(self.verifier, "model_id", getattr(self.verifier, "name", "?")),
                **result, "auto_merged": False}

    def _human_review(self, candidate: dict) -> dict:
        return {
            "mode": "human_required",
            "overall": "pending",
            "auto_merged": False,
            "checklist_vi": [
                "Tên thực thể Việt/Anh có đúng prompt không?",
                "Mỗi identity/canonical cue có trích dẫn tiếng Việt đúng nguyên văn không?",
                "Visual evidence có thực sự nhìn thấy trong một ảnh tĩnh không?",
                "Có nhầm chi tiết của prompt thành định nghĩa phổ quát của thực thể không?",
                "Confusable có gần về thị giác và khác biệt có quan sát được không?",
            ],
        }

    @staticmethod
    def _status(candidate: dict) -> str:
        if not candidate.get("required"):
            return "rejected_no_grounded_attributes"
        review = candidate.get("review", {})
        if review.get("overall") == "accept":
            return "model_checked_pending_human_approval"
        if review.get("overall") in {"revise", "reject"}:
            return "needs_revision"
        return "pending_human_review"


def _slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:80]


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _quote_in(quote: str, text: str) -> bool:
    quote_n, text_n = _normalise(quote), _normalise(text)
    return len(quote_n.split()) >= 4 and quote_n in text_n


def _quote_in_prompt(quote: str, text: str) -> bool:
    quote_n, text_n = _normalise(quote), _normalise(text)
    return len(quote_n.split()) >= 2 and quote_n in text_n


def _rule_observability(vi: str, en: str) -> str:
    vi_n, en_n = _normalise(vi), _normalise(en)
    if any(term in vi_n for term in _NONVISUAL_VI) or any(term in en_n for term in _NONVISUAL_EN):
        return "not_visual"
    hidden = ("bên trong", "inside", "under the wrapper", "dưới lớp", "concealed", "hidden")
    if any(term in vi_n or term in en_n for term in hidden):
        return "conditional"
    return "visible"


def _importance(value: Any) -> int:
    try:
        return max(1, min(3, int(value)))
    except (TypeError, ValueError):
        return 2
