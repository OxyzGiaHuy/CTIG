"""Stage 3 - Summary / Filter / Rank -> CulturalSpec."""

from __future__ import annotations

from ..kb import KnowledgeBase
from ..schema import AnalysisResult, CulturalSpec, Prompt, SearchResult


def run(agent, prompt: Prompt, analysis: AnalysisResult, search: SearchResult,
        kb: KnowledgeBase, max_entities: int, min_score: float) -> CulturalSpec:
    spec = agent.build_spec(prompt, analysis, search, kb, max_entities, min_score)
    keep, dropped = [], list(spec.dropped)
    for se in spec.entities:
        if se.required_attrs:
            keep.append(se)
        else:
            dropped.append([se.entity_id, "không có thuộc tính kiểm chứng được"])
    spec.entities, spec.dropped = keep, dropped
    sync_auto_entities(spec, kb)
    resolve_attr_conflicts(spec)
    return spec


def sync_auto_entities(spec: CulturalSpec, kb: KnowledgeBase) -> None:
    """v1.8: thực thể có bản KB tự sinh -> thuộc tính của SpecEntity lấy thẳng từ Entity (VI/EN/tags/kind/clip_label thẳng hàng),
    thay cho đường trộn item + dịch LLM (S012: dịch lệch số cụm -> EN rỗng -> Filter 0/0, mọi ảnh 'đạt')."""
    for se in spec.entities:
        ent = kb.get(se.entity_id)
        if ent is None or "KB tự sinh" not in (ent.notes or "") or not ent.must_have_en:
            continue
        se.required_attrs, se.required_attrs_en = list(ent.must_have), list(ent.must_have_en)
        se.forbidden_attrs, se.forbidden_attrs_en = list(ent.must_not), list(ent.must_not_en)
        se.confusables = list(ent.confusable_with)
        se.tags_en, se.neg_tags_en = list(ent.tags_en), list(ent.neg_tags_en)
        se.kind = ent.kind
        if ent.clip_label:
            se.clip_label = ent.clip_label
        spec.dropped.append([se.entity_id, f"thuộc tính lấy từ KB tự sinh ({len(se.required_attrs_en)} must_have, {len(se.forbidden_attrs_en)} must_not)"])


_STOP = {"with", "the", "and", "or", "of", "a", "an", "in", "on", "over", "under", "no", "not", "very", "hat", "dress", "long",
         "short", "small", "large", "wide", "flat", "round", "plain", "shape", "shaped", "style", "color", "colour", "worn", "made"}


def _words(s: str) -> set[str]:
    return {w for w in "".join(ch if ch.isalnum() else " " for ch in s.lower()).split() if len(w) >= 4 and w not in _STOP}


def resolve_attr_conflicts(spec: CulturalSpec) -> None:
    """v1.7.1 (p031): thuộc tính rút từ web cho nón lá thêm must_not 'conical shape' trong khi must_have KB là 'round conical hat'
    -> VQA thấy 'conical shape' >= 0.75 trên MỌI ảnh đúng và loại hết. Luật hẹp: một must_not / neg_tag bị bỏ khi TOÀN BỘ từ khoá
    của nó nằm trong MỘT must_have / tag của thực thể nào đó trong spec (nó chỉ nhắc lại điều phải có). Cặp đối lập cùng bộ phận
    ('one-piece dress with no trousers' so với 'over long trousers', 'crossed collar' so với 'stand-up collar') vẫn giữ vì có từ riêng."""
    positive: list[set[str]] = []
    for se in spec.entities:
        for a in list(se.required_attrs_en) + list(se.tags_en):
            ws = _words(a)
            if ws:
                positive.append(ws)

    def restated(a: str) -> bool:
        ws = _words(a)
        return bool(ws) and any(ws <= pos for pos in positive)

    for se in spec.entities:
        se.required_attrs_en = [a for a in se.required_attrs_en if a]
        kept_not, kept_tags = [], []
        for a in se.forbidden_attrs_en:
            if not a:
                continue
            if restated(a):
                spec.dropped.append([se.entity_id, f"bỏ must_not '{a[:40]}' vì chỉ nhắc lại must_have"])
            else:
                kept_not.append(a)
        for a in se.neg_tags_en:
            if not a:
                continue
            if restated(a):
                spec.dropped.append([se.entity_id, f"bỏ neg_tag '{a[:30]}' vì chỉ nhắc lại must_have"])
            else:
                kept_tags.append(a)
        se.forbidden_attrs_en, se.neg_tags_en = kept_not, kept_tags
