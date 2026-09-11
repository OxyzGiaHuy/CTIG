"""Stage 1 - Analysis Agent: prompt -> keywords -> keywords mới -> entity ứng viên (kể cả thực thể chưa có trong KB).

v1.2: cap số ứng viên. Lần chạy v1.1, Qwen 3B trả về TOÀN BỘ 37 thực thể trong KB cho prompt
"mừng năm mới âm lịch", kéo theo truy hồi 37 thực thể (12 phút) và spec giữ áo bà ba làm chủ thể
chính trong khi Tết bị cắt. Giờ: ứng viên phải có keyword hoặc alias chống lưng, xếp theo confidence,
giữ tối đa `max_candidates`; thực thể bối cảnh (kind=context) khớp keyword luôn được giữ.
"""

from __future__ import annotations

from ..kb import KnowledgeBase, contains
from ..schema import AnalysisResult, Keyword, Prompt


def run(agent, prompt: Prompt, kb: KnowledgeBase, max_candidates: int = 6) -> AnalysisResult:
    res = agent.analyze(prompt, kb)
    known = set(kb.entities)
    unknown = [e for e in res.candidate_entity_ids if e not in known]
    if unknown:
        res.candidate_entity_ids = [e for e in res.candidate_entity_ids if e in known]
        note = f"Bỏ {len(unknown)} id không có trong KB: {unknown[:5]}"
        res.notes = f"{res.notes} | {note}" if res.notes else note

    text = f"{prompt.text_vi} {prompt.text_en}"
    # Bù thực thể được NÊU TÊN thẳng mà agent bỏ sót.
    for ent, score, term in kb.match_text(text):
        if ent.id not in res.candidate_entity_ids:
            res.candidate_entity_ids.append(ent.id)
            res.keywords.append(Keyword(ent.name_vi, "entity", "surface", score, f"KB alias '{term}' (bù)"))

    # Thực thể mới do agent đề xuất: đăng ký vào KB (bộ nhớ), bằng chứng sẽ do stage 2/2b dựng.
    for ne in res.new_entities:
        if kb.match_text(ne.name_vi):
            continue  # thực ra đã có trong KB dưới alias khác
        ent = kb.add_adhoc(ne.name_vi, ne.name_en, ne.category, ne.region)
        if ent.id not in res.candidate_entity_ids:
            res.candidate_entity_ids.append(ent.id)
            res.keywords.append(Keyword(ent.name_vi, "entity", "expanded", 0.6,
                                        f"thực thể mới, chưa có trong KB: {ne.rationale or ''}"))

    # --- Cap ứng viên: xếp theo (được nêu tên, confidence keyword), giữ tối đa max_candidates ---
    if len(res.candidate_entity_ids) > max_candidates:
        conf = _support(res, kb, text)
        ranked = sorted(res.candidate_entity_ids, key=lambda eid: -conf.get(eid, 0.0))
        dropped = ranked[max_candidates:]
        res.candidate_entity_ids = ranked[:max_candidates]
        note = (f"Agent trả {len(ranked)} ứng viên (nổ danh mục); giữ {max_candidates} có căn cứ nhất, "
                f"bỏ: {dropped[:8]}{'…' if len(dropped) > 8 else ''}")
        res.notes = f"{res.notes} | {note}" if res.notes else note
    return res


def _support(res: AnalysisResult, kb: KnowledgeBase, text: str) -> dict[str, float]:
    """Mức căn cứ của từng ứng viên: nêu tên thẳng > keyword expanded > không có gì."""
    by_name = {e.name_vi: e.id for e in kb.all()}
    by_name.update({e.name_en.split("(")[0].strip(): e.id for e in kb.all()})
    conf: dict[str, float] = {}
    for kw in res.keywords:
        if kw.kind != "entity":
            continue
        eid = by_name.get(kw.term) or by_name.get(kw.term.split("(")[0].strip())
        if eid is None:
            continue
        base = 1.0 if kw.source == "surface" else 0.5
        conf[eid] = max(conf.get(eid, 0.0), base + min(kw.confidence, 1.0))
    for eid in res.candidate_entity_ids:
        ent = kb.get(eid)
        if ent is None:
            continue
        # Nêu tên thẳng trong prompt luôn thắng.
        if any(contains(text, t) for t in ent.search_terms if len(t) >= 3):
            conf[eid] = max(conf.get(eid, 0.0), 2.5)
        # Thực thể bối cảnh (Tết, chợ nổi...) có keyword chống lưng thì giữ trước vật thể suy ra.
        if ent.kind == "context" and eid in conf:
            conf[eid] += 0.4
    return conf
