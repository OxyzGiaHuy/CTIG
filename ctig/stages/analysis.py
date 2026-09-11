"""Stage 1 - Analysis Agent: prompt -> keywords -> keywords mới -> entity ứng viên (kể cả thực thể chưa có trong KB)."""

from __future__ import annotations

from ..kb import KnowledgeBase
from ..schema import AnalysisResult, Keyword, Prompt


def run(agent, prompt: Prompt, kb: KnowledgeBase) -> AnalysisResult:
    res = agent.analyze(prompt, kb)
    known = set(kb.entities)
    unknown = [e for e in res.candidate_entity_ids if e not in known]
    if unknown:
        res.candidate_entity_ids = [e for e in res.candidate_entity_ids if e in known]
        note = f"Bỏ {len(unknown)} id không có trong KB: {unknown[:5]}"
        res.notes = f"{res.notes} | {note}" if res.notes else note
    # Bù thực thể được NÊU TÊN thẳng mà agent bỏ sót.
    for ent, score, term in kb.match_text(f"{prompt.text_vi} {prompt.text_en}"):
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
    return res
