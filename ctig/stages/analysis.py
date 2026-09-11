"""Stage 1 - Analysis Agent: prompt -> keywords -> keywords mới -> entity ứng viên."""

from __future__ import annotations

from ..kb import KnowledgeBase
from ..schema import AnalysisResult, Prompt


def run(agent, prompt: Prompt, kb: KnowledgeBase) -> AnalysisResult:
    res = agent.analyze(prompt, kb)
    known = set(kb.entities)
    unknown = [e for e in res.candidate_entity_ids if e not in known]
    if unknown:
        res.candidate_entity_ids = [e for e in res.candidate_entity_ids if e in known]
        note = f"Bỏ {len(unknown)} id không có trong KB: {unknown[:5]}"
        res.notes = f"{res.notes} | {note}" if res.notes else note
    # Nếu LLM bỏ sót thực thể được NÊU TÊN thẳng, bù bằng khớp alias trong KB.
    for ent, score, term in kb.match_text(f"{prompt.text_vi} {prompt.text_en}"):
        if ent.id not in res.candidate_entity_ids:
            res.candidate_entity_ids.append(ent.id)
            from ..schema import Keyword

            res.keywords.append(Keyword(ent.name_vi, "entity", "surface", score, f"KB alias '{term}' (bù)"))
    return res
