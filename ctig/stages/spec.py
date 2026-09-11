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
    return spec
