"""Summary agent: tư liệu đã truy hồi về một thực thể -> CulturalBrief."""

from __future__ import annotations

from ..kb import KnowledgeBase
from ..schema import CulturalBrief, CulturalSpec, SearchResult


def run(agent, search: SearchResult, spec: CulturalSpec, kb: KnowledgeBase, max_sources: int = 6, log=print) -> dict[str, CulturalBrief]:
    """Mỗi thực thể trong spec: gom văn bản đã truy hồi (dài trước), gọi agent.summarize, kiểm câu gốc mờ."""
    from ..stages.extraction import quote_in_texts

    briefs: dict[str, CulturalBrief] = {}
    for se in spec.entities:
        texts = [it for it in search.items if it.entity_id == se.entity_id and it.kind in ("wiki_text", "web_text")
                 and it.snippet and len(it.snippet) > 80 and it.provenance != "extracted"]
        texts = sorted(texts, key=lambda t: -len(t.snippet))[:max_sources]
        ent = kb.get(se.entity_id)
        if not texts and ent is None:
            continue
        try:
            d = agent.summarize(se, ent, [{"title": t.title, "text": t.snippet} for t in texts])
        except Exception as exc:  # noqa: BLE001
            log(f"  [2c] summary {se.entity_id}: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        facts_vi = [str(x) for x in d.get("facts_vi", []) if x][:6]
        grounded = sum(1 for f in facts_vi if quote_in_texts(f, [t.snippet for t in texts])) if texts else 0
        b = CulturalBrief(
            entity_id=se.entity_id, facts_vi=facts_vi,
            facts_en=[str(x) for x in d.get("facts_en", []) if x][:6],
            confusions_en=[str(x) for x in d.get("confusions_en", []) if x][:4],
            depiction_en=str(d.get("depiction_en", "")).strip()[:300],
            sources=[t.title for t in texts], n_sources=len(texts), grounded=grounded,
        )
        briefs[se.entity_id] = b
        log(f"  [2c] {se.name_vi}: brief {len(b.facts_en)} facts EN, {b.grounded}/{len(facts_vi)} câu VI có gốc, {b.n_sources} nguồn")
    return briefs


def enrich_terms(briefs: dict[str, CulturalBrief], spec: CulturalSpec, max_words: int = 18) -> list[str]:
    """Cụm 'vẽ thế nào' để nối vào prompt khi agents.enrich_prompt bật; chỉ thực thể chính, ASCII, ngắn."""
    out = []
    for se in spec.entities:
        b = briefs.get(se.entity_id)
        if b and se.weight >= 0.8 and b.depiction_en and b.depiction_en.isascii() and len(b.depiction_en.split()) <= max_words:
            out.append(b.depiction_en.rstrip("."))
    return out
