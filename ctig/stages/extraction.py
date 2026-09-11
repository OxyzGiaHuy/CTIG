"""
Stage 2b - RÚT BẰNG CHỨNG từ văn bản truy hồi được (chạy lúc runtime).

Đây là bước làm cho Search thực sự TẠO RA bằng chứng thay vì chỉ xác nhận KB tay:

    văn bản Wikipedia / web  ->  VLM rút must_have, must_not, confusable_with  ->  EvidenceItem
                                 mỗi thuộc tính kèm trích đoạn gốc (attr_sources)

Quy tắc cho model: chỉ thuộc tính THỊ GIÁC kiểm chứng được bằng mắt, chỉ lấy từ văn bản,
không bịa. Không đủ văn bản thì trả ít, không trả bừa.

Cache theo entity_id (không theo prompt): cùng một thực thể xuất hiện ở nhiều prompt chỉ rút
một lần. Xoá <cache_dir>/evidence/<id>.json để rút lại. Với thực thể KB gốc, bằng chứng rút
được ĐI KÈM KB tay, nên bạn so được máy rút ra khớp bao nhiêu với người viết.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..kb import KnowledgeBase
from ..schema import EvidenceItem, SearchResult, to_dict


def run(agent, search: SearchResult, kb: KnowledgeBase, cfg, cache_dir: Path, log=print) -> SearchResult:
    if not cfg.extract or not hasattr(agent, "extract_evidence"):
        return search
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_entity: dict[str, list[EvidenceItem]] = {}
    for it in search.items:
        if it.kind == "wiki_text" and it.snippet and len(it.snippet) > 80:
            by_entity.setdefault(it.entity_id, []).append(it)

    for eid, texts in by_entity.items():
        ent = kb.get(eid)
        if ent is None:
            continue
        cache_file = cache_dir / f"{eid}.json"
        extracted = None
        if cfg.evidence_cache and cache_file.exists():
            try:
                extracted = json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                extracted = None
        if extracted is None:
            t0 = time.time()
            try:
                extracted = agent.extract_evidence(ent, [{"title": t.title, "url": t.url, "text": t.snippet} for t in texts])
            except Exception as exc:  # noqa: BLE001
                search.retrieval_errors.append(f"extract {eid}: {type(exc).__name__}: {exc}")
                continue
            extracted["_meta"] = {"entity_id": eid, "n_sources": len(texts), "seconds": round(time.time() - t0, 1)}
            if cfg.evidence_cache:
                cache_file.write_text(json.dumps(extracted, ensure_ascii=False, indent=1), encoding="utf-8")
        if not extracted.get("must_have"):
            continue
        srcs = "; ".join(t.title for t in texts)
        search.items.append(EvidenceItem(
            entity_id=eid, kind="wiki_text", title=f"Rút từ văn bản: {ent.name_vi}",
            snippet=f"Thuộc tính do VLM rút từ {len(texts)} nguồn: {srcs}",
            must_have=list(extracted.get("must_have", [])), must_not=list(extracted.get("must_not", [])),
            confusable_with=list(extracted.get("confusable_with", [])),
            url=texts[0].url, score=0.75, provenance="extracted",
            attr_sources=dict(extracted.get("attr_sources", {})),
        ))
        # Thực thể ad-hoc: nạp thuộc tính vào KB trong bộ nhớ để các stage sau (CLIP probe, plan) dùng.
        if not ent.must_have:
            ent.must_have = list(extracted.get("must_have", []))
            ent.must_not = list(extracted.get("must_not", []))
            ent.confusable_with = list(extracted.get("confusable_with", []))
        log(f"  [2b] {ent.name_vi}: rút {len(extracted.get('must_have', []))} must_have, "
            f"{len(extracted.get('must_not', []))} must_not, {len(extracted.get('confusable_with', []))} confusable"
            + (" (cache)" if "_meta" in extracted and cache_file.exists() and extracted["_meta"].get("seconds", 1) == 0 else ""))
    return search
