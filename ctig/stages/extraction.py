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
        if it.kind in ("wiki_text", "web_text") and it.snippet and len(it.snippet) > 80 \
                and not it.provenance.startswith("kb.notes") and it.provenance != "extracted":
            by_entity.setdefault(it.entity_id, []).append(it)

    for eid, texts in by_entity.items():
        ent = kb.get(eid)
        if ent is None:
            continue
        # Ưu tiên nguồn dài (toàn văn trang) hơn snippet; giới hạn số nguồn để VLM 3B không loạn và không tốn 70s.
        texts = sorted(texts, key=lambda t: -len(t.snippet))[: getattr(cfg, "extract_max_sources", 6)]
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
        for d in extracted.get("dropped_unsourced", []) or []:
            search.notes.append(f"extract {eid}: bỏ '{str(d)[:60]}' vì không có câu gốc trong văn bản")
        extracted, junk = clean_extracted(extracted, ent)
        for j in junk[:6]:
            search.notes.append(f"extract {eid}: loại rác '{j[:60]}'")
        for c in extracted.get("confirms_kb", [])[:4]:
            search.notes.append(f"extract {eid}: '{c[:60]}' khớp KB viết tay (xác nhận, không thêm)")
        if not extracted.get("must_have"):
            search.notes.append(f"extract {eid}: không rút được must_have MỚI từ {len(texts)} nguồn"
                                + (f" ({len(extracted.get('confirms_kb', []))} câu xác nhận KB)" if extracted.get("confirms_kb") else ""))
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
        elif extracted.get("confusable_with"):
            known = {c["name"] for c in ent.confusable_with}
            ent.confusable_with += [c for c in extracted["confusable_with"] if c.get("name") not in known]
        log(f"  [2b] {ent.name_vi}: rút {len(extracted.get('must_have', []))} must_have, "
            f"{len(extracted.get('must_not', []))} must_not, {len(extracted.get('confusable_with', []))} confusable"
            + (" (cache)" if "_meta" in extracted and cache_file.exists() and extracted["_meta"].get("seconds", 1) == 0 else ""))
    return search


#: Từ loại mà model 3B hay chép lại từ hướng dẫn thay vì viết thuộc tính thật (v1.1: "Hình dạng",
#: "Chất liệu", "cách mặc/bày" chiếm ~40% must_have rút được).
_CATEGORY_WORDS = {
    "hình dạng", "chất liệu", "màu sắc", "màu", "cách mặc/bày", "cách mặc", "cách bày", "kích thước",
    "đặc điểm", "thể loại", "độ phức tạp", "sự kiện", "thời gian", "tên tiếng anh", "tên tiếng việt",
    "đồ chơi", "trang phục", "giao hưởng", "shape", "color", "material", "ingredients", "cooking method",
}
_VIET_MARKERS = ("việt", "viet")


def clean_extracted(extracted: dict, ent) -> tuple[dict, list[str]]:
    """Lọc rác trong kết quả rút: thuộc tính là tên loại, must_not trùng must_have, confusable là chính nó."""
    from ..kb import normalize, tokens

    junk: list[str] = []

    def ok_attr(a: str) -> bool:
        low = a.strip().lower()
        if low in _CATEGORY_WORDS or len(tokens(a)) < 2:
            return False
        if any(low.startswith(w) and len(low) <= len(w) + 3 for w in _CATEGORY_WORDS):
            return False
        return True

    kb_tok = [tokens(a) for a in (ent.must_have or [])]
    confirms: list[str] = []
    mh = [a for a in extracted.get("must_have", []) if isinstance(a, str)]
    keep_mh: list[str] = []
    for a in mh:
        if not ok_attr(a):
            junk.append(a); continue
        ta = tokens(a)
        # v1.2.1 p001: "thân áo xẻ làm hai tà" rút được chỉ nói lại KB rồi được dịch tệ ("cut into two parts")
        # và lọt vào spec. Thuộc tính trùng KB là XÁC NHẬN KB (đếm riêng), không phải thuộc tính mới.
        if any(ta and len(ta & kt) / len(ta) >= 0.5 for kt in kb_tok):
            confirms.append(a); continue
        # bỏ trùng bên trong must_have (v1.2 p001: 3 mục thì 2 mục y chữ, 1 mục diễn đạt lại)
        if any(ta and len(ta & tokens(b)) / len(ta) >= 0.7 for b in keep_mh):
            junk.append(f"{a} (trùng must_have khác)"); continue
        keep_mh.append(a)

    mh_tok = [tokens(a) for a in keep_mh]
    mn = [a for a in extracted.get("must_not", []) if isinstance(a, str)]
    keep_mn = []
    for a in mn:
        if not ok_attr(a):
            junk.append(a); continue
        ta = tokens(a)
        if any(ta and len(ta & t) / len(ta) >= 0.8 for t in mh_tok):
            junk.append(f"{a} (trùng must_have)"); continue
        keep_mn.append(a)

    ent_names = {normalize(ent.name_vi), normalize(ent.name_en.split("(")[0])}
    keep_cf = []
    for c in extracted.get("confusable_with", []) or []:
        if not isinstance(c, dict):
            continue
        name = normalize(c.get("name") or c.get("name_en") or "")
        culture = (c.get("culture") or "").lower()
        if not name or name in ent_names or any(n and (n in name or name in n) for n in ent_names):
            junk.append(f"confusable '{c.get('name')}' là chính thực thể"); continue
        if any(m in culture for m in _VIET_MARKERS):
            junk.append(f"confusable '{c.get('name')}' cùng văn hoá Việt"); continue
        keep_cf.append(c)

    srcs = {k: v for k, v in (extracted.get("attr_sources") or {}).items() if k in keep_mh or k in keep_mn}
    out = dict(extracted)
    out.update({"must_have": keep_mh, "must_not": keep_mn, "confusable_with": keep_cf, "attr_sources": srcs,
                "confirms_kb": confirms})
    return out, junk


def quote_in_texts(quote: str, texts: list[str]) -> bool:
    """Câu trích có thật trong văn bản không. So mờ theo token vì model hay sửa dấu câu."""
    from ..kb import tokens

    q = tokens(quote)
    if len(q) < 3:
        return False
    for t in texts:
        tt = tokens(t)
        if len(q & tt) / len(q) >= 0.7:
            return True
    return False
