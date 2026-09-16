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


def run(agent, search: SearchResult, kb: KnowledgeBase, cfg, cache_dir: Path, log=print, kb_auto_dir: Path | None = None) -> SearchResult:
    if not cfg.extract or not (hasattr(agent, "extract_evidence") or hasattr(agent, "draft_kb_entry")):
        return search
    cache_dir.mkdir(parents=True, exist_ok=True)
    kb_auto_dir = Path(kb_auto_dir) if kb_auto_dir else cache_dir.parent / "kb_auto"
    by_entity: dict[str, list[EvidenceItem]] = {}
    for it in search.items:
        if it.kind in ("wiki_text", "web_text") and it.snippet and len(it.snippet) > 80 \
                and not it.provenance.startswith("kb.notes") and it.provenance not in ("extracted", "kb_auto"):
            by_entity.setdefault(it.entity_id, []).append(it)

    for eid, texts in by_entity.items():
        ent = kb.get(eid)
        if ent is None:
            continue
        # Ưu tiên nguồn dài (toàn văn trang) hơn snippet; giới hạn số nguồn để VLM 3B không loạn và không tốn 70s.
        texts = sorted(texts, key=lambda t: (0 if "wikipedia" in (t.provenance or "") else 1, -len(t.snippet)))[: getattr(cfg, "extract_max_sources", 6)]  # vi + en Wikipedia trước
        # v1.8 KB tự sinh trong Grounding. kb_mode "auto": dựng cho MỌI thực thể từ nguồn truy hồi (KB tay chỉ là danh mục tên
        # và đường lùi); "hand": chỉ thực thể thiếu bản tay; "hand_only": không dựng.
        mode = getattr(cfg, "kb_mode", "auto") if getattr(cfg, "auto_kb", True) else "hand_only"
        want_draft = hasattr(agent, "draft_kb_entry") and (mode == "auto" or (mode == "hand" and not ent.must_have_en))
        if want_draft:
            if draft_kb(agent, ent, texts, kb_auto_dir, search, cfg=cfg, log=log):
                # thuộc tính TAY trên item KB không được trộn vào spec nữa (spec chỉ dùng bản tự dựng)
                for it in search.items:
                    # chỉ item KB TAY (kb@<version>, kb.notes); KHÔNG đụng item 'kb_auto' vừa thêm (S012 v1.8: xoá nhầm -> spec 0 thuộc tính)
                    if it.entity_id == eid and it.provenance.startswith("kb") and it.provenance != "kb_auto":
                        it.must_have, it.must_not, it.confusable_with = [], [], []
                search.notes.append(f"kb {eid}: nguồn thuộc tính = tự dựng (auto)")
                continue
            if ent.must_have_en:
                search.notes.append(f"kb {eid}: tự dựng không đủ -> dùng bản tay (hand)")
                continue
            thin = _LAST_THIN_DRAFT.pop(eid, None)
            if thin and thin.get("must_have"):
                # bản nháp mỏng vẫn có thuộc tính kèm câu gốc -> dùng như kết quả rút, KHÔNG gọi LLM lần hai trên cùng văn bản
                search.items.append(EvidenceItem(
                    entity_id=eid, kind="wiki_text", title=f"Rút từ văn bản: {ent.name_vi}",
                    snippet=f"Thuộc tính (bản nháp KB mỏng) từ {len(texts)} nguồn: " + "; ".join(t.title for t in texts),
                    must_have=list(thin["must_have"]), must_not=list(thin.get("must_not", [])),
                    confusable_with=list(thin.get("confusable_with", [])), url=texts[0].url, score=0.7, provenance="extracted",
                    attr_sources=dict(thin.get("attr_sources", {}))))
                if not ent.must_have:
                    ent.must_have, ent.must_not = list(thin["must_have"]), list(thin.get("must_not", []))
                    ent.confusable_with = list(thin.get("confusable_with", []))
                continue
        if not hasattr(agent, "extract_evidence"):
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


#: bản nháp KB mỏng của lần gọi vừa rồi (theo id) để đường rút cũ dùng lại thay vì gọi LLM lần hai; id đã lỗi để không thử lại
#: trong cùng tiến trình (3 lần retry JSON x prompt ~6k token mỗi lần).
_LAST_THIN_DRAFT: dict[str, dict] = {}
_FAILED_DRAFT_IDS: set[str] = set()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def draft_usable(d: dict | None) -> bool:
    """Bản ghi tự sinh dùng được khi có >= 2 must_have kèm câu gốc (cùng một ngưỡng cho lúc dựng và lúc nạp lại)."""
    return bool(d) and isinstance(d.get("must_have_en"), list) and len(d["must_have_en"]) >= 2


def hand_snapshot(ent) -> dict:
    """Bản tay hiện có của thực thể (để export so auto/tay); rỗng nếu chưa có."""
    return {"must_have_en": list(ent.must_have_en), "must_not_en": list(ent.must_not_en),
            "tags_en": list(ent.tags_en), "neg_tags_en": list(ent.neg_tags_en)}


def clean_draft(d: dict, ent) -> tuple[dict, list[str]]:
    """Cho bản nháp qua CÙNG bộ lọc rác của đường rút thuộc tính cũ (clean_extracted): bỏ thuộc tính là tên loại, must_not trùng
    must_have, confusable là chính thực thể hoặc cùng văn hoá Việt; giữ thẳng hàng VI/EN. Không so với bản tay (kb_mode auto)."""
    from types import SimpleNamespace

    # Lọc trên bản TIẾNG ANH: tokens() bỏ từ < 3 chữ nên cụm Việt ngắn ("hai tà xẻ") chỉ còn 1 token và bị coi là rác.
    blank = SimpleNamespace(name_vi=ent.name_vi, name_en=ent.name_en, must_have=[])
    en2vi_h = dict(zip(d.get("must_have_en", []), d.get("must_have", [])))
    en2vi_n = dict(zip(d.get("must_not_en", []), d.get("must_not", [])))
    cleaned, junk = clean_extracted({"must_have": list(en2vi_h), "must_not": list(en2vi_n),
                                     "confusable_with": list(d.get("confusable_with", []) or [])}, blank)
    out = dict(d)
    out["must_have_en"] = list(cleaned.get("must_have", []))
    out["must_have"] = [en2vi_h[a] for a in out["must_have_en"]]
    out["must_not_en"] = list(cleaned.get("must_not", []))
    out["must_not"] = [en2vi_n[a] for a in out["must_not_en"]]
    out["confusable_with"] = list(cleaned.get("confusable_with", []))
    out["attr_sources"] = {k: v for k, v in (d.get("attr_sources") or {}).items() if k in out["must_have"] or k in out["must_not"]}
    return out, junk


def apply_kb_draft(ent, d: dict) -> None:
    """Nạp bản ghi KB tự sinh vào Entity trong bộ nhớ (dùng cho cả lúc dựng mới và lúc nạp lại từ cache). Chuẩn hoá tại đây
    để file do người duyệt sửa tay cũng an toàn: kind chỉ nhận 'context'/'object', prior kẹp [0, 1]."""
    ent.must_have, ent.must_have_en = list(d.get("must_have", [])), list(d.get("must_have_en", []))
    ent.must_not, ent.must_not_en = list(d.get("must_not", [])), list(d.get("must_not_en", []))
    ent.confusable_with = [c for c in (d.get("confusable_with") or []) if isinstance(c, dict) and (c.get("name") or c.get("name_en"))]
    for c in ent.confusable_with:
        c.setdefault("name", c.get("name_en", "")); c.setdefault("name_en", c.get("name", "")); c.setdefault("culture", ""); c.setdefault("why", "")
    ent.tags_en = [str(x) for x in (d.get("tags_en") or []) if str(x).strip()][:5]
    ent.neg_tags_en = [str(x) for x in (d.get("neg_tags_en") or []) if str(x).strip()][:4]
    if d.get("clip_label"):
        ent.clip_label = str(d["clip_label"])
    ent.analogy_en = str(d.get("analogy_en") or "").strip()
    ent.kind = "context" if str(d.get("kind", "")).strip().lower() == "context" else "object"
    try:
        ent.prior_strength = max(0.0, min(1.0, float(d.get("prior_strength", 0.2))))
    except (TypeError, ValueError):
        ent.prior_strength = 0.2
    if "KB tự sinh" not in (ent.notes or ""):
        ent.notes = (ent.notes or "") + " | KB tự sinh (source=auto)"


def load_kb_draft(ent, kb_auto_dir: Path) -> bool:
    d = _read_json(Path(kb_auto_dir) / f"{ent.id}.json")
    if not draft_usable(d):
        return False
    apply_kb_draft(ent, d)
    return True


def rehydrate(search: SearchResult, kb: KnowledgeBase, kb_auto_dir: Path) -> None:
    """Sau khi đọc SearchResult từ cache (Session hoặc Pipeline): dựng lại trạng thái KB bộ nhớ mà lúc chạy thật là tác dụng phụ
    của extraction.run. Thực thể ad-hoc chưa có trong KB (Analysis cũng đọc từ đĩa) được đăng ký lại từ _meta của bản nháp."""
    for it in search.items:
        if it.provenance == "kb_auto":
            ent = kb.get(it.entity_id)
            d = _read_json(Path(kb_auto_dir) / f"{it.entity_id}.json")
            if ent is None:
                meta = (d or {}).get("_meta", {})
                name_vi = meta.get("name_vi") or it.title.split(":", 1)[-1].strip() or it.entity_id
                ent = kb.add_adhoc(name_vi, meta.get("name_en") or name_vi)
                if ent.id != it.entity_id:
                    kb.entities[it.entity_id] = ent
            if "KB tự sinh" in (ent.notes or ""):
                continue
            if draft_usable(d):
                apply_kb_draft(ent, d)
            else:  # file bị xoá/hỏng: ít nhất giữ thuộc tính ghi trên item
                ent.must_have, ent.must_not, ent.confusable_with = list(it.must_have), list(it.must_not), list(it.confusable_with)
        elif it.provenance == "extracted":
            ent = kb.get(it.entity_id)
            if ent is not None and not ent.must_have:
                ent.must_have, ent.must_not, ent.confusable_with = list(it.must_have), list(it.must_not), list(it.confusable_with)


def draft_kb(agent, ent, texts: list[EvidenceItem], kb_auto_dir: Path, search: SearchResult, cfg=None, log=print) -> bool:
    """Dựng bản ghi KB tự sinh cho `ent` từ văn bản đã truy hồi. Cache theo id (tôn trọng cfg.evidence_cache; bản mỏng không
    được cache để lần sau có nguồn tốt hơn thì dựng lại). Thêm EvidenceItem provenance 'kb_auto'. Trả True nếu dùng được."""
    kb_auto_dir = Path(kb_auto_dir)
    kb_auto_dir.mkdir(parents=True, exist_ok=True)
    f = kb_auto_dir / f"{ent.id}.json"
    use_cache = bool(getattr(cfg, "evidence_cache", True)) if cfg is not None else True
    d = _read_json(f) if use_cache else None
    if not draft_usable(d):
        if ent.id in _FAILED_DRAFT_IDS:
            search.notes.append(f"kb_auto {ent.id}: đã lỗi trước đó trong tiến trình này -> không thử lại")
            return False
        t0 = time.time()
        try:
            raw = agent.draft_kb_entry(ent, [{"title": t.title, "url": t.url, "text": t.snippet} for t in texts])
        except Exception as exc:  # noqa: BLE001
            _FAILED_DRAFT_IDS.add(ent.id)
            search.retrieval_errors.append(f"kb_auto {ent.id}: {type(exc).__name__}: {exc}")
            return False
        d, junk = clean_draft(raw, ent)
        for j in junk[:4]:
            search.notes.append(f"kb_auto {ent.id}: loại rác '{str(j)[:50]}'")
        d["_meta"] = {"entity_id": ent.id, "name_vi": ent.name_vi, "name_en": ent.name_en, "n_sources": len(texts),
                      "sources": [t.title for t in texts], "seconds": round(time.time() - t0, 1), "source": "auto",
                      "hand": hand_snapshot(ent)}
        for x in (d.get("dropped_unsourced") or [])[:4]:
            search.notes.append(f"kb_auto {ent.id}: bỏ '{str(x)[:50]}' vì không có câu gốc")
        if draft_usable(d) and use_cache:
            try:
                f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError as exc:
                search.notes.append(f"kb_auto {ent.id}: không ghi được cache ({exc})")
    if not draft_usable(d):
        search.notes.append(f"kb_auto {ent.id}: chỉ {len((d or {}).get('must_have_en', []))} must_have có câu gốc -> không dùng bản tự sinh")
        if d:
            _LAST_THIN_DRAFT[ent.id] = d
        return False
    apply_kb_draft(ent, d)
    meta = d.get("_meta", {})
    log(f"  [2b] {ent.name_vi}: KB tự sinh {len(ent.must_have_en)} must_have, {len(ent.must_not_en)} must_not, "
        f"{len(ent.tags_en)} tags, kind={ent.kind}, prior={ent.prior_strength:.2f} ({meta.get('n_sources', len(texts))} nguồn)")
    search.items.append(EvidenceItem(
        entity_id=ent.id, kind="wiki_text", title=f"KB tự sinh: {ent.name_vi}",
        snippet=f"Bản ghi KB do LLM dựng từ {len(texts)} nguồn: " + "; ".join(t.title for t in texts),
        must_have=list(ent.must_have), must_not=list(ent.must_not), confusable_with=list(ent.confusable_with),
        url=texts[0].url if texts else None, score=0.8, provenance="kb_auto", attr_sources=dict(d.get("attr_sources", {})),
    ))
    return True


def quote_in_texts(quote: str, texts: list[str], min_overlap: float = 0.7) -> bool:
    """Câu trích có thật trong văn bản không. So mờ theo token vì model hay sửa dấu câu. Câu ngắn toàn từ phổ biến ("Áo dài có màu
    trắng") dễ qua ở 0,7 -> bản nháp KB dùng 0,8 và yêu cầu >= 8 từ."""
    from ..kb import tokens

    q = tokens(quote)
    if len(q) < 3:
        return False
    for t in texts:
        tt = tokens(t)
        if len(q & tt) / len(q) >= min_overlap:
            return True
    return False
