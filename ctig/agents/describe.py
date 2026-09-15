"""
Filter agent (v1.4): hai tầng, không hỏi có/không trên ảnh.

  tầng 1  VLM mô tả ảnh có cấu trúc (số người, chủ thể, trang phục, vật, nền)      -> ImageDescriptor
  tầng 2  so MÔ TẢ (văn bản) với must_have_en / must_not_en: agent văn bản trả present/absent/unsure
          kèm cụm trích từ mô tả; cụm trích không có trong mô tả -> hạ xuống unsure   -> FilterVerdict

Luật giữ/bỏ (rẻ, kiểm được, không phải LLM):
  reference: bỏ nếu (prompt một người mà ảnh >= 3 người) hoặc (không có must_have nào mà có must_not);
  candidate: bỏ nếu có must_not, hoặc sai số người khi prompt nói rõ một người.
  Luôn giữ ít nhất một ảnh (điểm cao nhất) để bước sau không rỗng.
"""

from __future__ import annotations

import re

from ..schema import CulturalSpec, FilterResult, FilterVerdict, ImageDescriptor

_SINGULAR = re.compile(r"\b(a|an|one|single)\s+(young\s+|old\s+|little\s+|elderly\s+)?(woman|girl|man|boy|person|lady|child|fisherman|farmer|vendor|monk|calligrapher|dancer|musician)\b", re.I)
_PLURAL = re.compile(r"\b(two|three|four|several|group|crowd|people|women|men|girls|boys|children|family|dancers|musicians|students|villagers|couple)\b", re.I)


def expected_people(prompt_en: str) -> int | None:
    """1 nếu prompt nói rõ một người và không có từ số nhiều; None nếu không ràng buộc."""
    if not prompt_en:
        return None
    if _PLURAL.search(prompt_en):
        return None
    return 1 if _SINGULAR.search(prompt_en) else None


def describe(agent, paths: list[str], log=print) -> list[ImageDescriptor]:
    out = []
    for p in paths:
        try:
            d = agent.describe_image(p)
        except Exception as exc:  # noqa: BLE001
            log(f"  [filter] mô tả {p}: {type(exc).__name__}: {str(exc)[:80]}")
            d = {}
        pc = d.get("people_count")
        try:
            pc = int(pc) if pc is not None else None
        except (TypeError, ValueError):
            pc = None
        out.append(ImageDescriptor(
            path=p, people_count=pc,
            subjects=[str(x) for x in d.get("subjects", []) if x][:6],
            garments=[str(x) for x in d.get("garments", []) if x][:8],
            objects=[str(x) for x in d.get("objects", []) if x][:8],
            background=str(d.get("background", ""))[:200],
            watermark_or_text=bool(d.get("watermark_or_text", False)),
        ))
    return out


def _garment_fields(desc: ImageDescriptor) -> list[dict]:
    """VLM hay trả mỗi garment là một dict (dạng chuỗi). Đọc lại thành dict; chuỗi thường thì gói vào {'text': ...}."""
    import ast

    out = []
    for g in desc.garments:
        d = None
        if isinstance(g, dict):
            d = g
        else:
            s = str(g).strip()
            if s.startswith("{"):
                try:
                    d = ast.literal_eval(s)
                except Exception:  # noqa: BLE001
                    d = None
        out.append({k: str(v).lower() for k, v in d.items()} if isinstance(d, dict) else {"text": s.lower()})
    return out


_NONE = {"", "none", "no", "null", "n/a", "not visible", "unknown"}


def garment_rules(desc: ImageDescriptor, attr: str) -> str | None:
    """Luật cứng trên các trường trang phục VLM đã mô tả (collar / lower_body / slits / sash_or_belt / type).
    Trả 'present' | 'absent' | None (không có luật -> để agent văn bản quyết). v1.4 p001: agent văn bản tính
    'high stand-up collar' là có trong khi mô tả ghi collar: crossed, lower_body: not visible."""
    a = attr.lower()
    gs = _garment_fields(desc)
    if not gs:
        return None

    def vals(field):
        return [g.get(field, "") for g in gs if g.get(field, "") not in _NONE]

    if "collar" in a:
        cs = vals("collar")
        if not cs:
            return None
        if any(k in a for k in ("stand", "mandarin", "high")):
            return "present" if any(("stand" in c or "mandarin" in c or "high" in c) for c in cs) else "absent"
        if "cross" in a or "y-shaped" in a or "v-neck" in a:
            return "present" if any(("cross" in c or "v" == c[:1] or "y" in c) for c in cs) else "absent"
    if "trouser" in a or "pants" in a or "bare legs" in a or "no trousers" in a:
        lb = vals("lower_body")
        if not lb:
            return None
        has_trousers = any(("trouser" in v or "pant" in v) for v in lb)
        bare = any(("bare" in v or "skirt" in v or "dress" in v or "gown" in v) for v in lb)
        if "no trousers" in a or "bare legs" in a or "without trousers" in a:
            return "present" if (bare and not has_trousers) else "absent"
        return "present" if has_trousers else ("absent" if bare else None)
    if "slit" in a or "split" in a:
        sl = vals("slits")
        if not sl:
            return None
        return "present" if any(v not in ("no", "none") for v in sl) else "absent"
    if "obi" in a or "sash" in a or "belt" in a:
        sb = vals("sash_or_belt")
        return "present" if sb and any(v not in ("no", "none") for v in sb) else "absent"
    if "one-piece" in a or "gown" in a or "floor-length" in a:
        ty = vals("type"); ln = vals("length"); lb = vals("lower_body")
        if any(("dress" in v or "gown" in v) for v in ty) and not any(("trouser" in v or "pant" in v) for v in lb):
            return "present"
        if any(("trouser" in v or "pant" in v) for v in lb):
            return "absent"
    return None


def compact_text(desc: ImageDescriptor, max_chars: int = 700) -> str:
    """Mô tả gọn cho agent văn bản: dict trang phục -> 'type=dress, fit=loose, ...'; cắt ở max_chars.
    (Mô tả thô có dict dài -> Qwen chép nguyên vào JSON -> vượt max_new_tokens -> không parse được.)"""
    parts = [f"people: {desc.people_count}"] if desc.people_count is not None else []
    if desc.subjects:
        parts.append("subjects: " + "; ".join(desc.subjects[:4]))
    for i, g in enumerate(_garment_fields(desc)[:3]):
        if "text" in g:
            parts.append(f"garment{i + 1}: {g['text'][:120]}")
        else:
            parts.append(f"garment{i + 1}: " + ", ".join(f"{k}={v}" for k, v in g.items() if v and v not in _NONE)[:160])
    if desc.objects:
        parts.append("objects: " + "; ".join(str(o)[:60] for o in desc.objects[:4]))
    if desc.background:
        parts.append("background: " + desc.background[:120])
    return " | ".join(parts)[:max_chars]


_PAIR_NOUNS = ("collar", "trousers", "pants", "sash", "obi", "skirt", "slit", "sleeve", "hat", "brim", "broth", "noodle")


def _counterpart(not_attr: str, have_attrs: list[str]) -> str | None:
    """must_have nói về cùng bộ phận với must_not (cổ áo, phần dưới, đai...) để CLIP so cặp."""
    a = not_attr.lower()
    nouns = [n for n in _PAIR_NOUNS if n in a]
    if "no trousers" in a or "bare legs" in a or "gown" in a or "one-piece" in a:
        nouns += ["trousers"]
    for h in have_attrs:
        hl = h.lower()
        if any(n in hl for n in nouns):
            return h
    return None


def clip_agrees_not(clip, path: str, name_en: str, not_attr: str, have_attr: str, margin: float = 0.60) -> bool | None:
    """CLIP so cặp trên chính ảnh: P('với must_not') so với P('với must_have'). True = CLIP cũng thấy must_not;
    False = CLIP nghiêng về must_have (VLM đọc sai); None = không kiểm được."""
    if clip is None or not hasattr(clip, "probs"):
        return None
    try:
        p = clip.probs(path, [f"a photo of a {name_en} with {have_attr}", f"a photo of a {name_en} with {not_attr}"])
        return p[1] >= (1.0 - margin)  # must_not phải chiếm >= 40% mới coi là CLIP đồng ý
    except Exception:  # noqa: BLE001
        return None


def _verdict(agent, desc: ImageDescriptor, spec: CulturalSpec, kind: str, n_people: int | None, clip=None) -> FilterVerdict:
    have_all: list[str] = []
    not_all: list[str] = []
    # v1.6 vast p050: prompt chỉ có thực thể bối cảnh (Tết) -> 0/0 thuộc tính, Filter mù. Thực thể context cũng có must_have
    # (cây quất, bao lì xì) / must_not (lồng đèn Trung Quốc) kiểm được trên mô tả objects/background -> đưa vào, vật thể trước.
    key_attrs: set[str] = set()
    # chỉ thực thể đủ trọng số (nêu tên hoặc suy ra chắc); thực thể đoán w<0.6 (C008: áo dài 0.55 trong prompt áo tứ thân)
    # không được áp must_have lên ảnh
    for se in sorted([s for s in spec.entities if s.weight >= 0.6] or spec.entities, key=lambda s: 0 if s.kind == "object" else 1):
        attrs = [a for a in se.required_attrs_en if a]
        have_all += attrs
        key_attrs.update(attrs[:2])  # KB viết tay xếp 2 thuộc tính ĐỊNH DANH (cổ đứng, hai tà...) lên đầu -> trọng số 2
        not_all += [a for a in se.forbidden_attrs_en if a]
    have_all, not_all = have_all[:8], not_all[:6]
    w = lambda a: 2.0 if a in key_attrs else 1.0
    matched_have, matched_not, reasons = [], [], []
    vqa: dict[str, float] = {}
    name_en = next((se.name_en.split("(")[0].strip() for se in spec.entities if se.kind == "object"), "outfit")
    if have_all or not_all:
        try:
            m = agent.match_descriptors(compact_text(desc), have_all, not_all)
        except Exception as exc:  # noqa: BLE001 - v1.5.1 p001: Qwen trả JSON quá dài bị cắt -> RuntimeError làm dừng cả bước
            m = {}
            reasons.append(f"agent văn bản lỗi ({type(exc).__name__}), chỉ dùng luật")
        matched_have = [a for a in m.get("present_must_have", []) if a in have_all]
        matched_not = [a for a in m.get("present_must_not", []) if a in not_all]
        # luật cứng trên trường trang phục ghi đè agent văn bản (cả hai chiều)
        for a in have_all:
            r = garment_rules(desc, a)
            if r == "present" and a not in matched_have:
                matched_have.append(a)
            elif r == "absent" and a in matched_have:
                matched_have.remove(a); reasons.append(f"luật: mô tả trái với '{a[:40]}'")
        for a in not_all:
            r = garment_rules(desc, a)
            if r == "present" and a not in matched_not:
                matched_not.append(a); reasons.append(f"luật: mô tả có '{a[:40]}'")
            elif r == "absent" and a in matched_not:
                matched_not.remove(a)
        # v1.5.1 p001: Qwen 3B ghi collar=crossed cho ảnh cổ đứng rõ -> 5/8 ảnh bị loại nhầm. must_not do VLM đọc ra
        # phải được CLIP xác nhận trên chính ảnh (so cặp với must_have cùng bộ phận); CLIP không đồng ý -> không tính.
        for a in list(matched_not):
            h = _counterpart(a, have_all)
            if h is None:
                continue
            ok = clip_agrees_not(clip, desc.path, name_en, a, h)
            if ok is False:
                matched_not.remove(a)
                if h not in matched_have and garment_rules(desc, h) != "absent":
                    pass  # không tự thêm must_have; chỉ gỡ must_not sai
                reasons.append(f"VLM nói '{a[:30]}' nhưng CLIP nghiêng về '{h[:30]}' -> bỏ")
        # v1.7.1: câu hỏi có/không trên ẢNH cho từng thuộc tính (Exposing Blindspots 2026: câu hỏi phủ định cho must_not).
        # Mô tả tự do hay bỏ sót chi tiết (cổ áo, đai) -> VQA là ý kiến thứ hai: >= 0.75 xác nhận, <= 0.25 bác.
        if hasattr(agent, "vqa_yes"):
            subj = name_en if any(se.kind == "object" for se in spec.entities) else "scene"
            for a in have_all + not_all:
                q = f"Look carefully. Does the {subj} in this photo have {a}? Answer Yes or No."
                pr = agent.vqa_yes(q, desc.path)
                if pr is None:
                    break
                vqa[a] = round(pr, 3)
            for a in have_all:
                pr = vqa.get(a)
                if pr is None:
                    continue
                if pr >= 0.75 and a not in matched_have and garment_rules(desc, a) != "absent":
                    matched_have.append(a); reasons.append(f"VQA xác nhận '{a[:30]}' ({pr:.2f})")
                elif pr <= 0.25 and a in matched_have and garment_rules(desc, a) != "present":
                    matched_have.remove(a); reasons.append(f"VQA bác '{a[:30]}' ({pr:.2f})")
            for a in not_all:
                pr = vqa.get(a)
                if pr is None:
                    continue
                if pr >= 0.75 and a not in matched_not:
                    h = _counterpart(a, have_all)
                    if h is None or clip_agrees_not(clip, desc.path, name_en, a, h) is not False:
                        matched_not.append(a); reasons.append(f"VQA thấy must_not '{a[:30]}' ({pr:.2f})")
                elif pr <= 0.25 and a in matched_not:
                    matched_not.remove(a); reasons.append(f"VQA bác must_not '{a[:30]}' ({pr:.2f})")
    missing = [a for a in have_all if a not in matched_have]
    tot = sum(w(a) for a in have_all)
    score = ((sum(w(a) for a in matched_have) - sum(w(a) for a in matched_not)) / tot) if tot else 0.0
    score = max(-1.0, min(1.0, score))
    keep = True
    if matched_not:
        reasons.append("có must_not: " + "; ".join(matched_not[:2]))
        if kind == "candidate" or not matched_have:
            keep = False
    if n_people == 1 and desc.people_count is not None and desc.people_count >= 3:
        reasons.append(f"prompt một người, ảnh {desc.people_count} người")
        keep = False
    if kind == "reference" and desc.watermark_or_text:
        reasons.append("có chữ/watermark")
    if not any(r.startswith(("có must_not", "prompt", "có chữ")) for r in reasons):
        reasons.append(f"{len(matched_have)}/{len(have_all)} must_have thấy trong mô tả")
    return FilterVerdict(path=desc.path, keep=keep, matched_must_have=matched_have, matched_must_not=matched_not,
                         missing_must_have=missing, people_count=desc.people_count, reasons=reasons, score=round(score, 3), vqa=vqa)


def run(agent, paths: list[str], spec: CulturalSpec, prompt_en: str, kind: str = "candidate", log=print, clip=None) -> FilterResult:
    n_people = expected_people(prompt_en)
    paths = list(dict.fromkeys(paths))  # hàng alias (+ref bị gate) chia sẻ đường dẫn -> không mô tả hai lần
    descs = describe(agent, paths, log=log)
    verdicts = [_verdict(agent, d, spec, kind, n_people, clip=clip) for d in descs]
    kept = [v.path for v in verdicts if v.keep]
    if not kept and verdicts:
        best = max(verdicts, key=lambda v: v.score)
        best.keep = True
        best.reasons.append("giữ lại vì là ảnh tốt nhất còn lại")
        kept = [best.path]
    log(f"  [filter:{kind}] giữ {len(kept)}/{len(paths)}" + (f", prompt nói rõ {n_people} người" if n_people else ""))
    return FilterResult(kind=kind, expected_people=n_people, verdicts=verdicts, kept=kept, descriptors=descs)
