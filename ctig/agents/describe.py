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


def _verdict(agent, desc: ImageDescriptor, spec: CulturalSpec, kind: str, n_people: int | None) -> FilterVerdict:
    have_all: list[str] = []
    not_all: list[str] = []
    for se in spec.entities:
        if se.kind != "object":
            continue
        have_all += [a for a in se.required_attrs_en if a]
        not_all += [a for a in se.forbidden_attrs_en if a]
    have_all, not_all = have_all[:8], not_all[:6]
    matched_have, matched_not, reasons = [], [], []
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
    missing = [a for a in have_all if a not in matched_have]
    score = ((len(matched_have) - len(matched_not)) / len(have_all)) if have_all else 0.0
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
                         missing_must_have=missing, people_count=desc.people_count, reasons=reasons, score=round(score, 3))


def run(agent, paths: list[str], spec: CulturalSpec, prompt_en: str, kind: str = "candidate", log=print) -> FilterResult:
    n_people = expected_people(prompt_en)
    descs = describe(agent, paths, log=log)
    verdicts = [_verdict(agent, d, spec, kind, n_people) for d in descs]
    kept = [v.path for v in verdicts if v.keep]
    if not kept and verdicts:
        best = max(verdicts, key=lambda v: v.score)
        best.keep = True
        best.reasons.append("giữ lại vì là ảnh tốt nhất còn lại")
        kept = [best.path]
    log(f"  [filter:{kind}] giữ {len(kept)}/{len(paths)}" + (f", prompt nói rõ {n_people} người" if n_people else ""))
    return FilterResult(kind=kind, expected_people=n_people, verdicts=verdicts, kept=kept, descriptors=descs)
