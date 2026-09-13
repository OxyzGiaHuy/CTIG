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
        m = agent.match_descriptors(desc.text(), have_all, not_all)
        matched_have = [a for a in m.get("present_must_have", []) if a in have_all]
        matched_not = [a for a in m.get("present_must_not", []) if a in not_all]
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
    if not reasons:
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
