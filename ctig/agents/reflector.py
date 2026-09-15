"""
Reflector (v1.7): đọc chẩn đoán của Reviewer, viết kế hoạch sửa cho Refiner, nhớ các vòng trước, quyết định dừng.

Ranh giới: Reviewer chỉ mô tả và so; Refiner chỉ thi hành. Reflector là nơi duy nhất quyết "sửa gì": phần luật
(plan_from_verdict) giữ là luật; LLM chỉ viết caption truy hồi cho thuộc tính thiếu (ImageRAG: caption > tên khái niệm)
và lời giải thích. Bộ nhớ (Idea2Img): cách sửa đã thử mà không tăng điểm thì không lặp lại, leo nấc khác.

Thang leo kênh khi bí: seed mới -> thêm ảnh tham chiếu theo thuộc tính -> tăng scale IP-Adapter -> tăng guidance.
"""

from __future__ import annotations

from dataclasses import replace

from ..schema import CulturalSpec, FilterVerdict, GenSpec, RevisionPlan
from .loop import needs_revision, plan_from_verdict

LADDER = ("attr_refs", "more_refs", "seed", "guidance")


def decide(v0: FilterVerdict | None, spec: CulturalSpec, gen: GenSpec, memory: list[dict], patience: int,
           agent=None, name_en: str = "", have_refs: bool = True, log=print) -> tuple[RevisionPlan | None, list[str], str]:
    """Trả (plan hoặc None nếu dừng, captions truy hồi, lý do). memory = [{'fix': str, 'improved': bool, 'score': float}]."""
    if v0 is None:
        return None, [], "không có chẩn đoán"
    if not needs_revision(v0):
        return None, [], "ứng viên đầu đạt: đủ thuộc tính, không must_not, đúng số người"
    recent = memory[-patience:] if patience > 0 else []
    if len(recent) >= patience and patience > 0 and not any(m.get("improved") for m in recent):
        return None, [], f"{patience} vòng liền không cải thiện -> dừng"

    plan = plan_from_verdict(v0, spec, gen)
    tried = {m.get("fix") for m in memory}
    fix = "attr_refs" if plan.use_reference_image and have_refs else ("negative" if plan.add_negative else "prompt")
    # leo nấc: cách sửa này đã thử mà không tăng -> đổi nấc
    if fix in tried and not any(m.get("improved") for m in memory if m.get("fix") == fix):
        nxt = None
        for step in LADDER:
            if step not in tried:
                nxt = step
                break
        if nxt is None:
            return None, [], "đã thử hết thang leo mà không cải thiện -> dừng"
        fix = nxt
        if fix == "more_refs":
            plan = replace(plan, use_reference_image=True, rationale=plan.rationale + "; leo nấc: thêm ảnh tham chiếu, scale +0.1")
        elif fix == "seed":
            plan = replace(plan, rationale=plan.rationale + "; leo nấc: seed mới, giữ prompt")
        elif fix == "guidance":
            plan = replace(plan, guidance_delta=max(plan.guidance_delta, 1.5), rationale=plan.rationale + "; leo nấc: guidance +1.5")
    plan = replace(plan, rationale=f"[{fix}] " + plan.rationale)

    captions: list[str] = []
    if plan.use_reference_image and v0.missing_must_have:
        captions = write_captions(agent, name_en, v0.missing_must_have[:3], log=log)
    return plan, captions, fix


def write_captions(agent, name_en: str, missing: list[str], log=print) -> list[str]:
    """Caption độc lập cho từng thuộc tính thiếu, để truy hồi ảnh (kho -> web). LLM viết; lỗi thì mẫu câu."""
    fallback = [f"close-up photo of a Vietnamese {name_en} showing {a}" for a in missing]
    if agent is None or not hasattr(agent, "write_retrieval_captions"):
        return fallback
    try:
        caps = agent.write_retrieval_captions(name_en, missing)
        caps = [str(c).strip() for c in caps if c and str(c).strip()]
        return caps[: len(missing)] or fallback
    except Exception as exc:  # noqa: BLE001
        log(f"  [reflector] LLM không viết được caption ({type(exc).__name__}) -> mẫu câu")
        return fallback
