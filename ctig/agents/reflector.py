"""
Reflector (v1.7): đọc chẩn đoán của Reviewer, viết kế hoạch sửa cho Refiner, nhớ các vòng trước, quyết định dừng.

Ranh giới: Reviewer chỉ mô tả và so; Refiner chỉ thi hành. Reflector là nơi duy nhất quyết "sửa gì": phần luật
(plan_from_verdict) giữ là luật; LLM chỉ viết caption truy hồi cho thuộc tính thiếu (ImageRAG: caption > tên khái niệm)
và lời giải thích. Bộ nhớ (Idea2Img): cách sửa đã thử mà không tăng điểm thì không lặp lại, leo nấc khác.

Thang leo kênh: ảnh tham chiếu ĐÃ CHỌN Ở GROUNDING (ground_refs) -> ảnh truy hồi theo caption thuộc tính thiếu (attr_refs,
ImageRAG) -> thêm ảnh / scale IP-Adapter +0,1 (more_refs) -> seed mới -> guidance +1,5. Mỗi vòng đổi MỘT thứ.
"""

from __future__ import annotations

from dataclasses import replace

from ..schema import CulturalSpec, FilterVerdict, GenSpec, RevisionPlan
from .loop import needs_revision, plan_from_verdict

LADDER = ("ground_refs", "attr_refs", "rewrite", "more_refs", "seed", "guidance")


def decide(v0: FilterVerdict | None, spec: CulturalSpec, gen: GenSpec, memory: list[dict], patience: int,
           agent=None, name_en: str = "", have_refs: bool = True, log=print, prior_fixes: list[str] | None = None,
           image: str | None = None, facts: list[str] | None = None) -> tuple[RevisionPlan | None, list[str], str]:
    """Trả (plan hoặc None nếu dừng, captions truy hồi, lý do). memory = [{'fix': str, 'improved': bool, 'score': float}].
    prior_fixes: cách sửa đã TĂNG điểm cho cùng thực thể ở prompt trước (bộ nhớ liên prompt, GenEvolve-lite) -> thử trước.
    image/facts: cho nấc 'rewrite' (VLM nhìn ảnh lỗi, viết lại prompt chính)."""
    if v0 is None:
        return None, [], "không có chẩn đoán"
    if not needs_revision(v0):
        return None, [], "ứng viên đầu đạt: đủ thuộc tính, không must_not, đúng số người"
    recent = memory[-patience:] if patience > 0 else []
    if len(recent) >= patience and patience > 0 and not any(m.get("improved") for m in recent):
        return None, [], f"{patience} vòng liền không cải thiện -> dừng"

    plan = plan_from_verdict(v0, spec, gen)
    ladder = [s for s in LADDER if have_refs or s not in ("ground_refs", "attr_refs", "more_refs")]
    base_fix = "ground_refs" if plan.use_reference_image and have_refs else ("negative" if plan.add_negative else "prompt")
    if not memory:
        fix = base_fix
        for pf in prior_fixes or []:  # bộ nhớ liên prompt: nấc từng thành công cho thực thể này đi trước
            if pf in ladder and (pf not in ("ground_refs", "attr_refs", "more_refs") or have_refs):
                fix = pf
                break
    elif memory[-1].get("improved"):
        fix = memory[-1].get("fix") or base_fix       # cách vừa rồi có tăng -> giữ nguyên nấc, đổi seed (iteration)
    else:
        tried = {m.get("fix") for m in memory}
        fix = next((s for s in ladder if s not in tried), None)  # lần gần nhất không tăng -> nấc kế tiếp chưa thử
        if fix is None:
            return None, [], "đã thử hết thang leo mà không cải thiện -> dừng"
    if fix in ("ground_refs", "attr_refs"):
        note = "; ảnh tham chiếu Grounding" if fix == "ground_refs" else "; ảnh truy hồi theo caption thuộc tính thiếu"
        plan = replace(plan, use_reference_image=True, rationale=plan.rationale + note)
    elif fix == "rewrite":
        new_p = ""
        if agent is not None and hasattr(agent, "rewrite_prompt") and image:
            try:
                new_p = agent.rewrite_prompt(image, gen.prompt_terms[0] if gen.prompt_terms else "", name_en,
                                             list(v0.missing_must_have), list(v0.matched_must_not), list(facts or []))
            except Exception as exc:  # noqa: BLE001
                log(f"  [reflector] VLM không viết lại được prompt ({type(exc).__name__})")
        if not new_p:
            # không viết được -> coi nấc này đã thử, nhảy nấc kế
            memory = memory + [{"fix": "rewrite", "improved": False, "score": None}]
            return decide(v0, spec, gen, memory, patience, agent=agent, name_en=name_en, have_refs=have_refs, log=log,
                          prior_fixes=None, image=None, facts=facts)
        plan = replace(plan, rewrite_prompt=new_p, add_positive=[], rationale=plan.rationale + f"; viết lại prompt: {new_p[:80]}")
    elif fix == "more_refs":
        plan = replace(plan, use_reference_image=True, rationale=plan.rationale + "; leo nấc: thêm ảnh tham chiếu, scale +0.1")
    elif fix == "seed":
        plan = replace(plan, rationale=plan.rationale + "; leo nấc: seed mới, giữ prompt")
    elif fix == "guidance":
        plan = replace(plan, guidance_delta=max(plan.guidance_delta, 1.5), rationale=plan.rationale + "; leo nấc: guidance +1.5")
    plan = replace(plan, rationale=f"[{fix}] " + plan.rationale)

    captions: list[str] = []
    if fix in ("attr_refs", "more_refs") and v0.missing_must_have:
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
