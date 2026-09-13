"""
Một vòng sửa (v1.4): ứng viên đầu sau Filter+Rank còn thiếu must_have hoặc còn must_not -> RevisionPlan -> sinh lại
trên model tốt nhất (cùng GenSpec + sửa, seed dịch sang vòng 1) -> lọc lại -> chọn ảnh cuối.

Khác vòng review v1: không hỏi VLM có/không, không cho LLM tự viết plan; plan suy từ FilterVerdict bằng luật
(thiếu gì thì thêm vào prompt, thấy must_not gì thì thêm vào negative), có giới hạn một vòng.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import CulturalSpec, FilterVerdict, GenSpec, RevisionPlan


def plan_from_verdict(v: FilterVerdict, spec: CulturalSpec, gen: GenSpec) -> RevisionPlan:
    pos = [a for a in v.missing_must_have[:2] if a and a not in gen.prompt_terms]
    neg = [a for a in v.matched_must_not[:2] if a and a not in gen.negative_terms]
    boost = {}
    # thuộc tính thiếu nhưng ĐÃ có trong prompt (attrs_in_prompt=3): không thêm lại được (trùng), thay bằng nhấn thực thể
    # lên đầu prompt + tăng guidance; thuộc tính thiếu chưa có trong prompt thì thêm thẳng.
    already = [a for a in v.missing_must_have if a in gen.prompt_terms]
    if v.matched_must_not or len(v.missing_must_have) >= 2:
        for se in spec.entities:
            if se.kind == "object" and se.weight >= 0.8:
                boost[se.entity_id] = 0.35
    if v.people_count is not None and v.people_count >= 3:
        pos.append("a single person, solo portrait")
        neg.append("group of people, crowd, multiple people")
    why = []
    if v.matched_must_not:
        why.append("ảnh đầu có must_not: " + "; ".join(v.matched_must_not[:2]))
    if v.missing_must_have:
        why.append("thiếu must_have: " + "; ".join(v.missing_must_have[:2]))
    if already and not pos:
        why.append("thuộc tính thiếu đã có trong prompt -> nhấn thực thể lên đầu, tăng guidance")
    return RevisionPlan(add_positive=pos, add_negative=neg, boost=boost,
                        guidance_delta=1.0 if (neg or already) else 0.0,
                        rationale="; ".join(why) or "không có gì để sửa")


def needs_revision(v: FilterVerdict | None) -> bool:
    return v is not None and (bool(v.matched_must_not) or len(v.missing_must_have) >= 2 or not v.keep)


def regenerate(gen: GenSpec, plan: RevisionPlan, spec: CulturalSpec, kb, model_key: str, cfg, out_dir: Path,
               clip=None, itm=None, prompt_en: str = "", log=print, aesthetic=None, ref_images=None):
    """Áp plan lên GenSpec, sinh lại MỘT model vào out_dir/revision. Trả ModelRun (có thể error)."""
    from dataclasses import replace

    from ..stages import multigen as mg
    from ..stages.generation import apply_plan

    g2 = apply_plan(gen, plan, spec, cfg.t2i)
    g2 = replace(g2, iteration=1, fast=False, steps=gen.steps, guidance=min(gen.guidance + plan.guidance_delta, 9.0))
    res = mg.run(g2, spec, kb, [model_key], cfg.multigen, Path(out_dir) / "revision", clip=clip, itm=itm,
                 prompt_en=prompt_en, log=log, ref_images=ref_images, aesthetic=aesthetic)
    return res.runs[0] if res.runs else None, g2
