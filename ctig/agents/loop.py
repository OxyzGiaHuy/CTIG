"""
Refiner (v1.7, trước là "một vòng sửa" v1.4): ứng viên đầu sau Filter+Rank còn thiếu must_have hoặc còn must_not -> RevisionPlan -> sinh lại
trên model tốt nhất (cùng GenSpec + sửa, seed dịch sang vòng 1) -> lọc lại -> chọn ảnh cuối.

Khác vòng review v1: không hỏi VLM có/không, không cho LLM tự viết plan; plan suy từ FilterVerdict bằng luật
(thiếu gì thì thêm vào prompt, thấy must_not gì thì thêm vào negative), có giới hạn một vòng.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import CulturalSpec, FilterVerdict, GenSpec, RevisionPlan


def _prompt_form(attr_en: str, spec: CulturalSpec, gen: GenSpec) -> str:
    """Dạng của must_have_en trong prompt hiện tại: render 'tags' dùng thẻ ngắn cùng vị trí trong KB, còn lại dùng nguyên câu."""
    if gen.render == "tags":
        for se in spec.entities:
            if attr_en in se.required_attrs_en:
                i = se.required_attrs_en.index(attr_en)
                if i < len(se.tags_en) and se.tags_en[i]:
                    return se.tags_en[i]
    return attr_en


def plan_from_verdict(v: FilterVerdict, spec: CulturalSpec, gen: GenSpec) -> RevisionPlan:
    forms = [_prompt_form(a, spec, gen) for a in v.missing_must_have[:3] if a]
    pos = [f for f in forms if f not in gen.prompt_terms][:2]
    neg = [a for a in v.matched_must_not[:2] if a and a not in gen.negative_terms]
    boost = {}
    # thuộc tính thiếu nhưng ĐÃ có trong prompt: không thêm lại được (trùng) -> nhấn bằng trọng số compel, đẩy thực thể lên
    # đầu prompt và tăng guidance; thuộc tính thiếu chưa có trong prompt thì thêm thẳng (dạng thẻ nếu render tags).
    already = [f for f in forms if f in gen.prompt_terms]
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
    weights = {a: 1.3 for a in already[:2]}
    if weights:
        why.append("nhấn compel ×1.3: " + "; ".join(a[:40] for a in weights))
    use_ref = len(v.missing_must_have) >= 1  # ImageRAG: model không tự vẽ được thuộc tính -> lần sinh lại kèm ảnh tham chiếu
    if use_ref:
        why.append("còn thuộc tính thiếu -> sinh lại KÈM ảnh tham chiếu (+ref)")
    return RevisionPlan(add_positive=pos, add_negative=neg, boost=boost, weights=weights, use_reference_image=use_ref,
                        guidance_delta=1.0 if (neg or already) else 0.0,
                        rationale="; ".join(why) or "không có gì để sửa")


def needs_revision(v: FilterVerdict | None) -> bool:
    """v1.7.1: 'đạt' = đủ MỌI must_have, không must_not, được giữ. Trước là thiếu >= 2 mới sửa -> p001 RealVis (+0.50, thiếu cổ
    đứng và tà bay) được coi là đạt dù ảnh còn sai."""
    return v is not None and (bool(v.matched_must_not) or len(v.missing_must_have) >= 1 or not v.keep)


def regenerate(gen: GenSpec, plan: RevisionPlan, spec: CulturalSpec, kb, model_key: str, cfg, out_dir: Path,
               clip=None, itm=None, prompt_en: str = "", log=print, aesthetic=None, ref_images=None,
               iteration: int = 1, ip_scale: float | None = None):
    """Refiner: áp plan lên GenSpec, sinh lại MỘT model vào out_dir/revision/iter<n>. Trả (ModelRun hoặc None, GenSpec).
    `iteration` dịch seed (seed + 1000·n) để mỗi vòng ra ảnh khác; `ip_scale` ghi đè scale IP-Adapter cho vòng này."""
    from dataclasses import replace

    from ..stages import multigen as mg
    from ..stages.generation import apply_plan

    from ..models.registry import get as get_model, parse_flags

    g2 = apply_plan(gen, plan, spec, cfg.t2i)
    g2 = replace(g2, iteration=iteration, fast=False, steps=gen.steps, guidance=min(gen.guidance + plan.guidance_delta, 9.0))
    try:
        base = get_model(model_key)
        fam, has_ip = base.family, base.ip_adapter
    except KeyError:
        fam, has_ip = "", False
    force = False
    if plan.use_reference_image and ref_images and fam in ("sdxl", "flux"):
        if "ref" not in parse_flags(model_key) and not has_ip:
            model_key = model_key + "+ref"
        force = True  # Filter đã nói model vẽ thiếu -> ảnh tham chiếu được phép bất kể auto_ref (ImageRAG: sinh trước, thiếu mới truy hồi)
        log(f"  [refiner] vòng {iteration}: sinh lại với ảnh tham chiếu: {model_key}")
    mcfg = cfg.multigen
    if ip_scale is not None:
        ov = dict(mcfg.overrides or {})
        ov[model_key] = {**ov.get(model_key, {}), "ip_scale": float(ip_scale)}
        mcfg = replace(mcfg, overrides=ov)
    sub = Path(out_dir) / "revision" / f"iter{iteration}"
    res = mg.run(g2, spec, kb, [model_key], mcfg, sub, clip=clip, itm=itm,
                 prompt_en=prompt_en, log=log, ref_images=ref_images, aesthetic=aesthetic, t2i_cfg=cfg.t2i, force_refs=force)
    return res.runs[0] if res.runs else None, g2
