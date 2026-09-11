"""
Logic hoà giải và lập bản sửa - DETERMINISTIC, dùng chung cho mọi agent.

Vì sao không để LLM tự hoà giải và tự lập bản sửa? Vì đây là chỗ cần tính nhất
quán hơn tính sáng tạo: cùng một bộ lỗi phải luôn cho cùng một hành động, nếu
không bạn không thể so hai lần chạy với nhau. LLM chỉ được dùng ở chỗ cần hiểu
ngôn ngữ (phân tích prompt, đọc ảnh, dịch thuộc tính).

"Debate" trong v1 là VLM đối chiếu với CLIP: hai tín hiệu độc lập về cùng một
câu hỏi "ảnh vẽ áo dài hay kimono". Bất đồng giữa chúng được ghi lại.
"""

from __future__ import annotations

from ..kb import KnowledgeBase, attr_covered, normalize, tokens
from ..schema import (
    Adjudication,
    Critique,
    CulturalSpec,
    Finding,
    GenSpec,
    Perception,
    RevisionPlan,
    SpecEntity,
    VisualElement,
)


def confusable_labels(name: str) -> list[str]:
    """'qipao / cheongsam (Trung Quốc)' -> ['qipao', 'cheongsam']."""
    base = name.split("(")[0]
    return [p.strip() for p in base.split("/") if p.strip()]


def locate(se: SpecEntity, elements: list[VisualElement]) -> tuple[VisualElement | None, str]:
    """Tìm thực thể trong danh sách quan sát: 'match' | 'drift' | 'missing'."""
    targets = {normalize(se.name_vi), normalize(se.name_en.split("(")[0])}
    for el in elements:
        lab = normalize(el.label)
        if lab in targets or any(t and (t in lab or lab in t) for t in targets if len(t) > 3):
            return el, "match"
    for el in elements:
        lab = normalize(el.label)
        for cf in se.confusables:
            for v in confusable_labels(cf.get("name", "")):
                nv = normalize(v)
                if nv and (nv in lab or lab in nv):
                    return el, "drift"
    return None, "missing"


def clip_target_prob(se: SpecEntity, perception: Perception) -> float | None:
    probs = perception.clip_probs.get(se.entity_id)
    if not probs:
        return None
    return probs.get("__target__")


def clip_drift(se: SpecEntity, perception: Perception, margin: float) -> tuple[bool, str | None, float]:
    """CLIP có nghiêng về một confusable hơn mục tiêu quá `margin` không."""
    probs = perception.clip_probs.get(se.entity_id)
    if not probs:
        return False, None, 0.0
    target = probs.get("__target__", 0.0)
    others = {k: v for k, v in probs.items() if k != "__target__"}
    if not others:
        return False, None, target
    best = max(others, key=others.get)
    return (others[best] - target) > margin, best, target


def adjudicate(
    critique: Critique, perception: Perception, spec: CulturalSpec,
    threshold: float, clip_weight: float, drift_margin: float,
) -> Adjudication:
    findings = list(critique.findings)
    disagreements: list[str] = []
    clip_scores: list[tuple[float, float]] = []  # (weight, prob)

    for se in spec.entities:
        drift, best, target = clip_drift(se, perception, drift_margin)
        if clip_target_prob(se, perception) is not None:
            clip_scores.append((se.weight, target))
        vlm_says_drift = any(
            f.entity_id == se.entity_id and f.severity == "critical" for f in critique.findings
        )
        if drift:
            findings.append(Finding(
                entity_id=se.entity_id, severity="critical",
                observed=f"CLIP nghiêng về '{best}' ({perception.clip_probs[se.entity_id][best]:.2f})",
                expected=f"{se.name_vi} ({target:.2f})",
                message=(f"CLIP cho rằng ảnh giống '{best}' hơn '{se.name_vi}'. "
                         f"Tín hiệu này độc lập với VLM."),
                evidence_ref="clip_probe",
            ))
            if not vlm_says_drift:
                disagreements.append(
                    f"{se.name_vi}: VLM không báo lệch nhưng CLIP nghiêng về '{best}'. "
                    f"Giữ phán quyết lệch vì thà sửa thừa còn hơn bỏ sót lỗi văn hoá."
                )
        elif vlm_says_drift and clip_target_prob(se, perception) is not None and target > 0.6:
            disagreements.append(
                f"{se.name_vi}: VLM báo lệch nhưng CLIP tin là đúng ({target:.2f}). "
                f"Giữ phán quyết của VLM vì nó đọc được chi tiết mà CLIP không đọc."
            )

    if clip_scores:
        clip_score = sum(w * p for w, p in clip_scores) / sum(w for w, _ in clip_scores)
        score = (1 - clip_weight) * critique.score + clip_weight * clip_score
        reasoning = (f"điểm = {1 - clip_weight:.2f}*VLM({critique.score:.2f}) + "
                     f"{clip_weight:.2f}*CLIP({clip_score:.2f}) = {score:.2f}")
    else:
        score = critique.score
        reasoning = f"điểm = VLM({critique.score:.2f}), không có CLIP"

    order = {"critical": 0, "major": 1, "minor": 2}
    findings.sort(key=lambda f: order[f.severity])
    has_critical = any(f.severity == "critical" for f in findings)
    verdict = "pass" if (score >= threshold and not has_critical) else "revise"
    return Adjudication(score=score, verdict=verdict, merged_findings=findings,
                        disagreements=disagreements, reasoning=reasoning)


def plan_revision(
    adjudication: Adjudication, spec: CulturalSpec, gen_spec: GenSpec,
    kb: KnowledgeBase, lora_available: bool, reference_available: bool,
) -> RevisionPlan:
    """Lỗi -> thao tác. Ưu tiên tiếng Anh vì T2I đọc tiếng Anh."""
    pos: list[str] = []
    neg: list[str] = []
    boost: dict[str, float] = {}
    lora = False
    ref = False
    guidance = 0.0
    why: list[str] = []

    for f in adjudication.merged_findings:
        se = spec.entity(f.entity_id)
        if se is None:
            continue
        ent = kb.get(se.entity_id)
        attrs_en = se.required_attrs_en or se.required_attrs
        low_prior = ent is not None and ent.prior_strength < 0.20

        if f.severity == "critical":
            for cf in se.confusables:
                neg.extend(confusable_labels(cf.get("name", "")))
            neg.extend((se.forbidden_attrs_en or se.forbidden_attrs)[:2])
            pos.append(f"authentic Vietnamese {se.name_en.split('(')[0].strip()}")
            pos.extend(attrs_en[:2])
            boost[se.entity_id] = boost.get(se.entity_id, 0.0) + 0.35
            guidance = max(guidance, 1.0)
            why.append(f"chặn thực thể văn hoá khác cho {se.name_vi}, tăng guidance")
            if reference_available and se.reference_image and not gen_spec.ip_adapter_image:
                ref = True
                why.append(f"dùng ảnh tham chiếu cho {se.name_vi} qua IP-Adapter")
            if low_prior and lora_available and not gen_spec.lora:
                lora = True
                why.append(f"{se.name_vi} prior {ent.prior_strength:.2f} thấp, gắn LoRA")

        elif f.observed in ("không thấy", "missing", "absent"):
            pos.append(se.name_en.split("(")[0].strip())
            pos.extend(attrs_en[:2])
            boost[se.entity_id] = boost.get(se.entity_id, 0.0) + 0.30
            why.append(f"thêm {se.name_vi} vào prompt vì ảnh bỏ sót")
            if reference_available and se.reference_image and not gen_spec.ip_adapter_image:
                ref = True
            if low_prior and lora_available and not gen_spec.lora:
                lora = True
                why.append(f"{se.name_vi} prior thấp và bị bỏ qua, gắn LoRA")

        else:  # thiếu thuộc tính
            # f.expected là thuộc tính tiếng Việt; tìm bản tiếng Anh tương ứng.
            idx = next((i for i, a in enumerate(se.required_attrs) if a == f.expected), None)
            pos.append(attrs_en[idx] if idx is not None and idx < len(attrs_en) else f.expected)
            boost[se.entity_id] = boost.get(se.entity_id, 0.0) + 0.15
            why.append(f"bổ sung thuộc tính cho {se.name_vi}")

    pos = list(dict.fromkeys(p for p in pos if p and p not in gen_spec.prompt))
    neg = list(dict.fromkeys(n for n in neg if n and n not in gen_spec.negative_prompt))
    return RevisionPlan(add_positive=pos, add_negative=neg, boost=boost,
                        attach_lora=lora, use_reference_image=ref,
                        guidance_delta=guidance,
                        rationale="; ".join(dict.fromkeys(why)) or "không có thao tác nào áp dụng được")


def rule_critique(spec: CulturalSpec, perception: Perception) -> Critique:
    """Phê bình bằng luật trên danh sách quan sát. Dùng cho RuleAgent và làm sàn cho PromptAgent."""
    findings: list[Finding] = []
    if not spec.entities:
        return Critique(reviewer=PERSONA, score=0.0, verdict="revise",
                        findings=[Finding("-", "major", "spec rỗng", "ít nhất một thực thể",
                                          "Không kiểm chứng được: stage 3 không giữ thực thể nào.")],
                        reasoning="spec rỗng")
    wt, ws = 0.0, 0.0
    for se in spec.entities:
        el, status = locate(se, perception.elements)
        wt += se.weight
        ev = se.evidence_titles[0] if se.evidence_titles else None
        if status == "drift":
            findings.append(Finding(se.entity_id, "critical", el.label, se.name_vi,
                                    f"Ảnh vẽ '{el.label}' thay vì '{se.name_vi}'.", ev))
            continue
        if status == "missing":
            findings.append(Finding(se.entity_id, "major", "không thấy", se.name_vi,
                                    f"'{se.name_vi}' không xuất hiện trong ảnh.", ev))
            continue
        missing = [a for a in se.required_attrs if not attr_covered(a, el.attrs)]
        req_vocab: set[str] = set()
        for a in se.required_attrs:
            req_vocab |= tokens(a)
        violated = [a for a in se.forbidden_attrs if attr_covered(a, el.attrs, exclude=req_vocab)]
        for a in violated:
            findings.append(Finding(se.entity_id, "critical", a, f"không được có: {a}",
                                    f"'{se.name_vi}' mang chi tiết bị cấm: {a}", ev))
        ratio = len(missing) / max(1, len(se.required_attrs))
        for a in missing:
            findings.append(Finding(se.entity_id, "major" if ratio > 0.5 else "minor",
                                    "thiếu", a, f"'{se.name_vi}' thiếu: {a}", ev))
        ws += se.weight * max(0.0, 1.0 - 0.45 * ratio - 0.5 * len(violated))
    score = ws / wt if wt else 0.0
    crit = any(f.severity == "critical" for f in findings)
    return Critique(reviewer=PERSONA, findings=findings, score=score,
                    verdict="revise" if (crit or score < 0.8) else "pass",
                    reasoning=f"{len(findings)} phát hiện trên {len(spec.entities)} thực thể")


PERSONA = "CulturalExpert"
