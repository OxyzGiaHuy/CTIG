"""
Stage 4 + tri giác + 5 - vòng lặp: sinh -> chọn ứng viên -> nhìn -> phê bình -> hoà giải -> sửa.

Reviewer chỉ nhận `Perception`. `GenOutput.oracle` (chỉ có với stub) không đi vào đây.
"""

from __future__ import annotations

from pathlib import Path

from ..kb import KnowledgeBase
from ..schema import CulturalSpec, GenOutput, Prompt, ReviewIteration, ReviewOutcome, RevisionPlan
from .generation import apply_plan, build_initial_spec, to_final_render


def select_candidate(out: GenOutput, spec: CulturalSpec, clip) -> None:
    """Chọn ứng viên có CLIP fidelity cao nhất. Không có CLIP thì giữ ứng viên 0."""
    if clip is None or not spec.entities:
        return
    best, best_score = 0, -1.0
    for i, c in enumerate(out.candidates):
        probs = clip.entity_probs(c.path, spec)
        c.clip_probs = probs
        wt = sum(se.weight for se in spec.entities)
        c.clip_fidelity = sum(se.weight * probs.get(se.entity_id, {}).get("__target__", 0.0)
                              for se in spec.entities) / wt
        if c.clip_fidelity > best_score:
            best, best_score = i, c.clip_fidelity
    out.chosen = best


def generate_only(generator, clip, prompt: Prompt, spec: CulturalSpec, kb: KnowledgeBase,
                  prompt_en: str | None, cfg, out_dir: Path, log=print) -> ReviewOutcome:
    """review.enabled = false: một vòng sinh, chọn ứng viên bằng CLIP, không gọi VLM.

    Cùng hình dạng ReviewOutcome nên eval, báo cáo, bundle không đổi. Điểm = CLIP fidelity.
    """
    from ..schema import Adjudication, Critique, Perception

    gen = build_initial_spec(prompt, spec, prompt_en, cfg.t2i, cfg.seed,
                             init_negatives=getattr(cfg.t2i, "init_negatives", True))
    out = generator.generate(gen, spec, kb, out_dir)
    select_candidate(out, spec, clip)
    chosen = out.candidates[out.chosen]
    perception = Perception(out.image_path, [], caption="(review tắt, không gọi VLM)",
                            clip_probs=chosen.clip_probs, perceiver="clip-only")
    score = chosen.clip_fidelity if chosen.clip_probs else 0.0
    verdict = "pass" if score >= cfg.review.pass_threshold else "revise"
    critique = Critique(reviewer="clip-only", score=score, verdict=verdict,
                        reasoning="review tắt; điểm = CLIP fidelity của ứng viên được chọn")
    adj = Adjudication(score=score, verdict=verdict, reasoning="review tắt; điểm = CLIP fidelity")
    log(f"  [4] sinh 1 vòng, không review: CLIP {score:.2f} -> {verdict}")
    it = ReviewIteration(0, gen, out, perception, [critique], adj, RevisionPlan(rationale="review tắt"))
    return ReviewOutcome(prompt.id, [it], verdict == "pass", out.image_path, score)


def run(agent, generator, perceiver, clip, prompt: Prompt, spec: CulturalSpec, kb: KnowledgeBase,
        prompt_en: str | None, cfg, out_dir: Path, log=print) -> ReviewOutcome:
    gen = build_initial_spec(prompt, spec, prompt_en, cfg.t2i, cfg.seed,
                             init_negatives=getattr(cfg.t2i, "init_negatives", True))
    iterations: list[ReviewIteration] = []
    for n in range(cfg.review.max_iters + 1):
        out = generator.generate(gen, spec, kb, out_dir)
        select_candidate(out, spec, clip)
        perception = perceiver.perceive(out, spec)
        # CLIP của ứng viên đã chọn tái dùng cho perception nếu perceiver không tự tính.
        if not perception.clip_probs and out.candidates[out.chosen].clip_probs:
            perception.clip_probs = out.candidates[out.chosen].clip_probs
        critique = agent.critique(prompt, spec, perception)
        adj = agent.adjudicate(critique, perception, spec, cfg.review.pass_threshold,
                               cfg.review.clip_weight, cfg.perception.drift_margin)
        plan = (RevisionPlan(rationale="đạt") if adj.verdict == "pass" else
                agent.plan_revision(adj, spec, gen, kb, generator.lora_available, generator.reference_available))
        iterations.append(ReviewIteration(n, gen, out, perception, [critique], adj, plan))
        ids = {e: c.get("identity") for e, c in perception.checklist.items()}
        log(f"  [4/5] vòng {n}{' (nhanh)' if gen.fast else ''}: checklist {critique.score:.2f} | hợp {adj.score:.2f} -> {adj.verdict}"
            + (f" | danh tính {ids}" if ids else "")
            + (f" | LoRA" if gen.lora else "") + (f" | ref" if gen.ip_adapter_image else ""))
        if adj.verdict == "pass" or not spec.entities or n == cfg.review.max_iters or plan.is_empty():
            break
        gen = apply_plan(gen, plan, spec, cfg.t2i, getattr(generator, "lora_id", None))

    # Vòng nhanh (LCM) đã xong -> render đủ bước cùng prompt và kiểm lại một lần.
    if iterations[-1].gen_spec.fast and spec.entities:
        gen_hq = to_final_render(iterations[-1].gen_spec, cfg.t2i)
        out = generator.generate(gen_hq, spec, kb, out_dir)
        perception = perceiver.perceive(out, spec)
        if not perception.clip_probs and clip is not None:
            select_candidate(out, spec, clip)
            perception.clip_probs = out.candidates[out.chosen].clip_probs
        critique = agent.critique(prompt, spec, perception)
        adj = agent.adjudicate(critique, perception, spec, cfg.review.pass_threshold,
                               cfg.review.clip_weight, cfg.perception.drift_margin)
        iterations.append(ReviewIteration(len(iterations), gen_hq, out, perception, [critique], adj,
                                          RevisionPlan(rationale="render đủ bước"), final_render=True))
        log(f"  [4/5] render đủ bước: hợp {adj.score:.2f} -> {adj.verdict}")
    last = iterations[-1]
    return ReviewOutcome(prompt.id, iterations, last.adjudication.verdict == "pass",
                         last.gen_output.image_path, last.adjudication.score)
