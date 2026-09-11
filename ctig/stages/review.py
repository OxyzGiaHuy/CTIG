"""
Stage 4 + tri giác + 5 - vòng lặp: sinh -> chọn ứng viên -> nhìn -> phê bình -> hoà giải -> sửa.

Reviewer chỉ nhận `Perception`. `GenOutput.oracle` (chỉ có với stub) không đi vào đây.
"""

from __future__ import annotations

from pathlib import Path

from ..kb import KnowledgeBase
from ..schema import CulturalSpec, GenOutput, Prompt, ReviewIteration, ReviewOutcome, RevisionPlan
from .generation import apply_plan, build_initial_spec


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
        log(f"  [4/5] vòng {n}: VLM {critique.score:.2f} | hợp {adj.score:.2f} -> {adj.verdict}"
            + (f" | {len(adj.merged_findings)} phát hiện" if adj.merged_findings else "")
            + (f" | LoRA" if gen.lora else "") + (f" | ref" if gen.ip_adapter_image else ""))
        if adj.verdict == "pass" or not spec.entities or n == cfg.review.max_iters or plan.is_empty():
            break
        gen = apply_plan(gen, plan, spec, cfg.t2i, getattr(generator, "lora_id", None))
    last = iterations[-1]
    return ReviewOutcome(prompt.id, iterations, last.adjudication.verdict == "pass",
                         last.gen_output.image_path, last.adjudication.score)
