"""Stage 6 - Evaluation: bảng Prompt | Kết quả | Evidence, CSV user study, báo cáo HTML."""

from __future__ import annotations

import csv
import html
from pathlib import Path

from ..schema import CulturalSpec, EvalRecord, Prompt, ReviewOutcome, SearchResult


def retrieval_recall(prompt: Prompt, search: SearchResult) -> float:
    if not prompt.gold_entities:
        return -1.0
    found = {i.entity_id for i in search.items}
    return sum(1 for g in prompt.gold_entities if g in found) / len(prompt.gold_entities)


def clip_fidelity(outcome: ReviewOutcome, spec: CulturalSpec, n: int = -1) -> float:
    it = outcome.iterations[n]
    probs = it.perception.clip_probs
    if not probs or not spec.entities:
        return 0.0
    wt = sum(se.weight for se in spec.entities)
    return sum(se.weight * probs.get(se.entity_id, {}).get("__target__", 0.0) for se in spec.entities) / wt


def oracle_fidelity(outcome: ReviewOutcome, spec: CulturalSpec, n: int = -1) -> float | None:
    o = outcome.iterations[n].gen_output.oracle
    if o is None or not spec.entities:
        return None
    return sum(1 for se in spec.entities if o.get(se.entity_id) == se.name_vi) / len(spec.entities)


class ITMJudge:
    """Judge ĐỘC LẬP với reviewer: BLIP-2 ITM (họ model khác Qwen) + CLIP, không LLM.

    Skill blip-2: đầu ITM trả xác suất ảnh khớp một câu. Ta chấm ba nhóm câu:
      identity     "a photo of <clip_label>"                       so với câu của từng confusable
      completeness "<name_en> with <attr_en>" cho từng must_have   (chỉ khi có bản dịch)
      purity       1 - max P(câu confusable)
    Lần chạy đầu judge là chính Qwen: chấm 0.98 khi reviewer fail, trả lời bằng tiếng Trung.
    """

    name = "blip2_itm+clip"

    def __init__(self, cfg, clip=None):
        import torch
        from transformers import Blip2ForImageTextRetrieval, Blip2Processor

        self.torch = torch
        self.clip = clip
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.offload = bool(getattr(cfg, "offload", True)) and self.device.startswith("cuda")
        dt = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = Blip2ForImageTextRetrieval.from_pretrained(cfg.blip2_model, torch_dtype=dt).eval()
        self.model.to("cpu" if self.offload else self.device)
        self.proc = Blip2Processor.from_pretrained(cfg.blip2_model)
        self.dt = dt

    def _on_gpu(self):
        if self.offload:
            self.model.to(self.device)

    def _off_gpu(self):
        if self.offload:
            self.model.to("cpu")
            self.torch.cuda.empty_cache()

    def itm(self, image_path: str, texts: list[str]) -> list[float]:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        out: list[float] = []
        for t in texts:
            inputs = self.proc(images=img, text=t, return_tensors="pt").to(self.device, self.dt)
            with self.torch.inference_mode():
                res = self.model(**inputs, use_image_text_matching_head=True)
            logits = res.logits_per_image if hasattr(res, "logits_per_image") else res[0]
            out.append(float(logits.softmax(dim=-1)[0, 1]))
        return out

    def judge(self, prompt: Prompt, spec: CulturalSpec, perception) -> tuple[float, str]:
        if not spec.entities:
            return 0.0, "spec rỗng"
        from ..llm.shared import confusable_clip_label

        self._on_gpu()
        try:
            return self._judge(spec, perception, confusable_clip_label)
        finally:
            self._off_gpu()

    def _judge(self, spec, perception, confusable_clip_label):
        wt, ident, comp, pur, notes = 0.0, 0.0, 0.0, 0.0, []
        for se in spec.entities:
            target = se.clip_label or f"a photo of Vietnamese {se.name_en.split('(')[0].strip()}"
            p_t = self.itm(perception.image_path, [target])[0]
            cf_labels = [confusable_clip_label(c) for c in se.confusables[:3]]
            p_cf = max(self.itm(perception.image_path, cf_labels)) if cf_labels else 0.0
            attrs = se.required_attrs_en[:4]
            p_attr = (sum(self.itm(perception.image_path, [f"{se.name_en.split('(')[0].strip()} with {a}" for a in attrs])) / len(attrs)
                      if attrs else p_t)
            wt += se.weight
            ident += se.weight * p_t
            comp += se.weight * p_attr
            pur += se.weight * (1.0 - p_cf)
            notes.append(f"{se.name_vi}: itm {p_t:.2f}, attrs {p_attr:.2f}, confusable {p_cf:.2f}")
        a, b, c = ident / wt, comp / wt, pur / wt
        return (a + b + c) / 3, f"identity {a:.2f} | completeness {b:.2f} | purity {c:.2f} | " + "; ".join(notes)


class CLIPJudge:
    name = "clip"

    def __init__(self, clip):
        self.clip = clip

    def judge(self, prompt, spec, perception) -> tuple[float, str]:
        if not spec.entities or self.clip is None:
            return 0.0, "không có CLIP hoặc spec rỗng"
        probs = perception.clip_probs or self.clip.entity_probs(perception.image_path, spec)
        objs = [se for se in spec.entities if se.entity_id in probs]
        if not objs:
            return 0.0, "không có thực thể object để CLIP chấm"
        wt = sum(se.weight for se in objs)
        s = sum(se.weight * probs[se.entity_id].get("__target__", 0.0) for se in objs) / wt
        return s, "CLIP target prob có trọng số: " + "; ".join(f"{se.name_vi} {probs[se.entity_id].get('__target__', 0):.2f}" for se in objs)


class AgentJudge:
    def __init__(self, agent):
        self.agent = agent
        self.name = f"vlm:{getattr(agent, 'name', '?')}"

    def judge(self, prompt, spec, perception):
        return self.agent.judge(prompt, spec, perception)


def get_judge(cfg, agent, clip):
    if cfg.backend == "blip2_itm":
        try:
            return ITMJudge(cfg, clip)
        except Exception as exc:  # noqa: BLE001
            print(f"[judge] không tải được BLIP-2 ITM ({type(exc).__name__}: {exc}); dùng CLIP")
            return CLIPJudge(clip)
    if cfg.backend == "clip":
        return CLIPJudge(clip)
    return AgentJudge(agent)


def run(judge, prompt: Prompt, spec: CulturalSpec, search: SearchResult, outcome: ReviewOutcome) -> EvalRecord:
    last = outcome.iterations[-1]
    if spec.entities:
        try:
            judge_score, judge_reason = judge.judge(prompt, spec, last.perception)
        except Exception as exc:  # noqa: BLE001
            judge_score, judge_reason = 0.0, f"judge lỗi: {type(exc).__name__}: {exc}"
    else:
        judge_score, judge_reason = 0.0, "spec rỗng"
    return EvalRecord(
        prompt_id=prompt.id, prompt_text=prompt.text_vi, final_image_path=outcome.final_image_path,
        passed=outcome.passed, iterations=outcome.n_iterations, verifiable=bool(spec.entities),
        retrieval_recall=retrieval_recall(prompt, search), review_score=outcome.final_score,
        clip_fidelity=clip_fidelity(outcome, spec), judge_score=judge_score, judge_reasoning=judge_reason,
        judge_backend=getattr(judge, "name", "?"),
        oracle_fidelity=oracle_fidelity(outcome, spec),
        evidence_summary=[f"[{i.kind}] {i.title}" + (f" <{i.url}>" if i.url else "") for i in search.items],
        residual_findings=[f"({f.severity}) {f.message}" for f in last.adjudication.merged_findings
                           if f.severity in ("critical", "major")],
        iteration_images=[it.gen_output.image_path for it in outcome.iterations],
    )


def export_user_study(records: list[EvalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id", "prompt", "image_path", "evidence",
                    "rating_dung_thuc_the_1_5", "rating_du_chi_tiet_1_5", "rating_khong_lan_van_hoa_khac_1_5",
                    "nhan_xet", "he_thong_passed", "review_score", "clip_fidelity", "judge_score"])
        for r in records:
            w.writerow([r.prompt_id, r.prompt_text, r.final_image_path, " || ".join(r.evidence_summary),
                        "", "", "", "", r.passed, f"{r.review_score:.3f}", f"{r.clip_fidelity:.3f}", f"{r.judge_score:.3f}"])


def write_html_report(results, summary, path: Path) -> None:
    """Một file HTML: mọi prompt, mọi vòng, ảnh và phán quyết. Mở trên Kaggle bằng IPython.display.HTML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root = path.parent

    def rel(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(root.resolve()))
        except ValueError:
            return p

    rows = []
    for r in results:
        rec, spec = r.record, r.spec
        ents = ", ".join(f"{e.name_vi} (w{e.weight})" for e in spec.entities) or "<i>spec rỗng</i>"
        iters = []
        for it in r.outcome.iterations:
            adj = it.adjudication
            finds = "".join(f"<li class='{f.severity}'>{html.escape(f.message)}</li>" for f in adj.merged_findings[:6])
            disag = "".join(f"<li>{html.escape(d)}</li>" for d in adj.disagreements)
            probs = "".join(
                f"<div class='clip'>{html.escape(spec.entity(eid).name_vi if spec.entity(eid) else eid)}: "
                + ", ".join(f"{html.escape(k if k != '__target__' else 'mục tiêu')} {v:.2f}" for k, v in p.items())
                + "</div>" for eid, p in it.perception.clip_probs.items())
            iters.append(f"""
            <div class='iter {adj.verdict}'>
              <img src='{html.escape(rel(it.gen_output.image_path))}' loading='lazy'>
              <div class='meta'>
                <b>vòng {it.n}</b> · điểm {adj.score:.2f} · <span class='v'>{adj.verdict}</span>
                {' · LoRA' if it.gen_spec.lora else ''}{' · ref' if it.gen_spec.ip_adapter_image else ''}
                <div class='cap'>{html.escape(it.perception.caption[:220])}</div>
                {probs}
                <ul class='f'>{finds}</ul>
                {('<div class="dis"><b>bất đồng VLM/CLIP</b><ul>' + disag + '</ul></div>') if disag else ''}
                <details><summary>prompt sinh</summary><pre>{html.escape(it.gen_spec.prompt)}</pre>
                <pre class='neg'>NEG: {html.escape(it.gen_spec.negative_prompt)}</pre></details>
              </div>
            </div>""")
        rows.append(f"""
        <section class='{'ok' if rec.passed else 'fail'}'>
          <h2>{rec.prompt_id} <small>{html.escape(r.prompt.difficulty)}</small> — {html.escape(rec.prompt_text)}</h2>
          <div class='spec'>Thực thể: {ents}</div>
          <div class='scores'>đạt: <b>{'Y' if rec.passed else 'n'}</b> · review {rec.review_score:.2f} · CLIP {rec.clip_fidelity:.2f}
            · judge {rec.judge_score:.2f} · recall {('n/a' if rec.retrieval_recall < 0 else f'{rec.retrieval_recall:.2f}')}
            · {rec.iterations} vòng</div>
          <div class='iters'>{''.join(iters)}</div>
          <details><summary>judge</summary><p>{html.escape(rec.judge_reasoning)}</p></details>
        </section>""")

    s = summary
    doc = f"""<!doctype html><meta charset='utf-8'><title>CTIG report {html.escape(s.run_id)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#faf9f6;color:#1c1e22}}
h1{{font-size:20px}} h2{{font-size:15px;margin:0 0 6px}} h2 small{{color:#888;font-weight:normal}}
section{{background:#fff;border-left:5px solid #c4302b;padding:12px 16px;margin:14px 0;border-radius:6px}}
section.ok{{border-left-color:#168052}}
.iters{{display:flex;gap:12px;overflow-x:auto;padding:6px 0}}
.iter{{min-width:300px;max-width:300px;border:1px solid #ddd;border-radius:6px;padding:8px;background:#f4f3ef}}
.iter.pass{{border-color:#168052}} .iter img{{width:100%;border-radius:4px}}
.meta{{font-size:12px}} .v{{font-weight:bold}} .cap{{color:#666;margin:4px 0}}
.clip{{font-size:11px;color:#444}} ul.f{{padding-left:16px;margin:4px 0}} li.critical{{color:#c4302b}} li.major{{color:#96691e}} li.minor{{color:#666}}
.dis{{font-size:11px;background:#fff3cd;padding:4px 6px;border-radius:4px;margin-top:4px}}
pre{{white-space:pre-wrap;font-size:10px;background:#eee;padding:4px}} .neg{{color:#c4302b}}
table{{border-collapse:collapse}} td,th{{padding:3px 10px;text-align:left;border-bottom:1px solid #ddd}}
.spec,.scores{{font-size:12px;color:#444;margin:2px 0}}
</style>
<h1>CTIG — {html.escape(s.run_id)}</h1>
<table>
<tr><th>prompt</th><td>{s.n_prompts}</td><th>loại (spec rỗng)</th><td>{s.n_unverifiable}</td></tr>
<tr><th>đạt cuối / vòng 0</th><td>{s.pass_rate:.3f} / {s.pass_rate_iter0:.3f}</td><th>CLIP fidelity cuối / vòng 0</th><td>{s.mean_clip_fidelity:.3f} / {s.mean_clip_fidelity_iter0:.3f}</td></tr>
<tr><th>judge</th><td>{s.mean_judge_score:.3f}</td><th>retrieval recall</th><td>{('n/a' if s.mean_retrieval_recall < 0 else f'{s.mean_retrieval_recall:.3f}')}</td></tr>
<tr><th>số vòng TB</th><td>{s.mean_iterations:.2f}</td><th>thời gian</th><td>{s.wall_seconds / 60:.1f} phút</td></tr>
</table>
{''.join(rows)}
"""
    path.write_text(doc, encoding="utf-8")
