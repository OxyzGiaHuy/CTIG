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


def run(agent, prompt: Prompt, spec: CulturalSpec, search: SearchResult, outcome: ReviewOutcome) -> EvalRecord:
    last = outcome.iterations[-1]
    if spec.entities:
        try:
            judge_score, judge_reason = agent.judge(prompt, spec, last.perception)
        except Exception as exc:  # noqa: BLE001
            judge_score, judge_reason = 0.0, f"judge lỗi: {type(exc).__name__}: {exc}"
    else:
        judge_score, judge_reason = 0.0, "spec rỗng"
    return EvalRecord(
        prompt_id=prompt.id, prompt_text=prompt.text_vi, final_image_path=outcome.final_image_path,
        passed=outcome.passed, iterations=outcome.n_iterations, verifiable=bool(spec.entities),
        retrieval_recall=retrieval_recall(prompt, search), review_score=outcome.final_score,
        clip_fidelity=clip_fidelity(outcome, spec), judge_score=judge_score, judge_reasoning=judge_reason,
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
