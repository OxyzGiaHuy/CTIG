"""
Chạy walkthrough bằng dòng lệnh (không cần Jupyter): cùng các bước và cùng HTML với notebook, cho máy thuê (vast.ai) hoặc chạy lô.

    python scripts/run_walkthrough.py --config configs/vast_a100.yaml --ids p001,p012 [--models realvis_xl,realvis_xl+ref]
        [--run-name walkthrough] [--no-agents] [--report]

Key đọc từ biến môi trường (CIVITAI_TOKEN, HF_TOKEN, SERPER_API_KEY); trên vast: export trước khi chạy hoặc ghi vào ~/.bashrc.
Mỗi prompt ghi runs/<run>/<pid>/walkthrough.html, grid.png; --report dựng progress_report.html cho cả run.
Log in ra stdout kèm thời gian từng bước, để đọc lại được sau khi ssh ngắt (chạy với nohup hoặc tmux).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.config import Config, set_dotted  # noqa: E402
from ctig.pipeline import load_prompts  # noqa: E402
from ctig.schema import MultiGenResult, Prompt  # noqa: E402
from ctig.session import Session  # noqa: E402
from ctig import viz  # noqa: E402


def run_one(cfg: Config, prompt: Prompt, run_dir: Path, agents: bool, log) -> Path:
    t0 = time.time()
    s = Session(cfg, prompt, run_dir=run_dir, log=log)
    report = viz.Report(f"CTIG walkthrough · {prompt.id}")
    report.parts.append(viz.prompt_card(prompt))

    def step(name, fn):
        t = time.time()
        out = fn()
        log(f"[{prompt.id}] {name}: {time.time() - t:.0f}s")
        return out

    g, src = step("1 grounding", s.grounding)
    report.parts.append(viz.grounding_table(g, s.kb, source=src))
    # chẩn đoán từng bước con (đã memo, không tốn thêm)
    a, src = s.analysis()
    report.parts.append(viz.keywords_table(a, s.kb, cfg.max_spec_entities, source=src))
    cmp, src = s.compare()
    report.parts.append(viz.query_comparison(cmp, cfg.search_viz.k_text, cfg.search_viz.k_images, source=src))
    search, src = s.retrieve()
    report.parts.append(viz.evidence_table(search, s.kb, source=src))
    if g["briefs"]:
        report.parts.append(viz.brief_card(g["briefs"], g["spec"], source=src))
    spec, src = s.spec()
    report.parts.append(viz.spec_card(spec, source=src))
    gen, src = step("1b genspec", s.genspec)
    report.parts.append(viz.genspec_card(gen, source=src))
    if cfg.multigen.device == cfg.llm.device and cfg.agents.reload_vlm:
        s.free_vlm()
    res, src = step("2 generate", lambda: s.multigen(cfg.models, on_model_done=lambda r: log(
        f"    hàng {r.model_key}: " + (r.error[:80] if r.error else f"{len(r.output.candidates)} ảnh, {r.seconds:.0f}s"))))
    report.parts.append(viz.model_grid(res, spec, source=src))
    if "ref_filter" in s.steps:
        report.parts.append(viz.filter_table(s.steps["ref_filter"].value, title="Grounding · Filter agent trên ảnh tham chiếu", source=s.steps["ref_filter"].source))
    report.parts.append(viz.score_table(res, source=src))
    cr = None
    if agents and cfg.agents.enabled and cfg.agents.candidate_review:
        cr, src = step("3 agentic review loop", s.candidate_review)
        report.parts.append(viz.candidate_review_html(cr, source=src))
        log(f"[{prompt.id}] ảnh cuối: {cr.final_path} ({cr.final_source}, {cr.best_model}) · {len(cr.iterations)} vòng · {cr.stop_reason}")
    report.parts.append(viz.paired_table(res, cr))
    report.parts.append(viz.vram_html())
    out = report.save(s.out_dir / "walkthrough.html")
    log(f"[{prompt.id}] xong {time.time() - t0:.0f}s -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", default="p001", help="danh sách id cách nhau bằng dấu phẩy, hoặc 'dev10'")
    ap.add_argument("--text", default=None, help="prompt tự do thay cho --ids")
    ap.add_argument("--models", default=None, help="ghi đè models: trong config")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--set", action="append", default=[], help="ghi đè config dạng a.b=c")
    ap.add_argument("--no-agents", action="store_true")
    ap.add_argument("--report", action="store_true", help="dựng progress_report.html cho cả run sau khi xong")
    a = ap.parse_args(argv)

    overrides: dict = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        set_dotted(overrides, k, v)
    cfg = Config.load(a.config, overrides)
    if a.models:
        cfg.models = [m.strip() for m in a.models.split(",") if m.strip()]
    run_dir = Path(cfg.runs_dir) / (a.run_name or cfg.run_name or "walkthrough")
    log = lambda *x: print(*x, flush=True)  # noqa: E731

    if a.text:
        prompts = [Prompt("adhoc", a.text, a.text)]
    else:
        ids = a.ids.split(",")
        if a.ids == "dev10":
            ids = [l.strip() for l in Path("data/dev10.txt").read_text().splitlines() if l.strip()]
        allp = {p.id: p for p in load_prompts(cfg.prompts_path)}
        prompts = [allp[i.strip()] for i in ids if i.strip() in allp]
    log(f"config {a.config} · {len(prompts)} prompt · models {cfg.models} · run_dir {run_dir}")
    for p in prompts:
        try:
            run_one(cfg, p, run_dir, not a.no_agents, log)
        except Exception as exc:  # noqa: BLE001 - một prompt lỗi không dừng cả lô
            log(f"[{p.id}] LỖI {type(exc).__name__}: {exc}")
    if a.report:
        from ctig.progress_report import build

        build(run_dir, run_dir / "progress_report.html", cfg=cfg, log=log)


if __name__ == "__main__":
    main()
