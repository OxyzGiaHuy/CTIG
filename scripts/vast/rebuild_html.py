"""Dựng lại walkthrough.html cho các prompt đã chạy, không gọi model (đọc step_*.json + multigen.json)."""
import sys
from pathlib import Path
sys.path.insert(0, "/workspace/ctig17")
from ctig import viz
from ctig.config import Config
from ctig.pipeline import load_prompts
from ctig.schema import CandidateReview, CulturalSpec, MultiGenResult, from_dict
import json

run = Path(sys.argv[1]); cfg = Config.load(sys.argv[2] if len(sys.argv) > 2 else "configs/vast_a100.yaml")
pp = sys.argv[3] if len(sys.argv) > 3 else "data/prompts_simple.json"
allp = {p.id: p for p in load_prompts(pp)}
for d in sorted(x for x in run.iterdir() if x.is_dir() and (x / "multigen.json").exists()):
    pid = d.name
    res = from_dict(MultiGenResult, json.loads((d / "multigen.json").read_text()))
    spec = from_dict(CulturalSpec, json.loads((d / "step_spec.json").read_text())["value"])
    crf = d / "step_candidate_review.json"
    cr = from_dict(CandidateReview, json.loads(crf.read_text())["value"]) if crf.exists() else None
    rep = viz.Report(f"CTIG · {pid}")
    if pid in allp:
        rep.parts.append(viz.prompt_card(allp[pid]))
    if cr is not None:
        rep.parts.append(viz.final_grid(res, cr))
    rep.parts.append(viz.model_grid(res, spec, cr=cr))
    rep.parts.append(viz.score_table(res))
    if cr is not None:
        rep.parts.append(viz.candidate_review_html(cr, res=res))
    rep.parts.append(viz.paired_table(res, cr))
    out = rep.save(d / "walkthrough.html")
    print(pid, "->", out)
