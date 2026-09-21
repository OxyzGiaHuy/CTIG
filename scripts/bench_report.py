"""Tổng hợp thời gian theo giai đoạn từ một hoặc nhiều lô run_kor (rec["thoi_gian"]) -> mean ± sd (giây/prompt).

    python scripts/bench_report.py --run SDXL=/workspace/runs/bench_sdxl_r1 --run SDXL=/workspace/runs/bench_sdxl_r2 ...
"""
import argparse, json, statistics as st
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--run", action="append", required=True); ap.add_argument("-o", default=""); ap.add_argument("--skip", default="S050", help="prompt warm-up bỏ khỏi thống kê"); a = ap.parse_args()
by = {}
for spec in a.run:
    lab, _, d = spec.partition("=")
    for u in json.loads((Path(d) / "kor.json").read_text(encoding="utf-8"))["don_vi"]:
        if u["prompt_id"] in a.skip.split(","): continue
        t = u.get("thoi_gian") or {}
        for k, v in t.items(): by.setdefault(lab, {}).setdefault(k, []).append(v)
        by[lab].setdefault("agent_total", []).append(sum(v for k, v in t.items() if k.startswith("agent_")))
        by[lab].setdefault("savier_total(I1)", []).append(sum(v for k, v in t.items() if k.startswith("agent_")) + t.get("gen_C", 0.0))
        by[lab].setdefault("n_actions", []).append(len(u.get("actions") or []))
out = []
for lab, d in by.items():
    out += [f"## {lab} — n={len(d.get('agent_total', []))} đơn vị", "", "| giai đoạn | mean (s) | sd | min | max |", "|---|---:|---:|---:|---:|"]
    order = ["gen_A", "gen_B", "agent_C", "agent_O", "agent_R", "agent_total", "gen_C", "savier_total(I1)", "n_actions"]
    for k in order + [k for k in d if k not in order]:
        if k not in d: continue
        v = d[k]; out.append(f"| {k} | {st.mean(v):.1f} | {st.pstdev(v):.1f} | {min(v):.1f} | {max(v):.1f} |")
    out.append("")
txt = "\n".join(out); print(txt)
if a.o: Path(a.o).write_text(txt, encoding="utf-8")
