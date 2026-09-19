"""Xuất toàn bộ giao tiếp agent (C/K, O, R) theo prompt để vẽ figure pipeline.

Mỗi prompt -> results/transcripts/<model>/<pid>.json: {unit: {P0, P_ct, P1, seed, cards, report_I0, gap, actions, drop_phrases, images},
 transcript: [ {t, agent, buoc, system, user, images, response, giay} ... ]}  (đủ system prompt, user prompt, ảnh đính kèm, JSON trả về, thời gian)
Kèm <model>/_summary.json gọn (P0 → Preservation Card → Evidence Card → Observer → Gap → Actions → P1) và <model>/_transcript.md.

    python scripts/export_transcripts.py --src docs/report_assets --dst results/transcripts
"""
import argparse, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); a = ap.parse_args()
S, D = Path(a.src), Path(a.dst)
for model, run in (("sdxl", "kor_20260918_1349"), ("flux", "kor_flux_20260918_1530")):
    kj = json.loads((S / run / "kor.json").read_text(encoding="utf-8")); out = D / model; out.mkdir(parents=True, exist_ok=True)
    by = {}
    for line in kj.get("giao_tiep", []): by.setdefault(line["prompt_id"], []).append(line)
    summ = {}
    for u in kj["don_vi"]:
        pid = u["prompt_id"]
        unit = {k: u.get(k) for k in ("prompt_id", "prompt_vi", "P_orig", "P_ct", "P1", "seed", "model", "cards", "report_I0", "gap", "actions", "drop_phrases", "refs")}
        unit["images"] = {k: Path(v).name for k, v in (u.get("images") or {}).items() if v}
        (out / f"{pid}.json").write_text(json.dumps({"unit": unit, "transcript": by.get(pid, [])}, ensure_ascii=False, indent=1), encoding="utf-8")
        c = u.get("cards") or {}
        summ[pid] = {"prompt_vi": u.get("prompt_vi"), "P0": u.get("P_orig"), "P_ct_expansion": (u.get("P_ct") or "")[len(u.get("P_orig") or ""):].strip(),
                     "C_preservation_card": c.get("prompt_preservation"), "C_evidence_card": c.get("cultural_evidence"),
                     "O_report_I0": u.get("report_I0"), "R_gap": {k: v for k, v in (u.get("gap") or {}).items() if k != "repair_actions"},
                     "R_actions_final": u.get("actions"), "R_drop_phrases": u.get("drop_phrases"), "P1": u.get("P1"),
                     "n_calls": len(by.get(pid, [])), "seconds_llm": round(sum(float(x.get("giay") or 0) for x in by.get(pid, [])), 1)}
    (out / "_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    md = S / run / "kor_transcript.md"
    if md.exists(): shutil.copy(md, out / "_transcript.md")
    print(f"{model}: {len(summ)} prompt, {sum(len(v) for v in by.values())} lượt gọi -> {out}")
