"""Sao chép ảnh từ các lô (docs/report_assets) sang bố cục results/<model>/<pid>/{A,I0,I1,refs_only,keep_only}.png của repo đích.

    python scripts/export_results.py --src docs/report_assets --dst ../SAVIER/results
"""
import argparse, json, shutil
from pathlib import Path

def _loc(p, rd):
    parts = Path(p).parts; return rd.joinpath(*parts[len(parts) - 1 - parts[::-1].index(rd.name) + 1:])
def units(rd): return {u["prompt_id"]: u for u in json.loads((rd / "kor.json").read_text(encoding="utf-8"))["don_vi"]}

ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); a = ap.parse_args()
S, D = Path(a.src), Path(a.dst)
plan = {"sdxl": ("kor_20260918_1349", "abl_sdxl_refsonly", "abl_sdxl_keeponly"), "flux": ("kor_flux_20260918_1530", "abl_flux_refsonly", "abl_flux_keeponly")}
n = 0
for model, (main, ro, ko) in plan.items():
    um, ur, uk = units(S / main), units(S / ro), units(S / ko)
    for pid, u in um.items():
        out = D / model / pid; out.mkdir(parents=True, exist_ok=True)
        for key, name, us, rd in (("A", "A", um, main), ("B (I0)", "I0", um, main), ("C (I1)", "I1", um, main), ("C (I1)", "refs_only", ur, ro), ("C (I1)", "keep_only", uk, ko)):
            p = us.get(pid, {}).get("images", {}).get(key)
            if not p: print("thiếu", model, pid, name); continue
            src = _loc(p, S / rd); shutil.copy(src, out / f"{name}.png"); n += 1
print(f"đã chép {n} ảnh -> {D}")
