"""Thay CHỈ ảnh I1 (+P1, cờ) của một số đơn vị trong lô chính bằng ảnh từ lô sinh lại theo P1 (--p1-from); giữ cards/gap/actions cũ.

    python scripts/merge_i1.py --main RUN --from RUN_P1FIX --ids S001,... [--tag p1fix]
"""
import argparse, json, shutil, time
from pathlib import Path
def _loc(p, rd):
    parts = Path(p).parts; return rd.joinpath(*parts[len(parts) - 1 - parts[::-1].index(rd.name) + 1:])
ap = argparse.ArgumentParser(); ap.add_argument("--main", required=True); ap.add_argument("--from", dest="src", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--tag", default="p1fix"); a = ap.parse_args()
M, S = Path(a.main), Path(a.src); ids = set(a.ids.split(","))
shutil.copy(M / "kor.json", M / f"kor_truoc_{a.tag}_{time.strftime('%H%M')}.json")
mj = json.loads((M / "kor.json").read_text(encoding="utf-8")); sj = {u["prompt_id"]: u for u in json.loads((S / "kor.json").read_text(encoding="utf-8"))["don_vi"]}; n = 0
for u in mj["don_vi"]:
    pid = u["prompt_id"]
    if pid not in ids or pid not in sj: continue
    nu = sj[pid]; assert nu["seed"] == u["seed"], pid
    src = _loc(nu["images"]["C (I1)"], S); dst = M / pid / "C_ref" / a.tag / src.name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst)
    u["images"]["C (I1)"] = str(dst); u["P1"] = nu["P1"]; u[a.tag] = True; u.pop("noop_ref_fix", None)
    (M / pid / "kor.json").write_text(json.dumps(u, ensure_ascii=False, indent=1), encoding="utf-8"); n += 1
(M / "kor.json").write_text(json.dumps(mj, ensure_ascii=False, indent=1), encoding="utf-8"); print(f"đã thay I1 của {n} đơn vị trong {M.name} (tag {a.tag})")
