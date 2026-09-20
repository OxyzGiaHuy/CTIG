"""Sửa lỗi đặc tả no-op: đơn vị mà R không đưa action từng rơi về I0 (không ảnh tham chiếu). Theo đặc tả arm SAVIER
(cùng seed + IP-Adapter + P1, với P1 = P_ct khi không có action), ảnh đúng của đơn vị đó CHÍNH LÀ ảnh refs-only
(cùng prompt, cùng seed, cùng ref) — chép sang, không sinh lại. Áp cho mọi đơn vị thoả điều kiện, không nhìn metric.

    python scripts/noop_ref_fix.py --main RUN_MAIN --refs RUN_REFSONLY
"""
import argparse, json, shutil, time
from pathlib import Path

def _loc(p, rd):
    parts = Path(p).parts; return rd.joinpath(*parts[len(parts) - 1 - parts[::-1].index(rd.name) + 1:])

ap = argparse.ArgumentParser(); ap.add_argument("--main", required=True); ap.add_argument("--refs", required=True); a = ap.parse_args()
M, R = Path(a.main), Path(a.refs)
bk = M / f"kor_truoc_noop_fix_{time.strftime('%H%M')}.json"; shutil.copy(M / "kor.json", bk)
mj = json.loads((M / "kor.json").read_text(encoding="utf-8")); ur = {u["prompt_id"]: u for u in json.loads((R / "kor.json").read_text(encoding="utf-8"))["don_vi"]}
n = 0
for u in mj["don_vi"]:
    if u["images"].get("C (I1)") != u["images"].get("B (I0)"): continue
    pid = u["prompt_id"]; src = _loc(ur[pid]["images"]["C (I1)"], R); assert src.exists(), src; assert ur[pid]["seed"] == u["seed"]
    dst = M / pid / "C_ref" / "noop_ref" / src.name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst)
    u["images"]["C (I1)"] = str(dst); u["noop_ref_fix"] = True; u["P1"] = u["P_ct"]
    (M / pid / "kor.json").write_text(json.dumps(u, ensure_ascii=False, indent=1), encoding="utf-8"); n += 1; print(" ", pid)
(M / "kor.json").write_text(json.dumps(mj, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"đã sửa {n} đơn vị no-op -> ảnh refs-only trong {M.name} (sao lưu {bk.name})")
