"""Ghép I1 (và bản ghi đơn vị) từ một lô chạy lại vào lô chính, sau khi sao lưu lô chính.

    python scripts/merge_units.py --main /workspace/runs/kor_20260918_1349 --from /workspace/runs/v4_sdxl_main --ids S001,...
Điều kiện: cùng seed; A và I0 của lô mới là bản chép từ lô chính (--reuse-from) nên chỉ I1 + cards/gap/actions đổi.
"""
import argparse, json, shutil, time
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--main", required=True); ap.add_argument("--from", dest="src", required=True); ap.add_argument("--ids", required=True)
a = ap.parse_args(); M, S = Path(a.main), Path(a.src); ids = a.ids.split(",")
bk = M.with_name(M.name + f"_bak_{time.strftime('%H%M')}"); shutil.copytree(M, bk, dirs_exist_ok=True); print("sao lưu:", bk)
mj = json.loads((M / "kor.json").read_text(encoding="utf-8")); sj = json.loads((S / "kor.json").read_text(encoding="utf-8"))
new = {u["prompt_id"]: u for u in sj["don_vi"]}; n = 0
for i, u in enumerate(mj["don_vi"]):
    pid = u["prompt_id"]
    if pid not in ids or pid not in new: continue
    nu = new[pid]; assert nu["seed"] == u["seed"], pid
    for key in ("C (I1)", "C-text"):
        p = nu["images"].get(key)
        if p:
            parts = Path(p).parts; k = len(parts) - 1 - parts[::-1].index(S.name); rel = Path(*parts[k + 1:])
            dst = M / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(S / rel, dst)
            nu["images"][key] = str(dst)
    nu["images"]["A"], nu["images"]["B (I0)"] = u["images"]["A"], u["images"]["B (I0)"]
    (M / pid / "kor.json").write_text(json.dumps(nu, ensure_ascii=False, indent=1), encoding="utf-8")
    mj["don_vi"][i] = nu; n += 1
mj["giao_tiep"] = [r for r in mj.get("giao_tiep", []) if r["prompt_id"] not in ids] + [r for r in sj.get("giao_tiep", []) if r["prompt_id"] in ids]
(M / "kor.json").write_text(json.dumps(mj, ensure_ascii=False, indent=1), encoding="utf-8"); print(f"đã ghép {n} đơn vị vào {M}")
