"""CAIRE: (1) stage — chép ảnh I1 của các lô vào một thư mục tên băm (mù), ghi mapping; (2) parse — đọc combined_outputs.csv
của CAIRE ra caire.json {"<label>|<pid>|I1": {"Vietnam": s, "China": s}} và bảng trung bình.

    python scripts/caire_stage.py stage --run SDXL=/workspace/runs/kor_... --run SDXL-refs=/workspace/runs/abl_sdxl_refsonly ... -o /workspace/runs/caire_in
    python scripts/caire_stage.py parse --map /workspace/runs/caire_in/map.json --csv <combined_outputs.csv> -o /workspace/runs/caire_abl.json
"""
import argparse, ast, csv, hashlib, json, shutil, statistics as st
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["stage", "parse"]); ap.add_argument("--run", action="append", default=[])
ap.add_argument("--arms", default="I1"); ap.add_argument("--map"); ap.add_argument("--csv"); ap.add_argument("-o", required=True); a = ap.parse_args()
KEY = {"A": "A", "I0": "B (I0)", "I1": "C (I1)"}
if a.mode == "stage":
    out = Path(a.o); (out / "img").mkdir(parents=True, exist_ok=True); m = {}
    for spec in a.run:
        lab, _, d = spec.partition("="); rd = Path(d)
        for u in json.loads((rd / "kor.json").read_text(encoding="utf-8"))["don_vi"]:
            for arm in a.arms.split(","):
                p = u["images"].get(KEY[arm])
                if not p: continue
                parts = Path(p).parts; src = rd.joinpath(*parts[len(parts) - 1 - parts[::-1].index(rd.name) + 1:]) if rd.name in parts else Path(p)
                if not src.exists(): print("thiếu", lab, u["prompt_id"], arm); continue
                h = hashlib.sha1(f"{lab}|{u['prompt_id']}|{arm}".encode()).hexdigest()[:12]
                shutil.copy(src, out / "img" / f"{h}.png"); m[h] = {"run": lab, "pid": u["prompt_id"], "arm": arm}
    (out / "map.json").write_text(json.dumps(m, indent=1), encoding="utf-8"); print(f"staged {len(m)} ảnh -> {out/'img'}")
else:
    m = json.loads(Path(a.map).read_text(encoding="utf-8")); res = {}
    for row in csv.DictReader(open(a.csv, encoding="utf-8")):
        h = Path(row["Image_Path"]).stem
        if h not in m: continue
        sc = ast.literal_eval(row["Scores"]); k = m[h]; res[f"{k['run']}|{k['pid']}|{k['arm']}"] = {"Vietnam": sc.get("Vietnam"), "China": sc.get("China"), "entity": row.get("Matched Entity")}
    Path(a.o).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    runs = sorted({k.split("|")[0] for k in res})
    print(f"parsed {len(res)}/{len(m)}"); print("| lô | n | CAIRE-VN | CAIRE-CN |\n|---|---:|---:|---:|")
    for r in runs:
        v = [x for k, x in res.items() if k.startswith(r + "|")]
        print(f"| {r} | {len(v)} | {st.mean(x['Vietnam'] for x in v):.2f} | {st.mean(x['China'] for x in v):.2f} |")
