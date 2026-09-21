"""VQAScore CHUẨN (t2v_metrics, clip-flant5-xxl) cho mọi ảnh results/<model>/<pid>/<arm>.png với câu hỏi mặc định của gói
("Does this figure show "{text}"? Please answer yes or no.") trên P0 tiếng Anh. Ghi {"<MODEL>|<pid>|<arm>": score}.

    /workspace/venv_t2v/bin/python scripts/vqascore_official.py --root /workspace/vqa_in --prompts data/prompts_simple.json -o /workspace/runs/vqascore_official.json
"""
import argparse, json, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("-o", required=True)
ap.add_argument("--cache", default="/workspace/.hf_home/hub"); ap.add_argument("--batch", type=int, default=8); a = ap.parse_args()
import t2v_metrics
P = {p["id"]: p["text_en"] for p in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}
LAB = {"sdxl": "SDXL", "flux": "FLUX.1-dev"}
m = t2v_metrics.VQAScore(model="clip-flant5-xxl", cache_dir=a.cache); out = {}; t0 = time.time()
for mk, lab in LAB.items():
    for d in sorted((Path(a.root) / mk).glob("S0*")):
        pid = d.name; imgs = sorted(d.glob("*.png"))
        if not imgs or pid not in P: continue
        for i in range(0, len(imgs), a.batch):
            chunk = imgs[i:i + a.batch]; sc = m(images=[str(x) for x in chunk], texts=[P[pid]])
            for x, s in zip(chunk, sc[:, 0].tolist()): out[f"{lab}|{pid}|{x.stem}"] = round(float(s), 5)
    print(lab, "xong", len(out), f"{time.time()-t0:.0f}s", flush=True)
Path(a.o).write_text(json.dumps(out, indent=1), encoding="utf-8"); print("->", a.o, len(out), "điểm")
