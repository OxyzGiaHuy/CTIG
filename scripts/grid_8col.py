"""Lưới 10 prompt × 8 cột: mỗi model 4 bản — <model> · + refined prompt · + refined prompt + reference · + SAVIER (ours).

    python scripts/grid_8col.py --sdxl-main DIR --sdxl-refs DIR --flux-main DIR --flux-refs DIR --ids S001,...,S010 -o out.png --cell 768
"""
import argparse, json, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def _font(sz):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; return ImageFont.truetype(p, sz) if Path(p).exists() else ImageFont.load_default()
def _loc(p, rd):
    parts = Path(p).parts; return rd.joinpath(*parts[len(parts) - 1 - parts[::-1].index(rd.name) + 1:]) if rd.name in parts else Path(p)
def _units(d): rd = Path(d); return rd, {u["prompt_id"]: u for u in json.loads((rd / "kor.json").read_text(encoding="utf-8"))["don_vi"]}

ap = argparse.ArgumentParser()
for k in ("sdxl-main", "sdxl-refs", "flux-main", "flux-refs"): ap.add_argument(f"--{k}", required=True)
ap.add_argument("--ids", required=True); ap.add_argument("-o", "--out", required=True); ap.add_argument("--cell", type=int, default=768)
a = ap.parse_args(); ids = a.ids.split(",")
runs = {"SDXL": (_units(a.sdxl_main), _units(a.sdxl_refs)), "FLUX.1-dev": (_units(a.flux_main), _units(a.flux_refs))}
cols = []
for m, ((rm, um), (rr, ur)) in runs.items():
    cols += [(m, rm, um, "A"), (f"{m} + refined prompt", rm, um, "B (I0)"), (f"{m} + refined prompt + reference", rr, ur, "C (I1)"), (f"{m} + SAVIER (ours)", rm, um, "C (I1)")]
k = max(1.0, a.cell / 220); cell, gut, head, pad = a.cell, int(200 * k), int(34 * k), int(6 * k); FS, LH = int(11 * k), int(15 * k)
cw, ch = cell + pad, cell + pad
im = Image.new("RGB", (gut + len(cols) * cw + pad, head + len(ids) * ch + pad), "white"); d = ImageDraw.Draw(im)
for ci, (lab, _, _, _) in enumerate(cols): d.text((gut + ci * cw + 3, int(8 * head / 34)), lab, font=_font(int(12 * k)), fill=(20, 20, 20))
for r, pid in enumerate(ids):
    y = head + r * ch; pv = next((u[pid].get("prompt_vi", "") for _, (_, u) in [(0, runs["SDXL"][0])] if pid in u), "")
    for j, line in enumerate(textwrap.wrap(f"{pid}: {pv}", 28)[:9]): d.text((4, y + 6 + j * LH), line, font=_font(FS), fill=(20, 20, 20))
    for ci, (_, rd, us, key) in enumerate(cols):
        u = us.get(pid); x = gut + ci * cw
        p = _loc(u["images"].get(key, ""), rd) if u and u["images"].get(key) else None
        if not p or not p.exists(): d.text((x + 6, y + 6), "—", font=_font(FS), fill=(150, 150, 150)); continue
        t = Image.open(p).convert("RGB"); t.thumbnail((cell, cell)); im.paste(t, (x + (cell - t.width) // 2, y))
Path(a.out).parent.mkdir(parents=True, exist_ok=True); im.save(a.out, compress_level=1); print(f"-> {a.out} · {len(ids)}×{len(cols)} · {Path(a.out).stat().st_size // 1024} KB")
