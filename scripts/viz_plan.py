"""Viz kế hoạch lọc: hai lưới (đơn vị ĐỔI / KHÔNG ĐỔI), cột I0 · refs-only · SAVIER v1 · SAVIER v2 (nếu có), nhãn hàng = lý do luật.

    python scripts/viz_plan.py --plan plan_sdxl.json --main DIR --refs DIR [--v2 DIR] -o out_prefix
"""
import argparse, json, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def _font(sz):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; return ImageFont.truetype(p, sz) if Path(p).exists() else ImageFont.load_default()
def _loc(p, rd):
    parts = Path(p).parts; return rd.joinpath(*parts[len(parts) - 1 - parts[::-1].index(rd.name) + 1:]) if rd.name in parts else Path(p)
def units(d): rd = Path(d); return rd, {u["prompt_id"]: u for u in json.loads((rd / "kor.json").read_text(encoding="utf-8"))["don_vi"]}

ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--main", required=True); ap.add_argument("--refs", required=True)
ap.add_argument("--v2", default=""); ap.add_argument("-o", required=True); ap.add_argument("--cell", type=int, default=200); a = ap.parse_args()
plan = json.loads(Path(a.plan).read_text(encoding="utf-8")); rm, um = units(a.main); rr, ur = units(a.refs); rv, uv = units(a.v2) if a.v2 else (None, {})
cols = [("I0 (refined prompt)", rm, um, "B (I0)"), ("+ reference (refs-only)", rr, ur, "C (I1)"), ("SAVIER v1", rm, um, "C (I1)")] + ([("SAVIER v2 (lọc máy)", rv, uv, "C (I1)")] if a.v2 else [])
def tom_tat(p):
    ly = [x.split(":")[0].split("(")[0].strip() for x in p["ly_do"]]; c = {}
    for x in ly: c[x] = c.get(x, 0) + 1
    return "; ".join(f"{k}×{v}" if v > 1 else k for k, v in c.items())
for ten, sel in (("doi", True), ("khong_doi", False)):
    ids = [k for k, v in plan.items() if v["changed"] == sel]
    if not ids: continue
    cell, gut, head, pad, FS, LH = a.cell, 360, 30, 6, 11, 14; cw = cell + pad; ch = cell + pad
    im = Image.new("RGB", (gut + len(cols) * cw + pad, head + len(ids) * ch + pad), "white"); d = ImageDraw.Draw(im); f = _font(FS)
    for ci, (lab, _, _, _) in enumerate(cols): d.text((gut + ci * cw + 3, 8), lab, font=_font(12), fill=(20, 20, 20))
    for r, pid in enumerate(ids):
        y = head + r * ch; p = plan[pid]; u = um[pid]
        lines = textwrap.wrap(f"{pid}: {u.get('prompt_vi','')}", 52)[:2]
        if sel: lines += textwrap.wrap("luật: " + tom_tat(p), 52)[:2] + textwrap.wrap("action mới: " + ("; ".join(p["actions_new"]) or "(không — = refs-only)"), 52)[:5] + (textwrap.wrap("drop: " + "; ".join(p["drop_new"]), 52)[:1] if p["drop_new"] else [])
        else: lines += textwrap.wrap("action: " + ("; ".join(p["actions_new"]) or "(không)"), 52)[:4]
        for j, line in enumerate(lines[:12]): d.text((4, y + 4 + j * LH), line, font=f, fill=(20, 20, 20))
        for ci, (_, rd, us, key) in enumerate(cols):
            x = gut + ci * cw; uu = us.get(pid); pth = _loc(uu["images"].get(key, ""), rd) if uu and uu["images"].get(key) else None
            if not pth or not pth.exists(): d.text((x + 6, y + 6), "—", font=f, fill=(150, 150, 150)); continue
            t = Image.open(pth).convert("RGB"); t.thumbnail((cell, cell)); im.paste(t, (x, y))
    out = f"{a.o}_{ten}.png"; im.save(out, compress_level=1); print(f"-> {out} · {len(ids)} hàng · {Path(out).stat().st_size//1024} KB")
