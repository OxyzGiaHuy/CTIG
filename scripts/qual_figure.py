"""Hình định tính cho bài: hàng = prompt (cột đầu ghi EN + VI), cột = {SDXL, FLUX} × {Original (A), Culture-TRIP (I0), SAVIER (I1)}.

    python scripts/qual_figure.py --ids S004,S010,S024,S034,S039,S042 --results results -o results/figures/qual_6x6.png --cell 512
"""
import argparse, json, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def font(sz, bold=False):
    p = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"; return ImageFont.truetype(p, sz) if Path(p).exists() else ImageFont.load_default()

ap = argparse.ArgumentParser(); ap.add_argument("--ids", required=True); ap.add_argument("--results", default="results"); ap.add_argument("--prompts", default="data/prompts_simple.json")
ap.add_argument("-o", required=True); ap.add_argument("--cell", type=int, default=512); ap.add_argument("--models", default="sdxl:SDXL 1.0,flux:FLUX.1-dev"); ap.add_argument("--font-scale", type=float, default=1.0, help="hệ số phóng chữ cột prompt (2.0 = gấp đôi)"); a = ap.parse_args()
ids = a.ids.split(","); P = {p["id"]: p for p in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}
models = [m.split(":") for m in a.models.split(",")]; arms = [("A", "Original prompt"), ("I0", "Culture-TRIP ($I_0$)".replace("$I_0$", "I0")), ("I1", "SAVIER (ours, I1)")]
k = a.cell / 256; cell, pad, gut = a.cell, int(8 * k), int(300 * k * max(1.0, a.font_scale)); head1, head2 = int(30 * k), int(26 * k); FS = int(12 * k * a.font_scale); LH = int(16 * k * a.font_scale); FH = int(12 * k)
ncol = len(models) * len(arms); W = gut + ncol * (cell + pad) + pad; H = head1 + head2 + len(ids) * (cell + pad) + pad
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im); fb, fr, fh = font(int(14 * k * a.font_scale), True), font(FS), font(FH)
for mi, (mk, mlab) in enumerate(models):
    x0 = gut + mi * len(arms) * (cell + pad); x1 = x0 + len(arms) * (cell + pad) - pad
    d.text(((x0 + x1) // 2 - d.textlength(mlab, font=fb) // 2, int(6 * k)), mlab, font=fb, fill=(0, 0, 0)); d.line((x0, head1 - int(4 * k), x1, head1 - int(4 * k)), fill=(120, 120, 120), width=max(1, int(k)))
    for ai, (_, alab) in enumerate(arms):
        x = x0 + ai * (cell + pad); d.text((x + (cell - d.textlength(alab, font=fh)) // 2, head1 + int(4 * k)), alab, font=fh, fill=(20, 20, 20))
for r, pid in enumerate(ids):
    y = head1 + head2 + r * (cell + pad); pr = P[pid]; wrap = int(gut / (FS * 0.62))
    lines = [f"{pid}"] + textwrap.wrap("EN: " + pr["text_en"], wrap) + [""] + textwrap.wrap("VI: " + pr["text_vi"], wrap)
    for j, line in enumerate(lines[:int(cell / LH)]): d.text((int(6 * k), y + int(6 * k) + j * LH), line, font=fb if j == 0 else fr, fill=(0, 0, 0))
    for mi, (mk, _) in enumerate(models):
        for ai, (arm, _) in enumerate(arms):
            p = Path(a.results) / mk / pid / f"{arm}.png"; x = gut + (mi * len(arms) + ai) * (cell + pad)
            if not p.exists(): d.text((x + 10, y + 10), "—", font=fr, fill=(150, 150, 150)); continue
            t = Image.open(p).convert("RGB"); t.thumbnail((cell, cell)); im.paste(t, (x, y))
Path(a.o).parent.mkdir(parents=True, exist_ok=True); im.save(a.o, compress_level=1); print(f"-> {a.o} {im.size} {Path(a.o).stat().st_size // 1024} KB")
