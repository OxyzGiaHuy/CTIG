"""Hình định tính cho bài: hàng = prompt (cột trái: mã, EN, VI), cột = {SDXL, FLUX} × {Original prompt, Culture-TRIP (I₀), SAVIER (I₁)}.

Thiết kế: nền trắng, hai nhóm cột cách nhau một khe, tiêu đề nhóm đậm có gạch dưới, tiêu đề cột và chữ prompt cùng hệ số phóng
(--font-scale), VI in nghiêng xám, đường kẻ mảnh giữa các hàng.

    python scripts/qual_figure.py --ids S004,S010,S024,S034,S039,S042 --results results -o results/figures/qual_6prompts_512.png --cell 512 --font-scale 2
"""
import argparse, json, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FONTS = {"r": "DejaVuSans.ttf", "b": "DejaVuSans-Bold.ttf", "i": "DejaVuSans-Oblique.ttf"}
def font(sz, kind="r"):
    p = Path("/usr/share/fonts/truetype/dejavu") / FONTS[kind]; return ImageFont.truetype(str(p), sz) if p.exists() else ImageFont.load_default()

ap = argparse.ArgumentParser(); ap.add_argument("--ids", required=True); ap.add_argument("--results", default="results"); ap.add_argument("--prompts", default="data/prompts_simple.json")
ap.add_argument("-o", required=True); ap.add_argument("--cell", type=int, default=512); ap.add_argument("--models", default="sdxl:SDXL 1.0,flux:FLUX.1-dev")
ap.add_argument("--font-scale", type=float, default=2.0); a = ap.parse_args()
ids = a.ids.split(","); P = {p["id"]: p for p in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}
models = [m.split(":") for m in a.models.split(",")]; arms = [("A", "Original prompt"), ("I0", "Culture-TRIP (I₀)"), ("I1", "SAVIER (ours, I₁)")]
k = a.cell / 256; fs = a.font_scale
cell, pad, gap = a.cell, int(6 * k), int(28 * k)                      # pad giữa ô, gap giữa hai nhóm model
F_PID, F_TXT, F_ARM, F_MOD = int(13 * k * fs), int(11.5 * k * fs), int(11 * k * fs), int(14 * k * fs)
LH = int(F_TXT * 1.32); gut = int(330 * k * max(1.0, fs * 0.9))       # cột prompt
head = int(F_MOD * 1.6) + int(F_ARM * 1.7) + int(10 * k)
ncol = len(models) * len(arms); W = gut + ncol * (cell + pad) + (len(models) - 1) * gap + pad; H = head + len(ids) * (cell + pad) + pad
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
f_pid, f_txt, f_vi, f_arm, f_mod = font(F_PID, "b"), font(F_TXT), font(F_TXT, "i"), font(F_ARM), font(F_MOD, "b")
INK, GREY, LINE = (20, 20, 20), (95, 95, 95), (200, 200, 200)
def col_x(mi, ai): return gut + mi * (len(arms) * (cell + pad) + gap) + ai * (cell + pad)
for mi, (mk, mlab) in enumerate(models):
    x0, x1 = col_x(mi, 0), col_x(mi, len(arms) - 1) + cell
    d.text(((x0 + x1) / 2 - d.textlength(mlab, font=f_mod) / 2, int(6 * k)), mlab, font=f_mod, fill=INK)
    yl = int(6 * k) + int(F_MOD * 1.35); d.line((x0, yl, x1, yl), fill=INK, width=max(1, int(1.2 * k)))
    for ai, (_, alab) in enumerate(arms):
        x = col_x(mi, ai); d.text((x + (cell - d.textlength(alab, font=f_arm)) / 2, yl + int(8 * k)), alab, font=f_arm, fill=INK)
wrap = max(18, int(gut / (F_TXT * 0.56)))
for r, pid in enumerate(ids):
    y = head + r * (cell + pad); pr = P[pid]
    if r: d.line((int(6 * k), y - pad // 2, W - pad, y - pad // 2), fill=LINE, width=max(1, int(k)))
    d.text((int(8 * k), y + int(4 * k)), pid, font=f_pid, fill=INK); yy = y + int(4 * k) + int(F_PID * 1.5)
    for line in textwrap.wrap(pr["text_en"], wrap): d.text((int(8 * k), yy), line, font=f_txt, fill=INK); yy += LH
    yy += LH // 2
    for line in textwrap.wrap(pr["text_vi"], wrap):
        if yy + LH > y + cell: break
        d.text((int(8 * k), yy), line, font=f_vi, fill=GREY); yy += LH
    for mi, (mk, _) in enumerate(models):
        for ai, (arm, _) in enumerate(arms):
            p = Path(a.results) / mk / pid / f"{arm}.png"; x = col_x(mi, ai)
            if not p.exists(): d.text((x + 10, y + 10), "—", font=f_txt, fill=GREY); continue
            t = Image.open(p).convert("RGB"); t.thumbnail((cell, cell)); im.paste(t, (x, y))
Path(a.o).parent.mkdir(parents=True, exist_ok=True); im.save(a.o, compress_level=1); print(f"-> {a.o} {im.size} {Path(a.o).stat().st_size // 1024} KB")
