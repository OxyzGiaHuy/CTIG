"""Lưới gộp nhiều model: mỗi model 3 cột — <model> · <model> + Culture-TRIP · <model> + <phương pháp>.

    python scripts/kor_grid_models.py --run SDXL=/path/kor_2026..._1349 --run FLUX.1-dev=/path/kor_flux_... \
        -o docs/report_assets/grid_models.png [--method-name CG-MAPR]

Hàng = hợp các prompt có trong các lô; ô thiếu để trống. Cột thứ ba là ảnh I1 của phương pháp (không phải
ảnh cổng chọn), dưới ô ghi cổng chọn I0 hay I1 để người đọc thấy cả hai thông tin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _font(size, bold=False):
    from PIL import ImageFont
    n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    p = f"/usr/share/fonts/truetype/dejavu/{n}"
    return ImageFont.truetype(p, size) if Path(p).exists() else ImageFont.load_default()


def _theo_run(duong_dan: str, run_dir: Path) -> Path:
    """Đường dẫn trong kor.json là tuyệt đối trên máy thuê; đổi về vị trí thật của lô đã kéo về."""
    parts = Path(duong_dan).parts
    ten = run_dir.name
    if ten in parts:
        i = len(parts) - 1 - parts[::-1].index(ten)
        return run_dir.joinpath(*parts[i + 1:])
    return Path(duong_dan)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="<nhãn model>=<thư mục lô>")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--method-name", default="CG-MAPR")
    ap.add_argument("--cell", type=int, default=260)
    a = ap.parse_args(argv)
    from PIL import Image, ImageDraw

    runs = []
    for spec in a.run:
        nhan, _, d = spec.partition("=")
        rd = Path(d); j = json.loads((rd / "kor.json").read_text(encoding="utf-8"))
        runs.append((nhan, rd, {u["prompt_id"]: u for u in j["don_vi"]}))
    pids = sorted({p for _, _, u in runs for p in u})
    cot = []                                  # (nhãn cột, chỉ số run, khoá ảnh)
    for i, (nhan, _, _) in enumerate(runs):
        cot += [(nhan, i, "A"), (f"{nhan} + Culture-TRIP", i, "B (I0)"), (f"{nhan} + {a.method_name}", i, "C (I1)")]

    cell, gut, head, pad, cap = a.cell, 60, 34, 6, 20
    cw, ch = cell + pad, cell + cap + pad
    W, H = gut + len(cot) * cw + pad, head + len(pids) * ch + pad
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    for k, (nh, _, _) in enumerate(cot):
        d.text((gut + k * cw + 3, 8), nh, font=_font(12, True), fill=(20, 20, 20))
    for r, pid in enumerate(pids):
        y = head + r * ch
        d.text((4, y + 6), pid, font=_font(12), fill=(20, 20, 20))
        for k, (_, ri, key) in enumerate(cot):
            _, rd, us = runs[ri]; u = us.get(pid); x = gut + k * cw
            p = _theo_run(u["images"].get(key, ""), rd) if u and u["images"].get(key) else None
            if not p or not p.exists():
                d.text((x + 6, y + 6), "—", font=_font(12), fill=(150, 150, 150)); continue
            t = Image.open(p).convert("RGB"); t.thumbnail((cell, cell))
            im.paste(t, (x + (cell - t.width) // 2, y))
            if key == "C (I1)":
                g = (u.get("gate_C") or {}); txt = f"cổng: {g.get('selection', '?')}" if g else ("no-op" if not u.get("actions") else "")
                d.text((x + 3, y + cell + 3), txt, font=_font(11), fill=(110, 110, 110))
        d.line([(0, y - 2), (W, y - 2)], fill=(225, 225, 225))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); im.save(out)
    print(f"-> {out} · {len(pids)} hàng × {len(cot)} cột · {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
