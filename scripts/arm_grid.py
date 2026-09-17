"""Lưới so NHIỀU NHÁNH: mỗi hàng một prompt, mỗi cột một (nhánh × model nền).

    python scripts/arm_grid.py A=runs/S001_armA B=runs/S001_armB C=runs/S001_armC -o arms.png [--cell 300] [--ids S001]

Khác `overview_grid.py` ở chỗ đó so bare với system BÊN TRONG một run; cái này so GIỮA các run, đúng thiết kế
ba nhánh A (prompt gốc) / B (Culture-TRIP) / C (B + agentic loop), vì mỗi nhánh chạy thành một run riêng.

Ô lấy ảnh CUỐI mà mỗi nhánh chọn (`per_model[].final_path`), lùi về ảnh c0 nếu chưa có vòng review.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

BORDER = {0: (150, 150, 150), 1: (196, 48, 43), 2: (22, 128, 82), 3: (28, 90, 168)}


def _font(size, bold=False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    cands = [f"/usr/share/fonts/truetype/dejavu/{name}"]
    try:
        import matplotlib

        cands.insert(0, str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name))
    except Exception:  # noqa: BLE001
        pass
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def read_run(run: str, ids: set[str] | None) -> dict[str, dict[str, tuple]]:
    """{prompt_id: {model: (path, score, nguồn, số vòng)}}"""
    out: dict[str, dict[str, tuple]] = {}
    for mg in sorted(glob.glob(f"{run}/*/multigen.json")):
        pid = Path(mg).parent.name
        if ids and pid not in ids:
            continue
        crp = f"{run}/{pid}/step_candidate_review.json"
        cr = json.load(open(crp))["value"] if os.path.exists(crp) else None
        ver = {v["path"]: v for v in (cr or {}).get("filter", {}).get("verdicts", [])}
        for x in ((cr or {}).get("per_model") or []):
            for it in x.get("iterations") or []:
                for v in ((it.get("filter") or {}).get("verdicts") or []):
                    ver[v["path"]] = v
        cells: dict[str, tuple] = {}
        for x in ((cr or {}).get("per_model") or []):
            fp = x.get("final_path")
            if fp and os.path.exists(fp):
                v = ver.get(fp, {})
                cells[x.get("base_model") or "?"] = (fp, v.get("score"), x.get("final_source", ""),
                                                     len(x.get("iterations") or []))
        if not cells:                      # chưa chạy review -> lấy ảnh đầu của mỗi hàng
            for r in json.load(open(mg))["runs"]:
                if r.get("output") and r["output"].get("candidates"):
                    k = r["model_key"].split("+")[0].split("#")[0]
                    cells.setdefault(k, (r["output"]["candidates"][0]["path"], None, "multigen", 0))
        if cells:
            out[pid] = cells
    return out


def build(arms: list[tuple[str, str]], out_png: str, cell: int = 300, ids: set[str] | None = None, log=print) -> str:
    from PIL import Image, ImageDraw

    data = {lab: read_run(run, ids) for lab, run in arms}
    pids = sorted({p for d in data.values() for p in d})
    models = []
    for d in data.values():
        for cells in d.values():
            for m in cells:
                if m not in models:
                    models.append(m)
    if not pids or not models:
        raise SystemExit("không đọc được run nào")

    pad, gut, head, cap = 8, 190, 74, 40
    cw, ch = cell + pad, cell + cap + pad
    cols = [(m, lab) for m in models for lab, _ in arms]
    W, H = gut + len(cols) * cw + pad, head + len(pids) * ch + pad
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f9, f11, f13 = _font(13), _font(15), _font(18, True)

    for i, (m, lab) in enumerate(cols):
        x = gut + i * cw
        if i == 0 or cols[i - 1][0] != m:
            d.text((x + 4, 6), m, font=f13, fill=(20, 20, 20))
            d.line([(x - 4, 2), (x - 4, H)], fill=(210, 210, 210), width=2)
        col = BORDER.get([l for l, _ in arms].index(lab) + 1, BORDER[0])
        d.text((x + 4, 44), lab, font=f11, fill=col)

    for r, pid in enumerate(pids):
        y = head + r * ch
        d.text((6, y + 6), pid, font=f13, fill=(20, 20, 20))
        for i, (m, lab) in enumerate(cols):
            x = gut + i * cw
            got = data[lab].get(pid, {}).get(m)
            if not got:
                d.text((x + 8, y + 8), "—", font=f11, fill=(150, 150, 150))
                continue
            path, score, src, nr = got
            try:
                p = Image.open(path).convert("RGB")
            except Exception:  # noqa: BLE001
                d.text((x + 8, y + 8), "không mở được", font=f9, fill=(196, 48, 43))
                continue
            p.thumbnail((cell, cell))
            ox, oy = x + (cell - p.width) // 2, y + (cell - p.height) // 2
            im.paste(p, (ox, oy))
            col = BORDER.get([l for l, _ in arms].index(lab) + 1, BORDER[0])
            d.rectangle([ox - 3, oy - 3, ox + p.width + 2, oy + p.height + 2], outline=col, width=3)
            st = "chưa chấm" if score is None else f"{score:+.2f}"
            d.text((x + 4, y + cell + 4), f"Reviewer {st}", font=f11, fill=col)
            d.text((x + 4, y + cell + 22), (f"{src} · {nr} vòng" if nr else src)[:34], font=f9, fill=(120, 120, 120))
        d.line([(0, y - 2), (W, y - 2)], fill=(225, 225, 225), width=1)

    im.save(out_png)
    log(f"{out_png} · {len(pids)} prompt × {len(models)} model × {len(arms)} nhánh · {os.path.getsize(out_png)//1024} KB")
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="NHÃN=đường_dẫn_run, ví dụ A=runs/S001_armA")
    ap.add_argument("-o", "--out", default="arms.png")
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--ids", default=None)
    a = ap.parse_args(argv)
    arms = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in a.arms if "=" in s]
    build(arms, a.out, a.cell, {i.strip() for i in a.ids.split(",")} if a.ids else None)


if __name__ == "__main__":
    main()
