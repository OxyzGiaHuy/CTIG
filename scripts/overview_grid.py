"""Lưới so sánh TỔNG của một run: mỗi HÀNG là một prompt, mỗi CẶP CỘT là một model nền (bare rồi system).

    python scripts/overview_grid.py <run_dir> [out.png] [--cell 300] [--ids S001,S012] [--html]

Ô bare = ảnh bare tốt nhất của model nền đó theo điểm Reviewer.
Ô system = ảnh CUỐI mà vòng sửa thật sự chọn cho model nền đó (không phải ảnh tốt nhất của lô).
Viền đỏ = bare, viền xanh = system; ô nào thắng trong cặp thì viền dày và có dấu sao.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BARE, SYS, WIN, TIE = (196, 48, 43), (22, 128, 82), (30, 30, 30), (140, 140, 140)


def _font(size, bold=False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    cands = [f"/usr/share/fonts/truetype/dejavu/{name}", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"]
    try:
        import matplotlib

        cands.insert(0, str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name))
    except Exception:  # noqa: BLE001
        pass
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def base_of(k: str) -> str:
    return k.split("+")[0].split("#")[0].split("@")[0]


def is_bare(k: str) -> bool:
    return "#bare" in k


def collect(run: str, ids: list[str] | None):
    """[(prompt_id, prompt_en, {model nền: {'bare': (path, score), 'system': (path, score, nguồn, số vòng)}})]"""
    out, models = [], []
    for mg in sorted(glob.glob(f"{run}/*/multigen.json")):
        pid = Path(mg).parent.name
        if ids and pid not in ids:
            continue
        d = json.load(open(mg))
        crp = f"{run}/{pid}/step_candidate_review.json"
        cr = json.load(open(crp))["value"] if os.path.exists(crp) else None
        ver = {v["path"]: v for v in cr["filter"]["verdicts"]} if cr else {}
        for x in ((cr or {}).get("per_model") or []):          # ảnh vòng sửa nằm ở verdict riêng của từng vòng
            for it in x.get("iterations") or []:
                for v in ((it.get("filter") or {}).get("verdicts") or []):
                    ver[v["path"]] = v
        finals = {x.get("base_model"): x for x in ((cr or {}).get("per_model") or []) if x.get("final_path")}

        groups: dict[tuple[str, str], list] = {}
        for r in d["runs"]:
            if not r.get("output"):
                continue
            for c in r["output"]["candidates"]:
                groups.setdefault((base_of(r["model_key"]), "bare" if is_bare(r["model_key"]) else "system"), []).append(c)

        def sc(path):
            v = ver.get(path)
            return v["score"] if v else None

        row: dict[str, dict] = {}
        for b in dict.fromkeys(base_of(r["model_key"]) for r in d["runs"]):
            cell = {}
            cands = groups.get((b, "bare"), [])
            if cands:
                c = max(cands, key=lambda c: ((sc(c["path"]) if sc(c["path"]) is not None else -1.0), c.get("ensemble") or 0))
                cell["bare"] = (c["path"], sc(c["path"]), "bare tốt nhất", 0)
            f = finals.get(b)
            if f and os.path.exists(f["final_path"]):
                cell["system"] = (f["final_path"], sc(f["final_path"]), f.get("final_source", ""), len(f.get("iterations") or []))
            elif groups.get((b, "system")):
                c = max(groups[(b, "system")], key=lambda c: ((sc(c["path"]) if sc(c["path"]) is not None else -1.0), c.get("ensemble") or 0))
                cell["system"] = (c["path"], sc(c["path"]), "chưa có loop", 0)
            if cell:
                row[b] = cell
                if b not in models:
                    models.append(b)
        if row:
            out.append((pid, d.get("prompt_en", ""), row))
    return out, models


def build(run: str, out_png: str, cell: int = 300, ids: list[str] | None = None, log=print) -> str:
    from PIL import Image, ImageDraw

    rows, models = collect(run, ids)
    if not rows:
        raise SystemExit(f"không có prompt nào trong {run}")

    pad, gut, head, cap = 8, 190, 62, 40
    cw, ch = cell + pad, cell + cap + pad
    W = gut + len(models) * 2 * cw + pad
    H = head + len(rows) * ch + pad
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f9, f11, f13b = _font(13), _font(15), _font(18, True)

    for i, m in enumerate(models):
        x = gut + i * 2 * cw
        d.text((x + 4, 6), m, font=f13b, fill=(20, 20, 20))
        d.line([(x - 4, 2), (x - 4, H)], fill=(225, 225, 225), width=1)
        d.text((x + 4, 32), "bare", font=f11, fill=BARE)
        d.text((x + cw + 4, 32), "system", font=f11, fill=SYS)

    for r, (pid, prompt_en, row) in enumerate(rows):
        y = head + r * ch
        d.text((6, y + 6), pid, font=f13b, fill=(20, 20, 20))
        words, line, lines = prompt_en.split(), "", []
        for w in words:
            if len(line) + len(w) > 26:
                lines.append(line); line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        for j, l in enumerate(lines[:7]):
            d.text((6, y + 28 + j * 16), l, font=f9, fill=(110, 110, 110))

        for i, m in enumerate(models):
            got = row.get(m, {})
            pair = {k: got[k][1] for k in ("bare", "system") if k in got and got[k][1] is not None}
            best = max(pair, key=lambda k: pair[k]) if len(pair) == 2 and len(set(pair.values())) > 1 else None
            for j, lab in enumerate(("bare", "system")):
                x = gut + (i * 2 + j) * cw
                if lab not in got:
                    d.text((x + 8, y + 8), "—", font=f11, fill=TIE)
                    continue
                path, score, src, nrounds = got[lab]
                try:
                    p = Image.open(path).convert("RGB")
                except Exception:  # noqa: BLE001
                    d.text((x + 8, y + 8), "không mở được ảnh", font=f9, fill=BARE)
                    continue
                p.thumbnail((cell, cell))
                ox, oy = x + (cell - p.width) // 2, y + (cell - p.height) // 2
                im.paste(p, (ox, oy))
                col = BARE if lab == "bare" else SYS
                bw = 5 if best == lab else 2
                d.rectangle([ox - bw, oy - bw, ox + p.width + bw - 1, oy + p.height + bw - 1], outline=col, width=bw)
                star = "★ " if best == lab else ""
                st = "chưa chấm" if score is None else f"{score:+.2f}"
                d.text((x + 4, y + cell + 4), f"{star}Reviewer {st}", font=f11, fill=col)
                note = src if lab == "system" else ""
                if lab == "system" and nrounds:
                    note = f"{src} · {nrounds} vòng"
                d.text((x + 4, y + cell + 22), note[:34], font=f9, fill=(120, 120, 120))
        d.line([(0, y - 2), (W, y - 2)], fill=(225, 225, 225), width=1)

    im.save(out_png)
    log(f"{out_png} · {len(rows)} prompt × {len(models)} model nền · {W}×{H}px · {os.path.getsize(out_png) // 1024} KB")
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--ids", default=None)
    a = ap.parse_args(argv)
    out = a.out or str(Path(a.run) / "overview_grid.png")
    build(a.run, out, a.cell, [i.strip() for i in a.ids.split(",")] if a.ids else None)


if __name__ == "__main__":
    main()
