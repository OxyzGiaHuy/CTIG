"""Theo dõi lô bốn nhánh và dựng LẠI một lưới tích luỹ mỗi khi có prompt mới xong.

    python scripts/watch_arms.py /workspace/runs/arms16 -o /workspace/arms_live.png [--every 30]

Vì sao cần: lô 16 prompt chạy hơn một tiếng. Xem từng ảnh lẻ thì không thấy được QUY LUẬT — ví dụ chuyện
khung hình lệch có hệ thống (B/T/M cận cảnh còn I0/S toàn thân) chỉ lộ ra khi nhìn nhiều hàng cạnh nhau.
Script này chạy song song, không đụng vào lô, và cứ có thêm một đơn vị xong là vẽ lại cả lưới.

Mỗi HÀNG là một (prompt, lần lặp); năm cột là I0, B, T, S, M. Dưới mỗi ô ghi chú ngắn:
ô M ghi "no-op" nếu vòng sửa không tìm được lỗi nào, vì khi đó M trùng đúng ảnh B.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

COT = ("draft", "B", "T", "S", "M")
NHAN = {"draft": "nháp I0", "B": "B không agent", "T": "T không nhìn ảnh",
        "S": "S một agent", "M": "M đủ ba agent"}


def _font(size, bold=False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for p in (f"/usr/share/fonts/truetype/dejavu/{name}",
              "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def doc(run: Path) -> list[dict]:
    out = []
    for f in sorted(run.glob("*/arms.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue          # đang ghi dở
        d["_tag"] = f.parent.name
        out.append(d)
    return out


def ve(rows: list[dict], out_png: Path, cell: int = 260, log=print) -> None:
    from PIL import Image, ImageDraw

    if not rows:
        return
    pad, gut, head, cap = 6, 92, 34, 26
    cw, ch = cell + pad, cell + cap + pad
    W, H = gut + len(COT) * cw + pad, head + len(rows) * ch + pad + 20
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f9, f11, f12b = _font(12), _font(14), _font(15, True)

    for i, c in enumerate(COT):
        d.text((gut + i * cw + 4, 9), NHAN[c], font=f12b, fill=(20, 20, 20))

    for r, rec in enumerate(rows):
        y = head + r * ch
        d.text((5, y + 6), rec["_tag"], font=f11, fill=(20, 20, 20))
        if rec.get("noop_M"):
            d.text((5, y + 24), "M no-op", font=f9, fill=(196, 48, 43))
        for i, c in enumerate(COT):
            path = rec["draft"] if c == "draft" else (rec.get("images") or {}).get(c)
            x = gut + i * cw
            if not path or not Path(path).exists():
                d.text((x + 6, y + 6), "—", font=f11, fill=(150, 150, 150))
                continue
            p = Image.open(path).convert("RGB")
            p.thumbnail((cell, cell))
            im.paste(p, (x + (cell - p.width) // 2, y))
            # chú thích: mệnh đề sửa rút gọn, để soi được nhánh nào yêu cầu gì
            note = ""
            if c in ("T", "S", "M"):
                cl = ((rec.get("proposals") or {}).get(c) or {}).get("clause") or ""
                note = (cl[:40] + "…") if len(cl) > 40 else (cl or "no-op")
            d.text((x + 3, y + cell + 3), note, font=f9, fill=(110, 110, 110))
        d.line([(0, y - 2), (W, y - 2)], fill=(228, 228, 228), width=1)

    n_noop = sum(1 for x in rows if x.get("noop_M"))
    d.text((5, head + len(rows) * ch + 4),
           f"{len(rows)} đơn vị · M no-op {n_noop} ({100 * n_noop / len(rows):.0f}%)"
           f" · hàng = (prompt, lần lặp) · bốn nhánh cùng seed, chỉ khác câu prompt",
           font=f11, fill=(90, 90, 90))
    im.save(out_png)
    log(f"{out_png} · {len(rows)} hàng · {W}×{H}px · {out_png.stat().st_size // 1024} KB", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--every", type=int, default=30, help="giây giữa hai lần kiểm")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run)
    out = Path(a.out or run.parent / "arms_live.png")

    truoc = -1
    while True:
        rows = doc(run)
        if len(rows) != truoc:
            truoc = len(rows)
            try:
                ve(rows, out)
            except Exception as exc:  # noqa: BLE001
                print(f"vẽ lỗi {type(exc).__name__}: {exc}", flush=True)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
