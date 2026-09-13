"""
Cắt ảnh tham chiếu về đúng THỰC THỂ trước khi đưa vào IP-Adapter (v1.4.2).

Vì sao: đưa cả bức ảnh thì embedding mang theo nền, số người, bố cục (v1.3: ảnh nhóm nữ sinh -> ảnh sinh 3-4 người).
Cắt vùng chứa thực thể rồi mới mã hoá thì embedding chỉ còn "vật đó trông thế nào". Nhiều ảnh cắt trung bình lại thì phần
riêng của từng bức triệt tiêu nhau.

Cách cắt: quét lưới ô vuông ở vài tỉ lệ (0,45 / 0,6 / 0,8 cạnh ngắn), chấm mỗi ô bằng CLIP: sim(ô, nhãn thực thể) - sim(ô,
nhãn nền/đám đông). Ô tốt nhất được nới 10 %, cắt vuông, lưu vào cache theo hash (ảnh, nhãn). Không cần detector.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

DISTRACTORS = ["a photo of a crowd of many people", "a photo of a street background with buildings and trees", "a blank wall"]


def _boxes(w: int, h: int, scales=(0.45, 0.6, 0.8), stride_frac: float = 0.5):
    s = min(w, h)
    for sc in scales:
        side = int(s * sc)
        if side < 64:
            continue
        step = max(16, int(side * stride_frac))
        ys = list(range(0, max(1, h - side + 1), step)) or [0]
        xs = list(range(0, max(1, w - side + 1), step)) or [0]
        if ys[-1] != h - side:
            ys.append(max(0, h - side))
        if xs[-1] != w - side:
            xs.append(max(0, w - side))
        for y in ys:
            for x in xs:
                yield (x, y, x + side, y + side)


def crop_to_entity(clip, path: str | Path, label: str, cache_dir: Path, margin: float = 0.10, min_gain: float = 0.02,
                   out_side: int = 768) -> tuple[str, dict]:
    """Trả (đường dẫn ảnh đã cắt, info). Không cắt (trả ảnh gốc) khi ô tốt nhất không hơn cả bức ảnh >= min_gain."""
    from PIL import Image

    path = Path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{path.name}|{path.stat().st_size}|{label}|{margin}".encode()).hexdigest()[:16]
    out = cache_dir / f"{key}.jpg"
    if out.exists():
        return str(out), {"cached": True}
    img = Image.open(path).convert("RGB")
    W, H = img.size
    labels = [label] + DISTRACTORS

    def score(im) -> float:
        s = clip.similarity_image(im, labels)
        return s[0] - max(s[1:])

    base = score(img)
    best, best_s = None, base
    for box in _boxes(W, H):
        s = score(img.crop(box))
        if s > best_s:
            best, best_s = box, s
    if best is None or best_s - base < min_gain:
        return str(path), {"cached": False, "cropped": False, "gain": round(best_s - base, 4)}
    x0, y0, x1, y1 = best
    m = int((x1 - x0) * margin)
    x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    crop = img.crop((x0, y0, x1, y1))
    side = max(crop.size)
    sq = Image.new("RGB", (side, side), (245, 244, 240))
    sq.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    if side > out_side:
        sq = sq.resize((out_side, out_side), Image.LANCZOS)
    sq.save(out, "JPEG", quality=92)
    return str(out), {"cached": False, "cropped": True, "box": [x0, y0, x1, y1], "gain": round(best_s - base, 4)}
