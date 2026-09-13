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


_OWL = {}


def detect_owlvit(img, label: str, device: str = "cuda:0", model_id: str = "google/owlvit-base-patch32",
                  min_score: float = 0.12, min_area: float = 0.04) -> tuple[tuple[int, int, int, int], float] | None:
    """v1.5: OWL-ViT phát hiện vật thể theo câu chữ -> hộp tốt nhất (x0, y0, x1, y1) và điểm; None nếu không có hộp đủ tin.
    Model ~600 MB, nạp lười một lần, giữ trên cùng GPU với CLIP. Mọi lỗi -> None để lùi về CLIP quét lưới."""
    try:
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        if "m" not in _OWL:
            dev = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
            _OWL["p"] = OwlViTProcessor.from_pretrained(model_id)
            _OWL["m"] = OwlViTForObjectDetection.from_pretrained(model_id).to(dev).eval()
            _OWL["dev"] = dev
        proc, model, dev = _OWL["p"], _OWL["m"], _OWL["dev"]
        short = label.split(",")[0].strip()
        texts = [[f"a photo of {short}" if not short.lower().startswith(("a ", "an ")) else short]]
        inputs = proc(text=texts, images=img, return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model(**inputs)
        target = torch.tensor([[img.height, img.width]], device=dev)
        res = proc.post_process_object_detection(outputs=out, threshold=min_score, target_sizes=target)[0]
        W, H = img.size
        best = None
        for box, sc in zip(res["boxes"].tolist(), res["scores"].tolist()):
            x0, y0, x1, y1 = [int(v) for v in box]
            area = max(0, x1 - x0) * max(0, y1 - y0) / (W * H)
            if area < min_area:
                continue
            if best is None or sc > best[1]:
                best = ((max(0, x0), max(0, y0), min(W, x1), min(H, y1)), float(sc))
        return best
    except Exception:  # noqa: BLE001
        return None


def crop_to_entity(clip, path: str | Path, label: str, cache_dir: Path, margin: float = 0.10, min_gain: float = 0.02,
                   out_side: int = 768, detector: str = "clip", device: str = "cuda:0") -> tuple[str, dict]:
    """Trả (đường dẫn ảnh đã cắt, info). detector="owlvit": hộp từ OWL-ViT, lỗi/không tin -> CLIP quét lưới.
    Không cắt (trả ảnh gốc) khi ô tốt nhất không hơn cả bức ảnh >= min_gain (đường CLIP)."""
    from PIL import Image

    path = Path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{path.name}|{path.stat().st_size}|{label}|{margin}|{detector}".encode()).hexdigest()[:16]
    out = cache_dir / f"{key}.jpg"
    if out.exists():
        return str(out), {"cached": True}
    img = Image.open(path).convert("RGB")
    W, H = img.size
    best, how = None, "clip"
    if detector == "owlvit":
        det = detect_owlvit(img, label, device=device)
        if det is not None:
            best, how = det[0], f"owlvit {det[1]:.2f}"
    best_s = base = None
    if best is None:
        labels = [label] + DISTRACTORS

        def score(im) -> float:
            s = clip.similarity_image(im, labels)
            return s[0] - max(s[1:])

        base = score(img)
        best_s = base
        for box in _boxes(W, H):
            s = score(img.crop(box))
            if s > best_s:
                best, best_s = box, s
        if best is None or best_s - base < min_gain:
            return str(path), {"cached": False, "cropped": False, "gain": round(best_s - base, 4), "how": how}
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
    return str(out), {"cached": False, "cropped": True, "box": [x0, y0, x1, y1], "how": how,
                      "gain": round(best_s - base, 4) if (best_s is not None and base is not None) else None}
