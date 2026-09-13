"""
Bộ chấm THẨM MỸ theo sở thích người (v1.3): PickScore v1.

PickScore = CLIP ViT-H/14 tinh chỉnh trên ~500k so sánh cặp ảnh do người chọn (Pick-a-Pic). Cho một
(prompt, ảnh) nó trả một số ~19-23; ảnh cùng prompt so được với nhau, khác prompt thì không. Vì thế
multigen chuẩn hoá min-max TRONG một lần chạy (cùng prompt) thành `aesthetic` ∈ [0, 1] rồi mới đưa
vào điểm tổng. Nó đo "người thích ảnh nào hơn", KHÔNG đo đúng văn hoá; đúng văn hoá là việc của
CLIP attr / ITM attr.

Nạp lười, để CPU, lên GPU khi chấm rồi về CPU (như BLIP-2) để không chiếm VRAM lúc model sinh ảnh
đang chạy. Tải ~3,9 GB lần đầu. Không nạp được (mạng, VRAM) -> None, các cột khác vẫn có.
"""

from __future__ import annotations


class PickScorer:
    name = "pickscore"

    def __init__(self, cfg, log=print):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.offload = bool(cfg.offload) and self.device.startswith("cuda")
        dt = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.proc = AutoProcessor.from_pretrained(cfg.processor)
        self.model = AutoModel.from_pretrained(cfg.model, torch_dtype=dt).eval()
        self.model.to("cpu" if self.offload else self.device)
        log(f"[aesthetic] PickScore nạp xong ({cfg.model}, {'offload CPU' if self.offload else self.device})")

    def _on_gpu(self):
        if self.offload:
            self.model.to(self.device)

    def _off_gpu(self):
        if self.offload:
            self.model.to("cpu")
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

    def score(self, prompt: str, image_paths: list[str]) -> list[float]:
        """PickScore thô cho từng ảnh với cùng prompt (không softmax, để so giữa các lần gọi)."""
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        dev = self.device
        with self.torch.inference_mode():
            im = self.proc(images=imgs, return_tensors="pt").to(dev)
            tx = self.proc(text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
            im["pixel_values"] = im["pixel_values"].to(self.model.dtype)
            # transformers mới: get_*_features có thể trả ModelOutput thay tensor (v1.4 p001: 'no attribute norm').
            ie = _feat(self.model.get_image_features(**im))
            te = _feat(self.model.get_text_features(**tx))
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            s = self.model.logit_scale.exp() * (te.float() @ ie.float().T)
        return [round(float(v), 3) for v in s[0].tolist()]


def _feat(x):
    """Tensor đặc trưng từ đầu ra get_image_features/get_text_features, bất kể phiên bản transformers."""
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        if hasattr(x, name) and getattr(x, name) is not None:
            return getattr(x, name)
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


LAST_ERROR: str | None = None


def get_scorer(cfg, log=print):
    """PickScorer hoặc None nếu tắt / không nạp được (log lý do, giữ trong LAST_ERROR để báo cáo hiện được)."""
    global LAST_ERROR
    LAST_ERROR = None
    if not getattr(cfg, "enabled", False):
        return None
    try:
        return PickScorer(cfg, log=log)
    except Exception as exc:  # noqa: BLE001
        LAST_ERROR = f"{type(exc).__name__}: {str(exc)[:160]}"
        log(f"[aesthetic] không nạp được PickScore ({LAST_ERROR}) -> bỏ cột thẩm mỹ")
        return None


def normalize(values: list[float | None]) -> list[float | None]:
    """Min-max trong một lần multigen; tất cả bằng nhau -> 0.5."""
    xs = [v for v in values if v is not None]
    if not xs:
        return list(values)
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-6:
        return [None if v is None else 0.5 for v in values]
    return [None if v is None else round((v - lo) / (hi - lo), 4) for v in values]
