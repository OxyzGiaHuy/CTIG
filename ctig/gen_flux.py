"""Bộ sinh FLUX.1-dev TỐI GIẢN, tách khỏi multigen — cho flow K/O/R.

Vì sao tách: đường multigen mang theo thiết kế cũ (biến thể `#bare` xoá cờ `+ref`, cổng `should_use_refs`,
render lại prompt theo KB, ghi chú/twin/alias…). Ba lô FLUX liên tiếp hỏng cột ref theo ba cách khác nhau
mà không lỗi nào lộ ra ở log chính. Ở đây chỉ còn đúng bốn việc: nạp pipe · (tuỳ) gắn XLabs IP-Adapter ·
sinh một ảnh với seed cho trước · lưu. Không negative prompt (FLUX guidance-distilled không có), không KB.

    g = FluxGen(device="cuda:0", offload=True)
    g.sinh(prompt, seed, out_path)                      # text2img
    g.sinh(prompt, seed, out_path, refs=[p1, p2])       # + IP-Adapter (XLabs nhận 1 ảnh: lấy ảnh đầu)

Bộ nhớ: FLUX bf16 ~33 GB + Mistral-24B thường trú ~53 GB > 80 GB, nên mặc định `enable_model_cpu_offload()`
(~40 s/ảnh trên A100). Encoder ảnh của adapter là openai/clip-vit-large-patch14 — PHẢI có sẵn trong cache
HF khi chạy offline; thiếu thì raise ngay lúc gắn, không lặn vào ảnh None.
"""

from __future__ import annotations

import time
from pathlib import Path

REPO = "black-forest-labs/FLUX.1-dev"
IPA_REPO, IPA_FILE, IPA_ENCODER = "XLabs-AI/flux-ip-adapter", "ip_adapter.safetensors", "openai/clip-vit-large-patch14"


class FluxGen:
    def __init__(self, device: str = "cuda:0", offload: bool = True, steps: int = 28, guidance: float = 3.5,
                 width: int = 1024, height: int = 1024, ip_scale: float = 0.6, log=print):
        import torch
        from diffusers import FluxPipeline

        self.torch, self.log = torch, log
        self.steps, self.guidance, self.width, self.height, self.ip_scale = steps, guidance, width, height, ip_scale
        t0 = time.time()
        self.pipe = FluxPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
        if offload:
            self.pipe.enable_model_cpu_offload(device=device)
        else:
            self.pipe.to(device)
        self._ipa = False
        log(f"  [FluxGen] nạp FLUX.1-dev bf16 {'(cpu offload)' if offload else device} trong {time.time() - t0:.0f}s")

    def _gan_ipa(self):
        if self._ipa:
            return
        t0 = time.time()
        self.pipe.load_ip_adapter(IPA_REPO, weight_name=IPA_FILE, image_encoder_pretrained_model_name_or_path=IPA_ENCODER)
        # encoder nạp về CPU; với model offload thì pipe tự chuyển, nhưng dtype phải khớp bf16
        enc = getattr(self.pipe, "image_encoder", None)
        if enc is not None:
            enc.to(dtype=self.torch.bfloat16)
        self._ipa = True
        self.log(f"  [FluxGen] gắn XLabs IP-Adapter (encoder {IPA_ENCODER}) trong {time.time() - t0:.0f}s")

    def sinh(self, prompt: str, seed: int, out_path: str | Path, refs: list[str] | None = None) -> str:
        from PIL import Image

        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        kw = dict(prompt=prompt, num_inference_steps=self.steps, guidance_scale=self.guidance,
                  width=self.width, height=self.height, max_sequence_length=512,     # T5 đọc hết P1 dài
                  generator=self.torch.Generator("cpu").manual_seed(int(seed)))
        if refs:
            self._gan_ipa()
            self.pipe.set_ip_adapter_scale(self.ip_scale)
            kw["ip_adapter_image"] = Image.open(refs[0]).convert("RGB").resize((self.width, self.height))   # XLabs: 1 ảnh
        elif self._ipa:
            self.pipe.set_ip_adapter_scale(0.0)          # cùng pipe, tắt adapter cho hàng không ref
        t0 = time.time()
        img = self.pipe(**kw).images[0]
        img.save(out_path)
        self.log(f"  [FluxGen] {'ref ' + Path(refs[0]).name + ' ' if refs else ''}seed {seed} -> {out_path.name} ({time.time() - t0:.0f}s)")
        return str(out_path)
