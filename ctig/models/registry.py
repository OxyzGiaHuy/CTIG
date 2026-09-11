"""
Danh mục model sinh ảnh cho bước so nhiều model (multigen).

Chỉ chọn model VỪA một T4 16 GB khi nạp tuần tự (một model tại một thời điểm). Model gắn
`experimental=True` cần token hoặc chưa thử trên T4, không nằm trong config mặc định.

Thêm model mới: thêm một ModelSpec ở đây và đưa khoá vào `models:` trong YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SDXL_VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"

Family = Literal["sdxl", "sdxl_turbo", "sd15", "playground", "sd3", "hunyuan", "stub"]


@dataclass
class ModelSpec:
    key: str
    repo: str
    family: Family
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 6.5
    negative_ok: bool = True
    #: VAE thay thế (SDXL fp16 hay ra ảnh đen nếu không dùng bản fix). None = giữ VAE của repo.
    vae: str | None = None
    variant: str | None = "fp16"
    load_kwargs: dict = field(default_factory=dict)
    #: LoRA đi kèm: {"source": "civitai"|"hf"|"path", "version_id"/"repo"/"path", "file", "trigger", "scale"}
    lora: dict | None = None
    #: Chỉ chạy hàng này khi spec có ít nhất một trong các thực thể (vd LoRA áo dài).
    only_if_entity: list[str] | None = None
    #: Dùng ảnh tham chiếu (từ Search, đã qua CLIP) qua IP-Adapter. Chỉ family sdxl.
    ip_adapter: bool = False
    ip_adapter_scale: float = 0.3
    est_vram_gb: float = 0.0
    experimental: bool = False
    notes: str = ""

    @property
    def display(self) -> str:
        return f"{self.key} ({self.repo.split('/')[-1]})"


REGISTRY: dict[str, ModelSpec] = {
    "sdxl_turbo": ModelSpec(
        "sdxl_turbo", "stabilityai/sdxl-turbo", "sdxl_turbo", 512, 512, steps=4, guidance=0.0,
        negative_ok=False, vae=SDXL_VAE_FIX, est_vram_gb=5.5,
        notes="Chưng cất từ SDXL; 1-4 bước, guidance 0 nên KHÔNG dùng negative prompt. Nhanh nhất."),
    "dreamshaper8": ModelSpec(
        "dreamshaper8", "Lykon/dreamshaper-8", "sd15", 512, 512, steps=30, guidance=7.0,
        est_vram_gb=2.5, load_kwargs={"safety_checker": None, "requires_safety_checker": False},
        notes="SD 1.5 fine-tune phổ biến; nhẹ, hỗ trợ negative. Đại diện thế hệ SD1.5."),
    "sdxl_base": ModelSpec(
        "sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0", "sdxl", 1024, 1024, steps=30, guidance=6.5,
        vae=SDXL_VAE_FIX, est_vram_gb=7.0,
        notes="Baseline chính của pipeline (cùng model với đường review)."),
    "sdxl_aodai": ModelSpec(
        "sdxl_aodai", "stabilityai/stable-diffusion-xl-base-1.0", "sdxl", 1024, 1024, steps=30, guidance=6.5,
        vae=SDXL_VAE_FIX, est_vram_gb=7.0,
        lora={"source": "civitai", "version_id": 590793, "file": "jay_ao_dai_xl.safetensors",
              "trigger": "aodaixl", "scale": 0.8, "license": "CreativeML Open RAIL++-M, tác giả ghi 'no commercial use'"},
        only_if_entity=["ao_dai"],
        notes="SDXL + LoRA áo dài 'JAY - AO DAI XL' (Civitai model 531599). Cần CIVITAI_TOKEN để tải."),
    "sdxl_ref": ModelSpec(
        "sdxl_ref", "stabilityai/stable-diffusion-xl-base-1.0", "sdxl", 1024, 1024, steps=30, guidance=6.5,
        vae=SDXL_VAE_FIX, est_vram_gb=8.0, ip_adapter=True, ip_adapter_scale=0.3,
        notes="SDXL + IP-Adapter với ảnh tham chiếu Commons đã qua CLIP (kiểm H4). Bỏ qua nếu spec không có ảnh tham chiếu."),
    "playground25": ModelSpec(
        "playground25", "playgroundai/playground-v2.5-1024px-aesthetic", "playground", 1024, 1024, steps=30, guidance=3.0,
        vae=None, est_vram_gb=7.0,
        notes="Kiến trúc SDXL, huấn luyện lại theo thẩm mỹ; scheduler EDM có sẵn trong repo, guidance thấp (3). KHÔNG thay VAE."),
    "sd3_medium": ModelSpec(
        "sd3_medium", "stabilityai/stable-diffusion-3-medium-diffusers", "sd3", 1024, 1024, steps=28, guidance=7.0,
        variant=None, load_kwargs={"text_encoder_3": None, "tokenizer_3": None}, est_vram_gb=8.0, experimental=True,
        notes="Repo gated: cần HF_TOKEN và chấp nhận điều khoản. Bỏ T5 để vừa T4."),
    "hunyuan_dit": ModelSpec(
        "hunyuan_dit", "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", "hunyuan", 1024, 1024, steps=30, guidance=5.0,
        variant=None, est_vram_gb=12.0, experimental=True,
        notes="Model Trung Quốc song ngữ; đối chứng 'có kéo ảnh về Trung Quốc không'. Nặng (mT5), chỉ với offload."),
    "stub": ModelSpec(
        "stub", "-", "stub", 512, 512, steps=1, guidance=0.0, variant=None, est_vram_gb=0.0,
        notes="StubGenerator (thẻ chẩn đoán) cho test offline."),
}


def get(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Model '{key}' không có trong registry. Có: {', '.join(REGISTRY)}")
    return REGISTRY[key]
