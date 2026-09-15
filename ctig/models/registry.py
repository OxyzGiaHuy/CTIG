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
    #: "base" = ip-adapter_sdxl (1 ảnh, encoder ViT-bigG) | "plus" = ip-adapter-plus_sdxl_vit-h (nhiều ảnh, chi tiết hơn).
    ip_adapter_kind: str = "base"
    #: Ghi đè scheduler của multigen.scheduler cho riêng model này (None = theo config; "keep" = giữ của repo).
    scheduler: str | None = None
    #: Có chạy hires fix (img2img phóng to) được không. Turbo/SD3 không.
    hires_ok: bool = True
    #: Negative khuyến nghị riêng của checkpoint (model card), nối sau negative chung (v1.4.1).
    extra_negative: list[str] = field(default_factory=list)
    #: Cách render prompt mặc định của model: None = theo t2i.render; sd3 -> "sentence".
    render: str | None = None
    est_vram_gb: float = 0.0
    experimental: bool = False
    notes: str = ""

    @property
    def display(self) -> str:
        return f"{self.key} ({self.repo.split('/')[-1]})"


REGISTRY: dict[str, ModelSpec] = {
    "sdxl_turbo": ModelSpec(
        "sdxl_turbo", "stabilityai/sdxl-turbo", "sdxl_turbo", 512, 512, steps=4, guidance=0.0,
        negative_ok=False, vae=SDXL_VAE_FIX, est_vram_gb=5.5, scheduler="keep", hires_ok=False,
        notes="Chưng cất từ SDXL; 1-4 bước, guidance 0 nên KHÔNG dùng negative prompt. Nhanh nhất."),
    "dreamshaper8": ModelSpec(
        "dreamshaper8", "Lykon/dreamshaper-8", "sd15", 512, 512, steps=30, guidance=7.0,
        est_vram_gb=2.5, load_kwargs={"safety_checker": None, "requires_safety_checker": False},
        extra_negative=["bad anatomy", "bad hands", "extra fingers", "poorly drawn face", "mutated", "lowres"],
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
    "realvis_xl": ModelSpec(
        "realvis_xl", "SG161222/RealVisXL_V4.0", "sdxl", 1024, 1024, steps=30, guidance=5.0,
        vae=SDXL_VAE_FIX, est_vram_gb=7.0,
        extra_negative=["worst quality", "low quality", "illustration", "3d", "2d", "painting", "cartoons", "sketch", "open mouth"],
        notes="SDXL fine-tune ảnh thực (người, vải, ánh sáng) - ứng viên thay sdxl_base làm baseline chất lượng (v1.3)."),
    "realvis_aodai": ModelSpec(
        "realvis_aodai", "SG161222/RealVisXL_V4.0", "sdxl", 1024, 1024, steps=30, guidance=5.0,
        vae=SDXL_VAE_FIX, est_vram_gb=7.0,
        extra_negative=["worst quality", "low quality", "illustration", "3d", "painting", "cartoons", "sketch", "open mouth"],
        lora={"source": "civitai", "version_id": 590793, "file": "jay_ao_dai_xl.safetensors",
              "trigger": "aodaixl", "scale": 0.8, "license": "CreativeML Open RAIL++-M, tác giả ghi 'no commercial use'"},
        only_if_entity=["ao_dai"],
        notes="RealVisXL + LoRA áo dài (cùng gốc SDXL nên LoRA dùng được). Đối chứng sdxl_aodai trên nền tốt hơn."),
    "sdxl_refplus": ModelSpec(
        "sdxl_refplus", "SG161222/RealVisXL_V4.0", "sdxl", 1024, 1024, steps=30, guidance=5.0,
        vae=SDXL_VAE_FIX, est_vram_gb=9.0, ip_adapter=True, ip_adapter_kind="plus", ip_adapter_scale=0.4,
        extra_negative=["worst quality", "low quality", "illustration", "painting", "cartoons", "sketch", "open mouth"],
        notes="RealVisXL + IP-Adapter Plus (ViT-H) với tối đa multigen.ref_images ảnh Commons đã qua CLIP (v1.3, H4). "
              "Scale 0.5 kéo cả bố cục ảnh tham chiếu (ảnh nhóm -> nhiều người); 0.4 và xếp ảnh theo độ khớp prompt."),
    "playground25": ModelSpec(
        "playground25", "playgroundai/playground-v2.5-1024px-aesthetic", "playground", 1024, 1024, steps=30, guidance=3.0,
        vae=None, est_vram_gb=9.5, scheduler="keep",
        notes="Kiến trúc SDXL, huấn luyện lại theo thẩm mỹ; scheduler EDM có sẵn trong repo, guidance thấp (3). GIỮ VAE repo: "
              "VAE của Playground mang latents_mean/std riêng, thay bằng fp16-fix (v1.3) ra ảnh bạc màu, mờ sương. Giải mã fp32 ~9,5 GB."),
    "sd35_medium": ModelSpec(
        "sd35_medium", "stabilityai/stable-diffusion-3.5-medium", "sd3", 1024, 1024, steps=28, guidance=4.5,
        variant=None, load_kwargs={"text_encoder_3": None, "tokenizer_3": None}, est_vram_gb=9.0, experimental=True,
        scheduler="keep", hires_ok=False, render="sentence",
        notes="SD3.5 Medium (2,5B MMDiT), bám prompt tốt hơn SDXL. Repo gated: cần HF_TOKEN + chấp nhận điều khoản. "
              "Bỏ T5 để vừa T4; T4 không có bf16 nên chạy fp16 - có thể ra ảnh lỗi số, vì vậy experimental."),
    "sd3_medium": ModelSpec(
        "sd3_medium", "stabilityai/stable-diffusion-3-medium-diffusers", "sd3", 1024, 1024, steps=28, guidance=7.0,
        variant=None, load_kwargs={"text_encoder_3": None, "tokenizer_3": None}, est_vram_gb=8.0, experimental=True,
        scheduler="keep", hires_ok=False, render="sentence",
        notes="Repo gated: cần HF_TOKEN và chấp nhận điều khoản. Bỏ T5 để vừa T4."),
    "hunyuan_dit": ModelSpec(
        "hunyuan_dit", "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", "hunyuan", 1024, 1024, steps=30, guidance=5.0,
        variant=None, est_vram_gb=12.0, experimental=True, scheduler="keep", hires_ok=False,
        notes="Model Trung Quốc song ngữ; đối chứng 'có kéo ảnh về Trung Quốc không'. Nặng (mT5), chỉ với offload."),
    "stub": ModelSpec(
        "stub", "-", "stub", 512, 512, steps=1, guidance=0.0, variant=None, est_vram_gb=0.0, scheduler="keep", hires_ok=False,
        notes="StubGenerator (thẻ chẩn đoán) cho test offline."),
}


RENDER_VARIANTS = ("legacy", "tags", "tags_w", "legacy_negtags", "sentence", "bare")


def parse_flags(key: str) -> set[str]:
    """Cờ '+ref' (bật IP-Adapter Plus với ảnh tham chiếu đã cắt cho hàng bất kỳ họ SDXL). 'realvis_aodai+ref#tags@0.8'."""
    flags = set()
    for part in key.split("+")[1:]:
        f = part.split("#")[0].split("@")[0]
        if f not in ("ref",):
            raise KeyError(f"Cờ '+{f}' của '{key}' không biết (chỉ có +ref)")
        flags.add(f)
    return flags


def _strip_flags(key: str) -> str:
    head, *rest = key.split("+")
    # hậu tố @/# có thể đứng sau cờ: gom lại
    tail = "".join(p[len(p.split("#")[0].split("@")[0]):] for p in rest)
    return head + tail


def parse_key(key: str) -> tuple[str, float | None]:
    """'sdxl_aodai@0.6#legacy' -> ('sdxl_aodai', 0.6). Hậu tố @ = LoRA scale, # = cách render prompt, +ref = ảnh tham chiếu."""
    key = _strip_flags(key)
    base, _, _variant = key.partition("#")
    base, _, tail = base.partition("@")
    if not tail:
        return base, None
    try:
        return base, float(tail)
    except ValueError as exc:
        raise KeyError(f"Hậu tố '@{tail}' của '{key}' phải là số (LoRA scale)") from exc


def parse_variant(key: str) -> str | None:
    """'realvis_xl#legacy' -> 'legacy'; không có # -> None (theo model/config)."""
    key = _strip_flags(key)
    _, _, v = key.partition("#")
    v = v.split("@")[0]
    if not v:
        return None
    if v not in RENDER_VARIANTS:
        raise KeyError(f"Hậu tố '#{v}' của '{key}' phải là một trong {RENDER_VARIANTS}")
    return v


def get(key: str) -> ModelSpec:
    key, _ = parse_key(key)
    if key not in REGISTRY:
        raise KeyError(f"Model '{key}' không có trong registry. Có: {', '.join(REGISTRY)}")
    return REGISTRY[key]
