"""Cấu hình đọc từ YAML. Xem configs/*.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LLMConfig:
    #: "qwen_vl" (local, GPU) | "anthropic" (API) | "rule" (offline, không cần model)
    backend: str = "qwen_vl"
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cuda:0"
    dtype: str = "auto"          # auto | float16 | bfloat16
    max_new_tokens: int = 1024
    temperature: float = 0.2
    #: Số lần thử lại khi đầu ra không phải JSON hợp lệ.
    json_retries: int = 2


@dataclass
class T2IConfig:
    #: "sdxl" (diffusers, GPU) | "stub" (offline)
    backend: str = "sdxl"
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae: str | None = "madebyollin/sdxl-vae-fp16-fix"
    device: str = "cuda:0"
    width: int = 768
    height: int = 768
    steps: int = 25
    guidance: float = 6.5
    n_candidates: int = 2
    #: Giảm VRAM khi VLM và SDXL cùng một GPU. Tắt nếu có 2 GPU.
    cpu_offload: bool = True
    #: LoRA văn hoá (đường dẫn hoặc repo HF). None = không có.
    lora_path: str | None = None
    lora_scale: float = 0.8
    #: IP-Adapter dùng ảnh tham chiếu từ Search.
    ip_adapter: bool = True
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_weight: str = "ip-adapter_sdxl.bin"
    ip_adapter_scale: float = 0.45


@dataclass
class PerceptionConfig:
    #: "vlm_clip" | "stub"
    backend: str = "vlm_clip"
    clip_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda:0"
    #: P(confusable) - P(target) vượt ngưỡng này thì CLIP báo lệch.
    drift_margin: float = 0.15


@dataclass
class RetrievalConfig:
    #: "local" | "wiki"
    backend: str = "wiki"
    max_evidence_per_entity: int = 3
    download_images: bool = True
    #: CLIP tối thiểu để ảnh Commons được dùng làm tham chiếu.
    ref_image_min_clip: float = 0.55
    timeout: float = 10.0


@dataclass
class ReviewConfig:
    max_iters: int = 2
    pass_threshold: float = 0.75
    #: Trọng số CLIP khi hợp điểm với VLM.
    clip_weight: float = 0.35


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    t2i: T2IConfig = field(default_factory=T2IConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    seed: int = 1234
    max_spec_entities: int = 4
    min_entity_score: float = 0.30
    kb_path: str = str(ROOT / "data" / "kb" / "entities.json")
    prompts_path: str = str(ROOT / "data" / "prompts_vi.jsonl")
    runs_dir: str = str(ROOT / "runs")
    run_name: str | None = None

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        cfg = cls()
        if path:
            import yaml

            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            cfg = _merge(cfg, raw)
        if overrides:
            cfg = _merge(cfg, overrides)
        return cfg

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(cfg: Any, raw: dict[str, Any]) -> Any:
    for k, v in raw.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Cấu hình không có trường '{k}'")
        cur = getattr(cfg, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            _merge(cur, v)
        else:
            setattr(cfg, k, v)
    return cfg


def set_dotted(overrides: dict[str, Any], dotted: str, value: str) -> None:
    """'t2i.steps=30' -> overrides['t2i']['steps'] = 30 (ép kiểu đơn giản)."""
    keys = dotted.split(".")
    d = overrides
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    v: Any = value
    if value.lower() in ("true", "false"):
        v = value.lower() == "true"
    elif value.lower() in ("none", "null"):
        v = None
    else:
        try:
            v = int(value)
        except ValueError:
            try:
                v = float(value)
            except ValueError:
                pass
    d[keys[-1]] = v
