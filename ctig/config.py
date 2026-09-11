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
    #: Cache mọi lần gọi model theo (backend, model, system, user, ảnh) trên đĩa.
    #: Chạy lại cell trong notebook không tốn API/VLM. Tắt nếu muốn đầu ra đa dạng.
    cache: bool = True


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
    #: Skill SD để 0.6 cho style transfer; ta giữ danh tính vật thể trong cảnh phức tạp nên thấp hơn.
    ip_adapter_scale: float = 0.3
    #: Có thêm negative confusable ngay từ vòng 0 không (tắt để đo tác dụng của vòng review).
    init_negatives: bool = True
    #: Vòng sửa dùng LCM-LoRA ít bước; chỉ ảnh đạt mới render đủ bước.
    fast_iters: bool = False
    lcm_lora: str = "latent-consistency/lcm-lora-sdxl"
    fast_steps: int = 8
    fast_guidance: float = 1.5


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
    #: CLIP tối thiểu để ảnh tìm được dùng làm tham chiếu IP-Adapter. Thấp hơn thì
    #: ảnh vẫn được ghi lại làm bằng chứng nhưng không đưa vào bộ sinh.
    ref_image_min_clip: float = 0.75
    timeout: float = 10.0
    #: Số ký tự văn bản Wikipedia lấy về cho bước rút thuộc tính (0 = chỉ tóm tắt).
    wiki_chars: int = 3000
    #: Dùng VLM rút must_have / must_not / confusable_with từ văn bản truy hồi được.
    extract: bool = True
    #: Cache bằng chứng đã rút theo entity_id để không rút lại (không phụ thuộc prompt).
    evidence_cache: bool = True
    #: Web search. "ddg" (DuckDuckGo, không cần key, mặc định) | "serper" (SERPER_API_KEY) | "none".
    web_api: str = "ddg"
    web_results: int = 5
    #: Tìm web bằng cả tiếng Việt và tiếng Anh.
    web_langs: list[str] = field(default_factory=lambda: ["vi", "en"])


@dataclass
class CacheConfig:
    #: Bỏ qua analysis / search / spec khi prompt và cấu hình liên quan không đổi.
    enabled: bool = True
    #: Bỏ cache, chạy lại tất cả.
    refresh: bool = False
    dir: str | None = None  # mặc định <runs_dir>/_cache


@dataclass
class JudgeConfig:
    #: "blip2_itm" (BLIP-2 ITM + CLIP, độc lập với reviewer) | "clip" | "vlm" | "rule"
    backend: str = "blip2_itm"
    blip2_model: str = "Salesforce/blip2-itm-vit-g"
    device: str = "cuda:0"
    #: Giữ BLIP-2 ở CPU, chỉ chuyển lên GPU khi chấm (mỗi prompt một lần, ~2 giây chuyển). Tiết kiệm 2.5 GB VRAM.
    offload: bool = True


@dataclass
class ReviewConfig:
    #: False = chỉ sinh vòng 0 và chọn ứng viên bằng CLIP, không gọi VLM phê bình (v1.2 mặc định trong notebook).
    enabled: bool = True
    max_iters: int = 2
    pass_threshold: float = 0.75
    #: Trọng số CLIP khi hợp điểm với VLM.
    clip_weight: float = 0.35


@dataclass
class MultiGenConfig:
    """So nhiều model sinh ảnh trong một lần chạy (v1.2). Model nạp tuần tự, giải phóng sau mỗi model."""

    enabled: bool = False
    device: str = "cuda:0"
    cpu_offload: bool = True
    n_candidates: int = 2
    #: Kẹp cạnh dài của ảnh (SDXL 1024 -> 768 trên 1xT4 để tránh OOM).
    max_side: int = 768
    #: Chấm BLIP-2 ITM cho từng ảnh (dùng judge.blip2_model, offload CPU).
    itm: bool = True
    #: Ghi đè tham số theo model: {sdxl_turbo: {steps: 4}}.
    overrides: dict[str, dict] = field(default_factory=dict)
    #: Thư mục cache LoRA tải từ Civitai/HF.
    lora_dir: str | None = None


@dataclass
class SearchVizConfig:
    """Bước so sánh truy vấn keyword vs prompt gốc trong notebook."""

    k_text: int = 5
    k_images: int = 6
    #: Tối đa bao nhiêu thực thể ứng viên tạo truy vấn (chống nổ danh mục).
    max_entities: int = 6


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    t2i: T2IConfig = field(default_factory=T2IConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    multigen: MultiGenConfig = field(default_factory=MultiGenConfig)
    search_viz: SearchVizConfig = field(default_factory=SearchVizConfig)
    #: Khoá model trong ctig/models/registry.py dùng cho multigen.
    models: list[str] = field(default_factory=lambda: ["sdxl_base"])
    #: Tối đa số thực thể ứng viên stage 1 giữ lại (chống nổ danh mục như p050 v1.1: 37 ứng viên).
    max_candidate_entities: int = 6
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
