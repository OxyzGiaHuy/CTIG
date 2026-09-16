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
    #: số must_have_en của mỗi thực thể chính đưa vào prompt (v1.2.1: 2 -> 3 để giữ 'quần dài')
    attrs_in_prompt: int = 3
    #: Cách render prompt mặc định. v1.4 p001: "tags" KÉM hơn "legacy" (attr 0,49 vs 0,68 trên RealVis cùng seed) nên mặc định
    #: trở lại "legacy"; các biến thể "tags" (thẻ, không trọng số), "tags_w" (thẻ + trọng số), "legacy_negtags" (prompt v1.3 +
    #: negative thẻ) dùng qua hậu tố '#variant' để tách thủ phạm. Model họ sd3 luôn dùng "sentence".
    render: str = "legacy"
    #: trọng số compel cho thẻ phân biệt đầu tiên của thực thể chính (1.0 = không nhấn)
    emphasis_weight: float = 1.2


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
    #: v1.8.2: thư mục ảnh tham chiếu THEO PROMPT do nhóm chuẩn bị: <ref_dir>/selected/<prompt_id>/*.jpg (3 ảnh chọn tay) và
    #: <ref_dir>/candidates/<prompt_id>/*.jpg (~20 ảnh ứng viên). Có thì dùng làm TẦNG ƯU TIÊN NHẤT cho IP-Adapter và cho bước
    #: kiểm KB (ảnh thật của đúng prompt, không phải ảnh tìm bằng CLIP trong kho trộn). Nhiều thư mục cách nhau bằng dấu phẩy.
    ref_dir: str | None = None
    #: dùng cả ảnh 'candidates' (nhiều, chưa lọc tay) hay chỉ 'selected'
    ref_dir_candidates: bool = False
    #: v1.8: tri thức thực thể do Grounding TỰ DỰNG lúc chạy (LLM đọc Wikipedia/web -> bản ghi KB theo mẫu: 2 thuộc tính định
    #: danh trước, must_not theo cặp dễ nhầm, tags, clip_label, prior; mỗi thuộc tính kèm câu gốc; cache runs/_cache/kb_auto/).
    #:   "auto"      (mặc định): dựng cho MỌI thực thể trong spec, kể cả thực thể có bản tay. KB tay chỉ còn là danh mục
    #:               tên/alias để Analysis nhận thực thể, và là đường lùi khi nguồn không đủ (< 2 thuộc tính có câu gốc).
    #:   "hand"      : KB tay như v1.7, tự sinh chỉ cho thực thể thiếu must_have_en.
    #:   "hand_only" : không tự sinh (tái lập v1.7).
    kb_mode: str = "auto"
    #: (cũ, giữ tương thích) tắt hẳn tự sinh = kb_mode "hand_only"
    auto_kb: bool = True
    max_evidence_per_entity: int = 3
    download_images: bool = True
    #: CLIP tối thiểu để ảnh tìm được dùng làm tham chiếu IP-Adapter. Thấp hơn thì
    #: ảnh vẫn được ghi lại làm bằng chứng nhưng không đưa vào bộ sinh.
    ref_image_min_clip: float = 0.75
    #: v1.6 (ImageRAG): kho ảnh tham chiếu đánh chỉ mục CLIP (ctig/stages/refindex.py). Đường dẫn .npz; None = không dùng.
    #: Là nguồn TẦNG 0 cho ảnh tham chiếu và cho truy hồi theo caption thuộc tính trong vòng sửa; web search là đường lùi.
    ref_index: str | None = None
    ref_index_min_sim: float = 0.26
    ref_index_k: int = 6
    timeout: float = 10.0
    #: Số ký tự văn bản Wikipedia lấy về cho bước rút thuộc tính (0 = chỉ tóm tắt).
    wiki_chars: int = 3000
    #: Tải TOÀN VĂN trang web của top-k kết quả DuckDuckGo (snippet chỉ 100-300 ký tự, không đủ để rút thuộc tính).
    fetch_pages: int = 3
    page_chars: int = 4000
    #: Số nguồn văn bản tối đa đưa vào VLM rút bằng chứng cho một thực thể.
    extract_max_sources: int = 6
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
class HiresConfig:
    """Hires fix (v1.3): sinh ở kích cỡ gốc, phóng `scale` lần rồi img2img `strength` thấp cùng prompt.

    Tăng nét vải, hoa văn, mặt. Tốn thêm ~60% thời gian và VRAM đỉnh cao hơn (SDXL 1536px ~10 GB không offload).
    Chỉ family sdxl/sd15; OOM ở bước này thì giữ ảnh gốc, không làm hỏng hàng.
    """

    enabled: bool = False
    scale: float = 1.5
    strength: float = 0.3
    steps: int = 20
    #: hàng IP-Adapter bỏ hires: encoder ViT-H + img2img 1536 px vượt T4 (v1.4 p001 OOM)
    skip_ip_adapter: bool = True


@dataclass
class AestheticConfig:
    """Bộ chấm thẩm mỹ theo sở thích người (v1.3): PickScore v1 (CLIP-H tinh chỉnh trên 500k so sánh Pick-a-Pic).

    Nạp lười, để CPU và chỉ lên GPU khi chấm (offload) như BLIP-2. Tải ~3,9 GB lần đầu.
    """

    enabled: bool = True
    model: str = "yuvalkirstain/PickScore_v1"
    processor: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    device: str = "cuda:0"
    offload: bool = True


@dataclass
class AdaptiveConfig:
    """v1.5 best-of-N thích nghi (Ma et al. 2025, inference-time scaling): sinh `min` ứng viên, verifier (CLIP attr tốt nhất)
    chưa đạt `target_attr` thì sinh thêm `step`, tới `max`. Model đã đạt sớm thì tiết kiệm thời gian cho model yếu."""

    enabled: bool = True
    min: int = 2
    max: int = 6
    step: int = 2
    target_attr: float = 0.70


@dataclass
class AutoRefConfig:
    """v1.5 (ImageRAG): ảnh tham chiếu chỉ được cấp cho hàng '+ref' khi thực thể chính có prior thấp (model không tự vẽ được)
    hoặc không có trong KB. Prior cao thì ảnh chỉ mang rủi ro chép bố cục (v1.3 p001)."""

    enabled: bool = True
    prior_max: float = 0.45


@dataclass
class MultiGenConfig:
    """So nhiều model sinh ảnh trong một lần chạy (v1.2). Model nạp tuần tự, giải phóng sau mỗi model."""

    enabled: bool = False
    device: str = "cuda:0"
    cpu_offload: bool = True
    #: v1.3: 4 thay vì 2. p001 cho thấy phương sai theo seed > phương sai giữa model; best-of-N với điểm
    #: thuộc tính là cách rẻ nhất để ảnh cuối "chuẩn".
    n_candidates: int = 4
    #: Scheduler cho family sdxl/sd15: "dpmpp_2m_karras" | "euler" | None (giữ của repo). Turbo/Playground/SD3 giữ nguyên.
    scheduler: str | None = "dpmpp_2m_karras"
    #: Prompt > 75 token CLIP: nối embedding bằng compel thay vì để pipeline cắt lặng lẽ (SDXL/SD1.5).
    long_prompt: bool = True
    #: Số ảnh tham chiếu (đã qua CLIP) đưa vào IP-Adapter Plus cho hàng *_refplus / +ref.
    ref_images: int = 3
    #: v1.4.2: cắt ảnh tham chiếu về vùng thực thể (CLIP quét lưới) trước khi đưa vào IP-Adapter.
    ref_crop: bool = True
    #: Scale IP-Adapter cho hàng mang cờ '+ref' (hàng sdxl_refplus dùng scale trong registry).
    ref_scale: float = 0.4
    #: Ngưỡng "chép": cosine CLIP ảnh sinh - ảnh tham chiếu vượt ngưỡng thì bị trừ vào điểm tổng.
    copy_threshold: float = 0.88
    hires: HiresConfig = field(default_factory=HiresConfig)
    aesthetic: AestheticConfig = field(default_factory=AestheticConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    auto_ref: AutoRefConfig = field(default_factory=AutoRefConfig)
    #: v1.5: điểm chọn = trung bình hạng các verifier (Ma et al.) thay trung bình giá trị (verifier đơn bị "hack").
    ensemble: bool = True
    #: cách tìm vùng thực thể để cắt ảnh tham chiếu: "owlvit" (phát hiện theo chữ, lùi về clip khi lỗi) | "clip" (quét lưới)
    ref_detector: str = "owlvit"
    #: v1.5.1: CLIP identity và BLIP-2 ITM danh tính bão hoà 0,95-1,00 trên mọi ảnh (p001 v1.3-v1.5) -> mặc định KHÔNG tính
    #: ITM danh tính, không hiện hai cột này, không đưa vào điểm chọn. True để bật lại cho prompt khó.
    saturated_metrics: bool = False
    #: Kẹp cạnh dài của ảnh (SDXL 1024 -> 768 trên 1xT4 để tránh OOM).
    max_side: int = 768
    #: Chấm BLIP-2 ITM cho từng ảnh (dùng judge.blip2_model, offload CPU).
    itm: bool = True
    #: Ghi đè tham số theo model: {sdxl_turbo: {steps: 4}}.
    overrides: dict[str, dict] = field(default_factory=dict)
    #: Thư mục cache LoRA tải từ Civitai/HF.
    lora_dir: str | None = None


@dataclass
class AgentsConfig:
    """v1.4: ba agent cơ bản theo sơ đồ gốc (1 Summary, 2 Filter, 3 Rank) + một vòng sửa.

    Nguyên tắc (từ Culture-TRIP, CULTIVate, Marmot): VLM chỉ MÔ TẢ ảnh; việc 'có thuộc tính X không' là so văn bản
    mô tả với must_have/must_not (LLM văn bản + kiểm câu trích), không hỏi có/không trên ảnh để tránh thiên lệch 'có'.
    """

    enabled: bool = True
    #: Summary agent: tóm tắt tư liệu mỗi thực thể thành brief (facts EN/VI, khác gì với confusable, một câu vẽ thế nào)
    summary: bool = True
    #: nối depiction_en của brief vào prompt sinh (Culture-TRIP kiểu refine). Tắt mặc định để so A/B.
    enrich_prompt: bool = False
    #: Filter agent lọc ảnh tham chiếu trước IP-Adapter (bỏ ảnh nhóm, ảnh không có thực thể)
    ref_filter: bool = True
    #: Filter + Rank trên top-k ứng viên sau multigen
    candidate_review: bool = True
    k_candidates: int = 8
    #: số vòng Reflector/Refiner tối đa (v1.7: 3; A100 ~30 s/vòng, T4 ~4 phút/vòng). 0 = chỉ Reviewer + Rank.
    max_revisions: int = 3
    #: dừng sớm khi liên tiếp `patience` vòng không cải thiện điểm Reviewer
    patience: int = 2
    #: Reflector nhờ LLM viết caption truy hồi cho thuộc tính thiếu (ImageRAG); False = mẫu câu cố định
    llm_captions: bool = True
    #: model nền có hồ sơ chính (hiện trong cell/ảnh cuối của walkthrough); None = model nền đầu tiên trong models:
    primary_model: str | None = None
    #: chỉ chạy Reflector/Refiner cho các model nền này (tiết kiệm giờ GPU); None = tất cả. Model ngoài danh sách vẫn có Reviewer + Rank + ảnh cuối.
    loop_models: list[str] | None = None
    #: mô tả ảnh cần VLM; trên 1xT4 sau bước 4 phải nạp lại Qwen (~1 phút)
    reload_vlm: bool = True


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
    agents: AgentsConfig = field(default_factory=AgentsConfig)
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
