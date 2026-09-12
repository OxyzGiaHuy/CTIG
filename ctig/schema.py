"""
Kiểu dữ liệu đi qua từng khối của pipeline CTIG.

    Prompt          -> Input
    AnalysisResult  -> Analysis Agent (keywords + keywords mới)
    SearchResult    -> Search (kb / wiki_text / image)
    CulturalSpec    -> Summary / Filter / Rank
    GenSpec         -> đầu vào Gen
    GenOutput       -> đầu ra Gen (N ứng viên, một ảnh được chọn)
    Perception      -> khối tri giác: ảnh -> những gì nhìn thấy (VLM + CLIP)
    Critique / Adjudication / RevisionPlan -> Loop Review
    EvalRecord      -> bảng Prompt | Kết quả | Evidence

Bất biến quan trọng: reviewer chỉ nhìn `Perception`, không nhìn bất kỳ sự thật
nội bộ nào của bộ sinh. Với bộ sinh thật thì không có sự thật nội bộ; với bộ sinh
stub (dùng cho test offline) thì `StubT2I` ghi sự thật vào `GenOutput.oracle`
và stage review không được đọc trường đó.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class Prompt:
    id: str
    text_vi: str
    text_en: str
    category: str = "general"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    #: Nhãn vàng, CHỈ dùng ở stage evaluate.
    gold_entities: list[str] = field(default_factory=list)
    note: str | None = None


# ---------------------------------------------------------------- stage 1

KeywordKind = Literal["entity", "scene", "attribute", "style", "region"]


@dataclass
class Keyword:
    term: str
    kind: KeywordKind
    source: Literal["surface", "expanded"]
    confidence: float = 1.0
    rationale: str | None = None


@dataclass
class NewEntity:
    """Thực thể văn hoá agent thấy cần nhưng KB chưa có. Sẽ được dựng bằng chứng lúc chạy."""

    name_vi: str
    name_en: str
    category: str = "other"
    region: str = "toan_quoc"
    rationale: str | None = None


@dataclass
class AnalysisResult:
    prompt_id: str
    keywords: list[Keyword]
    candidate_entity_ids: list[str]
    region_hint: str | None = None
    new_entities: list[NewEntity] = field(default_factory=list)
    #: Prompt tiếng Anh do agent viết lại cho bộ sinh (T2I hiểu tiếng Anh tốt hơn).
    prompt_en: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------- stage 2

EvidenceKind = Literal["kb", "wiki_text", "web_text", "image"]


@dataclass
class EvidenceItem:
    entity_id: str
    kind: EvidenceKind
    title: str
    snippet: str
    must_have: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    confusable_with: list[dict[str, str]] = field(default_factory=list)
    url: str | None = None
    #: Đường dẫn file ảnh tham chiếu đã tải về (kind == "image").
    local_path: str | None = None
    #: CLIP kiểm ảnh tham chiếu có đúng chủ thể không (kind == "image").
    clip_match: float | None = None
    score: float = 0.0
    provenance: str = "unknown"
    #: Với bằng chứng do VLM rút từ văn bản: thuộc tính -> trích đoạn gốc làm căn cứ.
    attr_sources: dict[str, str] = field(default_factory=dict)
    #: Truy vấn đã sinh ra item này (để bảng hiển thị nhóm theo truy vấn). None với KB.
    query: str | None = None
    #: "keyword" = truy vấn từ thực thể/keyword; "prompt" = truy vấn từ prompt gốc.
    query_group: str = "keyword"
    #: Ảnh này có được chọn làm tham chiếu IP-Adapter không (chỉ một ảnh mỗi thực thể).
    is_reference: bool = False


@dataclass
class SearchResult:
    prompt_id: str
    items: list[EvidenceItem]
    queries_used: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    retrieval_errors: list[str] = field(default_factory=list)
    #: Ghi chú không phải lỗi: thuộc tính bị loại vì không có câu gốc, nguồn bị bỏ, v.v.
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- stage 3

@dataclass
class SpecEntity:
    entity_id: str
    name_vi: str
    name_en: str
    required_attrs: list[str]
    forbidden_attrs: list[str]
    confusables: list[dict[str, str]]
    weight: float = 1.0
    evidence_titles: list[str] = field(default_factory=list)
    #: Bản tiếng Anh của thuộc tính, để ghép vào prompt T2I (T2I hiểu tiếng Anh tốt hơn).
    required_attrs_en: list[str] = field(default_factory=list)
    forbidden_attrs_en: list[str] = field(default_factory=list)
    #: Nhãn tiếng Anh MÔ TẢ cho CLIP (không phải tên trần). Ví dụ
    #: "a woman wearing a Vietnamese ao dai, a long split tunic over wide trousers".
    clip_label: str = ""
    #: Loại thực thể: "object" (vật thể, CLIP và IP-Adapter dùng được) hay "context"
    #: (sự kiện, cảnh; chỉ VLM checklist mới kiểm được).
    kind: str = "object"
    #: Ảnh tham chiếu đã qua kiểm CLIP, dùng cho IP-Adapter.
    reference_image: str | None = None


@dataclass
class CulturalSpec:
    prompt_id: str
    entities: list[SpecEntity]
    scene_notes: list[str] = field(default_factory=list)
    dropped: list[list[str]] = field(default_factory=list)

    def entity(self, entity_id: str) -> SpecEntity | None:
        return next((e for e in self.entities if e.entity_id == entity_id), None)


# ---------------------------------------------------------------- stage 4

@dataclass
class GenSpec:
    prompt_id: str
    #: Prompt là DANH SÁCH cụm, render bằng .prompt. Giữ danh sách để dedupe và
    #: giới hạn nhấn được; bản v1 dùng chuỗi nên lặp "Ao dai, Ao dai, Ao dai".
    prompt_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    #: entity_id -> số lần đã nhấn (đẩy lên đầu prompt). Tối đa 1.
    emphasis: dict[str, int] = field(default_factory=dict)
    #: entity_id -> mức nhấn 0..1. Bộ sinh thật hiện thực bằng thứ tự từ trong
    #: prompt và guidance; bộ sinh stub dùng thẳng con số.
    conditioning: dict[str, float] = field(default_factory=dict)
    lora: str | None = None
    lora_scale: float = 0.8
    #: Ảnh tham chiếu cho IP-Adapter, None nếu không dùng.
    ip_adapter_image: str | None = None
    ip_adapter_scale: float = 0.5
    seed: int = 0
    steps: int = 25
    guidance: float = 6.5
    width: int = 768
    height: int = 768
    n_candidates: int = 2
    iteration: int = 0
    #: Vòng nhanh (LCM-LoRA, ít bước) hay render đủ bước.
    fast: bool = False

    @property
    def prompt(self) -> str:
        return ", ".join(dict.fromkeys(t for t in self.prompt_terms if t))

    @property
    def negative_prompt(self) -> str:
        return ", ".join(dict.fromkeys(t for t in self.negative_terms if t))


@dataclass
class Candidate:
    path: str
    seed: int
    #: CLIP: P(đúng thực thể) trung bình có trọng số trên các thực thể trong spec.
    clip_fidelity: float = 0.0
    #: entity_id -> {nhãn: xác suất} từ CLIP probe.
    clip_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Model đã sinh ảnh này (multigen).
    model_id: str | None = None
    #: BLIP-2 ITM P(ảnh khớp mô tả thực thể), trung bình có trọng số (multigen).
    itm_score: float | None = None
    #: Cosine CLIP giữa ảnh và prompt_en (so được giữa ảnh, khác probs softmax).
    clip_prompt_sim: float | None = None
    #: CLIP tương phản mức THUỘC TÍNH: phần xác suất rơi vào các câu "<thực thể> with <must_have_en>"
    #: so với các câu "<thực thể> with <must_not_en>". Phân biệt được "áo dài có quần" với "váy liền".
    attr_contrast: float | None = None
    #: BLIP-2 ITM trung bình trên các câu must_have_en (độ đầy đủ thuộc tính).
    itm_attrs: float | None = None
    #: PickScore (CLIP-H tinh chỉnh theo sở thích người) thô, ~19-23; so được giữa ảnh cùng prompt (v1.3).
    pick_score: float | None = None
    #: PickScore chuẩn hoá min-max trong cùng một lần multigen (0..1), để vào điểm tổng.
    aesthetic: float | None = None
    #: Ảnh gốc trước hires fix (path trỏ tới ảnh cuối).
    base_path: str | None = None


@dataclass
class GenOutput:
    prompt_id: str
    iteration: int
    candidates: list[Candidate]
    #: Chỉ số ứng viên được chọn.
    chosen: int = 0
    #: CHỈ có với StubT2I: entity_id -> nhãn đã vẽ. Reviewer KHÔNG được đọc.
    oracle: dict[str, str] | None = None

    @property
    def image_path(self) -> str:
        return self.candidates[self.chosen].path


# ---------------------------------------------------------------- stage 4b: nhiều model

@dataclass
class ModelRun:
    """Một model sinh N ứng viên cho cùng GenSpec (đã điều chỉnh theo model)."""

    model_key: str
    repo: str
    gen_spec: GenSpec
    output: GenOutput | None = None
    seconds: float = 0.0
    peak_vram_gb: float | None = None
    error: str | None = None
    #: "computed" | "disk" - ảnh lấy lại từ lần chạy trước khi GenSpec và seed không đổi.
    source: str = "computed"
    #: Số token CLIP của prompt cuối (giới hạn 77; >75 thì cần compel, không thì bị cắt lặng lẽ).
    prompt_tokens: int | None = None
    #: Ghi chú không phải lỗi: "prompt dài, dùng compel", "hires bỏ qua vì OOM", "LoRA scale 0.6"...
    notes: list[str] = field(default_factory=list)


@dataclass
class MultiGenResult:
    prompt_id: str
    runs: list[ModelRun]
    prompt_en: str = ""
    grid_path: str | None = None
    #: sha1 của GenSpec đầu vào; đổi prompt/negative/seed là đổi hash -> ảnh cũ không được dùng lại.
    genspec_hash: str = ""


# ---------------------------------------------------------------- so sánh truy vấn (hiển thị)

@dataclass
class TextHit:
    query: str
    source: str
    title: str
    snippet: str
    url: str | None = None


@dataclass
class ImageHit:
    query: str
    source: str
    title: str
    url: str
    local_path: str | None = None
    #: cosine CLIP(ảnh, prompt_en) - so được giữa các ảnh.
    clip_prompt_sim: float | None = None
    #: P(ảnh khớp clip_label của thực thể) nếu truy vấn gắn với một thực thể.
    clip_entity_prob: float | None = None
    entity_id: str | None = None


@dataclass
class QueryColumn:
    label: str
    queries: list[str]
    text: list[TextHit] = field(default_factory=list)
    images: list[ImageHit] = field(default_factory=list)
    note: str = ""


@dataclass
class QueryComparison:
    prompt_id: str
    columns: list[QueryColumn]
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- tri giác

@dataclass
class VisualElement:
    label: str
    category: str = "object"
    attrs: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Perception:
    """Những gì khối tri giác nhìn thấy trong một ảnh. Đây là TẤT CẢ reviewer được biết."""

    image_path: str
    elements: list[VisualElement]
    caption: str = ""
    #: entity_id -> {nhãn: xác suất} từ CLIP. Tín hiệu độc lập với VLM.
    clip_probs: dict[str, dict[str, float]] = field(default_factory=dict)
    #: entity_id -> {"identity": "target"|"confusable:<tên>"|"absent"|"unsure",
    #:               "attrs": ["yes"|"no"|"unsure" theo thứ tự required_attrs],
    #:               "forbidden": ["yes"|"no"|"unsure" theo thứ tự forbidden_attrs]}
    #: VLM chỉ trả lời câu hỏi đóng. Điểm và findings suy ra bằng luật từ đây.
    checklist: dict[str, dict[str, Any]] = field(default_factory=dict)
    perceiver: str = "unknown"


# ---------------------------------------------------------------- stage 5

Severity = Literal["critical", "major", "minor"]


@dataclass
class Finding:
    entity_id: str
    severity: Severity
    observed: str
    expected: str
    message: str
    evidence_ref: str | None = None


@dataclass
class Critique:
    reviewer: str
    findings: list[Finding] = field(default_factory=list)
    score: float = 0.0
    verdict: Literal["pass", "revise"] = "pass"
    reasoning: str = ""


@dataclass
class Adjudication:
    score: float
    verdict: Literal["pass", "revise"]
    merged_findings: list[Finding] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class RevisionPlan:
    add_positive: list[str] = field(default_factory=list)
    add_negative: list[str] = field(default_factory=list)
    boost: dict[str, float] = field(default_factory=dict)
    attach_lora: bool = False
    use_reference_image: bool = False
    guidance_delta: float = 0.0
    rationale: str = ""

    def is_empty(self) -> bool:
        return not (self.add_positive or self.add_negative or self.boost
                    or self.attach_lora or self.use_reference_image or self.guidance_delta)


@dataclass
class ReviewIteration:
    n: int
    gen_spec: GenSpec
    gen_output: GenOutput
    perception: Perception
    critiques: list[Critique]
    adjudication: Adjudication
    plan: RevisionPlan
    #: True nếu đây là lần render đủ bước sau khi vòng nhanh đã đạt.
    final_render: bool = False


@dataclass
class ReviewOutcome:
    prompt_id: str
    iterations: list[ReviewIteration]
    passed: bool
    final_image_path: str
    final_score: float

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)


# ---------------------------------------------------------------- stage 6

@dataclass
class EvalRecord:
    prompt_id: str
    prompt_text: str
    final_image_path: str
    passed: bool
    iterations: int
    verifiable: bool = True
    retrieval_recall: float = -1.0
    review_score: float = 0.0
    clip_fidelity: float = 0.0
    judge_score: float = 0.0
    judge_reasoning: str = ""
    #: Judge theo model gì ("blip2_itm+clip" | "clip" | "vlm" | "rule").
    judge_backend: str = ""
    #: Chỉ có với stub.
    oracle_fidelity: float | None = None
    evidence_summary: list[str] = field(default_factory=list)
    residual_findings: list[str] = field(default_factory=list)
    iteration_images: list[str] = field(default_factory=list)


@dataclass
class RunSummary:
    run_id: str
    config: dict[str, Any]
    n_prompts: int
    n_unverifiable: int
    pass_rate: float
    pass_rate_iter0: float
    mean_iterations: float
    mean_clip_fidelity: float
    mean_clip_fidelity_iter0: float
    mean_retrieval_recall: float
    mean_judge_score: float
    mean_oracle_fidelity: float | None = None
    wall_seconds: float = 0.0


def from_dict(cls, data: Any) -> Any:
    """Dựng lại dataclass (lồng nhau) từ dict do to_dict() ghi ra. Dùng cho cache stage."""
    import typing
    from dataclasses import fields, is_dataclass

    if data is None or not is_dataclass(cls):
        return data
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints[f.name], data[f.name])
    return cls(**kwargs)


def _coerce(hint, value):
    import types
    import typing
    from dataclasses import is_dataclass

    if value is None:
        return None
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (types.UnionType, typing.Union):
        non_none = [a for a in args if a is not type(None)]
        return _coerce(non_none[0], value) if len(non_none) == 1 else value
    if origin is list:
        return [_coerce(args[0], v) for v in value] if args else list(value)
    if origin is dict:
        return {k: _coerce(args[1], v) for k, v in value.items()} if args else dict(value)
    if is_dataclass(hint):
        return from_dict(hint, value)
    return value


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj
