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
class AnalysisResult:
    prompt_id: str
    keywords: list[Keyword]
    candidate_entity_ids: list[str]
    region_hint: str | None = None
    #: Prompt tiếng Anh do agent viết lại cho bộ sinh (T2I hiểu tiếng Anh tốt hơn).
    prompt_en: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------- stage 2

EvidenceKind = Literal["kb", "wiki_text", "image"]


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


@dataclass
class SearchResult:
    prompt_id: str
    items: list[EvidenceItem]
    queries_used: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    retrieval_errors: list[str] = field(default_factory=list)


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
    prompt: str
    negative_prompt: str = ""
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


@dataclass
class Candidate:
    path: str
    seed: int
    #: CLIP: P(đúng thực thể) trung bình có trọng số trên các thực thể trong spec.
    clip_fidelity: float = 0.0
    #: entity_id -> {nhãn: xác suất} từ CLIP probe.
    clip_probs: dict[str, dict[str, float]] = field(default_factory=dict)


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


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj
