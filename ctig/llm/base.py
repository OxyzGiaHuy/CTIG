"""
Hai tầng interface:

  LLMBackend  - gọi model thô: chat(system, user, images) -> str.
                Cài cho Qwen local (GPU) và Anthropic (API).
  Agent       - theo NHIỆM VỤ: analyze / build_spec / critique / adjudicate /
                plan_revision / judge. PromptAgent cài trên một LLMBackend bất kỳ;
                RuleAgent cài bằng luật để test offline.

Các stage chỉ biết Agent. Đổi model = đổi backend, không sửa stage.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..kb import KnowledgeBase
from ..schema import (
    Adjudication,
    AnalysisResult,
    Critique,
    CulturalSpec,
    GenSpec,
    Perception,
    Prompt,
    RevisionPlan,
    SearchResult,
)
from .json_utils import JSONExtractError, extract_json, schema_to_hint

PERSONA_CULTURAL = "CulturalExpert"


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str: ...

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any],
        images: list[str] | None = None,
    ) -> dict[str, Any]: ...


class JSONChatMixin:
    """Cài complete_json cho backend chỉ có chat(): nhắc schema, parse, thử lại. Có cache trên đĩa."""

    json_retries: int = 2
    name: str = "?"
    model_id: str = "?"

    def complete_json(self, system, user, schema, images=None):
        from . import cache as llm_cache

        c = llm_cache.current()
        key = c.key(self.name, getattr(self, "model_id", "?"), system, user, images) if c.enabled else None
        if key:
            hit = c.get(key)
            if hit is not None:
                return hit

        hint = schema_to_hint(schema)
        sys_full = (
            system
            + "\n\nCHỈ trả về MỘT object JSON hợp lệ, không có chữ nào ngoài JSON, "
            "không dùng markdown. Cấu trúc bắt buộc:\n" + hint
        )
        last_err: Exception | None = None
        prompt = user
        for attempt in range(self.json_retries + 1):
            text = self.chat(sys_full, prompt, images)  # type: ignore[attr-defined]
            try:
                result = extract_json(text)
                if key:
                    c.put(key, result, {"backend": self.name, "attempt": attempt})
                return result
            except JSONExtractError as exc:
                last_err = exc
                prompt = (
                    user + "\n\nLần trước bạn trả về không phải JSON hợp lệ. "
                    "Hãy trả về đúng một object JSON, bắt đầu bằng '{' và kết thúc bằng '}'."
                )
        raise RuntimeError(f"Không lấy được JSON sau {self.json_retries + 1} lần: {last_err}")


@runtime_checkable
class Agent(Protocol):
    name: str

    def analyze(self, prompt: Prompt, kb: KnowledgeBase) -> AnalysisResult: ...

    def build_spec(
        self, prompt: Prompt, analysis: AnalysisResult, search: SearchResult,
        kb: KnowledgeBase, max_entities: int, min_score: float,
    ) -> CulturalSpec: ...

    def critique(self, prompt: Prompt, spec: CulturalSpec, perception: Perception) -> Critique: ...

    def adjudicate(
        self, critique: Critique, perception: Perception, spec: CulturalSpec,
        threshold: float, clip_weight: float, drift_margin: float,
    ) -> Adjudication: ...

    def plan_revision(
        self, adjudication: Adjudication, spec: CulturalSpec, gen_spec: GenSpec,
        kb: KnowledgeBase, lora_available: bool, reference_available: bool,
    ) -> RevisionPlan: ...

    def judge(
        self, prompt: Prompt, spec: CulturalSpec, perception: Perception,
    ) -> tuple[float, str]: ...


def get_backend(cfg) -> LLMBackend:
    if cfg.backend == "qwen_vl":
        from .qwen_vl import QwenVLBackend

        return QwenVLBackend(cfg.model, cfg.device, cfg.dtype, cfg.max_new_tokens,
                             cfg.temperature, cfg.json_retries)
    if cfg.backend == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(cfg.model, cfg.max_new_tokens)
    raise ValueError(f"LLM backend không rõ: {cfg.backend!r}")


def get_agent(cfg) -> Agent:
    if cfg.backend == "rule":
        from .rule_agent import RuleAgent

        return RuleAgent()
    from .prompt_agent import PromptAgent

    return PromptAgent(get_backend(cfg))
