"""Backend Claude API. Cần ANTHROPIC_API_KEY (trên Kaggle: Add-ons > Secrets)."""

from __future__ import annotations

import base64
import mimetypes
from typing import Any

from .json_utils import extract_json


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 16000, effort: str = "high"):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.model_id = model
        self.max_tokens = max(max_tokens, 4096)
        self.effort = effort
        self.calls = 0

    def _content(self, user: str, images: list[str] | None):
        blocks: list[dict[str, Any]] = []
        for p in images or []:
            mt = mimetypes.guess_type(p)[0] or "image/png"
            data = base64.standard_b64encode(open(p, "rb").read()).decode()
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        blocks.append({"type": "text", "text": user})
        return blocks

    def _create(self, **params):
        try:
            resp = self.client.messages.create(**params)
        except self._anthropic.AuthenticationError as exc:
            raise RuntimeError("Không xác thực được Claude API. Đặt ANTHROPIC_API_KEY.") from exc
        except self._anthropic.RateLimitError as exc:
            raise RuntimeError("Claude API: bị giới hạn tần suất.") from exc
        except self._anthropic.APIStatusError as exc:
            raise RuntimeError(f"Claude API lỗi {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise RuntimeError("Không kết nối được Claude API.") from exc
        if resp.stop_reason == "refusal":
            raise RuntimeError("Claude API từ chối yêu cầu.")
        self.calls += 1
        return next(b.text for b in resp.content if b.type == "text")

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        return self._create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": self._content(user, images)}],
            output_config={"effort": self.effort},
        )

    def complete_json(self, system, user, schema, images=None, max_new_tokens=None):  # max_new_tokens: tương thích JSONChatMixin
        from . import cache as llm_cache

        c = llm_cache.current()
        key = c.key(self.name, self.model, system, user, images) if c.enabled else None
        if key:
            hit = c.get(key)
            if hit is not None:
                return hit
        text = self._create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": self._content(user, images)}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        result = extract_json(text)
        if key:
            c.put(key, result, {"backend": self.name, "model": self.model})
        return result
