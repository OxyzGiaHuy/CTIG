"""
Cache mọi lần gọi LLM/VLM trên đĩa.

Khoá = sha1(backend, model, system, user, sha1 từng ảnh). Trùng khoá thì trả kết quả cũ, không gọi
model. Nhờ đó chạy lại một cell notebook không tốn API hay VLM, và cache sống qua restart kernel
(nằm trong runs/_cache/llm, đi cùng runs_cache.zip).

Đặt ở tầng backend (JSONChatMixin.complete_json và AnthropicBackend.complete_json) nên MỌI stage
dùng LLM đều hưởng: analysis, dịch, rút bằng chứng, checklist, judge.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        h.update(path.encode())
    return h.hexdigest()


class LLMCache:
    def __init__(self, cache_dir: Path | str | None):
        self.dir = Path(cache_dir) if cache_dir else None
        self.hits = 0
        self.misses = 0
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def key(self, backend: str, model: str, system: str, user: str, images: list[str] | None) -> str:
        sig = json.dumps({
            "backend": backend, "model": model, "system": system, "user": user,
            "images": [_file_sha1(p) for p in (images or [])],
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(sig.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if self.dir is None:
            return None
        p = self.dir / f"{key}.json"
        if not p.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.hits += 1
            return data.get("result")
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None

    def put(self, key: str, result: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        if self.dir is None:
            return
        try:
            (self.dir / f"{key}.json").write_text(
                json.dumps({"result": result, "meta": meta or {}}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def stats(self) -> str:
        return f"llm cache: {self.hits} hit / {self.misses} miss" if self.dir else "llm cache: tắt"


#: Cache dùng chung cho mọi backend trong tiến trình; pipeline/session gọi configure() một lần.
_GLOBAL = LLMCache(None)


def configure(cache_dir: Path | str | None) -> LLMCache:
    global _GLOBAL
    _GLOBAL = LLMCache(cache_dir)
    return _GLOBAL


def current() -> LLMCache:
    return _GLOBAL
