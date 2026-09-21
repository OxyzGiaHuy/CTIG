"""Rút JSON từ đầu ra model. Model mở không có ràng buộc schema nên phải chịu lỗi."""

from __future__ import annotations

import json
import re
from typing import Any


class JSONExtractError(ValueError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Tìm object JSON đầu tiên trong text và parse. Thử vài cách sửa nhẹ."""
    if not text or not text.strip():
        raise JSONExtractError("đầu ra rỗng")

    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates.extend(fenced)
    candidates.append(_first_balanced(text))
    candidates.append(text.strip())

    for cand in candidates:
        if not cand:
            continue
        for fixer in (lambda s: s, _strip_trailing_commas, _quote_single):
            try:
                obj = json.loads(fixer(cand))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    fixed = repair_truncated(text)
    if fixed is not None:
        fixed["_truncated"] = True          # JSON bị cụt ở trần token, đã cứu phần tử hoàn chỉnh (audit 21/09)
        return fixed
    raise JSONExtractError(f"không parse được JSON từ: {text[:200]!r}")


def repair_truncated(text: str) -> dict[str, Any] | None:
    """Cứu JSON bị cắt giữa chừng (hết max_new_tokens): lùi về dấu '}' đóng phần tử hoàn chỉnh gần cuối nhất,
    bỏ dấu phẩy treo, đóng các ngoặc còn mở rồi parse. Trả None nếu không cứu được."""
    start = text.find("{")
    if start < 0:
        return None
    s = re.sub(r"```(?:json)?", "", text[start:])
    ends = [m.start() for m in re.finditer(r"\}", s)]
    for cut in reversed(ends[-60:]):
        head = s[:cut + 1]
        stack, in_str, esc = [], False, False
        for ch in head:
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"': in_str = True
            elif ch in "{[": stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack: stack.pop()
        if in_str:
            continue
        cand = _strip_trailing_commas(head.rstrip().rstrip(",")) + "".join(reversed(stack))
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _first_balanced(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def _quote_single(s: str) -> str:
    # Chỉ dùng khi mọi cách khác thất bại: đổi nháy đơn thành nháy đôi.
    return s.replace("'", '"')


def schema_to_hint(schema: dict[str, Any]) -> str:
    """Biến JSON schema thành ví dụ rút gọn để nhắc model không có ràng buộc schema."""
    return json.dumps(_skeleton(schema), ensure_ascii=False, indent=1)


def _skeleton(node: dict[str, Any]) -> Any:
    t = node.get("type")
    if t == "object":
        return {k: _skeleton(v) for k, v in node.get("properties", {}).items()}
    if t == "array":
        return [_skeleton(node.get("items", {"type": "string"}))]
    if "enum" in node:
        return " | ".join(str(e) for e in node["enum"])
    return {"string": "...", "number": 0.0, "integer": 0, "boolean": True}.get(t, "...")
