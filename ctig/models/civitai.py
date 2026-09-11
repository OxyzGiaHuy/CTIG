"""
Tải LoRA từ Civitai.

Civitai bắt đăng nhập để tải phần lớn file. Lấy token: civitai.com -> ảnh đại diện -> Account
settings -> API Keys -> Add API key. Trên Kaggle: Add-ons -> Secrets -> CIVITAI_TOKEN, rồi đọc
vào os.environ (xem notebook). Không có token: hàm trả None và ghi log, hàng LoRA bị bỏ qua.
"""

from __future__ import annotations

import os
from pathlib import Path

DOWNLOAD_URL = "https://civitai.com/api/download/models/{version_id}"


def download_civitai(version_id: int | str, cache_dir: Path | str, filename: str | None = None,
                     token: str | None = None, timeout: float = 600.0, log=print) -> Path | None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (filename or f"civitai_{version_id}.safetensors")
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    token = token or os.getenv("CIVITAI_TOKEN")
    try:
        import requests

        params = {"type": "Model", "format": "SafeTensor"}
        headers = {"User-Agent": "CTIG/0.2"}
        if token:
            params["token"] = token
        with requests.get(DOWNLOAD_URL.format(version_id=version_id), params=params, headers=headers,
                          stream=True, timeout=timeout, allow_redirects=True) as r:
            if r.status_code in (401, 403):
                log(f"[civitai] HTTP {r.status_code} cho version {version_id}: cần CIVITAI_TOKEN hợp lệ "
                    f"(civitai.com -> Account settings -> API Keys). Bỏ qua hàng LoRA này.")
                return None
            if r.status_code != 200:
                log(f"[civitai] HTTP {r.status_code} cho version {version_id}. Bỏ qua.")
                return None
            ctype = r.headers.get("Content-Type", "")
            if "text/html" in ctype:
                log("[civitai] Máy chủ trả về HTML (trang đăng nhập) thay vì file. Cần CIVITAI_TOKEN. Bỏ qua.")
                return None
            tmp = target.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        fh.write(chunk)
            tmp.rename(target)
        log(f"[civitai] đã tải {target.name} ({target.stat().st_size / 2**20:.0f} MB)")
        return target
    except Exception as exc:  # noqa: BLE001
        log(f"[civitai] lỗi tải version {version_id}: {type(exc).__name__}: {exc}")
        return None
