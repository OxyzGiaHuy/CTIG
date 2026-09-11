"""
Gọi mạng cho bước Search, tách khỏi WikiRetriever để hai nơi dùng chung:

  * WikiRetriever (pipeline)        - truy hồi theo thực thể
  * compare_queries (notebook)      - so cột "keywords" với cột "prompt gốc"

Mọi lời gọi đi qua WebClient có CACHE JSON trên đĩa theo sha1(loại, truy vấn, vùng, n). Chạy lại
một cell không gọi DuckDuckGo lần hai (DDG giới hạn tần suất), và cache đi cùng runs_cache.zip.

Nguồn: DuckDuckGo (mặc định, không key, gói `ddgs`), Serper (SERPER_API_KEY), Wikimedia Commons.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

from ..kb import KnowledgeBase
from ..schema import AnalysisResult, ImageHit, Prompt, QueryColumn, QueryComparison, TextHit

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
SERPER_URL = "https://google.serper.dev/{kind}"
UA = "CTIG/0.2 (research prototype)"


def entity_queries(ent, langs: list[str]) -> list[tuple[str, str]]:
    """Truy vấn văn bản cho một thực thể: (câu hỏi, vùng). Dùng chung để cache trúng nhau."""
    en = ent.name_en.split("(")[0].strip()
    out = []
    if "vi" in langs:
        out.append((f"{ent.name_vi} đặc điểm cấu tạo hình dáng", "vn-vi"))
    if "en" in langs:
        out.append((f"Vietnamese {en} what it looks like characteristics", "wt-wt"))
    return out


def entity_image_queries(ent) -> list[str]:
    en = ent.name_en.split("(")[0].strip()
    return [en, ent.name_vi, f"{ent.name_vi} Việt Nam"]


class WebClient:
    """Mọi lời gọi mạng của bước Search, có cache trên đĩa."""

    def __init__(self, cfg, cache_dir: Path | str, sleep_after_ddg: float = 0.5):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.web_dir = self.cache_dir / "web"
        self.img_dir = self.cache_dir / "ref_images"
        self.web_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_after_ddg = sleep_after_ddg
        self.errors: list[str] = []
        self.calls = 0
        self.cache_hits = 0
        self._session = None
        self.serper_key = os.getenv("SERPER_API_KEY") if cfg.web_api == "serper" else None
        if cfg.web_api == "serper" and not self.serper_key:
            print("[web] web_api=serper nhưng không có SERPER_API_KEY, chỉ dùng Wikipedia/Commons")
        self._ddg = None
        if cfg.web_api == "ddg":
            try:
                from ddgs import DDGS

                self._ddg = DDGS()
            except ImportError:
                print("[web] web_api=ddg nhưng chưa cài gói ddgs (pip install ddgs); chỉ dùng Wikipedia/Commons")

    # ------------------------------------------------------------------ hạ tầng
    def _s(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers["User-Agent"] = UA
        return self._session

    def _cached(self, key: dict, fetch: Callable[[], list]) -> list:
        h = hashlib.sha1(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
        p = self.web_dir / f"{h}.json"
        if p.exists():
            try:
                self.cache_hits += 1
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        out = fetch()
        self.calls += 1
        try:
            p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return out

    # ------------------------------------------------------------------ text
    def text(self, q: str, region: str = "vn-vi", n: int = 5) -> list[dict]:
        """[{title, body, href, provenance}]"""
        def fetch():
            if self._ddg is not None:
                try:
                    out = self._ddg.text(q, region=region, max_results=n) or []
                    time.sleep(self.sleep_after_ddg)
                    return [{"title": r.get("title"), "body": r.get("body"), "href": r.get("href"),
                             "provenance": "duckduckgo"} for r in out]
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(f"ddg text '{q[:30]}': {type(exc).__name__}")
                    return []
            if self.serper_key:
                return [{"title": r.get("title"), "body": r.get("snippet"), "href": r.get("link"),
                         "provenance": "serper"} for r in self._serper("search", q, n)]
            return []

        return self._cached({"kind": "text", "api": self.cfg.web_api, "q": q, "region": region, "n": n}, fetch)

    # ------------------------------------------------------------------ ảnh
    def images(self, q: str, n: int = 6, region: str = "vn-vi") -> list[dict]:
        """[{image, title, provenance}] từ web image search."""
        def fetch():
            if self._ddg is not None:
                try:
                    out = self._ddg.images(q, region=region, max_results=n) or []
                    time.sleep(self.sleep_after_ddg)
                    return [{"image": r["image"], "title": r.get("title", q), "provenance": "duckduckgo-images"}
                            for r in out if r.get("image")]
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(f"ddg images '{q[:30]}': {type(exc).__name__}")
                    return []
            if self.serper_key:
                return [{"image": r["imageUrl"], "title": r.get("title", q), "provenance": "serper-images"}
                        for r in self._serper("images", q, n) if r.get("imageUrl")]
            return []

        return self._cached({"kind": "images", "api": self.cfg.web_api, "q": q, "region": region, "n": n}, fetch)

    def page_text(self, url: str, max_chars: int = 4000) -> str:
        """Toàn văn một trang web dạng text thô (bỏ script/style/tag), có cache. Rỗng nếu lỗi."""
        def fetch():
            try:
                r = self._s().get(url, timeout=self.cfg.timeout, headers={"Accept": "text/html"})
                if r.status_code != 200 or "html" not in r.headers.get("Content-Type", "html"):
                    return []
                return [_html_to_text(r.text)[: max_chars * 3]]
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"page {url[:40]}: {type(exc).__name__}")
                return []

        out = self._cached({"kind": "page", "url": url}, fetch)
        return out[0][:max_chars] if out else ""

    def commons(self, q: str, n: int = 3) -> list[dict]:
        """[{image, title, provenance:'commons'}] từ Wikimedia Commons."""
        def fetch():
            try:
                r = self._s().get(COMMONS_API_URL, params={
                    "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{q} Vietnam",
                    "gsrnamespace": "6", "gsrlimit": str(n), "prop": "imageinfo", "iiprop": "url|mime",
                    "iiurlwidth": "768",
                }, timeout=self.cfg.timeout)
                out = []
                for page in r.json().get("query", {}).get("pages", {}).values():
                    info = (page.get("imageinfo") or [{}])[0]
                    if info.get("mime", "").startswith("image/") and info.get("thumburl"):
                        out.append({"image": info["thumburl"], "title": page.get("title", q), "provenance": "commons"})
                return out
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"commons '{q[:30]}': {type(exc).__name__}")
                return []

        return self._cached({"kind": "commons", "q": q, "n": n}, fetch)

    def download(self, url: str) -> str | None:
        path = self.img_dir / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg")
        if path.exists():
            return str(path)
        try:
            r = self._s().get(url, timeout=self.cfg.timeout)
            if r.status_code != 200:
                return None
            from io import BytesIO
            from PIL import Image

            im = Image.open(BytesIO(r.content)).convert("RGB")
            im.thumbnail((1024, 1024))
            im.save(path, "JPEG", quality=90)
            self.calls += 1
            return str(path)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"download: {type(exc).__name__}")
            return None

    # ------------------------------------------------------------------ serper
    def _serper(self, kind: str, q: str, n: int) -> list[dict]:
        try:
            r = self._s().post(SERPER_URL.format(kind=kind), json={"q": q, "num": n, "gl": "vn", "hl": "vi"},
                               headers={"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
                               timeout=self.cfg.timeout)
            if r.status_code != 200:
                self.errors.append(f"serper {kind}: HTTP {r.status_code}")
                return []
            return r.json().get("organic" if kind == "search" else "images", [])[:n]
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"serper {kind}: {type(exc).__name__}")
            return []

    def drain_errors(self) -> list[str]:
        e, self.errors = list(self.errors), []
        return e


def _html_to_text(raw: str) -> str:
    """HTML -> text: bỏ script/style/nav, gộp khoảng trắng. Đủ cho VLM đọc, không cần thư viện ngoài."""
    import html as _h
    import re

    raw = re.sub(r"(?is)<(script|style|noscript|nav|footer|header|form)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", raw)
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = _h.unescape(txt)
    txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt)
    # Bỏ dòng quá ngắn (menu, nút) và dòng là JSON/script sót lại - giữ đoạn văn
    lines = []
    for l in txt.split("\n"):
        l = l.strip()
        if len(l) < 40:
            continue
        if sum(l.count(ch) for ch in "{}[]\"=") > 0.08 * len(l):
            continue
        lines.append(l)
    return "\n".join(lines)


# ======================================================================
# So sánh truy vấn: keywords vs prompt gốc (hiển thị trong notebook)
# ======================================================================

def compare_queries(web: WebClient, analysis: AnalysisResult, prompt: Prompt, kb: KnowledgeBase, clip=None,
                    k_text: int = 5, k_images: int = 6, max_entities: int = 6,
                    langs: list[str] | None = None) -> QueryComparison:
    """Hai cột: truy vấn sinh từ thực thể/keyword, và truy vấn từ prompt gốc.

    Cùng một hệ search, chỉ khác câu hỏi. Đây là cách rẻ nhất để thấy bước Analysis
    có giúp gì cho Search hay không: cột keywords ra ảnh đúng chủ thể hơn không?
    """
    langs = langs or ["vi", "en"]
    prompt_en = analysis.prompt_en or prompt.text_en

    # ---- cột 1: keywords / thực thể ----
    ids = list(analysis.candidate_entity_ids)
    note = ""
    if len(ids) > max_entities:
        note = f"Có {len(ids)} ứng viên, chỉ lấy {max_entities} đầu để tránh nổ truy vấn."
        ids = ids[:max_entities]
    kw_queries: list[tuple[str, str, str | None]] = []  # (q, region, entity_id)
    kw_img_queries: list[tuple[str, str | None]] = []
    for eid in ids:
        ent = kb.get(eid)
        if ent is None:
            continue
        kw_queries += [(q, region, eid) for q, region in entity_queries(ent, langs)]
        kw_img_queries += [(entity_image_queries(ent)[0], eid)]
    col_kw = _fill_column("Từ keywords / thực thể", kw_queries, kw_img_queries, web, kb, clip, prompt_en,
                          k_text, k_images, note)

    # ---- cột 2: prompt gốc ----
    raw_queries: list[tuple[str, str, str | None]] = [(prompt.text_vi, "vn-vi", None)]
    if "en" in langs and prompt_en:
        raw_queries.append((prompt_en, "wt-wt", None))
    col_raw = _fill_column("Từ prompt gốc", raw_queries, [(prompt.text_vi, None)], web, kb, clip, prompt_en,
                           k_text, k_images, "")

    return QueryComparison(prompt.id, [col_kw, col_raw], errors=web.drain_errors())


def _fill_column(label, text_queries, img_queries, web: WebClient, kb, clip, prompt_en, k_text, k_images, note) -> QueryColumn:
    col = QueryColumn(label=label, queries=[q for q, _, _ in text_queries], note=note)
    per_q = max(1, k_text // max(1, len(text_queries)))
    for q, region, _eid in text_queries:
        for r in web.text(q, region, per_q):
            if r.get("body"):
                col.text.append(TextHit(q, r.get("provenance", "web"), (r.get("title") or "")[:120],
                                        (r.get("body") or "")[:400], r.get("href")))
    col.text = col.text[:k_text]

    per_iq = max(1, k_images // max(1, len(img_queries)))
    for q, eid in img_queries:
        hits = web.commons(q, n=max(1, per_iq // 2)) + web.images(q, n=per_iq)
        for r in hits:
            local = web.download(r["image"])
            hit = ImageHit(q, r.get("provenance", "web"), (r.get("title") or q)[:80], r["image"], local, entity_id=eid)
            if local and clip is not None:
                try:
                    hit.clip_prompt_sim = clip.similarity(local, [prompt_en])[0] if prompt_en else None
                    ent = kb.get(eid) if eid else None
                    if ent is not None:
                        from ..llm.shared import confusable_clip_label

                        hit.clip_entity_prob = clip.image_matches(
                            local, ent.clip_label or f"a photo of Vietnamese {ent.name_en.split('(')[0].strip()}",
                            [confusable_clip_label(c) for c in ent.confusable_with])
                except Exception as exc:  # noqa: BLE001
                    web.errors.append(f"clip: {type(exc).__name__}")
            col.images.append(hit)
    col.images.sort(key=lambda h: -(h.clip_prompt_sim if h.clip_prompt_sim is not None else -1))
    col.images = col.images[:k_images]
    return col
