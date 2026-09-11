"""
Stage 2 - SEARCH: kb / wiki_text / image, gọi API lúc chạy.

Nguồn (tất cả không cần key trừ Serper):
  * KB gốc trong repo - bằng chứng đã có sẵn, provenance "kb".
  * Wikipedia tiếng Việt: tìm bài (list=search) khi chưa biết tên bài, rồi lấy văn bản dài
    (prop=extracts) để stage extraction rút thuộc tính.
  * Wikimedia Commons: ảnh tham chiếu, tải về, CLIP kiểm trước khi dùng cho IP-Adapter.
  * DuckDuckGo (mặc định, KHÔNG cần key, gói `ddgs`): web tiếng Việt và tiếng Anh + ảnh.
    Wikipedia cho lịch sử; bài blog / báo tiếng Việt mới cho "sườn nón là nan tre, quai buộc đối xứng".
  * Serper (tuỳ chọn, SERPER_API_KEY): Google web + images.

Stage này chỉ THU văn bản. Việc biến văn bản thành must_have/must_not nằm ở stages/extraction.py.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..kb import KnowledgeBase
from ..schema import AnalysisResult, EvidenceItem, SearchResult

WIKI_API = "https://vi.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://vi.wikipedia.org/api/rest_v1/page/summary/{title}"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
SERPER_URL = "https://google.serper.dev/{kind}"
UA = "CTIG/0.1 (research prototype)"


def _confidence_map(analysis: AnalysisResult, kb: KnowledgeBase) -> dict[str, float]:
    out: dict[str, float] = {}
    by_name = {e.name_vi: e.id for e in kb.all()}
    for kw in analysis.keywords:
        if kw.kind == "entity" and kw.term in by_name:
            eid = by_name[kw.term]
            out[eid] = max(out.get(eid, 0.0), kw.confidence)
    return out


class LocalRetriever:
    name = "local"

    def __init__(self, cfg, clip_probe=None, cache_dir: Path | None = None):
        self.cfg = cfg
        self.clip = clip_probe
        self.cache_dir = cache_dir or Path("runs/_cache/ref_images")

    def search(self, analysis: AnalysisResult, kb: KnowledgeBase) -> SearchResult:
        conf = _confidence_map(analysis, kb)
        items, misses, queries = [], [], []
        for eid in analysis.candidate_entity_ids:
            ent = kb.get(eid)
            if ent is None:
                misses.append(eid)
                continue
            base = conf.get(eid, 0.5)
            queries.append(ent.name_vi)
            if ent.must_have:  # thực thể ad-hoc chưa có gì trong KB thì không tạo item kb rỗng
                items.append(EvidenceItem(
                    entity_id=eid, kind="kb", title=f"KB: {ent.name_vi}",
                    snippet=f"{ent.name_vi} ({ent.name_en}). {ent.notes or ''}".strip(),
                    must_have=list(ent.must_have), must_not=list(ent.must_not),
                    confusable_with=list(ent.confusable_with), score=base, provenance=f"kb@{kb.version}",
                ))
            if ent.notes and not ent.id.startswith("x_"):
                items.append(EvidenceItem(
                    entity_id=eid, kind="wiki_text", title=f"Ghi chú: {ent.name_vi}", snippet=ent.notes,
                    score=base * 0.75, provenance="kb.notes (offline)",
                ))
        return SearchResult(analysis.prompt_id, items, queries, misses)


class WikiRetriever(LocalRetriever):
    name = "wiki"

    def __init__(self, cfg, clip_probe=None, cache_dir=None):
        super().__init__(cfg, clip_probe, cache_dir)
        self._session = None
        self.errors: list[str] = []
        self.serper_key = os.getenv("SERPER_API_KEY") if cfg.web_api == "serper" else None
        if cfg.web_api == "serper" and not self.serper_key:
            print("[retrieval] web_api=serper nhưng không có SERPER_API_KEY, chỉ dùng Wikipedia")
        self._ddg = None
        if cfg.web_api == "ddg":
            try:
                from ddgs import DDGS

                self._ddg = DDGS()
            except ImportError:
                print("[retrieval] web_api=ddg nhưng chưa cài gói ddgs (pip install ddgs); chỉ dùng Wikipedia")

    def _s(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers["User-Agent"] = UA
        return self._session

    # ------------------------------------------------------------------
    def search(self, analysis, kb) -> SearchResult:
        res = super().search(analysis, kb)
        for eid in analysis.candidate_entity_ids:
            ent = kb.get(eid)
            if ent is None:
                continue
            conf = _confidence_map(analysis, kb).get(eid, 0.5)

            # --- Wikipedia: tìm tên bài nếu chưa biết, rồi lấy văn bản dài ---
            title = ent.wiki_title_vi or self._wiki_find_title(ent.name_vi)
            if title:
                text, url = self._wiki_text(title)
                if text:
                    res.items = [i for i in res.items if not (i.entity_id == eid and i.provenance.startswith("kb.notes"))]
                    res.items.append(EvidenceItem(eid, "wiki_text", f"Wikipedia: {title}", text,
                                                  url=url, score=0.7 * max(conf, 0.5), provenance="vi.wikipedia.org"))
                    if not ent.wiki_title_vi:
                        ent.wiki_title_vi = title

            # --- Web search: tiếng Việt trước, tiếng Anh sau ---
            en = ent.name_en.split("(")[0].strip()
            web_queries = []
            if "vi" in self.cfg.web_langs:
                web_queries.append((f"{ent.name_vi} đặc điểm cấu tạo hình dáng", "vn-vi"))
            if "en" in self.cfg.web_langs:
                web_queries.append((f"Vietnamese {en} what it looks like characteristics", "wt-wt"))
            for q, region in web_queries:
                for r in self._web_text(q, region, self.cfg.web_results):
                    if r.get("body"):
                        res.items.append(EvidenceItem(eid, "web_text", f"Web: {r.get('title', '')[:60]}",
                                                      r["body"][:1500], url=r.get("href"), score=0.5,
                                                      provenance=r.get("provenance", "web")))

            # --- Ảnh tham chiếu: Commons (EN + VI) rồi web images ---
            if self.cfg.download_images:
                cands = self._commons(en, n=3) + self._commons(ent.name_vi, n=2)
                cands += self._web_images(f"{ent.name_vi} Việt Nam", 3)
                for img_url, ititle in cands:
                    local = self._download(img_url)
                    if not local:
                        continue
                    match = None
                    if self.clip is not None:
                        try:
                            match = self.clip.image_matches(
                                local, ent.clip_label or f"a photo of Vietnamese {en}",
                                [c.get("name_en") or c["name"] for c in ent.confusable_with])
                        except Exception as exc:  # noqa: BLE001
                            self.errors.append(f"clip: {type(exc).__name__}")
                    # Thực thể "context" (sự kiện, cảnh) không dùng làm ảnh tham chiếu IP-Adapter:
                    # tham chiếu một bức pháo hoa sẽ kéo cả ảnh về pháo hoa.
                    ok = (match is None or match >= self.cfg.ref_image_min_clip) and ent.kind == "object"
                    res.items.append(EvidenceItem(
                        eid, "image", f"Ảnh: {ititle[:60]}",
                        "Ảnh tham chiếu đã qua CLIP" if ok else
                        ("Thực thể bối cảnh, không dùng làm tham chiếu" if ent.kind != "object" else "Bị CLIP loại: không đúng chủ thể"),
                        must_have=list(ent.must_have[:2]) if ok else [], url=img_url,
                        local_path=local if ok else None, clip_match=match,
                        score=0.65 if ok else 0.1, provenance="image-search",
                    ))
                    if ok:
                        break
        res.items.sort(key=lambda i: (i.entity_id, -i.score))
        res.retrieval_errors = list(self.errors)
        self.errors.clear()
        return res

    # ------------------------------------------------------------------ Wikipedia
    def _wiki_find_title(self, name: str) -> str | None:
        try:
            r = self._s().get(WIKI_API, params={"action": "query", "list": "search", "srsearch": name,
                                                "srlimit": "1", "format": "json"}, timeout=self.cfg.timeout)
            hits = r.json().get("query", {}).get("search", [])
            return hits[0]["title"] if hits else None
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"wiki-search {name}: {type(exc).__name__}")
            return None

    def _wiki_text(self, title: str) -> tuple[str | None, str | None]:
        url = f"https://vi.wikipedia.org/wiki/{title.replace(' ', '_')}"
        try:
            if self.cfg.wiki_chars > 0:
                # exchars bị MediaWiki chặn ở 1200, nên lấy toàn văn rồi cắt phía client.
                r = self._s().get(WIKI_API, params={
                    "action": "query", "prop": "extracts", "explaintext": "1", "exsectionformat": "plain",
                    "exlimit": "1", "titles": title, "format": "json", "redirects": "1",
                }, timeout=self.cfg.timeout)
                pages = r.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    if p.get("extract"):
                        return p["extract"][: self.cfg.wiki_chars], url
            r = self._s().get(WIKI_SUMMARY_URL.format(title=title.replace(" ", "_")), timeout=self.cfg.timeout)
            if r.status_code == 200:
                return r.json().get("extract"), url
            self.errors.append(f"wiki {title}: HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"wiki {title}: {type(exc).__name__}")
        return None, None

    # ------------------------------------------------------------------ Web (DDG / Serper)
    def _web_text(self, q: str, region: str, n: int) -> list[dict]:
        if self._ddg is not None:
            try:
                out = self._ddg.text(q, region=region, max_results=n) or []
                return [{"title": r.get("title"), "body": r.get("body"), "href": r.get("href"),
                         "provenance": "duckduckgo"} for r in out]
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"ddg text '{q[:30]}': {type(exc).__name__}")
                return []
        if self.serper_key:
            return [{"title": r.get("title"), "body": r.get("snippet"), "href": r.get("link"), "provenance": "serper"}
                    for r in self._serper("search", q, n)]
        return []

    def _web_images(self, q: str, n: int) -> list[tuple[str, str]]:
        if self._ddg is not None:
            try:
                out = self._ddg.images(q, region="vn-vi", max_results=n) or []
                return [(r["image"], r.get("title", q)) for r in out if r.get("image")]
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"ddg images '{q[:30]}': {type(exc).__name__}")
                return []
        if self.serper_key:
            return [(r["imageUrl"], r.get("title", q)) for r in self._serper("images", q, n) if r.get("imageUrl")]
        return []

    # ------------------------------------------------------------------ Commons / Serper
    def _commons(self, query, n=3):
        try:
            r = self._s().get(COMMONS_API_URL, params={
                "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{query} Vietnam",
                "gsrnamespace": "6", "gsrlimit": str(n), "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": "768",
            }, timeout=self.cfg.timeout)
            out = []
            for page in r.json().get("query", {}).get("pages", {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime", "").startswith("image/") and info.get("thumburl"):
                    out.append((info["thumburl"], page.get("title", query)))
            return out
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"commons {query}: {type(exc).__name__}")
            return []

    def _serper(self, kind: str, q: str, n: int) -> list[dict]:
        try:
            r = self._s().post(SERPER_URL.format(kind=kind), json={"q": q, "num": n, "gl": "vn", "hl": "vi"},
                               headers={"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
                               timeout=self.cfg.timeout)
            if r.status_code != 200:
                self.errors.append(f"serper {kind}: HTTP {r.status_code}")
                return []
            d = r.json()
            return d.get("organic" if kind == "search" else "images", [])[:n]
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"serper {kind}: {type(exc).__name__}")
            return []

    def _download(self, url):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".jpg")
        if path.exists():
            return str(path)
        try:
            r = self._s().get(url, timeout=self.cfg.timeout)
            if r.status_code != 200:
                return None
            from io import BytesIO
            from PIL import Image

            Image.open(BytesIO(r.content)).convert("RGB").save(path, "JPEG", quality=90)
            return str(path)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"download: {type(exc).__name__}")
            return None


def get_retriever(cfg, clip_probe=None, cache_dir=None):
    if cfg.backend == "local":
        return LocalRetriever(cfg, clip_probe, cache_dir)
    if cfg.backend == "wiki":
        return WikiRetriever(cfg, clip_probe, cache_dir)
    raise ValueError(f"retrieval backend không rõ: {cfg.backend!r}")
