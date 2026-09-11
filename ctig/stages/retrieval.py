"""
Stage 2 - SEARCH: kb / wiki_text / image.

Khác bản mô phỏng: ảnh Commons được TẢI VỀ và KIỂM bằng CLIP trước khi được dùng
làm ảnh tham chiếu cho IP-Adapter. Lý do: truy vấn "Phở" trên Commons từng trả về
một mâm cỗ Tết. Dùng ảnh sai làm tham chiếu thì IP-Adapter kéo ảnh sinh về phía sai.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..kb import KnowledgeBase
from ..schema import AnalysisResult, EvidenceItem, SearchResult

WIKI_SUMMARY_URL = "https://vi.wikipedia.org/api/rest_v1/page/summary/{title}"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
UA = "CTIG/0.1 (research prototype; https://github.com)"


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
            items.append(EvidenceItem(
                entity_id=eid, kind="kb", title=f"KB: {ent.name_vi}",
                snippet=f"{ent.name_vi} ({ent.name_en}). {ent.notes or ''}".strip(),
                must_have=list(ent.must_have), must_not=list(ent.must_not),
                confusable_with=list(ent.confusable_with),
                score=base, provenance=f"kb@{kb.version}",
            ))
            if ent.notes:
                items.append(EvidenceItem(
                    entity_id=eid, kind="wiki_text", title=f"Ghi chú: {ent.name_vi}",
                    snippet=ent.notes, score=base * 0.75, provenance="kb.notes (offline)",
                    url=f"https://vi.wikipedia.org/wiki/{ent.wiki_title_vi.replace(' ', '_')}" if ent.wiki_title_vi else None,
                ))
        return SearchResult(analysis.prompt_id, items, queries, misses)


class WikiRetriever(LocalRetriever):
    """Wikipedia tiếng Việt + Wikimedia Commons, có tải ảnh và kiểm CLIP. Lỗi mạng -> lùi về offline."""

    name = "wiki"

    def __init__(self, cfg, clip_probe=None, cache_dir=None):
        super().__init__(cfg, clip_probe, cache_dir)
        self._session = None
        self.errors: list[str] = []

    def _s(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers["User-Agent"] = UA  # phải là ASCII
        return self._session

    def search(self, analysis, kb) -> SearchResult:
        res = super().search(analysis, kb)
        for eid in list(dict.fromkeys(i.entity_id for i in res.items)):
            ent = kb.get(eid)
            if ent is None:
                continue
            if ent.wiki_title_vi:
                extract, url = self._wiki(ent.wiki_title_vi)
                if extract:
                    res.items = [i for i in res.items if not (i.entity_id == eid and i.provenance.startswith("kb.notes"))]
                    res.items.append(EvidenceItem(eid, "wiki_text", f"Wikipedia: {ent.wiki_title_vi}",
                                                  extract[:900], url=url, score=0.7,
                                                  provenance="vi.wikipedia.org"))
            if self.cfg.download_images:
                for img_url, title in self._commons(ent.name_en.split("(")[0].strip(), n=3):
                    local = self._download(img_url)
                    if not local:
                        continue
                    match = None
                    if self.clip is not None:
                        try:
                            match = self.clip.image_matches(local, ent.name_en.split("(")[0].strip(),
                                                            [c["name"] for c in ent.confusable_with])
                        except Exception as exc:  # noqa: BLE001
                            self.errors.append(f"clip {title}: {type(exc).__name__}")
                    ok = match is None or match >= self.cfg.ref_image_min_clip
                    res.items.append(EvidenceItem(
                        eid, "image", f"Commons: {title}",
                        ("Ảnh tham chiếu đã qua CLIP" if ok else "Ảnh Commons bị CLIP loại: không đúng chủ thể"),
                        must_have=list(ent.must_have[:2]) if ok else [],
                        url=img_url, local_path=local if ok else None, clip_match=match,
                        score=0.65 if ok else 0.1, provenance="commons.wikimedia.org",
                    ))
                    if ok:
                        break  # đủ một ảnh tốt
        res.items.sort(key=lambda i: (i.entity_id, -i.score))
        res.retrieval_errors = list(self.errors)
        self.errors.clear()
        return res

    def _wiki(self, title):
        try:
            r = self._s().get(WIKI_SUMMARY_URL.format(title=title.replace(" ", "_")), timeout=self.cfg.timeout)
            if r.status_code != 200:
                self.errors.append(f"wiki {title}: HTTP {r.status_code}")
                return None, None
            d = r.json()
            return d.get("extract"), d.get("content_urls", {}).get("desktop", {}).get("page")
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"wiki {title}: {type(exc).__name__}")
            return None, None

    def _commons(self, query, n=3):
        try:
            r = self._s().get(COMMONS_API_URL, params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": f"{query} Vietnam", "gsrnamespace": "6", "gsrlimit": str(n),
                "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": "768",
            }, timeout=self.cfg.timeout)
            if r.status_code != 200:
                self.errors.append(f"commons {query}: HTTP {r.status_code}")
                return []
            out = []
            for page in r.json().get("query", {}).get("pages", {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime", "").startswith("image/") and info.get("thumburl"):
                    out.append((info["thumburl"], page.get("title", query)))
            return out
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"commons {query}: {type(exc).__name__}")
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
