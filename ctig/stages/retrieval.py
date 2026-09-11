"""
Stage 2 - SEARCH: kb / wiki_text / web_text / image, gọi API lúc chạy.

Nguồn (tất cả không cần key trừ Serper):
  * KB gốc trong repo - bằng chứng đã có sẵn, provenance "kb".
  * Wikipedia tiếng Việt: tìm bài (list=search) khi chưa biết tên bài, rồi lấy văn bản dài.
  * DuckDuckGo web tiếng Việt và tiếng Anh + ảnh (mặc định, không key), qua WebClient có cache.
  * Wikimedia Commons: ảnh tham chiếu.
  * Serper (tuỳ chọn, SERPER_API_KEY).

v1.2: mọi item ghi `query` và `query_group` ("keyword" | "prompt"); ảnh tải và chấm CLIP tối đa
k ảnh mỗi thực thể (không dừng ở ảnh đầu đạt) để bảng hiển thị so được; chỉ MỘT ảnh tốt nhất
đạt ngưỡng và thuộc thực thể vật thể mới `is_reference=True` cho IP-Adapter.

Stage này chỉ THU văn bản. Biến văn bản thành must_have/must_not là stages/extraction.py.
"""

from __future__ import annotations

from pathlib import Path

from ..kb import KnowledgeBase
from ..llm.shared import confusable_clip_label
from ..schema import AnalysisResult, EvidenceItem, SearchResult
from .websearch import WebClient, entity_image_queries, entity_queries

WIKI_API = "https://vi.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://vi.wikipedia.org/api/rest_v1/page/summary/{title}"
UA = "CTIG/0.2 (research prototype)"


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
        # cache_dir là thư mục _cache gốc (chứa web/, ref_images/, ...)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("runs/_cache")

    def search(self, analysis: AnalysisResult, kb: KnowledgeBase, raw_prompt: str | None = None) -> SearchResult:
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

    def __init__(self, cfg, clip_probe=None, cache_dir=None, web: WebClient | None = None, k_images: int = 4):
        super().__init__(cfg, clip_probe, cache_dir)
        self.web = web or WebClient(cfg, self.cache_dir)
        self.k_images = k_images
        self.errors: list[str] = []

    def _s(self):
        return self.web._s()

    # ------------------------------------------------------------------
    def search(self, analysis, kb, raw_prompt: str | None = None) -> SearchResult:
        res = super().search(analysis, kb)
        conf_map = _confidence_map(analysis, kb)
        for eid in analysis.candidate_entity_ids:
            ent = kb.get(eid)
            if ent is None:
                continue
            conf = conf_map.get(eid, 0.5)
            en = ent.name_en.split("(")[0].strip()

            # --- Wikipedia: tìm tên bài nếu chưa biết, rồi lấy văn bản dài ---
            title = ent.wiki_title_vi or self._wiki_find_title(ent.name_vi)
            if title:
                text, url = self._wiki_text(title)
                if text:
                    res.items = [i for i in res.items if not (i.entity_id == eid and i.provenance.startswith("kb.notes"))]
                    res.items.append(EvidenceItem(eid, "wiki_text", f"Wikipedia: {title}", text, url=url,
                                                  score=0.7 * max(conf, 0.5), provenance="vi.wikipedia.org",
                                                  query=title, query_group="keyword"))
                    if not ent.wiki_title_vi:
                        ent.wiki_title_vi = title

            # --- Web text: tiếng Việt trước, tiếng Anh sau ---
            for q, region in entity_queries(ent, self.cfg.web_langs):
                for r in self.web.text(q, region, self.cfg.web_results):
                    if r.get("body"):
                        res.items.append(EvidenceItem(eid, "web_text", f"Web: {(r.get('title') or '')[:60]}",
                                                      r["body"][:1500], url=r.get("href"), score=0.5,
                                                      provenance=r.get("provenance", "web"), query=q,
                                                      query_group="keyword"))

            # --- Ảnh: Commons (EN, VI) + web images; tải và chấm tối đa k ảnh ---
            if self.cfg.download_images:
                iq = entity_image_queries(ent)
                cands = self.web.commons(iq[0], n=3) + self.web.commons(iq[1], n=2) + self.web.images(iq[2], n=3)
                seen: set[str] = set()
                scored: list[EvidenceItem] = []
                for r in cands:
                    if r["image"] in seen or len(scored) >= self.k_images:
                        continue
                    seen.add(r["image"])
                    local = self.web.download(r["image"])
                    if not local:
                        continue
                    match = None
                    if self.clip is not None:
                        try:
                            match = self.clip.image_matches(
                                local, ent.clip_label or f"a photo of Vietnamese {en}",
                                [confusable_clip_label(c) for c in ent.confusable_with])
                        except Exception as exc:  # noqa: BLE001
                            self.errors.append(f"clip: {type(exc).__name__}")
                    scored.append(EvidenceItem(
                        eid, "image", f"Ảnh: {(r.get('title') or '')[:60]}", "",
                        url=r["image"], local_path=local, clip_match=match, score=0.0,
                        provenance=r.get("provenance", "image-search"), query=iq[2 if r.get("provenance", "").startswith("duck") else 0],
                        query_group="keyword",
                    ))
                # Chỉ một ảnh tốt nhất, đạt ngưỡng, và thực thể là VẬT THỂ mới làm tham chiếu:
                # tham chiếu một bức pháo hoa cho "Tết" sẽ kéo cả ảnh về pháo hoa.
                best = max(scored, key=lambda i: (i.clip_match if i.clip_match is not None else -1.0), default=None)
                for it in scored:
                    ok = (it is best and ent.kind == "object"
                          and (it.clip_match is None or it.clip_match >= self.cfg.ref_image_min_clip))
                    it.is_reference = ok
                    it.must_have = list(ent.must_have[:2]) if ok else []
                    it.score = 0.65 if ok else (0.3 if (it.clip_match or 0) >= 0.5 else 0.1)
                    it.snippet = ("Ảnh tham chiếu (CLIP đạt, IP-Adapter)" if ok else
                                  "Thực thể bối cảnh, không dùng làm tham chiếu" if ent.kind != "object" else
                                  f"CLIP {it.clip_match:.2f}" if it.clip_match is not None else "chưa chấm CLIP")
                res.items.extend(scored)

        # --- Nhóm truy vấn từ prompt gốc (để so, không đưa vào must_have) ---
        if raw_prompt:
            for r in self.web.text(raw_prompt, "vn-vi", self.cfg.web_results):
                if r.get("body"):
                    res.items.append(EvidenceItem("-", "web_text", f"Web(prompt): {(r.get('title') or '')[:60]}",
                                                  r["body"][:800], url=r.get("href"), score=0.2,
                                                  provenance=r.get("provenance", "web"), query=raw_prompt,
                                                  query_group="prompt"))

        res.items.sort(key=lambda i: (i.entity_id, -i.score))
        res.retrieval_errors = list(self.errors) + self.web.drain_errors()
        self.errors.clear()
        return res

    # ------------------------------------------------------------------ Wikipedia
    def _wiki_find_title(self, name: str) -> str | None:
        def fetch():
            try:
                r = self._s().get(WIKI_API, params={"action": "query", "list": "search", "srsearch": name,
                                                    "srlimit": "1", "format": "json"}, timeout=self.cfg.timeout)
                hits = r.json().get("query", {}).get("search", [])
                return [hits[0]["title"]] if hits else []
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"wiki-search {name}: {type(exc).__name__}")
                return []

        out = self.web._cached({"kind": "wiki-title", "q": name}, fetch)
        return out[0] if out else None

    def _wiki_text(self, title: str) -> tuple[str | None, str | None]:
        url = f"https://vi.wikipedia.org/wiki/{title.replace(' ', '_')}"

        def fetch():
            try:
                if self.cfg.wiki_chars > 0:
                    # exchars bị MediaWiki chặn ở 1200, nên lấy toàn văn rồi cắt phía client.
                    r = self._s().get(WIKI_API, params={
                        "action": "query", "prop": "extracts", "explaintext": "1", "exsectionformat": "plain",
                        "exlimit": "1", "titles": title, "format": "json", "redirects": "1",
                    }, timeout=self.cfg.timeout)
                    for p in r.json().get("query", {}).get("pages", {}).values():
                        if p.get("extract"):
                            return [p["extract"][: self.cfg.wiki_chars]]
                r = self._s().get(WIKI_SUMMARY_URL.format(title=title.replace(" ", "_")), timeout=self.cfg.timeout)
                if r.status_code == 200 and r.json().get("extract"):
                    return [r.json()["extract"]]
                self.errors.append(f"wiki {title}: HTTP {r.status_code}")
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"wiki {title}: {type(exc).__name__}")
            return []

        out = self.web._cached({"kind": "wiki-text", "title": title, "chars": self.cfg.wiki_chars}, fetch)
        return (out[0], url) if out else (None, None)


def get_retriever(cfg, clip_probe=None, cache_dir=None, web: WebClient | None = None, k_images: int = 4):
    if cfg.backend == "local":
        return LocalRetriever(cfg, clip_probe, cache_dir)
    if cfg.backend == "wiki":
        return WikiRetriever(cfg, clip_probe, cache_dir, web=web, k_images=k_images)
    raise ValueError(f"retrieval backend không rõ: {cfg.backend!r}")
