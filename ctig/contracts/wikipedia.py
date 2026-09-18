"""Small bilingual Wikipedia retriever for contract evidence.

Vietnamese pages are always queried first. English pages are supplementary and
never silently replace missing Vietnamese evidence. Responses are cached as JSON
so a run is reproducible and polite to Wikimedia.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


USER_AGENT = "CTIG-VisualContract/0.1 (academic research; bilingual evidence retrieval)"


@dataclass(frozen=True)
class WikipediaPassage:
    source_id: str
    lang: str
    title: str
    url: str
    text: str
    query: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class BilingualWikipediaRetriever:
    def __init__(self, cache_dir: str | Path, timeout: int = 20, session=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._session = session

    def _http(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": USER_AGENT})
        return self._session

    def _cache_file(self, lang: str, query: str, limit: int) -> Path:
        raw = json.dumps({"lang": lang, "query": query, "limit": limit}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(raw.encode()).hexdigest()[:24] + ".json")

    def search(self, lang: str, query: str, limit: int = 2) -> list[WikipediaPassage]:
        if lang not in {"vi", "en"}:
            raise ValueError("Wikipedia language must be 'vi' or 'en'")
        cache = self._cache_file(lang, query, limit)
        if cache.exists():
            rows = json.loads(cache.read_text(encoding="utf-8"))
            return [WikipediaPassage(**row) for row in rows]

        endpoint = f"https://{lang}.wikipedia.org/w/api.php"
        response = self._http().get(endpoint, params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "0",
            "gsrlimit": limit,
            "prop": "extracts|info",
            "explaintext": "1",
            "exsectionformat": "plain",
            "inprop": "url",
            "redirects": "1",
        }, timeout=self.timeout)
        response.raise_for_status()
        pages: list[dict[str, Any]] = response.json().get("query", {}).get("pages", [])
        pages.sort(key=lambda page: int(page.get("index", 10**9)))
        out = []
        for index, page in enumerate(pages):
            text = " ".join(str(page.get("extract") or "").split())
            if len(text) < 80:
                continue
            out.append(WikipediaPassage(
                source_id=f"{lang.upper()}{index}",
                lang=lang,
                title=str(page.get("title") or query),
                url=str(page.get("fullurl") or ""),
                text=text[:12000],
                query=query,
            ))
        cache.write_text(
            json.dumps([x.to_dict() for x in out], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return out

    def retrieve(self, entity: dict, per_language: int = 2) -> list[WikipediaPassage]:
        vi_queries = _unique([entity.get("name_vi", ""), *(entity.get("search_queries_vi") or [])])
        en_queries = _unique([entity.get("name_en", ""), *(entity.get("search_queries_en") or [])])
        passages: list[WikipediaPassage] = []
        # Vietnamese is deliberately first so prompt truncation cannot silently
        # remove the evidence that a Vietnamese reviewer can inspect.
        for lang, queries in (("vi", vi_queries), ("en", en_queries)):
            for query in queries[:2]:
                try:
                    rows = self.search(lang, query, limit=per_language)
                except Exception:  # network errors are represented by missing evidence
                    rows = []
                known = {(x.lang, x.title) for x in passages}
                passages.extend(x for x in rows if (x.lang, x.title) not in known)
                if sum(x.lang == lang for x in passages) >= per_language:
                    break
        # Stable IDs after deduplication: VI0.. then EN0..
        renumbered = []
        counters = {"vi": 0, "en": 0}
        for row in passages:
            sid = f"{row.lang.upper()}{counters[row.lang]}"
            counters[row.lang] += 1
            renumbered.append(WikipediaPassage(sid, row.lang, row.title, row.url, row.text, row.query))
        return renumbered


def _unique(values) -> list[str]:
    out = []
    for value in values:
        value = " ".join(str(value or "").split())
        if value and value.casefold() not in {x.casefold() for x in out}:
            out.append(value)
    return out
