"""Curated Wikipedia sources for the Source-Aware Prompt Curator.

Pages are fetched ONE PER REQUEST. TextExtracts returns a full-article extract for a single page per
request and silently lowers `exlimit` to 1 otherwise; a combined search+extract call therefore returns an
empty extract for the top-ranked page (measured: the 'Áo dài' article came back with 0 characters and was
dropped, while 'Người Việt' was kept). Sources are chosen by hand (URLs in `contracts_v2.json`), not by
automatic entity linking.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "SAVIER-research/0.1 (academic; contact via repository)"}


def fetch_page(url: str) -> dict:
    lang = url.split("//")[1].split(".")[0]
    title = urllib.parse.unquote(url.rsplit("/wiki/", 1)[1])
    q = urllib.parse.urlencode({"action": "query", "format": "json", "formatversion": "2", "titles": title,
                                "prop": "extracts", "explaintext": "1", "exsectionformat": "plain", "redirects": "1"})
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"https://{lang}.wikipedia.org/w/api.php?{q}", headers=UA), timeout=30))
    pg = d["query"]["pages"][0]
    return {"lang": lang, "title": pg.get("title"), "url": url, "text": " ".join((pg.get("extract") or "").split())}


def cut_passage(text: str, keywords: list[str], max_chars: int = 4000) -> str:
    """Keep sentences containing entity keywords, in order, up to `max_chars`; fall back to the article head.

    Feeding 12k-character articles made the Curator's JSON overflow the generation budget and truncate;
    ~4k characters around the entity nouns is enough to quote verbatim from."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    kw = [k.lower() for k in keywords if k]
    keep, n = [], 0
    for s in sents:
        if any(k in s.lower() for k in kw) and n + len(s) <= max_chars:
            keep.append(s); n += len(s) + 1
    return " ".join(keep) if len(" ".join(keep)) >= 800 else text[:max_chars]


def load_curated(path: str | Path) -> dict[str, list[dict]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
