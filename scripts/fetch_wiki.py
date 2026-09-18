"""Build data/wiki_curated/<name>.json from the hand-chosen Wikipedia URLs in data/contracts_v2.json.

    python scripts/fetch_wiki.py --ids S001,...,S010 -o data/wiki_curated/S001_S010.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from savier.wiki import fetch_page  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--contracts", default=str(ROOT / "data/contracts_v2.json"))
    a = ap.parse_args()
    c = json.loads(Path(a.contracts).read_text(encoding="utf-8")); out = {}
    for pid in a.ids.split(","):
        urls = sorted({r.get("source", "") for r in c[pid]["required"] if "wikipedia.org/wiki/" in str(r.get("source", ""))},
                      key=lambda u: 0 if "vi.wikipedia" in u else 1)
        out[pid] = [p for p in (fetch_page(u) for u in urls[:2]) if len(p["text"]) > 200]
        if not any(p["lang"] == "vi" for p in out[pid]):          # the Curator needs a Vietnamese source to quote from
            t = c[pid].get("entity_vi", ""); p = fetch_page("https://vi.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")))
            if len(p["text"]) > 200: out[pid].insert(0, p)
        print(pid, [(p["lang"], p["title"], len(p["text"])) for p in out[pid]])
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
