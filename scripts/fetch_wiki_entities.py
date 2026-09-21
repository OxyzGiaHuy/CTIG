"""Lấy bài Wikipedia tiếng Việt theo TÊN THỰC THỂ (tìm kiếm rồi lấy bài đầu), ghi đúng định dạng wiki_curated:
{pid: [{lang, title, url, text, entity}]}. Dùng cho prompt phức (>=2 thực thể) không có contract/URL sẵn.

    python scripts/fetch_wiki_entities.py --prompts data/prompts_complex.json --ids C001 -o data/wiki_curated/C001.json
"""
import argparse, json, time, urllib.parse, urllib.request
from pathlib import Path
UA = {"User-Agent": "CTIG-research/1.0 (student project; contact via GitHub OxyzGiaHuy/CTIG)"}
def api(lang, **q):
    q.update(format="json", formatversion="2"); url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q)
    for i in range(4):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        except Exception as e:  # noqa: BLE001
            time.sleep(2 * (i + 1)); err = e
    raise err
def search_title(ent, lang="vi"):
    r = api(lang, action="query", list="search", srsearch=ent, srlimit=3); hits = r.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None
def page(title, lang="vi"):
    r = api(lang, action="query", prop="extracts", explaintext=1, redirects=1, titles=title); pg = r["query"]["pages"][0]
    return {"lang": lang, "title": pg["title"], "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(pg['title'].replace(' ', '_'))}", "text": pg.get("extract", "")}
ap = argparse.ArgumentParser(); ap.add_argument("--prompts", required=True); ap.add_argument("--ids", required=True); ap.add_argument("-o", required=True); a = ap.parse_args()
P = {p["id"]: p for p in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}; out = {}
for pid in a.ids.split(","):
    pages = []
    for ent in P[pid]["entities"]:
        t = search_title(ent)
        if not t: print(f"  {pid} '{ent}': không tìm thấy"); continue
        pg = page(t); pg["entity"] = ent; pages.append(pg); print(f"  {pid} '{ent}' -> {pg['title']} ({len(pg['text'])} ký tự)"); time.sleep(0.5)
    out[pid] = pages
Path(a.o).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"); print("->", a.o)
