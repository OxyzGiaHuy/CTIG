"""Gom các bản ghi KB tự sinh (runs/_cache/kb_auto/*.json) thành một file để nhóm duyệt, và (tuỳ chọn) so với KB tay.

    python scripts/kb_auto_export.py --cache runs/_cache/kb_auto --out data/kb/auto_entities.review.json [--kb data/kb/entities.json]

File ra: danh sách bản ghi theo mẫu entities.json + trường review: {status: "pending", note: ""} để người duyệt điền.
Bản ghi cùng id với KB tay được in cạnh nhau (auto so với tay) nhưng KHÔNG ghi đè KB tay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/_cache/kb_auto")
    ap.add_argument("--out", default="data/kb/auto_entities.review.json")
    ap.add_argument("--kb", default="data/kb/entities.json")
    a = ap.parse_args(argv)
    hand = {e["id"]: e for e in json.loads(Path(a.kb).read_text(encoding="utf-8"))["entities"]} if Path(a.kb).exists() else {}
    rows = []
    for f in sorted(Path(a.cache).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        meta = d.get("_meta", {})
        eid = meta.get("entity_id", f.stem)
        row = {
            "id": eid, "name_vi": meta.get("name_vi", ""), "name_en": meta.get("name_en", ""), "source": "auto",
            "must_have": d.get("must_have", []), "must_have_en": d.get("must_have_en", []),
            "must_not": d.get("must_not", []), "must_not_en": d.get("must_not_en", []),
            "confusable_with": d.get("confusable_with", []), "tags_en": d.get("tags_en", []), "neg_tags_en": d.get("neg_tags_en", []),
            "clip_label": d.get("clip_label", ""), "kind": d.get("kind", "object"), "prior_strength": d.get("prior_strength", 0.2),
            "attr_sources": d.get("attr_sources", {}), "sources": meta.get("sources", []), "dropped_unsourced": d.get("dropped_unsourced", []),
            "review": {"status": "pending", "note": ""},
        }
        if eid in hand:
            row["hand_written"] = {k: hand[eid].get(k) for k in ("must_have_en", "must_not_en", "tags_en", "neg_tags_en")}
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"n": len(rows), "entities": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} bản ghi -> {a.out}" + (f" ({sum(1 for r in rows if 'hand_written' in r)} có bản tay để so)" if rows else ""))


if __name__ == "__main__":
    main()
