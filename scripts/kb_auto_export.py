"""Gom các bản ghi KB tự sinh (runs/_cache/kb_auto/*.json) thành file để nhóm duyệt.

    python scripts/kb_auto_export.py --cache runs/_cache/kb_auto --out data/kb/auto_entities.review.json [--kb data/kb/entities.json]

File ra: {"entities": [bản ghi ĐÚNG mẫu Entity của entities.json -> dán thẳng vào KB được], "review": {id: {status, note, sources,
dropped_unsourced, attr_sources, hand_written}}}. Không ghi đè KB tay.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.kb import KnowledgeBase  # noqa: E402
from ctig.stages.extraction import _read_json, apply_kb_draft, draft_usable  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/_cache/kb_auto")
    ap.add_argument("--out", default="data/kb/auto_entities.review.json")
    ap.add_argument("--kb", default="data/kb/entities.json")
    a = ap.parse_args(argv)
    kb = KnowledgeBase.load(a.kb) if Path(a.kb).exists() else KnowledgeBase(version="", disclaimer="")
    hand = {e.id: e for e in kb.all()}
    ents, review, skipped = [], {}, []
    for f in sorted(Path(a.cache).glob("*.json")):
        d = _read_json(f)
        if not draft_usable(d):
            skipped.append(f.name); continue
        meta = d.get("_meta", {})
        eid = meta.get("entity_id", f.stem)
        base = hand.get(eid)
        ent = KnowledgeBase(version="", disclaimer="").add_adhoc(meta.get("name_vi") or (base.name_vi if base else eid),
                                                                 meta.get("name_en") or (base.name_en if base else eid),
                                                                 category=base.category if base else "other",
                                                                 region=base.region if base else "toan_quoc")
        ent.id = eid
        if base:
            ent.aliases = list(base.aliases); ent.wiki_title_vi = base.wiki_title_vi
        apply_kb_draft(ent, d)
        ent.notes = f"KB tự sinh (source=auto) từ {meta.get('n_sources', '?')} nguồn; chưa duyệt"
        ents.append(asdict(ent))
        review[eid] = {"status": "pending", "note": "", "sources": meta.get("sources", []),
                       "dropped_unsourced": d.get("dropped_unsourced", []), "attr_sources": d.get("attr_sources", {}),
                       "hand_written": {k: getattr(base, k) for k in ("must_have_en", "must_not_en", "tags_en", "neg_tags_en")} if base else None}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"version": f"auto-{len(ents)}", "entities": ents, "review": review}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(ents)} bản ghi -> {a.out} ({sum(1 for r in review.values() if r['hand_written']) } có bản tay để so; bỏ {len(skipped)} file mỏng/hỏng)")


if __name__ == "__main__":
    main()
