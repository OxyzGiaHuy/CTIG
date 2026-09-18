"""In ra ĐÚNG văn bản mà Critic nhìn thấy, cho từng prompt, để người duyệt trước khi chạy.

    python scripts/dump_contracts.py -o docs/contracts_review.md [--ids S001,S002]

Vì sao cần: contract là chỗ DUY NHẤT để chỉnh hệ thống theo từng prompt mà không phải đụng vào mã. Sửa
một dòng mô tả trong `data/contracts.json` là đổi hẳn thứ Critic đi tìm. Nhưng muốn sửa đúng thì phải thấy
được nó đang đọc gì — mà `_contract_text()` gộp và định dạng lại, nên đọc file JSON thô không đủ.

In thêm hai thứ cạnh nhau để soi được nhanh:
  - DANH SÁCH BỘ PHẬN mà Observer được chỉ đi soi (`_cho_can_ta`). Mục required nào không đẻ ra được bộ
    phận nào thì Observer sẽ không bao giờ nhắc tới nó, và Critic sẽ không bao giờ bắt được lỗi ở đó.
  - Mục nào CHƯA CÓ NGUỒN, để biết chỗ nào cần người Việt soi kỹ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.agents import vietrepair as vr  # noqa: E402
from ctig.pipeline import load_prompts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--contracts", default=None)
    a = ap.parse_args(argv)

    c = vr.load_contracts(a.contracts)
    prompts = {}
    for f in ("prompts_simple.json", "prompts_complex.json"):
        p = ROOT / "data" / f
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8")):
                prompts[r["id"]] = r
    ids = [i.strip() for i in a.ids.split(",")] if a.ids else sorted(c)

    out = ["# Contract — đúng thứ Critic đọc\n",
           "Sửa `data/contracts.json` là đổi hẳn thứ hệ thống đi tìm; không cần đụng vào mã.\n",
           "Ba chỗ đáng soi: **mục nào không có bộ phận tương ứng** thì Observer sẽ không nhắc tới và Critic "
           "không bao giờ bắt được lỗi ở đó; **mục CHƯA CÓ NGUỒN**; và **mục không phải lúc nào cũng xuất "
           "hiện** — mục kiểu đó làm Critic bắt lỗi oan rồi hệ thống sửa hỏng ảnh vốn đúng.\n"]

    thieu_nguon = 0
    for pid in ids:
        if pid not in c:
            continue
        v = c[pid]
        parts = vr._cho_can_ta(v)
        out.append(f"\n---\n\n## {pid} · {v.get('entity_vi', '')} — *{v.get('entity', '')}*\n")
        pr = prompts.get(pid, {})
        if pr.get("text_en"):
            out.append(f"> prompt gốc: {pr['text_en']}\n")
        out.append(f"\n**Observer được chỉ soi:** `{', '.join(parts) or '(KHÔNG CÓ — Observer sẽ tả tự do)'}`\n")
        out.append("\n**Critic đọc nguyên văn:**\n\n```\n" + vr._contract_text(v) + "\n```\n")
        out.append("\n| mục | mô tả | có trong danh sách soi? | nguồn |\n|---|---|---|---|\n")
        for r in v.get("required", []):
            co = any(w in parts for w in str(r["description"]).lower().replace(",", " ").split())
            src = str(r.get("source", ""))
            if "CHƯA" in src:
                thieu_nguon += 1
            src_txt = "**CHƯA CÓ NGUỒN**" if "CHƯA" in src else (src[:46] + "…" if len(src) > 46 else src)
            out.append(f"| `{r['id']}` | {r['description']} | {'có' if co else '**KHÔNG**'} | {src_txt} |\n")
        for x in v.get("confusables", []):
            out.append(f"| ~~`{x['id']}`~~ dễ nhầm | {x['description']} | — | {x.get('culture', '')} |\n")

    out.append(f"\n---\n\n{len(ids)} thực thể · "
               f"{sum(len(c[i].get('required', [])) for i in ids if i in c)} mục required · "
               f"**{thieu_nguon} mục chưa có nguồn**\n")
    txt = "".join(out)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"-> {a.out} · {len(txt.splitlines())} dòng")
    else:
        print(txt)


if __name__ == "__main__":
    main()
