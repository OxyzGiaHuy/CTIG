"""Đối chiếu NHÃN NGƯỜI với điểm máy: bộ chấm xếp hạng đúng được bao nhiêu?

    python scripts/label_score.py data/labels/labels_huy.json <run_dir> [run_dir2 ...] [--axis culture|overall]

Vì sao cần: mọi kết luận về vòng sửa tới nay đều dựa trên điểm của chính hệ thống, và ngày 2026-09-17 đã thấy
nó xếp ảnh áo hoa văn Trung Quốc (6,4) trên ảnh áo dài đúng (3,8). Script này biến câu "bộ chấm sai" từ một
nhận xét bằng mắt thành một CON SỐ, để mỗi lần sửa bộ chấm là biết ngay tốt lên hay tệ đi.

Ba số in ra:

  1. **Xếp cặp đúng**: trong mỗi prompt, lấy mọi cặp (ảnh người bảo ĐÚNG, ảnh người bảo SAI) rồi hỏi máy có
     cho ảnh đúng điểm cao hơn không. Đây là AUC, và là số quan trọng nhất: 0,5 nghĩa là bộ chấm ngang đoán mò.
  2. **Ảnh hệ thống giữ có đúng không**: hệ thống chọn ảnh điểm cao nhất; người có đồng ý đó là ảnh đúng không.
  3. **Thuộc tính**: tỉ lệ thuộc tính đạt theo người, để đối chiếu với phần `differences` của bộ chấm.

Nhãn `khong_chac` và `khong_thay` bị loại khỏi mọi phép tính, không đoán thay người gán.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path


def load_machine(runs: list[str]) -> dict[str, dict]:
    """image_path -> {overall, culture, prompt_id, kept_by_system}. Đọc cả run vòng sửa lẫn run thường."""
    out: dict[str, dict] = {}
    for run in runs:
        for lf in sorted(glob.glob(f"{run}/*/loop_v2.json")):
            d = json.load(open(lf))
            pid = Path(lf).parent.name
            final = d.get("final")
            # ảnh mốc: điểm nằm trong `best` nếu chính nó là ảnh tốt nhất, ngược lại không ghi lại được
            for r in d.get("rounds") or []:
                ev = r.get("eval") or {}
                if r.get("image"):
                    out[r["image"]] = {"prompt_id": pid, "overall": ev.get("overall"),
                                       "culture": (ev.get("axes") or {}).get("culture"),
                                       "kept": r["image"] == final, "round": r.get("n")}
            b = d.get("best") or {}
            if b.get("path") and b["path"] not in out:
                out[b["path"]] = {"prompt_id": pid, "overall": b.get("overall"),
                                  "culture": (b.get("axes") or {}).get("culture"),
                                  "kept": b["path"] == final, "round": 0}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--axis", default="culture", choices=("culture", "overall"))
    a = ap.parse_args(argv)

    rows = json.load(open(a.labels, encoding="utf-8"))
    machine = load_machine(a.runs)

    # --- gom nhãn theo ảnh
    human: dict[str, dict] = {}
    for r in rows:
        img = r["image"]
        h = human.setdefault(img, {"prompt_id": r["prompt_id"], "overall": None, "ok": 0, "no": 0})
        if r.get("overall"):
            h["overall"] = r["overall"]
        j, side = r.get("judgement"), r.get("side")
        if j in ("co", "khong"):                      # 'khong_thay' không tính vào tỉ lệ thuộc tính
            good = (j == "co") if side == "must_have" else (j == "khong")
            h["ok" if good else "no"] += 1

    seen = [i for i in human if i in machine]
    if not seen:
        raise SystemExit("không khớp được ảnh nào giữa nhãn và run; kiểm lại đường dẫn run_dir")
    print(f"{len(human)} ảnh có nhãn · {len(seen)} ảnh khớp được với điểm máy · trục dùng: {a.axis}\n")

    print(f"{'prompt':7s} {'vòng':>4s} {'người':>7s} {'thuộc tính':>11s} {'máy':>6s}  ảnh")
    for img in sorted(seen, key=lambda x: (machine[x]["prompt_id"], machine[x].get("round") or 0)):
        h, m = human[img], machine[img]
        tot = h["ok"] + h["no"]
        attr = f"{h['ok']}/{tot}" if tot else "-"
        sc = m.get(a.axis)
        print(f"{m['prompt_id']:7s} {str(m.get('round','?')):>4s} {str(h['overall'] or '-'):>7s} {attr:>11s} "
              f"{(f'{sc:.1f}' if sc is not None else '-'):>6s}  {'← hệ thống giữ' if m['kept'] else ''}")

    # --- 1. xếp cặp đúng (AUC)
    by_p: dict[str, list] = defaultdict(list)
    for img in seen:
        h, m = human[img], machine[img]
        if h["overall"] in ("dung", "sai") and m.get(a.axis) is not None:
            by_p[m["prompt_id"]].append((h["overall"] == "dung", m[a.axis]))
    win = tie = lose = 0
    for pid, items in by_p.items():
        for good, sg in [x for x in items if x[0]]:
            for _bad, sb in [x for x in items if not x[0]]:
                win += sg > sb
                tie += sg == sb
                lose += sg < sb
    n = win + tie + lose
    print()
    if n:
        auc = (win + 0.5 * tie) / n
        print(f"1. XẾP CẶP: {n} cặp (người bảo đúng vs người bảo sai) · máy xếp đúng {win}, hoà {tie}, "
              f"sai {lose} -> AUC {auc:.2f}")
        print("   " + ("0,5 = ngang đoán mò. Dưới 0,5 nghĩa là bộ chấm xếp NGƯỢC."
                       if auc <= 0.6 else "trên 0,6 thì bộ chấm bắt đầu dùng được."))
    else:
        print("1. XẾP CẶP: chưa đủ dữ liệu — cần ít nhất một ảnh 'đúng' và một ảnh 'sai' trong cùng một prompt")

    # --- 2. ảnh hệ thống giữ có được người chấp nhận không
    kept_ok = kept_bad = kept_unk = 0
    for img in seen:
        if not machine[img]["kept"]:
            continue
        o = human[img]["overall"]
        kept_ok += o == "dung"
        kept_bad += o == "sai"
        kept_unk += o not in ("dung", "sai")
    print(f"\n2. ẢNH HỆ THỐNG GIỮ: người bảo đúng {kept_ok}, sai {kept_bad}, chưa rõ {kept_unk}")
    if kept_bad:
        print(f"   {kept_bad} prompt hệ thống giữ lại đúng cái ảnh người bảo SAI.")

    # --- 3. thuộc tính
    tot_ok = sum(human[i]["ok"] for i in seen)
    tot_all = sum(human[i]["ok"] + human[i]["no"] for i in seen)
    if tot_all:
        print(f"\n3. THUỘC TÍNH: {tot_ok}/{tot_all} đạt theo người ({100 * tot_ok / tot_all:.0f}%)")
    print("\n(nhãn 'không chắc' và 'không thấy được' đã bị loại khỏi mọi phép tính)")


if __name__ == "__main__":
    main()
