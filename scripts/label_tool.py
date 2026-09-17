"""Dựng trang gán nhãn tay cho ảnh sinh: mỗi màn hình một ảnh, hỏi từng thuộc tính của spec.

    python scripts/label_tool.py <run_dir> [run_dir2 ...] -o labels.html [--ids S001,S012] [--per-prompt 6]
        [--annotator ten] [--seed 0]

Vì sao cần: mọi con số hiệu chỉnh Reviewer tới nay đều đo trên ảnh do chính hệ thống tự gán nhãn — vừa ra đề
vừa chấm bài. Trang này tạo ra thước đo ĐỘC LẬP: nhãn người theo TỪNG thuộc tính, cộng một nhãn tổng thể để
bắt trường hợp "đủ mục nhưng vẫn sai" (Goodhart).

Chống thiên lệch người gán:
  - KHÔNG hiện tên model, tên nhánh, hay điểm máy chấm. Người gán không biết ảnh nào của bare, ảnh nào của system.
  - Thứ tự ảnh xáo trộn theo seed cố định (lặp lại được).
  - Mỗi ảnh hỏi cả must_have lẫn must_not, trộn lẫn, không nói mục nào là "phải có" mục nào là "không được có".

Đầu ra JSON: [{prompt_id, image, attribute_en, side, judgement, overall, annotator, ms}]
  judgement ∈ {"co", "khong", "khong_thay"}   overall ∈ {"dung", "sai", "khong_chac"}
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.bundle import thumb_b64  # noqa: E402

CSS = """
:root{--fg:#1d1d1f;--mut:#6b6b70;--line:#dcdcd8;--ok:#168052;--no:#c4302b;--na:#8a8a8f;--bg:#faf9f7}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;z-index:5}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;background:var(--ok);width:0}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;grid-template-columns:minmax(0,440px) 1fr;gap:24px}
@media(max-width:820px){main{grid-template-columns:1fr}}
img.shot{width:100%;border:1px solid var(--line);border-radius:6px;display:block}
.prompt{font-size:14px;color:var(--mut);margin:10px 0 0}
.q{border-bottom:1px solid var(--line);padding:10px 0}
.q p{margin:0 0 6px;font-weight:600}
.btns{display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;padding:7px 14px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer}
button:hover{border-color:#999}
button.sel[data-v=co]{background:var(--ok);border-color:var(--ok);color:#fff}
button.sel[data-v=khong]{background:var(--no);border-color:var(--no);color:#fff}
button.sel[data-v=khong_thay]{background:var(--na);border-color:var(--na);color:#fff}
button.sel[data-v=dung]{background:var(--ok);border-color:var(--ok);color:#fff}
button.sel[data-v=sai]{background:var(--no);border-color:var(--no);color:#fff}
button.sel[data-v=khong_chac]{background:var(--na);border-color:var(--na);color:#fff}
.nav{display:flex;gap:10px;align-items:center;margin-top:18px}
.key{font:12px ui-monospace,monospace;color:var(--mut);border:1px solid var(--line);border-radius:4px;padding:1px 5px}
.muted{color:var(--mut);font-size:13px}
.done{color:var(--ok);font-weight:600}
"""

JS = """
const D = DATA, KEY = "ctig_labels_" + DATA_ID;
let i = 0, ans = {};
try { ans = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { ans = {}; }
let shownAt = Date.now();

const save = () => { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) {} };
const el = (id) => document.getElementById(id);
const answered = (k) => { const a = ans[k] || {}; const it = D[k];
  return it.attrs.every((_, j) => a["a" + j]) && !!a.overall; };
const nDone = () => Object.keys(D).filter(answered).length;

function render() {
  const keys = Object.keys(D), k = keys[i], it = D[k];
  ans[k] = ans[k] || {};
  el("shot").src = it.img;
  el("prompt").textContent = it.prompt_vi;
  el("pos").textContent = (i + 1) + " / " + keys.length;
  el("done").textContent = nDone() + " ảnh đã xong";
  el("bar").style.width = (100 * nDone() / keys.length) + "%";
  let h = "";
  it.attrs.forEach((a, j) => {
    h += `<div class="q"><p>${j + 1}. ${a}</p><div class="btns" data-f="a${j}">` +
      `<button data-v="co">có <span class="key">1</span></button>` +
      `<button data-v="khong">không <span class="key">2</span></button>` +
      `<button data-v="khong_thay">không thấy được <span class="key">3</span></button></div></div>`;
  });
  h += `<div class="q"><p>Tổng thể: ảnh này có đúng là <b>${it.subject}</b> không?</p>` +
    `<div class="btns" data-f="overall">` +
    `<button data-v="dung">đúng <span class="key">d</span></button>` +
    `<button data-v="sai">sai <span class="key">s</span></button>` +
    `<button data-v="khong_chac">không chắc <span class="key">k</span></button></div></div>`;
  el("qs").innerHTML = h;
  el("qs").querySelectorAll(".btns").forEach(g => {
    const f = g.dataset.f;
    g.querySelectorAll("button").forEach(b => {
      if (ans[k][f] === b.dataset.v) b.classList.add("sel");
      b.onclick = () => { pick(f, b.dataset.v); };
    });
  });
  shownAt = Date.now();
}

function pick(field, val) {
  const k = Object.keys(D)[i];
  ans[k][field] = val;
  ans[k].ms = (ans[k].ms || 0) + (Date.now() - shownAt);
  save(); render();
  if (answered(k) && i < Object.keys(D).length - 1) setTimeout(() => { i++; render(); }, 180);
}

function firstUnanswered(field) {
  const k = Object.keys(D)[i], it = D[k];
  for (let j = 0; j < it.attrs.length; j++) if (!ans[k]["a" + j]) return "a" + j;
  return null;
}

document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") { if (i < Object.keys(D).length - 1) { i++; render(); } return; }
  if (e.key === "ArrowLeft") { if (i > 0) { i--; render(); } return; }
  const m = {1: "co", 2: "khong", 3: "khong_thay"}[e.key];
  if (m) { const f = firstUnanswered(); if (f) pick(f, m); return; }
  const o = {d: "dung", s: "sai", k: "khong_chac"}[e.key.toLowerCase()];
  if (o) pick("overall", o);
});

function exportJson() {
  const out = [];
  for (const [k, it] of Object.entries(D)) {
    const a = ans[k]; if (!a) continue;
    it.attrs.forEach((attr, j) => {
      if (!a["a" + j]) return;
      out.push({prompt_id: it.prompt_id, image: it.path, attribute_en: it.attrs_en[j], side: it.sides[j],
                judgement: a["a" + j], overall: a.overall || null, annotator: ANNOT, ms: a.ms || null});
    });
  }
  const b = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"});
  const u = URL.createObjectURL(b), l = document.createElement("a");
  l.href = u; l.download = "labels_" + ANNOT + ".json"; l.click(); URL.revokeObjectURL(u);
}
render();
"""


def collect(runs: list[str], ids: set[str] | None, per_prompt: int, seed: int):
    """[(prompt_id, image_path)] xáo trộn, tối đa per_prompt ảnh mỗi prompt, trải đều các hàng model."""
    by_prompt: dict[str, list[str]] = {}
    for run in runs:
        # Run của vòng sửa v2 xếp ảnh theo iter{n}/ và ghi loop_v2.json, không có multigen.json ở cấp prompt.
        # Lấy đúng danh sách `kept` để mỗi VÒNG một ảnh — đây chính là thứ cần nhãn: vòng nào thật sự tốt hơn.
        for lf in sorted(glob.glob(f"{run}/*/loop_v2.json")):
            pid = Path(lf).parent.name
            if ids and pid not in ids:
                continue
            d = json.load(open(lf))
            by_prompt.setdefault(pid, []).extend([x for x in (d.get("kept") or []) if os.path.exists(x)])
        for mg in sorted(glob.glob(f"{run}/*/multigen.json")):
            pid = Path(mg).parent.name
            if ids and pid not in ids:
                continue
            rows: dict[str, list[str]] = {}
            for p in sorted(glob.glob(f"{run}/{pid}/*/*.png")) + sorted(glob.glob(f"{run}/{pid}/*/*/*/*/*.png")):
                if Path(p).name in ("mask.png", "grid.png"):
                    continue
                rows.setdefault(Path(p).parent.name, []).append(p)
            picked, r = [], random.Random(seed)
            names = sorted(rows)
            while len(picked) < per_prompt and any(rows[n] for n in names):   # vòng tròn qua các hàng
                for n in names:
                    if rows[n] and len(picked) < per_prompt:
                        picked.append(rows[n].pop(0))
            by_prompt.setdefault(pid, []).extend(picked)
    items = [(pid, p) for pid, ps in sorted(by_prompt.items()) for p in ps[:per_prompt]]
    random.Random(seed).shuffle(items)          # trộn để người gán không đoán được nhánh
    return items


def spec_of(run_dirs: list[str], pid: str):
    for run in run_dirs:
        f = Path(run) / pid / "step_spec.json"
        if f.exists():
            return json.load(open(f))["value"]
    return None


def prompt_text(run_dirs: list[str], pid: str) -> tuple[str, str]:
    for f in ("data/prompts_simple.json", "data/prompts_complex.json"):
        p = Path(__file__).resolve().parent.parent / f
        if p.exists():
            for rec in json.load(open(p)):
                if rec.get("id") == pid:
                    return rec.get("text_vi", ""), ", ".join(rec.get("entities", []) or [])
    return "", ""


def build(runs: list[str], out: str, ids: set[str] | None, per_prompt: int, annotator: str, seed: int, log=print) -> str:
    items = collect(runs, ids, per_prompt, seed)
    data, skipped = {}, 0
    for pid, path in items:
        sp = spec_of(runs, pid)
        if not sp:
            skipped += 1
            continue
        attrs_en, sides = [], []
        for se in sp["entities"]:
            for a in (se.get("required_attrs_en") or []):
                if a:
                    attrs_en.append(a); sides.append("must_have")
            for a in (se.get("forbidden_attrs_en") or []):
                if a:
                    attrs_en.append(a); sides.append("must_not")
        if not attrs_en:
            skipped += 1
            continue
        order = list(range(len(attrs_en)))
        random.Random(seed + hash(path) % 9973).shuffle(order)   # trộn must_have với must_not
        attrs_en = [attrs_en[j] for j in order]
        sides = [sides[j] for j in order]
        b64 = thumb_b64(path, 760, quality=80)
        if not b64:
            skipped += 1
            continue
        vi, subject = prompt_text(runs, pid)
        data[f"{pid}|{Path(path).name}"] = {
            "prompt_id": pid, "path": path, "img": b64, "prompt_vi": vi,
            "subject": subject or pid, "attrs": attrs_en, "attrs_en": attrs_en, "sides": sides,
        }
    if not data:
        raise SystemExit("không gom được ảnh nào; kiểm lại run_dir và --ids")

    n_q = sum(len(v["attrs"]) + 1 for v in data.values())
    page = (
        "<!doctype html><meta charset='utf-8'><title>CTIG · gán nhãn</title>"
        f"<style>{CSS}</style>"
        "<header><b>Gán nhãn ảnh sinh</b> <span class='muted'>phím <span class='key'>1</span> có · "
        "<span class='key'>2</span> không · <span class='key'>3</span> không thấy được · "
        "<span class='key'>d</span>/<span class='key'>s</span>/<span class='key'>k</span> cho câu tổng thể · "
        "<span class='key'>←</span><span class='key'>→</span> chuyển ảnh</span>"
        "<div class='bar'><i id='bar'></i></div></header>"
        "<main><div><img id='shot' class='shot' alt=''><p id='prompt' class='prompt'></p></div>"
        "<div><div id='qs'></div><div class='nav'>"
        "<button onclick='if(i>0){i--;render()}'>← trước</button>"
        "<button onclick='if(i<Object.keys(D).length-1){i++;render()}'>sau →</button>"
        "<span class='muted' id='pos'></span><span class='done' id='done'></span>"
        "<button onclick='exportJson()' style='margin-left:auto'>Xuất JSON</button>"
        "</div></div></main>"
        f"<script>const DATA={json.dumps(data, ensure_ascii=False)};"
        f"const DATA_ID={json.dumps(f'{annotator}_{seed}_{len(data)}')};"
        f"const ANNOT={json.dumps(annotator)};{JS}</script>"
    )
    Path(out).write_text(page, encoding="utf-8")
    log(f"{out} · {len(data)} ảnh · {n_q} câu hỏi · {os.path.getsize(out) // 1024} KB"
        + (f" · bỏ qua {skipped} ảnh thiếu spec" if skipped else ""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="một hoặc nhiều thư mục run")
    ap.add_argument("-o", "--out", default="labels.html")
    ap.add_argument("--ids", default=None, help="giới hạn prompt, cách nhau bằng dấu phẩy")
    ap.add_argument("--per-prompt", type=int, default=6, help="số ảnh mỗi prompt (trải đều các hàng model)")
    ap.add_argument("--annotator", default="a1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    build(a.runs, a.out, {i.strip() for i in a.ids.split(",")} if a.ids else None,
          a.per_prompt, a.annotator, a.seed)


if __name__ == "__main__":
    main()
