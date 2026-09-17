"""Trang đối chiếu prompt GỐC với prompt do Culture-TRIP viết lại, để duyệt trước khi đem đi sinh ảnh.

    python scripts/culture_trip_review.py [--dir data/culture_trip] [-o ct_review.html] [--md ct_review.md]

Hiện mỗi prompt: câu gốc, câu sẽ đưa vào bộ sinh, số từ, số token CLIP, cách bóc (regex / llm), tỉ lệ giữ
bối cảnh của bản thô, và chuỗi THÔ mà Culture-TRIP trả về (gập lại) để soi khi thấy nghi.

Cờ cần chú ý:
  · "llm"        phần bóc phải nhờ LLM vì regex không khớp -> nên liếc qua
  · "mất cảnh"   bản thô giữ dưới 50% từ khoá câu gốc; câu gốc đã được ghép lại nên vẫn dùng được
  · ">77 tok"    SDXL/RealVis nối embedding nên không sao; SD 3.5 Medium sẽ cắt phần cuối
"""

from __future__ import annotations

import argparse
import glob
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E = html.escape

CSS = """body{font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1d1d1f;background:#faf9f7}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #dcdcd8;padding:12px 20px;z-index:2}
main{max-width:1080px;margin:0 auto;padding:16px 20px 60px}
.card{background:#fff;border:1px solid #e6e5e1;border-radius:8px;padding:14px 16px;margin:12px 0}
.id{font-weight:700;font-size:17px}
.tag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:10px;background:#eee;color:#555;margin-left:6px}
.warn{background:#fdecea;color:#a3261f}.info{background:#e8f2ff;color:#1a56a8}.ok{background:#e7f5ee;color:#0f6b46}
.vi{color:#6b6b70;font-size:14px;margin:6px 0 0}
.old{margin:8px 0 0}.new{margin:8px 0 0;padding:10px 12px;background:#f6f8f6;border-left:3px solid #168052;border-radius:0 4px 4px 0}
.lbl{font-size:12px;color:#8a8a8f;text-transform:uppercase;letter-spacing:.04em}
details{margin-top:8px}summary{cursor:pointer;font-size:13px;color:#6b6b70}
pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,monospace;background:#f5f5f2;padding:10px;border-radius:4px;margin:6px 0 0}
table{border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #e6e5e1;padding:4px 10px;text-align:left}"""


def load(d: Path) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(d / "*.json"))):
        try:
            out.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out


def prompt_meta() -> dict:
    meta = {}
    for f in ("data/prompts_simple.json", "data/prompts_complex.json"):
        p = ROOT / f
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8")):
                meta[r["id"]] = r
    return meta


def build(src: Path, out_html: Path, out_md: Path | None, log=print) -> None:
    recs = load(src)
    if not recs:
        raise SystemExit(f"không có tệp nào trong {src}")
    meta = prompt_meta()
    tok = None
    try:
        from transformers import CLIPTokenizerFast

        tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    except Exception:  # noqa: BLE001
        log("[lưu ý] không nạp được tokenizer CLIP -> số token là ước lượng")

    n_tok = lambda t: (len(tok(t)["input_ids"]) if tok else int(len(t.split()) * 1.35))  # noqa: E731
    cards, rows = [], []
    n_llm = n_lost = n_over = 0
    for d in recs:
        pid = d["prompt_id"]
        m = meta.get(pid, {})
        new, old = d["refined_prompt"], d["prompt_en"]
        nt = n_tok(new)
        by = ",".join(sorted({s.get("cleaned_by", "?") for s in d.get("per_step", [])}))
        keep = d.get("scene_keep_raw", 1.0)
        n_llm += "llm" in by
        n_lost += keep < 0.5
        n_over += nt > 77
        tags = [f"<span class='tag'>{len(old.split())} → {len(new.split())} từ</span>",
                f"<span class='tag {'warn' if nt > 77 else 'ok'}'>{nt} token CLIP</span>",
                f"<span class='tag {'info' if 'llm' in by else ''}'>bóc: {E(by)}</span>"]
        if keep < 0.5:
            tags.append(f"<span class='tag warn'>bản thô mất cảnh ({keep:.0%})</span>")
        if d.get("chained"):
            tags.append(f"<span class='tag info'>nối chuỗi {len(d.get('culture_nouns') or [])} thực thể</span>")
        raw = "\n\n".join(f"[{s.get('culture_noun','?')}]\n{s.get('out_raw','')}" for s in d.get("per_step", []))
        cards.append(
            f"<div class='card'><span class='id'>{E(pid)}</span>{''.join(tags)}"
            f"<div class='vi'>{E(m.get('text_vi',''))}</div>"
            f"<div class='old'><span class='lbl'>câu gốc</span><br>{E(old)}</div>"
            f"<div class='new'><span class='lbl'>câu đưa vào bộ sinh</span><br>{E(new)}</div>"
            f"<details><summary>chuỗi thô Culture-TRIP trả về ({sum(s.get('words_raw',0) for s in d.get('per_step',[]))} từ)"
            f"</summary><pre>{E(raw[:6000])}</pre></details></div>")
        rows.append((pid, len(old.split()), len(new.split()), nt, by, keep, old, new))

    head = (f"<b>{len(recs)} prompt</b> · bóc bằng LLM ở {n_llm} · bản thô mất cảnh ở {n_lost} "
            f"(đã ghép lại câu gốc) · vượt 77 token ở {n_over} "
            "<span class='tag'>SDXL/RealVis nối embedding, không cắt; SD 3.5 Medium sẽ cắt</span>")
    out_html.write_text(
        "<!doctype html><meta charset='utf-8'><title>Culture-TRIP · đối chiếu prompt</title>"
        f"<style>{CSS}</style><header>{head}</header><main>{''.join(cards)}</main>", encoding="utf-8")
    log(f"{out_html} · {len(recs)} prompt · {out_html.stat().st_size // 1024} KB")

    if out_md:
        lines = ["# Culture-TRIP · đối chiếu prompt", "", head.replace("<b>", "**").replace("</b>", "**"), "",
                 "| id | từ | token | bóc | giữ cảnh |", "|---|---|---|---|---|"]
        lines += [f"| {p} | {a} → {b} | {t} | {c} | {k:.0%} |" for p, a, b, t, c, k, _, _ in rows]
        lines.append("")
        for p, _, _, _, _, _, old, new in rows:
            lines += [f"## {p}", "", f"**gốc:** {old}", "", f"**mới:** {new}", ""]
        out_md.write_text("\n".join(lines), encoding="utf-8")
        log(f"{out_md} · {out_md.stat().st_size // 1024} KB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/culture_trip")
    ap.add_argument("-o", "--out", default="ct_review.html")
    ap.add_argument("--md", default="ct_review.md")
    a = ap.parse_args(argv)
    d = Path(a.dir) if Path(a.dir).is_absolute() else ROOT / a.dir
    build(d, Path(a.out), Path(a.md) if a.md else None)


if __name__ == "__main__":
    main()
