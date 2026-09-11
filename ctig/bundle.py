"""
Gom kết quả một lần chạy thành HAI file tự chứa để tải từng file khỏi Kaggle:

    bundle.html   mọi prompt, mọi vòng, ảnh nhúng base64 (thu nhỏ), phán quyết, bất đồng VLM/CLIP
    bundle.json   summary + records + evidence đã rút + review chi tiết của các prompt chọn lọc
                  + ảnh thu nhỏ base64 -> đủ để người khác đánh giá mà không cần thư mục runs/

    python -m ctig.bundle runs/v1-full --max-side 448 --detail p001,p003,p004,p009,p012,p017,p048,p050
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from io import BytesIO
from pathlib import Path

DEFAULT_DETAIL = "p001,p003,p004,p009,p012,p017,p048,p050"


def thumb_b64(path: str, max_side: int, quality: int = 78) -> str | None:
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _join(g: dict, list_key: str, str_key: str) -> str:
    """GenSpec v1.1 lưu danh sách cụm; v1 lưu chuỗi. Đọc được cả hai."""
    if isinstance(g.get(list_key), list):
        return ", ".join(dict.fromkeys(t for t in g[list_key] if t))
    return str(g.get(str_key, ""))


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def make_bundle(run_dir: Path, max_side: int, detail_ids: set[str], out_html: Path, out_json: Path) -> None:
    summary = load(run_dir / "summary.json") or {}
    records = load(run_dir / "records.json") or []
    config = load(run_dir / "config.json") or {}
    cache_dir = Path(config.get("cache", {}).get("dir") or Path(config.get("runs_dir", run_dir.parent)) / "_cache")
    if not cache_dir.exists():
        cache_dir = run_dir.parent / "_cache"
    evidence = {p.stem: load(p) for p in sorted((cache_dir / "evidence").glob("*.json"))} if (cache_dir / "evidence").exists() else {}

    sections, detail = [], {}
    for rec in records:
        pid = rec["prompt_id"]
        review = load(run_dir / pid / "stage45_review.json") or {}
        spec = load(run_dir / pid / "stage3_spec.json") or {}
        analysis = load(run_dir / pid / "stage1_analysis.json") or {}
        iters = review.get("iterations", [])
        thumbs = [thumb_b64(it["gen_output"]["candidates"][it["gen_output"]["chosen"]]["path"], max_side) for it in iters]

        cards = []
        for it, th in zip(iters, thumbs):
            adj = it["adjudication"]
            finds = "".join(f"<li class='{f['severity']}'>{html.escape(f['message'][:160])}</li>" for f in adj["merged_findings"][:6])
            dis = "".join(f"<li>{html.escape(d[:200])}</li>" for d in adj.get("disagreements", []))
            clip = "<br>".join(
                html.escape(next((e["name_vi"] for e in spec.get("entities", []) if e["entity_id"] == eid), eid)) + ": "
                + ", ".join(f"{html.escape('mục tiêu' if k == '__target__' else k)} {v:.2f}" for k, v in p.items())
                for eid, p in it["perception"].get("clip_probs", {}).items())
            elems = ", ".join(html.escape(e["label"]) for e in it["perception"].get("elements", [])[:6])
            ck = "; ".join(f"{eid}: {c.get('identity')}" for eid, c in (it['perception'].get('checklist') or {}).items())
            cards.append(f"""<div class='it {adj['verdict']}'>{f"<img src='{th}'>" if th else '<div class=noimg>không có ảnh</div>'}
              <div class='m'><b>vòng {it['n']}{' · render đủ bước' if it.get('final_render') else ''}</b> · {adj['score']:.2f} · <b>{adj['verdict']}</b>{' · LoRA' if it['gen_spec'].get('lora') else ''}{' · ref' if it['gen_spec'].get('ip_adapter_image') else ''}
              <div class='cap'>VLM thấy: {elems or '—'}</div><div class='clip'>{clip}</div>{f"<div class='cap'>checklist: {html.escape(ck)}</div>" if ck else ''}
              <ul>{finds}</ul>{f"<div class='dis'>{dis}</div>" if dis else ''}
              <details><summary>prompt</summary><pre>{html.escape(_join(it['gen_spec'], 'prompt_terms', 'prompt')[:600])}</pre><pre class='neg'>NEG: {html.escape(_join(it['gen_spec'], 'negative_terms', 'negative_prompt')[:400])}</pre></details></div></div>""")
        ents = ", ".join(f"{e['name_vi']}" + (" <i>(ad-hoc)</i>" if e["entity_id"].startswith("x_") else "") for e in spec.get("entities", []))
        rec_r = "n/a" if rec["retrieval_recall"] < 0 else f"{rec['retrieval_recall']:.2f}"
        grid_p = run_dir / pid / "grid.png"
        grid_html = ""
        if grid_p.exists():
            gth = thumb_b64(str(grid_p), max_side * 2)
            grid_html = f"<h4>So nhiều model</h4><img src='{gth}' style='max-width:100%'>" if gth else ""
        sections.append(f"""<section class='{'ok' if rec['passed'] else 'fail'}'>
          <h2>{pid} — {html.escape(rec['prompt_text'])}</h2>
          <div class='s'>thực thể: {ents or '<i>spec rỗng</i>'} · đạt <b>{'Y' if rec['passed'] else 'n'}</b> · review {rec['review_score']:.2f} · CLIP {rec['clip_fidelity']:.2f} · judge {rec['judge_score']:.2f} · recall {rec_r} · {rec['iterations']} vòng</div>
          <div class='row'>{''.join(cards)}</div>{grid_html}
          <details><summary>judge</summary>{html.escape(rec.get('judge_reasoning', '')[:800])}</details></section>""")

        if pid in detail_ids:
            detail[pid] = {
                "record": rec, "analysis": analysis, "spec": spec,
                "iterations": [{
                    "n": it["n"], "gen_spec": it["gen_spec"], "chosen_clip_fidelity": it["gen_output"]["candidates"][it["gen_output"]["chosen"]].get("clip_fidelity"),
                    "perception": it["perception"], "critiques": it["critiques"], "adjudication": it["adjudication"], "plan": it["plan"],
                    "thumb": th,
                } for it, th in zip(iters, thumbs)],
            }

    s = summary
    doc = f"""<!doctype html><meta charset='utf-8'><title>CTIG bundle {html.escape(s.get('run_id', ''))}</title><style>
body{{font-family:system-ui,sans-serif;margin:18px;background:#faf9f6;color:#1c1e22;font-size:13px}}
h1{{font-size:18px}}h2{{font-size:14px;margin:0 0 4px}}section{{background:#fff;border-left:5px solid #c4302b;padding:10px 14px;margin:12px 0;border-radius:6px}}
section.ok{{border-left-color:#168052}}.row{{display:flex;gap:10px;overflow-x:auto;padding:6px 0}}
.it{{min-width:280px;max-width:280px;border:1px solid #ddd;border-radius:6px;padding:6px;background:#f4f3ef}}.it.pass{{border-color:#168052}}
.it img{{width:100%;border-radius:4px}}.m{{font-size:11px}}.cap,.clip{{color:#555;margin:3px 0}}ul{{padding-left:14px;margin:3px 0}}
li.critical{{color:#c4302b}}li.major{{color:#96691e}}li.minor{{color:#666}}.dis{{background:#fff3cd;padding:3px 5px;border-radius:4px;font-size:10px}}
pre{{white-space:pre-wrap;font-size:10px;background:#eee;padding:4px}}.neg{{color:#c4302b}}.s{{color:#444;margin-bottom:4px}}.noimg{{height:160px;background:#ddd}}
table td,table th{{padding:2px 10px;text-align:left;border-bottom:1px solid #ddd}}</style>
<h1>CTIG — {html.escape(s.get('run_id', run_dir.name))}</h1>
<table><tr><th>prompt</th><td>{s.get('n_prompts')}</td><th>spec rỗng</th><td>{s.get('n_unverifiable')}</td></tr>
<tr><th>đạt cuối / vòng 0</th><td>{s.get('pass_rate', 0):.3f} / {s.get('pass_rate_iter0', 0):.3f}</td><th>CLIP cuối / vòng 0</th><td>{s.get('mean_clip_fidelity', 0):.3f} / {s.get('mean_clip_fidelity_iter0', 0):.3f}</td></tr>
<tr><th>judge</th><td>{s.get('mean_judge_score', 0):.3f}</td><th>recall</th><td>{s.get('mean_retrieval_recall', -1):.3f}</td></tr>
<tr><th>vòng TB</th><td>{s.get('mean_iterations', 0):.2f}</td><th>thời gian</th><td>{s.get('wall_seconds', 0) / 60:.0f} phút</td></tr></table>
<p>Bằng chứng đã rút: {len(evidence)} thực thể ({', '.join(sorted(evidence)[:12])}{'…' if len(evidence) > 12 else ''})</p>
{''.join(sections)}"""
    out_html.write_text(doc, encoding="utf-8")

    bundle = {"summary": summary, "config": config, "records": records, "evidence_extracted": evidence, "detail": detail,
              "note": "thumb là JPEG base64 thu nhỏ; ảnh gốc trong runs/<run>/<prompt_id>/"}
    out_json.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    print(f"bundle.html {out_html.stat().st_size / 1e6:.1f} MB | bundle.json {out_json.stat().st_size / 1e6:.1f} MB | "
          f"{len(records)} prompt, chi tiết {len(detail)}, bằng chứng {len(evidence)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--detail", default=DEFAULT_DETAIL, help="id prompt lấy JSON chi tiết, cách nhau bằng dấu phẩy; 'all' = tất cả")
    ap.add_argument("--out", default=None, help="thư mục ghi (mặc định: chính run_dir)")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    out = Path(a.out) if a.out else run_dir
    records = load(run_dir / "records.json") or []
    ids = {r["prompt_id"] for r in records} if a.detail == "all" else set(a.detail.split(","))
    make_bundle(run_dir, a.max_side, ids, out / "bundle.html", out / "bundle.json")


if __name__ == "__main__":
    main()
