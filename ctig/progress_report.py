"""
BÁO CÁO TIẾN ĐỘ (v1.3) - một file HTML tự chứa để gửi người hướng dẫn, dựng từ thư mục runs/<run>/.

Khác `viz.Report` (nhật ký từng cell, ảnh thu nhỏ 220 px) và `bundle` (dữ liệu để tôi/anh chị chẩn đoán):
  * ảnh nhúng ở độ phân giải gốc (tối đa `img_side` px, JPEG chất lượng cao) -> in / zoom vẫn nét;
  * bảng và biểu đồ là HTML/SVG vector, không phải ảnh chụp;
  * có phần "kết luận đến nay" và "giả thuyết đang kiểm" lấy từ research/ nếu repo có thư mục đó;
  * thêm `grid_hires.png` (ô 768 px) cho slide.

Dùng:  python -m ctig.progress_report /kaggle/working/runs/walkthrough --out /kaggle/working/progress_report.html
       hoặc trong notebook: from ctig.progress_report import build; build(run_dir, out_path, title=...)
Mở file bằng trình duyệt, Ctrl+P -> "Save as PDF" nếu cần PDF (chữ vẫn là vector).
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import re
import subprocess
from io import BytesIO
from pathlib import Path

from .schema import AnalysisResult, CulturalSpec, GenSpec, MultiGenResult, SearchResult, from_dict

_e = html.escape

CSS = """
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-size:14px;color:#1c1e22;background:#fff;margin:0;padding:24px 32px;line-height:1.45}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px;border-bottom:2px solid #e5e2da;padding-bottom:4px}
h3{font-size:15px;margin:18px 0 6px}.muted{color:#6b7280}.small{font-size:12px}
table{border-collapse:collapse;font-size:13px;margin:6px 0}th,td{border:1px solid #e5e2da;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f4f2ec}tr.best td{background:#dcfce7}.bad{color:#b91c1c}.ok{color:#15803d}
.row{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;margin:6px 0 14px}
.cell img{width:100%;height:auto;display:block;border-radius:4px;border:1px solid #e5e2da}
.cell.chosen img{outline:3px solid #16a34a;outline-offset:-3px}.cap{font-size:12px;color:#444;margin-top:3px}
.hero{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.hero img{width:100%;border-radius:6px}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:9px;background:#eef2ff;color:#3730a3;margin-left:6px}
.kv td:first-child{font-weight:600;white-space:nowrap;background:#faf9f6}
details summary{cursor:pointer;color:#3730a3}
@media print{body{padding:0}.row{grid-template-columns:repeat(4,1fr)}h2{page-break-after:avoid}.cell{page-break-inside:avoid}}
</style>
"""

STEP_FILES = {"analysis": AnalysisResult, "retrieve": SearchResult, "spec": CulturalSpec, "genspec": GenSpec, "multigen": MultiGenResult}


# ------------------------------------------------------------------ đọc dữ liệu
def load_prompt_dir(pdir: Path) -> dict:
    """Đọc step_*.json (Session) và multigen.json của một prompt. Thiếu bước nào thì bỏ qua bước đó."""
    out: dict = {"id": pdir.name, "dir": pdir}
    for name, cls in STEP_FILES.items():
        f = pdir / f"step_{name}.json"
        if f.exists():
            try:
                out[name] = from_dict(cls, json.loads(f.read_text(encoding="utf-8"))["value"])
            except Exception:  # noqa: BLE001
                pass
    if "multigen" not in out and (pdir / "multigen.json").exists():
        try:
            out["multigen"] = from_dict(MultiGenResult, json.loads((pdir / "multigen.json").read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return out


def find_prompt_dirs(run_dir: Path) -> list[Path]:
    return sorted(p for p in Path(run_dir).iterdir() if p.is_dir() and not p.name.startswith("_")
                  and ((p / "multigen.json").exists() or any(p.glob("step_*.json"))))


# ------------------------------------------------------------------ ảnh
def img_b64(path: str | Path, max_side: int = 1024, quality: int = 90) -> str | None:
    """JPEG chất lượng cao ở độ phân giải gốc (kẹp max_side). 1024 px ~ 250-400 KB mỗi ảnh."""
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True, subsampling=0)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _img(path, max_side, cap: str = "", chosen: bool = False) -> str:
    b = img_b64(path, max_side)
    body = f"<img src='{b}' loading='lazy'>" if b else f"<div class='muted'>không đọc được {_e(Path(str(path)).name)}</div>"
    return f"<div class='cell{' chosen' if chosen else ''}'>{body}<div class='cap'>{cap}</div></div>"


# ------------------------------------------------------------------ biểu đồ SVG (vector, không cần matplotlib)
def bar_chart_svg(series: dict[str, list[tuple[str, float]]], width: int = 720, height: int = 260,
                  ymax: float = 1.0, title: str = "") -> str:
    """Nhóm cột: series = {tên chuỗi: [(nhãn, giá trị), ...]} với cùng thứ tự nhãn."""
    colors = ["#2563eb", "#16a34a", "#d97706", "#7c3aed"]
    names = list(series)
    labels = [l for l, _ in series[names[0]]] if names else []
    if not labels:
        return ""
    left, bottom, top = 44, 58, 28
    W, H = width, height
    plot_w, plot_h = W - left - 12, H - bottom - top
    group_w = plot_w / len(labels)
    bar_w = group_w / (len(names) + 1)
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' "
           f"style='font-family:system-ui,sans-serif;font-size:11px;max-width:100%;height:auto'>"]
    if title:
        out.append(f"<text x='{left}' y='16' font-size='13' font-weight='600' fill='#1c1e22'>{_e(title)}</text>")
    for k in range(5):
        y = top + plot_h - plot_h * k / 4
        out.append(f"<line x1='{left}' y1='{y:.1f}' x2='{W - 12}' y2='{y:.1f}' stroke='#e5e2da'/>")
        out.append(f"<text x='{left - 6}' y='{y + 4:.1f}' text-anchor='end' fill='#6b7280'>{ymax * k / 4:.2f}</text>")
    for gi, lab in enumerate(labels):
        gx = left + gi * group_w
        for si, name in enumerate(names):
            v = series[name][gi][1]
            if v is None:
                continue
            h = plot_h * max(0.0, min(v, ymax)) / ymax
            x = gx + bar_w * (si + 0.5)
            y = top + plot_h - h
            out.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w * 0.9:.1f}' height='{h:.1f}' fill='{colors[si % len(colors)]}' rx='2'/>")
            out.append(f"<text x='{x + bar_w * 0.45:.1f}' y='{y - 3:.1f}' text-anchor='middle' fill='#1c1e22'>{v:.2f}</text>")
        out.append(f"<text x='{gx + group_w / 2:.1f}' y='{top + plot_h + 16}' text-anchor='middle' fill='#1c1e22' "
                   f"transform='rotate(-18 {gx + group_w / 2:.1f},{top + plot_h + 16})'>{_e(lab)}</text>")
    lx = left
    for si, name in enumerate(names):
        out.append(f"<rect x='{lx}' y='{H - 14}' width='10' height='10' fill='{colors[si % len(colors)]}' rx='2'/>")
        out.append(f"<text x='{lx + 14}' y='{H - 5}' fill='#1c1e22'>{_e(name)}</text>")
        lx += 14 + 7 * len(name) + 18
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ các mục
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def model_stats(res: MultiGenResult) -> list[dict]:
    from .stages.multigen import combined_score

    rows = []
    for r in res.runs:
        if not r.output:
            rows.append({"model": r.model_key, "error": r.error or "không có ảnh"})
            continue
        cs = r.output.candidates
        best = max(cs, key=combined_score)
        rows.append({"model": r.model_key, "n": len(cs), "attr": _mean([c.attr_contrast for c in cs]),
                     "attr_best": best.attr_contrast, "itm_attr": _mean([c.itm_attrs for c in cs]),
                     "pick": _mean([c.pick_score for c in cs]), "aes": _mean([c.aesthetic for c in cs]),
                     "total": _mean([combined_score(c) for c in cs]), "best": combined_score(best),
                     "sec": r.seconds, "vram": r.peak_vram_gb, "notes": r.notes, "tokens": r.prompt_tokens,
                     "best_path": best.path})
    return rows


def _f(v, nd=2):
    return "" if v is None else f"{v:.{nd}f}"


def section_prompt(d: dict, kb, img_side: int, top_k: int = 3) -> str:
    from .stages.multigen import combined_score
    from . import viz

    pid = d["id"]
    a: AnalysisResult | None = d.get("analysis")
    sp: CulturalSpec | None = d.get("spec")
    g: GenSpec | None = d.get("genspec")
    res: MultiGenResult | None = d.get("multigen")
    parts = [f"<h2 id='{_e(pid)}'>Prompt {_e(pid)}</h2>"]
    # prompt + keywords
    kv = []
    if a is not None:
        kv.append(("Prompt (VI)", _e(a.prompt_id if False else "")))
    kv = []
    if res is not None and res.prompt_en:
        kv.append(("Prompt EN (đưa vào bộ sinh)", _e(res.prompt_en)))
    if a is not None:
        kws = ", ".join(f"{_e(k.term)} <span class='muted'>({_e(k.kind)})</span>" for k in a.keywords)
        kv.append(("Keywords (Analysis agent)", kws or "<span class='muted'>không có</span>"))
        kv.append(("Thực thể ứng viên", ", ".join(_e(x) for x in a.candidate_entity_ids) or "-"))
    if sp is not None:
        ents = []
        for se in sp.entities:
            mh = "; ".join(_e(x) for x in se.required_attrs_en[:4] if x) or "; ".join(_e(x) for x in se.required_attrs[:4])
            mn = "; ".join(_e(x) for x in se.forbidden_attrs_en[:3] if x)
            ents.append(f"<b>{_e(se.name_vi)}</b> <span class='muted'>({_e(se.name_en)}, w={se.weight:g}, {_e(se.kind)})</span>"
                        f"<br><span class='ok'>phải có:</span> {mh}" + (f"<br><span class='bad'>không được:</span> {mn}" if mn else ""))
        kv.append(("Hợp đồng văn hoá (spec)", "<br>".join(ents) or "-"))
    if g is not None:
        kv.append(("Prompt cuối", _e(g.prompt)))
        kv.append(("Negative", _e(g.negative_prompt)))
        kv.append(("Sinh", f"seed {g.seed} · {g.steps} bước · guidance {g.guidance:g} · {g.width}×{g.height} · {g.n_candidates} ứng viên/model"))
    if kv:
        parts.append("<table class='kv'>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in kv) + "</table>")

    if res is None:
        parts.append("<div class='muted'>Chưa có bước sinh ảnh cho prompt này.</div>")
        return "".join(parts)

    # top-K toàn cục
    cands = [(combined_score(c), r, c) for r in res.runs if r.output for c in r.output.candidates]
    cands.sort(key=lambda x: -x[0])
    if cands:
        parts.append(f"<h3>Ảnh tốt nhất theo điểm tổng (top {min(top_k, len(cands))} trong {len(cands)} ứng viên)</h3><div class='hero'>")
        for s, r, c in cands[:top_k]:
            cap = (f"<b>{_e(r.model_key)}</b> · tổng {s:.3f} · attr {_f(c.attr_contrast)} · ITM attr {_f(c.itm_attrs)}"
                   + (f" · đẹp {_f(c.aesthetic)}" if c.aesthetic is not None else "") + f" · seed {c.seed}")
            parts.append(_img(c.path, img_side, cap))
        parts.append("</div>")

    # bảng tóm tắt theo model + biểu đồ
    stats = model_stats(res)
    best_model = max((s for s in stats if "error" not in s), key=lambda s: s["best"], default=None)
    rows = ["<table><tr><th>model</th><th>ảnh</th><th>CLIP attr TB</th><th>CLIP attr tốt nhất</th><th>ITM attr TB</th>"
            "<th>PickScore TB</th><th>điểm tổng TB</th><th>tốt nhất</th><th>giây</th><th>VRAM đỉnh</th><th>ghi chú</th></tr>"]
    for s in stats:
        if "error" in s:
            rows.append(f"<tr><td>{_e(s['model'])}</td><td colspan='10' class='bad'>{_e(s['error'][:160])}</td></tr>")
            continue
        cls = " class='best'" if best_model and s["model"] == best_model["model"] else ""
        notes = "; ".join(n for n in (s["notes"] or []) if not n.startswith("scheduler"))
        rows.append(f"<tr{cls}><td><b>{_e(s['model'])}</b></td><td>{s['n']}</td><td>{_f(s['attr'], 3)}</td><td>{_f(s['attr_best'], 3)}</td>"
                    f"<td>{_f(s['itm_attr'], 3)}</td><td>{_f(s['pick'], 1)}</td><td>{_f(s['total'], 3)}</td><td><b>{_f(s['best'], 3)}</b></td>"
                    f"<td>{s['sec']:.0f}</td><td>{'' if s['vram'] is None else f'{s[chr(118)+chr(114)+chr(97)+chr(109)]} GB'}</td>"
                    f"<td class='small muted'>{_e(notes[:140])}</td></tr>")
    rows.append("</table>")
    parts.append("<h3>Tóm tắt theo model</h3>" + "".join(rows))
    ok = [s for s in stats if "error" not in s]
    if ok:
        series = {"CLIP attr (đúng thuộc tính)": [(s["model"], s["attr"]) for s in ok],
                  "ITM attr (đủ thuộc tính)": [(s["model"], s["itm_attr"]) for s in ok]}
        if any(s["aes"] is not None for s in ok):
            series["đẹp (PickScore chuẩn hoá)"] = [(s["model"], s["aes"]) for s in ok]
        parts.append(bar_chart_svg(series, title=f"{pid}: điểm trung bình trên {ok[0]['n']} ứng viên mỗi model"))
    for n in res.notes or []:
        parts.append(f"<div class='bad'>⚠ {_e(n)}</div>")

    # grid đầy đủ
    parts.append("<h3>Toàn bộ ứng viên</h3>")
    for r in res.runs:
        meta = f"<b>{_e(r.model_key)}</b> <span class='muted'>{_e(r.repo.split('/')[-1])} · {r.gen_spec.steps} bước · g{r.gen_spec.guidance:g} · {r.gen_spec.width}px · {r.seconds:.0f}s</span>"
        if r.notes:
            meta += "<span class='tag'>" + "</span><span class='tag'>".join(_e(n[:50]) for n in r.notes if not n.startswith("scheduler")) + "</span>"
        parts.append(f"<div>{meta}</div>")
        if not r.output:
            parts.append(f"<div class='bad'>{_e(r.error or 'không có ảnh')}</div>")
            continue
        parts.append("<div class='row'>")
        for j, c in enumerate(r.output.candidates):
            cap = f"attr {_f(c.attr_contrast)} · ITM attr {_f(c.itm_attrs)}" + (f" · đẹp {_f(c.aesthetic)}" if c.aesthetic is not None else "") + f" · seed {c.seed}"
            parts.append(_img(c.path, img_side, cap, chosen=(j == r.output.chosen and len(r.output.candidates) > 1)))
        parts.append("</div>")
    # bảng chi tiết
    parts.append("<details><summary>Bảng điểm chi tiết từng ứng viên</summary>" + viz.score_table(res) + "</details>")
    return "".join(parts)


def md_to_html(md: str) -> str:
    """Markdown tối giản (##, -, **) -> HTML, đủ cho research/findings.md."""
    out, in_ul = [], False
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("- ") or s.startswith("  - "):
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{_inline(s.lstrip()[2:])}</li>")
            continue
        if in_ul:
            out.append("</ul>"); in_ul = False
        if s.startswith("### "):
            out.append(f"<h4>{_inline(s[4:])}</h4>")
        elif s.startswith("## "):
            out.append(f"<h3>{_inline(s[3:])}</h3>")
        elif s.startswith("# "):
            continue
        elif s:
            out.append(f"<p>{_inline(s)}</p>")
    if in_ul:
        out.append("</ul>")
    return "".join(out)


def _inline(s: str) -> str:
    s = _e(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def section_research(repo_root: Path) -> str:
    parts = []
    f = repo_root / "research" / "findings.md"
    if f.exists():
        parts.append("<h2>Kết luận đến nay (research/findings.md)</h2>" + md_to_html(f.read_text(encoding="utf-8")))
    st = repo_root / "research" / "research-state.yaml"
    if st.exists():
        try:
            import yaml

            d = yaml.safe_load(st.read_text(encoding="utf-8"))
            rows = ["<table><tr><th>id</th><th>giả thuyết</th><th>trạng thái</th><th>dự đoán kiểm được</th></tr>"]
            for h in d.get("hypotheses", []):
                rows.append(f"<tr><td>{_e(h['id'])}</td><td>{_e(h['statement'])}</td><td>{_e(h.get('status', ''))}</td>"
                            f"<td class='small'>{_e(h.get('prediction', ''))}</td></tr>")
            rows.append("</table>")
            parts.append("<h2>Giả thuyết đang kiểm (research-state.yaml)</h2>" + "".join(rows))
            traj = d.get("experiments", {}).get("trajectory", [])
            if traj:
                rows = ["<table><tr><th>lần chạy</th><th>giả thuyết</th><th>thay đổi</th><th>ghi chú</th><th>ngày</th></tr>"]
                for r in traj:
                    rows.append(f"<tr><td>{_e(str(r.get('run_id')))}</td><td>{_e(str(r.get('hypothesis', '')))}</td>"
                                f"<td>{_e(str(r.get('change_summary', '')))}</td><td class='small'>{_e(str(r.get('notes', '')))}</td>"
                                f"<td>{_e(str(r.get('timestamp', '')))}</td></tr>")
                rows.append("</table>")
                parts.append("<h3>Các lần chạy</h3>" + "".join(rows))
        except Exception as exc:  # noqa: BLE001
            parts.append(f"<div class='muted'>không đọc được research-state.yaml: {_e(str(exc))}</div>")
    return "".join(parts)


PIPELINE_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1180 150' style='max-width:100%;height:auto;font-family:system-ui,sans-serif;font-size:12px'>
<defs><marker id='ar' markerWidth='8' markerHeight='8' refX='6' refY='4' orient='auto'><path d='M0,0 L8,4 L0,8 z' fill='#6b7280'/></marker></defs>
<g fill='#fff' stroke='#2563eb' stroke-width='1.5'>
<rect x='10' y='40' width='150' height='60' rx='8'/><rect x='200' y='40' width='170' height='60' rx='8'/><rect x='410' y='40' width='190' height='60' rx='8'/>
<rect x='640' y='40' width='170' height='60' rx='8'/><rect x='850' y='40' width='150' height='60' rx='8'/><rect x='1030' y='40' width='140' height='60' rx='8'/></g>
<g fill='#1c1e22' text-anchor='middle'>
<text x='85' y='64' font-weight='600'>1 · Analysis agent</text><text x='85' y='82' fill='#6b7280'>prompt VI → keywords, EN</text>
<text x='285' y='64' font-weight='600'>2 · Search</text><text x='285' y='82' fill='#6b7280'>Wikipedia · DDG · Commons</text>
<text x='505' y='64' font-weight='600'>2b/3 · Evidence → Spec</text><text x='505' y='82' fill='#6b7280'>must_have / must_not / confusable</text>
<text x='725' y='64' font-weight='600'>4 · Gen (nhiều model)</text><text x='725' y='82' fill='#6b7280'>SDXL, RealVis, LoRA, IP-Adapter</text>
<text x='925' y='64' font-weight='600'>5 · Chấm điểm</text><text x='925' y='82' fill='#6b7280'>CLIP attr · BLIP-2 · PickScore</text>
<text x='1100' y='64' font-weight='600'>Review loop</text><text x='1100' y='82' fill='#6b7280'>(cờ, đang tắt)</text></g>
<g stroke='#6b7280' stroke-width='1.5' marker-end='url(#ar)'><line x1='160' y1='70' x2='198' y2='70'/><line x1='370' y1='70' x2='408' y2='70'/>
<line x1='600' y1='70' x2='638' y2='70'/><line x1='810' y1='70' x2='848' y2='70'/><line x1='1000' y1='70' x2='1028' y2='70'/></g>
<text x='590' y='130' text-anchor='middle' fill='#6b7280'>Ba lớp cache (LLM · web · từng bước) để chạy lại không tốn API; mọi bước ghi đầu ra ra đĩa.</text>
</svg>
"""


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo_root), "log", "-1", "--format=%h %s"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def build(run_dir: str | Path, out_path: str | Path, title: str = "CTIG - báo cáo tiến độ", author: str = "",
          img_side: int = 1024, top_k: int = 3, hires_grid: bool = True, cfg=None, log=print) -> Path:
    from .config import Config
    from .kb import KnowledgeBase

    run_dir = Path(run_dir)
    repo_root = Path(__file__).resolve().parent.parent
    cfg = cfg or Config()
    try:
        kb = KnowledgeBase.load(cfg.kb_path)
    except Exception:  # noqa: BLE001
        kb = None
    pdirs = find_prompt_dirs(run_dir)
    data = [load_prompt_dir(p) for p in pdirs]
    data = [d for d in data if "multigen" in d or "analysis" in d]

    parts = [CSS, f"<h1>{_e(title)}</h1>",
             f"<div class='muted'>{_e(author) + ' · ' if author else ''}{_dt.date.today().isoformat()} · thư mục {_e(str(run_dir))}"
             + (f" · mã nguồn {_e(_git_commit(repo_root))}" if _git_commit(repo_root) else "") + "</div>",
             "<h2>Pipeline</h2>", PIPELINE_SVG]
    # tổng quan
    if data:
        rows = ["<table><tr><th>prompt</th><th>số model</th><th>ứng viên/model</th><th>model tốt nhất</th><th>điểm tốt nhất</th><th>CLIP attr TB (mọi ảnh)</th></tr>"]
        for d in data:
            res = d.get("multigen")
            if not res:
                rows.append(f"<tr><td><a href='#{_e(d['id'])}'>{_e(d['id'])}</a></td><td colspan='5' class='muted'>chưa sinh ảnh</td></tr>")
                continue
            st = [s for s in model_stats(res) if "error" not in s]
            bm = max(st, key=lambda s: s["best"], default=None)
            all_attr = _mean([c.attr_contrast for r in res.runs if r.output for c in r.output.candidates])
            rows.append(f"<tr><td><a href='#{_e(d['id'])}'>{_e(d['id'])}</a></td><td>{len(res.runs)}</td><td>{st[0]['n'] if st else ''}</td>"
                        f"<td>{_e(bm['model']) if bm else ''}</td><td>{_f(bm['best'], 3) if bm else ''}</td><td>{_f(all_attr, 3)}</td></tr>")
        rows.append("</table>")
        parts.append("<h2>Tổng quan</h2>" + "".join(rows))
        parts.append("<div class='small muted'>CLIP attr = phần xác suất CLIP dành cho câu “thực thể với &lt;must_have&gt;” so với “với &lt;must_not&gt;” "
                     "(đo đúng thuộc tính văn hoá, vd áo dài có quần vs váy liền). ITM attr = BLIP-2 xác nhận từng thuộc tính. "
                     "PickScore = mô hình sở thích người (đo đẹp, không đo đúng). Điểm tổng = trung bình các số có. Viền xanh = ứng viên CLIP chọn.</div>")
    for d in data:
        parts.append(section_prompt(d, kb, img_side, top_k))
        res = d.get("multigen")
        if hires_grid and res is not None and any(r.output for r in res.runs):
            try:
                from .stages.multigen import draw_grid

                sp = d.get("spec")
                gp = draw_grid(res, sp, d["dir"] / "grid_hires.png", cell=768)
                log(f"[report] grid nét: {gp}")
            except Exception as exc:  # noqa: BLE001
                log(f"[report] không vẽ được grid_hires: {type(exc).__name__}: {exc}")
    parts.append(section_research(repo_root))
    doc = f"<!doctype html><html lang='vi'><meta charset='utf-8'><title>{_e(title)}</title>{''.join(parts)}</html>"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    log(f"[report] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(data)} prompt)")
    return out_path


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Dựng báo cáo tiến độ HTML tự chứa từ thư mục runs/<run>/")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="mặc định <run_dir>/progress_report.html")
    ap.add_argument("--title", default="CTIG - Cultural Text-to-Image (Việt Nam): báo cáo tiến độ")
    ap.add_argument("--author", default="")
    ap.add_argument("--img-side", type=int, default=1024, help="cạnh dài tối đa của ảnh nhúng (px)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--no-hires-grid", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or str(Path(a.run_dir) / "progress_report.html")
    build(a.run_dir, out, title=a.title, author=a.author, img_side=a.img_side, top_k=a.top_k, hires_grid=not a.no_hires_grid)


if __name__ == "__main__":
    main()
