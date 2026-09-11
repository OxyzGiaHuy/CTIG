"""
Hiển thị từng bước của pipeline trong notebook.

Mọi hàm nhận dataclass và trả về MỘT chuỗi HTML tự chứa (ảnh nhúng base64 thu nhỏ), không phụ
thuộc torch hay IPython. `show()` hiển thị trong Jupyter; `Report` gom mọi fragment đã show rồi
lưu thành một file HTML để tải khỏi Kaggle.

Badge nguồn: mỗi bước có thể kèm nhãn "memory | disk | computed" cho biết cell vừa chạy có tốn
API/model không (xem ctig/session.py).
"""

from __future__ import annotations

import html as H
from pathlib import Path
from typing import Iterable

from .bundle import thumb_b64
from .schema import (
    AnalysisResult, CulturalSpec, GenSpec, ImageHit, MultiGenResult, Prompt, QueryComparison, ReviewOutcome,
    SearchResult, TextHit,
)

CSS = """
<style>
.ctig{font-family:system-ui,sans-serif;font-size:13px;color:#1c1e22;line-height:1.4}
.ctig h3{font-size:15px;margin:10px 0 6px}.ctig h4{font-size:13px;margin:8px 0 4px;color:#444}
.ctig table{border-collapse:collapse;max-width:100%}.ctig td,.ctig th{padding:3px 8px;border-bottom:1px solid #e3e1db;vertical-align:top;text-align:left}
.ctig th{background:#f4f3ef;font-weight:600}.ctig .muted{color:#7a7e86}.ctig .warn{background:#fff3cd;border-left:4px solid #c4302b;padding:6px 10px;margin:6px 0}
.ctig .ok{color:#168052}.ctig .bad{color:#c4302b}.ctig .badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:9px;background:#e6e4de;color:#333;margin-left:6px}
.ctig .badge.memory{background:#dbeafe}.ctig .badge.disk{background:#dcfce7}.ctig .badge.computed{background:#fde68a}
.ctig .grid{display:flex;flex-wrap:wrap;gap:8px}.ctig .cell{width:180px;background:#f4f3ef;border-radius:6px;padding:5px;font-size:11px}
.ctig .cell img{width:100%;border-radius:4px;display:block}.ctig .cell.ref{outline:3px solid #168052}
.ctig .cols{display:flex;gap:16px;align-items:flex-start}.ctig .col{flex:1;min-width:0}
.ctig pre{white-space:pre-wrap;font-size:11px;background:#f4f3ef;padding:6px;border-radius:4px;margin:4px 0}.ctig .neg{color:#c4302b}
.ctig .mg td img{width:220px;display:block;border-radius:4px}.ctig .mg td.chosen img{outline:3px solid #168052}
.ctig .chip{display:inline-block;background:#eee;border-radius:4px;padding:1px 6px;margin:1px 2px;font-size:11px}
</style>
"""


def _e(x) -> str:
    return H.escape(str(x)) if x is not None else ""


def _wrap(title: str, body: str, source: str | None = None) -> str:
    badge = f"<span class='badge {source}'>{ {'memory': 'bộ nhớ', 'disk': 'đĩa', 'computed': 'chạy mới'}.get(source, source) }</span>" if source else ""
    return f"{CSS}<div class='ctig'><h3>{_e(title)}{badge}</h3>{body}</div>"


def _img(path: str | None, side: int = 220) -> str:
    if not path:
        return "<div class='muted'>(không có ảnh)</div>"
    b = thumb_b64(path, side)
    return f"<img src='{b}'>" if b else f"<div class='muted'>không đọc được {_e(Path(path).name)}</div>"


# ---------------------------------------------------------------- bước 0
def prompt_card(prompt: Prompt, source: str | None = None) -> str:
    body = (f"<table><tr><th>id</th><td>{_e(prompt.id)}</td></tr>"
            f"<tr><th>tiếng Việt</th><td><b>{_e(prompt.text_vi)}</b></td></tr>"
            f"<tr><th>tiếng Anh</th><td>{_e(prompt.text_en)}</td></tr>"
            f"<tr><th>độ khó / loại</th><td>{_e(prompt.difficulty)} / {_e(prompt.category)}</td></tr>"
            + (f"<tr><th>nhãn vàng (chỉ để đo)</th><td>{_e(', '.join(prompt.gold_entities))}</td></tr>" if prompt.gold_entities else "")
            + "</table>")
    return _wrap("Prompt", body, source)


# ---------------------------------------------------------------- bước 1
def keywords_table(analysis: AnalysisResult, kb, max_spec_entities: int = 4, source: str | None = None) -> str:
    rows = []
    for src in ("surface", "expanded"):
        ks = [k for k in analysis.keywords if k.source == src]
        rows.append(f"<h4>{'Keywords bề mặt (nêu thẳng trong prompt)' if src == 'surface' else 'Keywords mới (agent suy ra)'} · {len(ks)}</h4>")
        if not ks:
            rows.append("<div class='muted'>không có</div>")
            continue
        rows.append("<table><tr><th>term</th><th>kind</th><th>conf</th><th>vì sao</th></tr>")
        for k in ks:
            kind_ok = k.kind in ("entity", "scene", "attribute", "style", "region")
            rows.append(f"<tr><td>{_e(k.term)}</td><td class='{'' if kind_ok else 'bad'}'>{_e(k.kind)}</td>"
                        f"<td>{k.confidence:.2f}</td><td class='muted'>{_e((k.rationale or '')[:140])}</td></tr>")
        rows.append("</table>")
    ids = analysis.candidate_entity_ids
    names = []
    for eid in ids:
        ent = kb.get(eid)
        names.append(f"<span class='chip'>{_e(eid)}{' · ' + _e(ent.name_vi) if ent else ''}{' · ' + _e(ent.kind) if ent and ent.kind == 'context' else ''}</span>")
    body = "".join(rows)
    body += f"<h4>Thực thể ứng viên · {len(ids)}</h4><div>{''.join(names) or '<span class=muted>không có</span>'}</div>"
    if len(ids) > 2 * max_spec_entities:
        body += (f"<div class='warn'><b>Nổ danh mục:</b> {len(ids)} ứng viên cho một prompt (spec chỉ giữ {max_spec_entities}). "
                 f"Agent đang đổ cả KB vào; stage 1 đã cap nhưng hãy kiểm rationale phía trên.</div>")
    body += (f"<table><tr><th>vùng suy ra</th><td>{_e(analysis.region_hint or '—')}</td></tr>"
             f"<tr><th>prompt tiếng Anh cho bộ sinh</th><td>{_e(analysis.prompt_en or '—')}</td></tr>"
             + (f"<tr><th>thực thể mới đề xuất</th><td>{_e(', '.join(n.name_vi for n in analysis.new_entities))}</td></tr>" if analysis.new_entities else "")
             + (f"<tr><th>ghi chú</th><td class='muted'>{_e(analysis.notes)}</td></tr>" if analysis.notes else "") + "</table>")
    return _wrap("Bước 1 · Phân tích prompt", body, source)


# ---------------------------------------------------------------- bước 2
def text_results_table(hits: Iterable[TextHit], k: int = 5) -> str:
    hits = list(hits)[:k]
    if not hits:
        return "<div class='muted'>không có kết quả text</div>"
    out = ["<table><tr><th>nguồn</th><th>tiêu đề</th><th>trích</th><th>truy vấn</th></tr>"]
    for h in hits:
        link = f"<a href='{_e(h.url)}' target='_blank'>{_e(h.title or h.url)}</a>" if h.url else _e(h.title)
        out.append(f"<tr><td class='muted'>{_e(h.source)}</td><td>{link}</td><td>{_e(h.snippet[:220])}</td><td class='muted'>{_e(h.query[:50])}</td></tr>")
    out.append("</table>")
    return "".join(out)


def image_grid(hits: Iterable[ImageHit], k: int = 6, side: int = 170) -> str:
    hits = [h for h in hits if h.local_path][:k]
    if not hits:
        return "<div class='muted'>không có ảnh tải được</div>"
    cells = []
    for h in hits:
        badges = []
        if h.clip_prompt_sim is not None:
            badges.append(f"sim(prompt) {h.clip_prompt_sim:.2f}")
        if h.clip_entity_prob is not None:
            badges.append(f"P(thực thể) {h.clip_entity_prob:.2f}")
        cells.append(f"<div class='cell'>{_img(h.local_path, side)}<div>{' · '.join(badges) or '&nbsp;'}</div>"
                     f"<div class='muted'>{_e(h.source)} · {_e(h.title[:40])}</div></div>")
    return f"<div class='grid'>{''.join(cells)}</div>"


def query_comparison(cmp: QueryComparison, k_text: int = 5, k_images: int = 6, source: str | None = None) -> str:
    cols = []
    for c in cmp.columns:
        q = "".join(f"<span class='chip'>{_e(x[:60])}</span>" for x in c.queries)
        cols.append(f"<div class='col'><h4>{_e(c.label)}</h4><div>{q}</div>"
                    + (f"<div class='muted'>{_e(c.note)}</div>" if c.note else "")
                    + f"<h4>Text top-{k_text}</h4>{text_results_table(c.text, k_text)}"
                    + f"<h4>Ảnh top-{k_images} (xếp theo CLIP sim với prompt)</h4>{image_grid(c.images, k_images)}</div>")
    body = f"<div class='cols'>{''.join(cols)}</div>"
    if cmp.errors:
        body += f"<div class='warn'>Lỗi mạng: {_e('; '.join(cmp.errors[:5]))}</div>"
    return _wrap("Bước 2 · Search: keywords vs prompt gốc", body, source)


# ---------------------------------------------------------------- bước 2b
def evidence_table(search: SearchResult, kb, source: str | None = None) -> str:
    by = {}
    for it in search.items:
        by.setdefault(it.entity_id, []).append(it)
    parts = []
    for eid, items in by.items():
        if eid == "-":
            continue
        ent = kb.get(eid)
        name = ent.name_vi if ent else eid
        kbi = [i for i in items if i.kind == "kb"]
        exi = [i for i in items if i.provenance == "extracted"]
        texts = [i for i in items if i.kind in ("wiki_text", "web_text") and i.provenance != "extracted"]
        imgs = [i for i in items if i.kind == "image"]
        parts.append(f"<h4>{_e(name)} <span class='muted'>({_e(eid)})</span> · {len(texts)} văn bản · {len(imgs)} ảnh</h4>")
        parts.append("<div class='cols'>")
        parts.append("<div class='col'><b>KB viết tay</b>" + (
            "<ul>" + "".join(f"<li>{_e(a)}</li>" for a in kbi[0].must_have) + "</ul>"
            f"<div class='muted'>không được có: {_e('; '.join(kbi[0].must_not[:3]))}</div>" if kbi else "<div class='muted'>không có trong KB</div>") + "</div>")
        if exi:
            ex = exi[0]
            parts.append("<div class='col'><b>Rút từ văn bản (có câu gốc)</b><ul>" + "".join(
                f"<li>{_e(a)}<div class='muted'>↳ {_e((ex.attr_sources.get(a) or '')[:150])}</div></li>" for a in ex.must_have) + "</ul>"
                + (f"<div class='muted'>không được có: {_e('; '.join(ex.must_not[:3]))}</div>" if ex.must_not else "")
                + (f"<div class='muted'>dễ nhầm: {_e(', '.join(c.get('name', '') for c in ex.confusable_with))}</div>" if ex.confusable_with else "")
                + "</div>")
        else:
            parts.append("<div class='col'><b>Rút từ văn bản</b><div class='muted'>không rút được (xem ghi chú)</div></div>")
        parts.append("</div>")
        if texts:
            trs = []
            for t in texts[:6]:
                title = f"<a href='{_e(t.url)}' target='_blank'>{_e(t.title[:70])}</a>" if t.url else _e(t.title[:70])
                trs.append(f"<tr><td class='muted'>{_e(t.provenance)}</td><td>{title}</td><td class='muted'>{len(t.snippet)} ký tự</td></tr>")
            parts.append("<table><tr><th>nguồn</th><th>tiêu đề</th><th>độ dài</th></tr>" + "".join(trs) + "</table>")
        if imgs:
            cells = []
            for i in sorted(imgs, key=lambda x: -(x.clip_match or 0))[:6]:
                cls = "cell ref" if i.is_reference else "cell"
                cells.append(f"<div class='{cls}'>{_img(i.local_path, 160)}<div>CLIP {i.clip_match:.2f}{' · <b>tham chiếu</b>' if i.is_reference else ''}</div><div class='muted'>{_e(i.provenance)}</div></div>"
                             if i.clip_match is not None else f"<div class='{cls}'>{_img(i.local_path, 160)}<div class='muted'>{_e(i.snippet[:40])}</div></div>")
            parts.append(f"<div class='grid'>{''.join(cells)}</div>")
    notes = [n for n in search.notes]
    body = "".join(parts) or "<div class='muted'>không có bằng chứng</div>"
    if notes:
        body += "<h4>Ghi chú rút bằng chứng</h4><ul class='muted'>" + "".join(f"<li>{_e(n)}</li>" for n in notes[:12]) + "</ul>"
    if search.retrieval_errors:
        body += f"<div class='warn'>Lỗi: {_e('; '.join(search.retrieval_errors[:5]))}</div>"
    return _wrap("Bước 2b · Bằng chứng đưa vào spec", body, source)


# ---------------------------------------------------------------- bước 3
def spec_card(spec: CulturalSpec, source: str | None = None) -> str:
    parts = []
    for se in spec.entities:
        parts.append(f"<h4>{_e(se.name_vi)} <span class='muted'>· {_e(se.name_en)} · w={se.weight} · {_e(se.kind)}</span></h4>")
        parts.append("<table><tr><th>phải có (VI)</th><th>phải có (EN → prompt)</th></tr>")
        for a, b in zip(se.required_attrs, se.required_attrs_en + [""] * len(se.required_attrs)):
            parts.append(f"<tr><td>{_e(a)}</td><td class='{'muted' if not b else ''}'>{_e(b) or '(chưa dịch, bỏ khỏi prompt)'}</td></tr>")
        parts.append("</table>")
        if se.forbidden_attrs:
            parts.append(f"<div class='muted'>không được có: {_e('; '.join(se.forbidden_attrs[:4]))}</div>")
        if se.confusables:
            cf_txt = ", ".join(f"{c.get('name')} ({c.get('culture', '')})" for c in se.confusables)
            parts.append(f"<div class='muted'>dễ nhầm: {_e(cf_txt)}</div>")
        if se.reference_image:
            parts.append(f"<div class='cell ref'>{_img(se.reference_image, 160)}<div>ảnh tham chiếu</div></div>")
    if spec.dropped:
        n_cap = sum(1 for d in spec.dropped if "vượt giới hạn" in d[1])
        others = [d for d in spec.dropped if "vượt giới hạn" not in d[1]]
        parts.append("<h4>Bị loại</h4><ul class='muted'>" + (f"<li>{n_cap} thực thể vượt giới hạn số thực thể</li>" if n_cap else "")
                     + "".join(f"<li>{_e(d[0])}: {_e(d[1][:120])}</li>" for d in others[:8]) + "</ul>")
    return _wrap("Bước 3 · Hợp đồng văn hoá (CulturalSpec)", "".join(parts) or "<div class='muted'>spec rỗng</div>", source)


def genspec_card(gen: GenSpec, source: str | None = None) -> str:
    body = ("<b>prompt</b><pre>" + "".join(f"<span class='chip'>{_e(t)}</span> " for t in gen.prompt_terms) + "</pre>"
            f"<b>negative</b><pre class='neg'>{_e(gen.negative_prompt)}</pre>"
            f"<table><tr><th>seed</th><td>{gen.seed}</td><th>bước</th><td>{gen.steps}</td><th>guidance</th><td>{gen.guidance}</td>"
            f"<th>kích cỡ</th><td>{gen.width}×{gen.height}</td><th>ứng viên</th><td>{gen.n_candidates}</td></tr></table>")
    if gen.ip_adapter_image:
        body += f"<div class='cell ref'>{_img(gen.ip_adapter_image, 160)}<div>IP-Adapter · scale {gen.ip_adapter_scale}</div></div>"
    return _wrap("Bước 3 · GenSpec đưa vào bộ sinh", body, source)


# ---------------------------------------------------------------- bước 4/5
def model_grid(res: MultiGenResult, spec: CulturalSpec, side: int = 220, source: str | None = None) -> str:
    rows = ["<table class='mg'><tr><th>model</th><th colspan='8'>ứng viên (viền xanh = CLIP chọn)</th></tr>"]
    for r in res.runs:
        meta = (f"<b>{_e(r.model_key)}</b><div class='muted'>{_e(r.repo.split('/')[-1])}</div>"
                f"<div class='muted'>{r.gen_spec.steps} bước · g{r.gen_spec.guidance:g} · {r.gen_spec.width}px</div>"
                f"<div class='muted'>{r.seconds:.0f}s" + (f" · đỉnh {r.peak_vram_gb} GB" if r.peak_vram_gb else "") + "</div>"
                + (f"<span class='badge disk'>ảnh từ lần trước</span>" if r.source == "disk" else ""))
        if r.error or not r.output:
            rows.append(f"<tr><td>{meta}</td><td colspan='8' class='bad'>{_e(r.error or 'không có ảnh')}</td></tr>")
            continue
        cells = []
        for j, c in enumerate(r.output.candidates):
            b = []
            if c.clip_probs:
                b.append(f"CLIP id {c.clip_fidelity:.2f}")
            if c.itm_score is not None:
                b.append(f"ITM {c.itm_score:.2f}")
            if c.clip_prompt_sim is not None:
                b.append(f"sim {c.clip_prompt_sim:.2f}")
            cls = "chosen" if (j == r.output.chosen and len(r.output.candidates) > 1) else ""
            cells.append(f"<td class='{cls}'>{_img(c.path, side)}<div>{' · '.join(b) or f'seed {c.seed}'}</div></td>")
        rows.append(f"<tr><td>{meta}</td>{''.join(cells)}</tr>")
    rows.append("</table>")
    return _wrap(f"Bước 4 · {len(res.runs)} model cùng một GenSpec", "".join(rows), source)


def score_table(res: MultiGenResult, source: str | None = None) -> str:
    from .stages.multigen import best_run

    best = best_run(res)
    rows = ["<table><tr><th>model</th><th>ứng viên</th><th>CLIP identity</th><th>ITM</th><th>CLIP sim(prompt)</th><th>giây</th><th>VRAM đỉnh</th></tr>"]
    for r in res.runs:
        if not r.output:
            rows.append(f"<tr><td>{_e(r.model_key)}</td><td colspan='6' class='bad'>{_e(r.error or '')}</td></tr>")
            continue
        for j, c in enumerate(r.output.candidates):
            hl = " style='background:#dcfce7'" if (best and r.model_key == best.model_key and j == r.output.chosen) else ""
            rows.append(f"<tr{hl}><td>{_e(r.model_key)}</td><td>{j}</td>"
                        f"<td>{c.clip_fidelity:.3f}</td><td>{'' if c.itm_score is None else f'{c.itm_score:.3f}'}</td>"
                        f"<td>{'' if c.clip_prompt_sim is None else f'{c.clip_prompt_sim:.3f}'}</td>"
                        f"<td>{r.seconds:.0f}</td><td>{'' if r.peak_vram_gb is None else f'{r.peak_vram_gb} GB'}</td></tr>")
    rows.append("</table>")
    note = (f"<div><b>Tốt nhất theo CLIP identity:</b> {_e(best.model_key)}</div>" if best else "")
    note += ("<div class='muted'>CLIP identity = P(ảnh giống mô tả thực thể Việt) so với các confusable, có trọng số. "
             "ITM = BLIP-2 P(ảnh khớp mô tả). sim = cosine CLIP với prompt tiếng Anh. Ba số này KHÔNG thay được mắt người; "
             "dùng để xếp thứ tự rồi tự nhìn grid.</div>")
    return _wrap("Bước 5 · Bảng điểm", "".join(rows) + note, source)


def review_summary(outcome: ReviewOutcome, spec: CulturalSpec, source: str | None = None) -> str:
    cells = []
    for it in outcome.iterations:
        adj = it.adjudication
        ck = "; ".join(f"{_e(spec.entity(e).name_vi if spec.entity(e) else e)}: {_e(c.get('identity'))}" for e, c in it.perception.checklist.items())
        finds = "".join(f"<li class='{f.severity}'>{_e(f.message[:120])}</li>" for f in adj.merged_findings[:5])
        cells.append(f"<div class='cell' style='width:260px'>{_img(it.gen_output.image_path, 250)}"
                     f"<div><b>vòng {it.n}</b>{' · render đủ bước' if it.final_render else ''} · {adj.score:.2f} · <b>{_e(adj.verdict)}</b></div>"
                     + (f"<div class='muted'>{ck}</div>" if ck else "") + f"<ul>{finds}</ul>"
                     + (f"<div class='muted'>bất đồng: {_e(adj.disagreements[0][:120])}</div>" if adj.disagreements else "") + "</div>")
    body = f"<div class='grid'>{''.join(cells)}</div><div>đạt: <b>{outcome.passed}</b> · điểm cuối {outcome.final_score:.2f}</div>"
    return _wrap("Bước 5b · Vòng review agent (tuỳ chọn)", body, source)


def vram_html() -> str:
    from .models.loader import vram_report

    r = vram_report()
    return f"{CSS}<div class='ctig'><div class='muted'>VRAM: {_e(r) if r else 'không có CUDA'}</div></div>"


# ---------------------------------------------------------------- hiển thị / gom báo cáo
def show(html: str) -> None:
    try:
        from IPython.display import HTML, display

        display(HTML(html))
    except Exception:  # noqa: BLE001 - không có IPython (chạy script)
        import re

        print(re.sub(r"<[^>]+>", " ", html)[:2000])


class Report:
    """Gom mọi fragment đã show rồi save() thành một file HTML tự chứa."""

    def __init__(self, title: str):
        self.title = title
        self.parts: list[str] = []

    def show(self, html: str) -> None:
        self.parts.append(html)
        show(html)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = (f"<!doctype html><meta charset='utf-8'><title>{_e(self.title)}</title>{CSS}"
               f"<body style='background:#faf9f6;margin:18px'><h1 style='font-family:system-ui;font-size:18px'>{_e(self.title)}</h1>"
               + "<hr>".join(self.parts) + "</body>")
        path.write_text(doc, encoding="utf-8")
        return path
