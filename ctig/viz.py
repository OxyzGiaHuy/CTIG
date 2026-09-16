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
        exi = [i for i in items if i.provenance in ("extracted", "kb_auto")]
        texts = [i for i in items if i.kind in ("wiki_text", "web_text") and i.provenance not in ("extracted", "kb_auto")]
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
def model_grid(res: MultiGenResult, spec: CulturalSpec, side: int = 220, source: str | None = None, cr=None) -> str:
    """v1.9: viền xanh = ảnh Reviewer chấm CAO NHẤT trong hàng (nếu đã có Reviewer), không phải ảnh có CLIP attr cao nhất.
    Trước đây viền theo metric nên grid có thể tô đậm một ảnh mà Reviewer đã loại vì must_not (S001 sdxl_base c5: attr 0,76
    nhưng Reviewer -0,17 vì 'one-piece dress'), khiến người đọc tưởng hệ thống chọn ảnh sai."""
    ver = {v.path: v for v in cr.filter.verdicts} if cr is not None else {}
    finals = {x.final_path for x in (getattr(cr, "per_model", None) or ([cr] if cr is not None else [])) if x.final_path}
    head = "ứng viên (viền xanh = ĐIỂM REVIEWER cao nhất trong hàng; ★ = ảnh cuối hệ thống)" if ver else \
           "ứng viên (viền xanh = điểm tổng cao nhất trong hàng)"
    rows = [f"<table class='mg'><tr><th>model</th><th colspan='8'>{head}</th></tr>"]
    for r in res.runs:
        meta = (f"<b>{_e(r.model_key)}</b><div class='muted'>{_e(r.repo.split('/')[-1])}</div>"
                f"<div class='muted'>{r.gen_spec.steps} bước · g{r.gen_spec.guidance:g} · {r.gen_spec.width}px</div>"
                f"<div class='muted'>{r.seconds:.0f}s" + (f" · đỉnh {r.peak_vram_gb} GB" if r.peak_vram_gb else "") + "</div>"
                + (f"<span class='badge disk'>ảnh từ lần trước</span>" if r.source == "disk" else "")
                + (f"<div class='muted'>{r.prompt_tokens} token</div>" if r.prompt_tokens else "")
                + "".join(f"<div class='muted'>· {_e(n[:90])}</div>" for n in (r.notes or [])[:5]))
        if r.error or not r.output:
            rows.append(f"<tr><td>{meta}</td><td colspan='8' class='bad'>{_e(r.error or 'không có ảnh')}</td></tr>")
            continue
        cells = []
        for j, c in enumerate(r.output.candidates):
            from .stages import multigen as _mg
            b = []
            if _mg.SATURATED and c.clip_probs:
                b.append(f"CLIP id {c.clip_fidelity:.2f}")
            if c.attr_contrast is not None:
                b.append(f"<b>attr {c.attr_contrast:.2f}</b>")
            if _mg.SATURATED and c.itm_score is not None:
                b.append(f"ITM {c.itm_score:.2f}")
            if c.itm_attrs is not None:
                b.append(f"ITMattr {c.itm_attrs:.2f}")
            if c.aesthetic is not None:
                b.append(f"đẹp {c.aesthetic:.2f}")
            if c.clip_prompt_sim is not None:
                b.append(f"sim {c.clip_prompt_sim:.2f}")
            if c.ref_sim is not None:
                b.append(f"<span class='{'bad' if c.ref_sim > 0.88 else ''}'>giống ref {c.ref_sim:.2f}</span>")
            v = ver.get(c.path)
            if v is not None:
                b.insert(0, f"<b class='{'ok' if (v.keep and not v.matched_must_not) else 'bad'}'>Reviewer {v.score:+.2f}</b>")
                if v.matched_must_not:
                    b.append(f"<span class='bad'>must_not: {_e('; '.join(a[:22] for a in v.matched_must_not[:2]))}</span>")
                elif v.missing_must_have:
                    b.append(f"<span class='muted'>thiếu: {_e('; '.join(a[:22] for a in v.missing_must_have[:2]))}</span>")
            if ver:
                scored = [(vv.score, cc.path) for cc in r.output.candidates for vv in [ver.get(cc.path)] if vv]
                top = max(scored)[1] if scored else None
                cls = "chosen" if (c.path == top and len(r.output.candidates) > 1) else ""
            else:
                cls = "chosen" if (j == r.output.chosen and len(r.output.candidates) > 1) else ""
            hr = " <span class='badge disk'>hires</span>" if c.base_path else ""
            star = " <b title='ảnh cuối của hệ thống'>★</b>" if c.path in finals else ""
            cells.append(f"<td class='{cls}'>{_img(c.path, side)}<div>{' · '.join(b) or f'seed {c.seed}'}{hr}{star}</div></td>")
        rows.append(f"<tr><td>{meta}</td>{''.join(cells)}</tr>")
    rows.append("</table>")
    return _wrap(f"Bước 4 · {len(res.runs)} model cùng một GenSpec", "".join(rows), source)


def score_table(res: MultiGenResult, source: str | None = None) -> str:
    from .stages import multigen as mg
    from .stages.multigen import best_run, combined_score, score_key

    sat = mg.SATURATED
    best = best_run(res)
    f3 = lambda v: "" if v is None else f"{v:.3f}"
    sat_h = "<th>CLIP identity</th>" if sat else ""
    itm_h = "<th>ITM</th>" if sat else ""
    rows = [f"<table><tr><th>model</th><th>ứng viên</th><th>hạng ensemble</th><th>tổng</th>{sat_h}<th>CLIP attr</th>{itm_h}<th>ITM attr</th><th>đẹp (PickScore)</th><th>sim(prompt)</th><th>giống ref</th><th>giây</th><th>VRAM đỉnh</th></tr>"]
    for r in res.runs:
        if not r.output:
            rows.append(f"<tr><td>{_e(r.model_key)}</td><td colspan='12' class='bad'>{_e(r.error or '')}</td></tr>")
            continue
        for j, c in enumerate(r.output.candidates):
            hl = " style='background:#dcfce7'" if (best and r.model_key == best.model_key and j == r.output.chosen) else ""
            sat_c = f"<td>{c.clip_fidelity:.3f}</td>" if sat else ""
            itm_c = f"<td>{f3(c.itm_score)}</td>" if sat else ""
            rows.append(f"<tr{hl}><td>{_e(r.model_key)}</td><td>{j}</td><td><b>{f3(c.ensemble)}</b></td><td>{combined_score(c):.3f}</td>"
                        f"{sat_c}<td>{f3(c.attr_contrast)}</td>{itm_c}<td>{f3(c.itm_attrs)}</td>"
                        f"<td>{f3(c.aesthetic)}{'' if c.pick_score is None else f' <span class=muted>({c.pick_score:.1f})</span>'}</td>"
                        f"<td>{f3(c.clip_prompt_sim)}</td>"
                        f"<td{' class=bad' if (c.ref_sim or 0) > 0.88 else ''}>{f3(c.ref_sim)}</td>"
                        f"<td>{r.seconds:.0f}</td><td>{'' if r.peak_vram_gb is None else f'{r.peak_vram_gb} GB'}</td></tr>")
    rows.append("</table>")
    note = (f"<div><b>Tốt nhất theo điểm tổng:</b> {_e(best.model_key)}</div>" if best else "")
    for n in (getattr(res, "notes", None) or []):
        note += f"<div class='bad'>⚠ {_e(n)}</div>"
    note += ("<div class='muted'>" + ("CLIP identity = P(giống mô tả thực thể Việt) so với confusable; bão hoà ~1.0 trên prompt dễ. " if sat else
             "CLIP identity và ITM danh tính ẩn vì bão hoà 0,95-1,00 trên mọi ảnh (bật lại bằng multigen.saturated_metrics). ") +
             "<b>CLIP attr</b> = phần xác suất rơi vào câu 'thực thể with &lt;must_have&gt;' so với 'with &lt;must_not&gt;' "
             "(vd có quần vs váy liền). ITM attr = BLIP-2 trung bình trên câu must_have. "
             "<b>đẹp</b> = PickScore (sở thích người) chuẩn hoá min-max trong lần chạy này, số thô trong ngoặc; đo 'thích', không đo đúng văn hoá. "
             "<b>hạng ensemble</b> = 1 − trung bình hạng trên các verifier có (attr, ITM attr, đẹp; CLIP id chỉ khi bật saturated_metrics), tính trên mọi ứng viên của lần chạy (Ma et al. 2025: verifier đơn bị 'hack'); dùng để chọn. "
             "sim = cosine CLIP với prompt. <b>giống ref</b> = cosine CLIP ảnh-ảnh lớn nhất với ảnh tham chiếu (đã cắt); > 0,88 coi là chép và bị trừ điểm tổng. "
             "Tổng = trung bình các số có. Không thay được mắt người; dùng để xếp thứ tự rồi nhìn grid.</div>")
    return _wrap("Bước 5 · Bảng điểm", "".join(rows) + note, source)


# ---------------------------------------------------------------- v1.7 grounding + bare/system
def grounding_table(g: dict, kb=None, source: str | None = None) -> str:
    """Một bảng cho cả khối Grounding: thực thể | thuộc tính dương (vào prompt) | âm / dễ nhầm (vào negative) | ảnh tham chiếu.
    Nguồn ghi ở cột cuối: KB viết tay, web (số trang), brief (Summary agent)."""
    sp: CulturalSpec = g["spec"]
    briefs = g.get("briefs") or {}
    search = g.get("search")
    n_pages = {}
    if search is not None:
        for it in getattr(search, "items", []) or []:
            n_pages[getattr(it, "entity_id", "")] = n_pages.get(getattr(it, "entity_id", ""), 0) + 1
    refs = g.get("refs") or []
    rows = ["<table><tr><th>thực thể</th><th>loại</th><th>dương → prompt</th><th>âm / dễ nhầm → negative</th><th>brief (Summary agent)</th><th>nguồn</th></tr>"]
    for se in sp.entities:
        b = briefs.get(se.entity_id)
        pos = "".join(f"<div>+ {_e(x)}</div>" for x in se.required_attrs_en[:5])
        neg = "".join(f"<div>− {_e(x)}</div>" for x in se.forbidden_attrs_en[:4])
        conf = ", ".join(str(c.get("name", "")) for c in (se.confusables or [])[:4])
        if conf:
            neg += f"<div class='muted small'>dễ nhầm: {_e(conf)}</div>"
        bf = ""
        if b is not None:
            bf = "".join(f"<div>· {_e(x)}</div>" for x in b.facts_en[:3])
            if b.depiction_en:
                bf += f"<div class='muted small'>{_e(b.depiction_en)}</div>"
        srcs = ["KB"]
        if n_pages.get(se.entity_id):
            srcs.append(f"web {n_pages[se.entity_id]} trang")
        if b is not None and b.n_sources:
            srcs.append(f"brief {b.n_sources} nguồn")
        rows.append(f"<tr><td><b>{_e(se.name_vi)}</b><div class='muted small'>{_e(se.name_en)} · w={se.weight:.2f}</div></td><td>{_e(se.kind)}</td>"
                    f"<td>{pos}</td><td>{neg}</td><td class='small'>{bf}</td><td class='small'>{_e(', '.join(srcs))}</td></tr>")
    rows.append("</table>")
    body = "".join(rows)
    a = g.get("analysis")
    if a is not None and getattr(a, "prompt_en", ""):
        body = f"<div><b>prompt_en:</b> {_e(a.prompt_en)}</div>" + body
    if refs:
        body += f"<h4>Ảnh tham chiếu ({len(refs)}, đã qua Filter và cắt theo thực thể)</h4><div class='grid'>" + "".join(_img(r, 110) for r in refs[:6]) + "</div>"
    else:
        body += "<div class='muted'>không có ảnh tham chiếu (tắt hoặc không ảnh nào đạt ngưỡng)</div>"
    body += ("<div class='muted'>Grounding = Analysis (VLM tách thực thể) + Search (Wikipedia/DDG/Commons, ảnh) + Summary agent + Spec. "
             "Dương/âm lấy từ KB viết tay khi thực thể có trong KB; web bổ sung thuộc tính/dễ nhầm và là nguồn ảnh tham chiếu. "
             "Chi tiết từng bước con xem các ô chẩn đoán phía dưới.</div>")
    return _wrap("Grounding · prompt → thực thể, thuộc tính, ảnh tham chiếu", body, source)


def _bare_pairs(res: MultiGenResult) -> list[tuple]:
    """Ghép hàng '<m>#bare' với hàng hệ thống cùng model nền '<m>' (hoặc '<m>+ref', '<m>@scale')."""
    from .models.registry import parse_key, parse_variant

    by_base: dict[str, list] = {}
    for r in res.runs:
        base, _ = parse_key(r.model_key)
        by_base.setdefault(base, []).append(r)
    pairs = []
    for base, runs in by_base.items():
        bare = [r for r in runs if parse_variant(r.model_key) == "bare"]
        sys_ = [r for r in runs if parse_variant(r.model_key) != "bare"]
        if bare and sys_:
            pairs.append((base, bare[0], sys_))
    return pairs


def paired_table(res: MultiGenResult, cr=None, source: str | None = None) -> str:
    """Bảng 'model nền M' so 'M + hệ thống' cùng seed (v1.7). Cột: CLIP attr, ITM attr, hạng ensemble trung bình; cột Filter
    khi có candidate_review (số ảnh qua Filter / có must_not)."""
    from .stages.multigen import combined_score

    pairs = _bare_pairs(res)
    if not pairs:
        return _wrap("Bare vs system", "<div class='muted'>không có hàng '#bare' để ghép (thêm 'realvis_xl#bare' vào models:)</div>", source)
    ver = {}
    vqa = dict(getattr(cr, "vqa", {}) or {}) if cr is not None else {}
    if cr is not None:
        for v in cr.filter.verdicts:
            ver[v.path] = v
    f3 = lambda v: "" if v is None else f"{v:.3f}"

    def agg(runs):
        cs = [c for r in runs if r.output for c in r.output.candidates]
        if not cs:
            return None
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None
        vs = [ver[c.path] for c in cs if c.path in ver]
        return {"n": len(cs), "attr": mean([c.attr_contrast for c in cs if c.attr_contrast is not None]),
                "itm": mean([c.itm_attrs for c in cs if c.itm_attrs is not None]),
                "ens": mean([c.ensemble for c in cs if c.ensemble is not None]),
                "tot": mean([combined_score(c) for c in cs]),
                "keep": (sum(1 for v in vs if v.keep and not v.matched_must_not and v.score >= 0.5), len(vs)) if vs else None,
                "rev": mean([v.score for v in vs]) if vs else None,
                "vqa": mean([vqa[c.path] for c in cs if c.path in vqa]) if any(c.path in vqa for c in cs) else None,
                "best": max(cs, key=combined_score).path}

    finals = {x.base_model: x for x in (getattr(cr, "per_model", None) or [])} if cr is not None else {}
    rows = ["<table><tr><th>model nền</th><th>nhánh</th><th>ảnh</th><th>CLIP attr</th><th>ITM attr</th><th>hạng ensemble</th><th>tổng</th><th>Reviewer TB</th><th>VQAScore TB</th><th>Filter đạt</th><th>ảnh tốt nhất (metric)</th><th>ảnh cuối loop</th></tr>"]
    for base, bare, sys_ in pairs:
        A, B = agg([bare]), agg(sys_)
        for lab, d, keys in (("bare", A, bare.model_key), ("system", B, ", ".join(r.model_key for r in sys_))):
            if d is None:
                rows.append(f"<tr><td>{_e(base)}</td><td>{lab}</td><td colspan='10' class='bad'>không có ảnh ({_e(keys)})</td></tr>")
                continue
            kp = "" if d["keep"] is None else f"{d['keep'][0]}/{d['keep'][1]}"
            rows.append(f"<tr><td>{_e(base)}</td><td><b>{lab}</b><div class='muted small'>{_e(keys)}</div></td><td>{d['n']}</td><td>{f3(d['attr'])}</td>"
                        f"<td>{f3(d['itm'])}</td><td>{f3(d['ens'])}</td><td>{f3(d['tot'])}</td><td>{'' if d['rev'] is None else f'{d['rev']:+.2f}'}</td><td>{f3(d['vqa'])}</td><td>{kp}</td><td>{_img(d['best'], 120)}</td>"
                        f"<td>{_img(finals[base].final_path, 120) + f'<div class=small>{_e(finals[base].final_source)} · {len(finals[base].iterations)} vòng</div>' if (lab == 'system' and base in finals and finals[base].final_path) else ''}</td></tr>")
        if A and B:
            dl = lambda k: "" if (A[k] is None or B[k] is None) else f"{B[k] - A[k]:+.3f}"
            cls = "ok" if (A["tot"] is not None and B["tot"] is not None and B["tot"] > A["tot"]) else "bad"
            rows.append(f"<tr class='{cls}'><td></td><td>Δ system − bare</td><td></td><td>{dl('attr')}</td><td>{dl('itm')}</td><td>{dl('ens')}</td><td><b>{dl('tot')}</b></td><td><b>{dl('rev')}</b></td><td>{dl('vqa')}</td><td></td><td></td><td></td></tr>")
    rows.append("</table>")
    note = ("<div class='muted'>bare = model nền với prompt dịch thẳng + negative chung, không KB, không LoRA, không ảnh tham chiếu, cùng seed. "
            "system = cùng model nền qua Grounding (+ LoRA/ảnh nếu hàng có). Δ &gt; 0 ủng hộ H_sys: hệ thống cải thiện mọi model nền, "
            "không phải chọn model tốt nhất. Metric bão hoà thì nhìn cột Reviewer (điểm Filter trung bình, âm = có must_not) và Filter đạt "
            "(giữ, không must_not, điểm ≥ 0,5) rồi nhìn grid. <b>VQAScore</b> = P(Yes | 'Does this figure show \"prompt\"?') bằng Qwen2.5-VL "
            "(Lin 2024), cột tham chiếu chuẩn của cộng đồng; không dùng để chọn.</div>")
    return _wrap("Bare vs system · cùng model nền, cùng seed", "".join(rows) + note, source)


# ---------------------------------------------------------------- v1.4 agents
def brief_card(briefs: dict, spec: CulturalSpec, source: str | None = None) -> str:
    if not briefs:
        return _wrap("Bước 2c · Summary agent", "<div class='muted'>tắt (agents.summary=false) hoặc không có tư liệu</div>", source)
    cells = []
    for se in spec.entities:
        b = briefs.get(se.entity_id)
        if b is None:
            continue
        facts = "".join(f"<li>{_e(en)}<div class='muted'>{_e(vi)}</div></li>" for en, vi in zip(b.facts_en, b.facts_vi + [""] * len(b.facts_en)))
        conf = "".join(f"<li>{_e(x)}</li>" for x in b.confusions_en)
        cells.append(f"<div class='cell' style='width:420px'><b>{_e(se.name_vi)}</b> <span class='muted'>{_e(se.name_en)} · "
                     f"{b.n_sources} nguồn · {b.grounded}/{len(b.facts_vi)} câu VI có gốc</span>"
                     f"<ul>{facts}</ul>" + (f"<div><b>khác với thứ dễ nhầm:</b><ul>{conf}</ul></div>" if conf else "")
                     + (f"<div><b>vẽ thế nào:</b> <i>{_e(b.depiction_en)}</i></div>" if b.depiction_en else "")
                     + ("<div class='muted small'>chiều CULTIVate: " + " · ".join(f"<b>{_e(k)}</b> {len(v)}" for k, v in b.dimensions.items() if v) + "</div>" if b.dimensions else "")
                     + f"<div class='muted small'>nguồn: {_e('; '.join(b.sources[:4]))}</div></div>")
    note = ("<div class='muted'>Summary agent chỉ được dùng thông tin trong văn bản truy hồi; facts VI được kiểm mờ xem có câu gốc. "
            "'Vẽ thế nào' chỉ vào prompt khi agents.enrich_prompt bật (đang " + "tắt" + " để so A/B).</div>")
    return _wrap("Bước 2c · Summary agent: brief văn hoá từ tư liệu", f"<div class='grid'>{''.join(cells)}</div>{note}", source)


def filter_table(flt, title: str = "Filter agent", side: int = 160, source: str | None = None) -> str:
    desc_by = {d.path: d for d in flt.descriptors}
    cells = []
    for v in flt.verdicts:
        d = desc_by.get(v.path)
        cls = "" if v.keep else " style='opacity:.55'"
        badge = "<span class='badge computed'>giữ</span>" if v.keep else "<span class='badge' style='background:#fee2e2;color:#991b1b'>bỏ</span>"
        have = "".join(f"<li class='ok'>✓ {_e(a[:60])}</li>" for a in v.matched_must_have[:4])
        notv = "".join(f"<li class='bad'>✗ {_e(a[:60])}</li>" for a in v.matched_must_not[:3])
        miss = "".join(f"<li class='muted'>? {_e(a[:60])}</li>" for a in v.missing_must_have[:3])
        dtxt = _e((d.text() if d else "")[:260])
        cells.append(f"<div class='cell' style='width:{side + 20}px'{cls}>{_img(v.path, side)}<div>{badge} điểm {v.score:+.2f}"
                     + (f" · {v.people_count} người" if v.people_count is not None else "") + "</div>"
                     f"<ul>{have}{notv}{miss}</ul><div class='muted small'>{_e('; '.join(v.reasons)[:160])}</div>"
                     f"<details><summary class='muted small'>mô tả VLM</summary><div class='small'>{dtxt}</div></details></div>")
    head = (f"giữ <b>{len(flt.kept)}</b>/{len(flt.verdicts)}"
            + (f" · prompt nói rõ {flt.expected_people} người" if flt.expected_people else " · prompt không ràng buộc số người"))
    return _wrap(title, f"<div>{head}</div><div class='grid'>{''.join(cells)}</div>", source)


def final_grid(res: MultiGenResult, cr=None, side: int = 300, source: str | None = None) -> str:
    """v1.8.2: lưới KẾT LUẬN cho một prompt: mỗi model nền một hàng, cột trái = ảnh model THUẦN (bare) tốt nhất, cột phải = ảnh CUỐI
    của hệ thống (sau loop), kèm Δ các số đo. Đây là hình người đọc cần nhìn đầu tiên, không phải bảng số."""
    from .stages.multigen import combined_score

    pairs = _bare_pairs(res)
    if not pairs:
        return ""
    ver = {v.path: v for v in cr.filter.verdicts} if cr is not None else {}
    vqa = dict(getattr(cr, "vqa", {}) or {}) if cr is not None else {}
    finals = {x.base_model: x for x in (getattr(cr, "per_model", None) or ([cr] if cr is not None else []))}
    f2 = lambda v: "–" if v is None else f"{v:.2f}"

    def stat(paths):
        cs = [c for r in res.runs if r.output for c in r.output.candidates if c.path in paths]
        m = lambda xs: (sum(xs) / len(xs)) if xs else None
        vs = [ver[p] for p in paths if p in ver]
        return {"attr": m([c.attr_contrast for c in cs if c.attr_contrast is not None]),
                "rev": m([v.score for v in vs]) if vs else None,
                "vqa": m([vqa[p] for p in paths if p in vqa]) if any(p in vqa for p in paths) else None}

    rows = []
    for base, bare, sys_ in pairs:
        bare_cs = [c for c in (bare.output.candidates if bare.output else [])]
        if not bare_cs:
            continue
        b_best = max(bare_cs, key=lambda c: ((ver[c.path].score if c.path in ver else -9), combined_score(c)))
        fx = finals.get(base)
        f_path = fx.final_path if fx else None
        sys_paths = [c.path for r in sys_ if r.output for c in r.output.candidates]
        sb, ss = stat([c.path for c in bare_cs]), stat(sys_paths)
        d = lambda k: ("" if (sb[k] is None or ss[k] is None) else
                       f"<span class='{'ok' if ss[k] > sb[k] else 'bad'}'>{ss[k] - sb[k]:+.2f}</span>")
        vb = ver.get(b_best.path); vf = ver.get(f_path) if f_path else None
        cap = lambda v: ("" if v is None else
                         (f"<div class='small {'ok' if (v.keep and not v.matched_must_not) else 'bad'}'>Reviewer {v.score:+.2f}"
                          + (f" · thiếu: {_e('; '.join(a[:26] for a in v.missing_must_have[:2]))}" if v.missing_must_have else "")
                          + (f" · <b>must_not:</b> {_e('; '.join(a[:22] for a in v.matched_must_not[:2]))}" if v.matched_must_not else "") + "</div>"))
        rows.append(
            f"<tr><td style='vertical-align:middle'><b>{_e(base)}</b><div class='small muted'>{_e(bare.repo.split('/')[-1])}</div></td>"
            f"<td><div class='small bad'><b>MODEL THUẦN</b> (bare)</div>{_img(b_best.path, side)}{cap(vb)}</td>"
            f"<td><div class='small ok'><b>HỆ THỐNG</b> ({_e(fx.final_source) if fx else '-'}"
            + (f", {len(fx.iterations)} vòng" if fx else "") + f")</div>{_img(f_path, side)}{cap(vf)}</td>"
            f"<td class='small' style='vertical-align:middle'>Δ CLIP attr {d('attr')}<br>Δ Reviewer {d('rev')}<br>Δ VQAScore {d('vqa')}"
            f"<div class='muted'>TB bare: attr {f2(sb['attr'])} · rev {f2(sb['rev'])}<br>TB system: attr {f2(ss['attr'])} · rev {f2(ss['rev'])}</div></td></tr>")
    if not rows:
        return ""
    note = ("<div class='muted'>Cùng model nền, cùng seed. <b>Model thuần</b>: prompt tiếng Anh dịch thẳng + negative chung, không KB, "
            "không LoRA, không ảnh tham chiếu. <b>Hệ thống</b>: Grounding (thuộc tính, negative, ảnh tham chiếu) + Agentic Review Loop; "
            "ảnh hiển thị là ảnh CUỐI hệ thống chọn. Δ tính trên trung bình mọi ảnh của mỗi nhánh.</div>")
    return _wrap("Kết luận · Model thuần so với Hệ thống, từng model nền",
                 "<table>" + "".join(rows) + "</table>" + note, source)


def per_model_table(cr, source: str | None = None) -> str:
    """v1.7.2: mỗi model nền một hệ thống -> một ảnh cuối, số vòng loop, lý do dừng. Không ensemble giữa các model."""
    pm = getattr(cr, "per_model", None) or []
    if not pm:
        return ""
    rows = ["<table><tr><th>model nền</th><th>ảnh cuối</th><th>nguồn</th><th>Reviewer</th><th>vòng loop</th><th>dừng</th></tr>"]
    for x in pm:
        sc = x.pool.get(x.final_path) if x.final_path else None
        rows.append(f"<tr{' style=background:#f0fdf4' if x.base_model == cr.base_model else ''}><td><b>{_e(x.base_model)}</b>"
                    f"<div class='muted small'>{_e(x.best_model or '')}</div></td><td>{_img(x.final_path, 160)}</td><td>{_e(x.final_source)}</td>"
                    f"<td>{'' if sc is None else f'{sc:+.2f}'}</td><td>{len(x.iterations)}</td><td class='small'>{_e(x.stop_reason)}</td></tr>")
    rows.append("</table>")
    note = ("<div class='muted'>Reviewer tầng 1 chấm chung mọi ảnh; Rank, Reflector/Refiner và ảnh cuối chạy riêng cho từng checkpoint "
            "(các hàng +ref / LoRA / IP-Adapter Plus của cùng checkpoint thuộc một nhóm). Hàng tô xanh là hồ sơ chính hiện chi tiết bên dưới.</div>")
    return _wrap("Agentic Review Loop · Ảnh cuối theo model nền", "".join(rows) + note, source)


def agent_dialog(cr, side: int = 170, source: str | None = None) -> str:
    """v1.9: NHẬT KÝ GIAO TIẾP giữa các agent trong Agentic Review Loop, đọc như một cuộc trao đổi:
    Reviewer (mô tả ảnh -> hỏi VQA từng thuộc tính -> kết luận) -> Rank (xếp hạng, so với metric) -> Reflector (chẩn đoán ->
    chọn nấc sửa -> viết caption/prompt) -> Refiner (prompt và ảnh thật sự dùng -> ảnh mới) -> Reviewer chấm lại.
    Mục đích: người đọc thấy chính xác agent nào nói gì với agent nào, và số nào dẫn tới quyết định nào."""
    if cr is None:
        return ""
    desc_by = {d.path: d for d in (cr.filter.descriptors or [])}
    ver_by = {v.path: v for v in cr.filter.verdicts}
    vqa_all = dict(getattr(cr, "vqa", {}) or {})

    def bubble(who: str, cls: str, body: str) -> str:
        return (f"<div style='margin:6px 0;padding:8px 10px;border-left:4px solid {cls};background:#fafaf8'>"
                f"<div class='small' style='color:{cls};font-weight:600'>{_e(who)}</div>{body}</div>")

    def vqa_rows(v):
        if not v.vqa:
            return ""
        fc, alt = dict(getattr(v, "vqa_fc", {}) or {}), dict(getattr(v, "alt_attrs", {}) or {})
        cells = "".join(
            f"<tr><td class='small'>{_e(a[:52])}</td><td class='small'><b class='{'ok' if p >= 0.6 else ('bad' if p <= 0.25 else 'muted')}'>"
            f"{p:.2f}</b></td><td class='small'>"
            + (f"trắc nghiệm, mô tả sai đối ứng: <i>{_e(alt.get(a, '')[:46])}</i>" if a in fc else "câu có/không")
            + f"</td><td class='small'>{'có' if a in v.matched_must_have else ('MUST_NOT' if a in v.matched_must_not else ('thiếu' if a in v.missing_must_have else '–'))}</td></tr>"
            for a, p in sorted(v.vqa.items(), key=lambda kv: -kv[1]))
        return ("<table style='margin:4px 0'><tr><th>hỏi VLM: thuộc tính</th><th>P(đúng)</th><th>cách hỏi</th>"
                f"<th>kết luận</th></tr>{cells}</table>")

    parts = []
    # --- hiệu chỉnh ngưỡng trên ảnh thật (v1.9.3) ---
    cal = dict(getattr(getattr(cr, "filter", None), "calibration", {}) or {})
    if cal:
        rows = "".join(
            f"<tr><td class='small'>{_e(a[:56])}</td><td class='small'>{'bắt buộc' if c.get('side') == 'have' else 'cấm'}</td>"
            f"<td class='small'>{c.get('ref_mean', 0):.2f}</td><td class='small'><b>{c.get('thr', 0):.2f}</b></td>"
            f"<td class='small'>{'—' if c.get('checkable') else '<b class=bad>không kiểm được, bỏ khỏi bảng kiểm</b>'}</td></tr>"
            for a, c in sorted(cal.items(), key=lambda kv: (kv[1].get("side") != "have", -kv[1].get("ref_mean", 0))))
        parts.append(bubble("REVIEWER ← ảnh tham chiếu thật (hiệu chỉnh ngưỡng từng thuộc tính)", "#0f766e",
                            "<div class='small'>Ngưỡng không đặt cứng mà lấy từ chính ảnh thật của prompt: ứng viên phải "
                            "giống thuộc tính gần bằng ảnh thật. Thuộc tính mà ảnh thật cũng không đạt thì VLM này không "
                            "kiểm được, bỏ khỏi bảng kiểm (vẫn giữ trong prompt sinh ảnh).</div>"
                            "<table style='margin:4px 0'><tr><th>thuộc tính</th><th>loại</th><th>ảnh thật</th>"
                            f"<th>ngưỡng</th><th>ghi chú</th></tr>{rows}</table>"))
    # --- Reviewer trên ảnh mốc ---
    base = cr.best_path or (cr.rank.final_order[0] if cr.rank.final_order else None)
    if base and base in ver_by:
        v, d = ver_by[base], desc_by.get(base)
        body = f"<div class='pair'><div>{_img(base, side)}</div><div style='flex:1'>"
        if d is not None:
            body += (f"<div class='small'><b>Bước 1, VLM mô tả ảnh (không phán xét văn hoá):</b> {_e(d.text()[:300])}</div>")
        body += f"<div class='small'><b>Bước 2, kiểm từng thuộc tính trên ảnh:</b></div>{vqa_rows(v)}"
        body += (f"<div class='small'><b>Bước 3, kết luận:</b> điểm {v.score:+.2f}, "
                 f"{'GIỮ' if v.keep else 'LOẠI'}" + (f", lý do: {_e('; '.join(v.reasons[:3]))}" if v.reasons else "") + "</div>")
        body += "</div></div>"
        parts.append(bubble("REVIEWER → (ảnh mốc của model nền " + _e(cr.base_model or "?") + ")", "#1d4ed8", body))
    # --- Rank ---
    rk = cr.rank
    if rk.final_order:
        agree = "trùng" if rk.agreement_top1 else "KHÁC"
        parts.append(bubble("RANK → REFLECTOR", "#7c3aed",
                            f"<div class='small'>Xếp {len(rk.final_order)} ảnh. Top-1 của agent và của metric <b>{agree}</b>"
                            + (f", Spearman {rk.spearman}" if rk.spearman is not None else "")
                            + (f". Bất đồng: {_e('; '.join(rk.disagreements[:2]))}" if rk.disagreements else "") + "</div>"))
    # --- từng vòng loop ---
    for it in (cr.iterations or []):
        fix = it.plan.rationale.split("]")[0].strip("[") if it.plan.rationale.startswith("[") else "?"
        body = (f"<div class='small'><b>Nhận từ Reviewer:</b> thiếu {_e('; '.join(a[:40] for a in (it.plan.add_positive or [])[:2]) or '—')}"
                f"<br><b>Chọn nấc:</b> <code>{_e(fix)}</code><br><b>Lý do:</b> {_e(it.plan.rationale[:260])}</div>")
        if it.captions:
            body += f"<div class='small'><b>Viết caption truy hồi:</b> {_e(' | '.join(it.captions))}</div>"
        if getattr(it.plan, "rewrite_prompt", ""):
            body += f"<div class='small'><b>Viết lại prompt:</b> “{_e(it.plan.rewrite_prompt)}”</div>"
        if it.plan.add_negative:
            body += f"<div class='small'><b>Thêm negative:</b> {_e('; '.join(it.plan.add_negative[:3]))}</div>"
        parts.append(bubble(f"REFLECTOR → REFINER · vòng {it.n}", "#b45309", body))

        rbody = ""
        if it.refs:
            rbody += f"<div class='small'><b>Ảnh tham chiếu nhận được ({len(it.refs)}):</b></div><div class='grid'>" + "".join(_img(r, 90) for r in it.refs[:4]) + "</div>"
        if it.run is not None and it.run.gen_spec is not None:
            gs = it.run.gen_spec
            rbody += (f"<div class='small'><b>Prompt thật sự gửi model ({_e(it.run.model_key)}):</b> <code>{_e(gs.prompt[:320])}</code></div>"
                      f"<div class='small muted'>negative: {_e(gs.negative_prompt[:180]) or '—'} · seed {gs.seed}+{gs.iteration}·1000 · guidance {gs.guidance:g}</div>")
            if it.run.notes:
                rbody += f"<div class='small muted'>{_e(' · '.join(it.run.notes[:3]))}</div>"
        if it.run is not None and it.run.output:
            rbody += "<div class='grid'>" + "".join(_img(c.path, side) for c in it.run.output.candidates[:4]) + "</div>"
        elif it.run is not None and it.run.error:
            rbody += f"<div class='bad small'>{_e(it.run.error)}</div>"
        parts.append(bubble(f"REFINER · vòng {it.n}", "#047857", rbody or "<div class='muted small'>không sinh được ảnh</div>"))

        if it.filter is not None and it.filter.verdicts:
            top = max(it.filter.verdicts, key=lambda v: v.score)
            parts.append(bubble(f"REVIEWER chấm lại · vòng {it.n}", "#1d4ed8",
                                f"<div class='small'>{len(it.filter.kept)}/{len(it.filter.verdicts)} ảnh qua. Ảnh tốt nhất {top.score:+.2f}"
                                + (f", còn thiếu: {_e('; '.join(a[:36] for a in top.missing_must_have[:2]))}" if top.missing_must_have else "")
                                + f". <b>{'CẢI THIỆN' if it.improved else 'không cải thiện'}</b> ({_e(it.note)})</div>"
                                + vqa_rows(top)))
    # --- kết ---
    fin = f"<div class='small'>Ảnh cuối: <b>{_e(Path(cr.final_path).name) if cr.final_path else '—'}</b> ({_e(cr.final_source)})"
    if cr.final_path and cr.final_path in vqa_all:
        fin += f", VQAScore {vqa_all[cr.final_path]:.2f}"
    fin += f". Dừng vì: {_e(cr.stop_reason)}</div>"
    if cr.final_path:
        fin += _img(cr.final_path, 260)
    parts.append(bubble("KẾT LUẬN", "#374151", fin))
    return _wrap(f"Nhật ký giao tiếp giữa các agent · {_e(cr.base_model or '')}", "".join(parts), source)


def candidate_review_html(cr, side: int = 200, source: str | None = None, res=None) -> str:
    rk = cr.rank
    parts = [final_grid(res, cr, source=source) if res is not None else "", per_model_table(cr, source),
             "".join(agent_dialog(x) for x in (getattr(cr, "per_model", None) or [cr])), filter_table(cr.filter, title="Agentic Review Loop · Reviewer tầng 1: Filter agent trên mọi ảnh", side=140)]
    # bảng xếp hạng
    rows = ["<table><tr><th>#</th><th>ảnh</th><th>hạng metric</th><th>hạng agent</th><th>lý do agent</th></tr>"]
    for i, p in enumerate(rk.final_order[:8]):
        rm = rk.order_metric.index(p) + 1 if p in rk.order_metric else "-"
        ra = rk.order_agent.index(p) + 1 if p in rk.order_agent else "-"
        rows.append(f"<tr><td>{i + 1}</td><td>{_img(p, 110)}<div class='muted small'>{_e(Path(p).stem)}</div></td><td>{rm}</td><td>{ra}</td>"
                    f"<td class='small'>{_e(rk.reasons.get(p, ''))}</td></tr>")
    rows.append("</table>")
    agree = (f"top-1 <b>{'trùng' if rk.agreement_top1 else 'KHÁC'}</b>" + (f" · Spearman {rk.spearman}" if rk.spearman is not None else "")
             + "".join(f"<div class='bad'>· {_e(d)}</div>" for d in rk.disagreements))
    parts.append(_wrap(f"Agentic Review Loop · Rank agent (đảo vị trí) so với metric · {_e(getattr(cr, 'base_model', '') or '')} top-{cr.k}", f"<div>{agree}</div>{''.join(rows)}"))
    # Agentic Review Loop: Reflector -> Refiner từng vòng
    body = ""
    its = getattr(cr, "iterations", None) or []
    if not its and cr.revision is not None:  # JSON cũ (v1.4-1.6) chỉ có một vòng
        its = [LoopIteration(n=1, plan=cr.revision, run=cr.regen, filter=cr.regen_filter)]
    for it in its:
        body += (f"<h4>Vòng {it.n}</h4><div><b>Reflector:</b> {_e(it.plan.rationale)}<br>+ prompt: {_e(', '.join(it.plan.add_positive))}"
                 f"<br>+ negative: {_e(', '.join(it.plan.add_negative))}"
                 + (f"<br>caption truy hồi: {_e(' | '.join(it.captions))}" if it.captions else "")
                 + (f"<br>ảnh tham chiếu: {len(it.refs)}" if it.refs else "") + "</div>")
        if it.refs:
            body += "<div class='grid'>" + "".join(_img(r, 90) for r in it.refs[:4]) + "</div>"
        if it.run is not None and it.run.output:
            body += "<div class='grid'>" + "".join(_img(c.path, side) for c in it.run.output.candidates) + "</div>"
            if it.filter is not None:
                body += filter_table(it.filter, title=f"Reviewer chấm lại ảnh vòng {it.n}", side=120)
        elif it.run is not None:
            body += f"<div class='bad'>{_e(it.run.error or '')}</div>"
        if it.note:
            body += f"<div class='{'good' if it.improved else 'muted'}'>· {_e(it.note)}</div>"
    body += "".join(f"<div class='muted'>· {_e(n)}</div>" for n in cr.notes)
    pool = getattr(cr, "pool", None) or {}
    if pool:
        top = sorted(pool.items(), key=lambda kv: -kv[1])[:6]
        body += "<h4>Chọn cuối trên toàn pool (" + str(len(pool)) + " ảnh)</h4><div class='grid'>" + "".join(
            f"<div>{_img(pth, 120)}<div class='small muted'>{sc:+.2f}</div></div>" for pth, sc in top) + "</div>"
    if cr.final_path:
        body += (f"<h4>Ảnh cuối ({_e(cr.final_source)}, {_e(cr.best_model or '')})</h4>{_img(cr.final_path, 320)}")
    n_it = len(its)
    parts.append(_wrap(f"Agentic Review Loop · Reflector + Refiner · {_e(getattr(cr, 'base_model', '') or '')} ({n_it} vòng)", body or "<div class='muted'>không chạy</div>", source))
    return "".join(parts)


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
        """Cùng tiêu đề (<h3>) thì THAY phần cũ: bấm lại một cell không nhân đôi mục trong file (v1.3 p001: Bước 5 lặp 3 lần)."""
        import re

        m = re.search(r"<h3>(.*?)(?:<span|</h3>)", html, re.S)
        title = m.group(1).strip() if m else None
        if title:
            for i, old in enumerate(self.parts):
                mo = re.search(r"<h3>(.*?)(?:<span|</h3>)", old, re.S)
                if mo and mo.group(1).strip() == title:
                    self.parts[i] = html
                    show(html)
                    return
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
