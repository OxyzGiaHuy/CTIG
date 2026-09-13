"""
Stage 4b - MULTIGEN: cùng một GenSpec, nhiều model sinh ảnh, so trên một grid.

Đây là cách rẻ nhất để trả lời "model nào vẽ văn hoá Việt đúng hơn" mà không cần vòng review:
mỗi model sinh N ứng viên, chấm CLIP danh tính (P(mục tiêu) vs confusable), BLIP-2 ITM
(ảnh khớp mô tả thực thể) và CLIP sim(prompt), rồi xếp cạnh nhau.

Ràng buộc VRAM: nạp MỘT model, sinh, chấm, giải phóng, rồi model kế. Model nhỏ chạy trước để
OOM ở model lớn không làm mất kết quả. Ảnh đã sinh với cùng (model, GenSpec, seed) được dùng lại
(source="disk") nên chạy lại cell không sinh lại.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from ..kb import KnowledgeBase
from ..models import loader as model_loader
from ..models.registry import ModelSpec, get as get_model, parse_flags, parse_key, parse_variant
from ..schema import Candidate, CulturalSpec, GenOutput, GenSpec, ModelRun, MultiGenResult, from_dict, to_dict
from .generation import DiffusersGenerator, StubGenerator, _font


def render_settings(cfg) -> dict:
    """Những tham số NGOÀI GenSpec làm ảnh khác đi (v1.3): scheduler, compel, hires. Vào hash để ảnh cũ không bị dùng nhầm."""
    h = getattr(cfg, "hires", None)
    return {
        "sched": getattr(cfg, "scheduler", None),
        "long": bool(getattr(cfg, "long_prompt", False)),
        "hires": [h.scale, h.strength, h.steps] if (h is not None and getattr(h, "enabled", False)) else None,
        "refs": int(getattr(cfg, "ref_images", 1)),
    }


def genspec_hash(gen: GenSpec, render: dict | None = None) -> str:
    sig = json.dumps({"p": gen.prompt_terms, "n": gen.negative_terms, "seed": gen.seed,
                      "steps": gen.steps, "g": gen.guidance, "w": gen.width, "h": gen.height,
                      **({"r": render} if render else {})},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


def _clamp(v: int, max_side: int) -> int:
    v = min(v, max_side)
    return max(256, (v // 8) * 8)


def adapt_spec(gen: GenSpec, mspec: ModelSpec, cfg, spec: CulturalSpec | None = None, prompt_en: str = "",
               variant: str | None = None, t2i_cfg=None) -> GenSpec:
    """GenSpec chung -> GenSpec cho một model: kích cỡ, bước, guidance, negative, trigger LoRA.

    v1.4.1: cách render prompt có thể khác theo model (sd3 -> sentence) hoặc theo hậu tố '#variant' của khoá; khi đó
    prompt/negative được render lại từ spec (cần `spec` và `prompt_en`). Negative riêng của checkpoint nối vào cuối.
    """
    ov = (cfg.overrides or {}).get(mspec.key, {})
    w, h = _clamp(ov.get("width", mspec.width), cfg.max_side), _clamp(ov.get("height", mspec.height), cfg.max_side)
    want = variant or mspec.render
    g = gen
    if want and want != gen.render and spec is not None and t2i_cfg is not None:
        from .generation import render_terms

        terms, neg, weights = render_terms(prompt_en or "", spec, t2i_cfg, want)
        g = replace(gen, prompt_terms=terms, negative_terms=neg, term_weights=weights, render=want)
    neg = list(g.negative_terms) if mspec.negative_ok else []
    if mspec.negative_ok and mspec.extra_negative:
        neg = list(dict.fromkeys(neg + list(mspec.extra_negative)))
    return replace(
        g,
        negative_terms=neg,
        steps=int(ov.get("steps", mspec.steps)), guidance=float(ov.get("guidance", mspec.guidance)),
        width=w, height=h, n_candidates=int(ov.get("n_candidates", cfg.n_candidates)),
        ip_adapter_image=None, lora=None, fast=False, iteration=0,
    )


def score_run(run: ModelRun, spec: CulturalSpec, clip, itm, prompt_en: str) -> None:
    """Điền CLIP identity (select_candidate), ITM và CLIP sim(prompt) cho từng ứng viên."""
    if run.output is None:
        return
    from .review import select_candidate

    if clip is not None:
        select_candidate(run.output, spec, clip)  # clip_fidelity + clip_probs, chọn ứng viên tốt nhất
        pairs = attribute_labels(spec)
        for c in run.output.candidates:
            if prompt_en:
                try:
                    c.clip_prompt_sim = round(clip.similarity(c.path, [prompt_en])[0], 4)
                except Exception:  # noqa: BLE001
                    c.clip_prompt_sim = None
            # v1.2 p001: CLIP identity 0.95-1.00 cho MỌI model, kể cả ảnh không có quần hay có đai đỏ.
            # Danh tính bão hoà trên prompt dễ; cần đối chiếu ở mức thuộc tính.
            if pairs:
                try:
                    c.attr_contrast = round(_attr_contrast(clip, c.path, pairs), 4)
                except Exception:  # noqa: BLE001
                    c.attr_contrast = None
    if itm is not None and spec.entities:
        objs = [se for se in spec.entities if se.kind == "object"] or list(spec.entities)
        labels = [se.clip_label or f"a photo of Vietnamese {se.name_en.split('(')[0].strip()}" for se in objs]
        wt = sum(se.weight for se in objs) or 1.0
        attr_sents = [f"{se.name_en.split('(')[0].strip()} with {a}" for se in objs for a in se.required_attrs_en if a][:8]
        try:
            itm._on_gpu()
            for c in run.output.candidates:
                ps = itm.itm(c.path, labels)
                c.itm_score = round(sum(se.weight * p for se, p in zip(objs, ps)) / wt, 4)
                if attr_sents:
                    pa = itm.itm(c.path, attr_sents)
                    c.itm_attrs = round(sum(pa) / len(pa), 4)
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                itm._off_gpu()
            except Exception:  # noqa: BLE001
                pass


def attribute_labels(spec: CulturalSpec) -> list[tuple[list[str], list[str]]]:
    """Với mỗi thực thể vật thể: (câu must_have_en, câu must_not_en) để CLIP tương phản."""
    pairs = []
    for se in spec.entities:
        if se.kind != "object":
            continue
        name = se.name_en.split("(")[0].strip()
        pos = [f"a photo of a Vietnamese {name} with {a}" for a in se.required_attrs_en if a][:4]
        neg = [f"a photo of a {name} with {a}" for a in se.forbidden_attrs_en if a][:4]
        if pos and neg:
            pairs.append((pos, neg))
    return pairs


def _attr_contrast(clip, path: str, pairs: list[tuple[list[str], list[str]]]) -> float:
    """Trung bình trên thực thể: tổng P(câu must_have) trong softmax chung với câu must_not."""
    vals = []
    for pos, neg in pairs:
        p = clip.probs(path, pos + neg)
        vals.append(sum(p[: len(pos)]))
    return sum(vals) / len(vals)


def should_use_refs(spec: CulturalSpec, kb: KnowledgeBase, cfg) -> tuple[bool, str]:
    """v1.5 (ImageRAG): chỉ cấp ảnh tham chiếu khi thực thể vật thể chính có prior thấp hoặc không có trong KB.
    Trả (dùng?, lý do)."""
    ar = getattr(cfg, "auto_ref", None)
    if ar is None or not getattr(ar, "enabled", False):
        return True, "auto_ref tắt"
    objs = [se for se in spec.entities if se.kind == "object"]
    if not objs:
        return False, "không có thực thể vật thể"
    main = max(objs, key=lambda se: se.weight)
    ent = kb.get(main.entity_id)
    if ent is None:
        return True, f"{main.entity_id} không có trong KB -> coi là prior thấp"
    if ent.prior_strength <= ar.prior_max:
        return True, f"prior {ent.prior_strength:.2f} <= {ar.prior_max} -> dùng ảnh tham chiếu"
    return False, f"prior {ent.prior_strength:.2f} > {ar.prior_max} -> model tự vẽ được, không dùng ảnh (tránh chép)"


def score_key(c: Candidate) -> float:
    """Điểm dùng để CHỌN: ensemble hạng (v1.5) nếu đã tính, không thì combined_score."""
    return c.ensemble if c.ensemble is not None else combined_score(c)


def ensemble_rank(result: MultiGenResult, penalty: bool = True) -> None:
    """Verifier Ensemble (Ma et al. 2025): trung bình hạng KHÔNG trọng số trên các verifier có mặt, tính trên mọi ứng viên
    của lần chạy. Verifier: CLIP id, CLIP attr, ITM attr, đẹp (PickScore). ensemble = 1 - hạngTB/(n-1), trừ phạt chép."""
    cands = [c for r in result.runs if r.output for c in r.output.candidates]
    n = len(cands)
    if n == 0:
        return
    if n == 1:
        cands[0].ensemble = 1.0
        return
    metrics = [
        lambda c: c.clip_fidelity if c.clip_probs else None,
        lambda c: c.attr_contrast,
        lambda c: c.itm_attrs,
        lambda c: c.aesthetic,
    ]
    ranks = {id(c): [] for c in cands}
    for m in metrics:
        vals = [(m(c), c) for c in cands]
        have = [(v, c) for v, c in vals if v is not None]
        if len(have) < 2:
            continue
        order = sorted(have, key=lambda vc: -vc[0])
        # hạng trung bình cho giá trị bằng nhau
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and abs(order[j + 1][0] - order[i][0]) < 1e-9:
                j += 1
            r = (i + j) / 2
            for k in range(i, j + 1):
                ranks[id(order[k][1])].append(r)
            i = j + 1
    for c in cands:
        rs = ranks[id(c)]
        if not rs:
            c.ensemble = None
            continue
        e = 1.0 - (sum(rs) / len(rs)) / (n - 1)
        if penalty and c.ref_sim is not None and c.ref_sim > COPY_THRESHOLD:
            e -= min(0.5, (c.ref_sim - COPY_THRESHOLD) * 3.0)
        c.ensemble = round(e, 4)


def combined_score(c: Candidate) -> float:
    """Điểm xếp hạng = trung bình các số có: danh tính CLIP, CLIP attr, ITM attr, thẩm mỹ (PickScore chuẩn hoá).

    Danh tính bão hoà ~1 trên prompt dễ nên thực chất thứ tự do attr + thẩm mỹ quyết định.
    """
    parts = []
    if c.clip_probs:
        parts.append(c.clip_fidelity)
    if c.attr_contrast is not None:
        parts.append(c.attr_contrast)
    if c.itm_attrs is not None:
        parts.append(c.itm_attrs)
    if c.aesthetic is not None:
        parts.append(c.aesthetic)
    if not parts and c.itm_score is not None:
        parts.append(c.itm_score)
    base = sum(parts) / len(parts) if parts else 0.0
    # v1.4.2: phạt "chép" ảnh tham chiếu. Cosine ảnh-ảnh > ngưỡng (0,88 mặc định) trừ dần; kênh ảnh không được thưởng vì sao chép.
    if c.ref_sim is not None and c.ref_sim > COPY_THRESHOLD:
        base -= min(0.5, (c.ref_sim - COPY_THRESHOLD) * 3.0)
    return base


COPY_THRESHOLD = 0.88


def score_ref_sim(result: MultiGenResult, clip, refs: list[str], log=print) -> None:
    """ref_sim cho mọi ứng viên còn thiếu: max cosine CLIP với các ảnh tham chiếu (ảnh đã cắt nếu có)."""
    if clip is None or not refs or not hasattr(clip, "image_similarity"):
        return
    try:
        for r in result.runs:
            if not r.output:
                continue
            for c in r.output.candidates:
                if c.ref_sim is None:
                    c.ref_sim = round(max(clip.image_similarity(c.path, ref) for ref in refs), 4)
    except Exception as exc:  # noqa: BLE001
        log(f"  [4c] ref_sim lỗi ({type(exc).__name__}: {str(exc)[:80]})")


def score_aesthetic(result: MultiGenResult, scorer, prompt_en: str, log=print) -> None:
    """PickScore thô cho ảnh còn thiếu, rồi chuẩn hoá min-max trên TẤT CẢ ứng viên của lần chạy."""
    from .aesthetic import normalize

    cands = [c for r in result.runs if r.output for c in r.output.candidates]
    if not cands:
        return
    todo = [c for c in cands if c.pick_score is None]
    if todo and scorer is not None and prompt_en:
        try:
            scorer._on_gpu()
            for i in range(0, len(todo), 4):
                chunk = todo[i:i + 4]
                for c, s in zip(chunk, scorer.score(prompt_en, [c.path for c in chunk])):
                    c.pick_score = s
        except Exception as exc:  # noqa: BLE001
            msg = f"PickScore lỗi khi chấm: {type(exc).__name__}: {str(exc)[:120]} -> cột 'đẹp' trống"
            log(f"  [4c] {msg}")
            result.notes.append(msg)
        finally:
            try:
                scorer._off_gpu()
            except Exception:  # noqa: BLE001
                pass
    for c, a in zip(cands, normalize([c.pick_score for c in cands])):
        c.aesthetic = a
    rechoose(result)


def rechoose(result: MultiGenResult) -> None:
    """Viền xanh = ứng viên ĐIỂM TỔNG cao nhất trong hàng (v1.3: CLIP id bão hoà nên chọn theo nó gần như ngẫu nhiên)."""
    for r in result.runs:
        if r.output and len(r.output.candidates) > 1:
            r.output.chosen = max(range(len(r.output.candidates)), key=lambda i: score_key(r.output.candidates[i]))


def _read_previous(out_dir: Path) -> MultiGenResult | None:
    """Đọc multigen.json của lần trước MỘT LẦN ở đầu run(). (v1.2.1: đọc lại sau mỗi hàng trong khi file đã bị
    ghi đè bởi kết quả đang chạy -> chỉ hàng đầu tiên được dùng lại, các hàng sau sinh lại dù GenSpec không đổi.)"""
    meta = out_dir / "multigen.json"
    if not meta.exists():
        return None
    try:
        return from_dict(MultiGenResult, json.loads(meta.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return None


def _load_previous(prev: MultiGenResult | None, key: str, ghash: str, n: int) -> ModelRun | None:
    """Ảnh của (model, GenSpec hash) đã có trên đĩa -> dùng lại, không sinh."""
    if prev is None or prev.genspec_hash != ghash:
        return None
    for r in prev.runs:
        if r.model_key == key and r.output and not r.error and len(r.output.candidates) >= n \
                and all(Path(c.path).exists() for c in r.output.candidates):
            return replace(r, source="disk")
    return None


def run(gen: GenSpec, spec: CulturalSpec, kb: KnowledgeBase, model_keys: list[str], cfg, out_dir: Path,
        clip=None, itm=None, prompt_en: str = "", log=print, on_model_done=None,
        loader=None, lora_dir: Path | str | None = None, ref_images: list[str] | None = None,
        aesthetic=None, t2i_cfg=None) -> MultiGenResult:
    """`ref_images`: ảnh tham chiếu đã qua CLIP (tốt nhất trước) cho hàng IP-Adapter; `aesthetic`: PickScorer hoặc None.
    Khoá model có thể mang hậu tố '@<scale>' để ghi đè LoRA scale (sweep: sdxl_aodai@0.6, sdxl_aodai@1.0)."""
    loader = loader or model_loader.load_pipeline
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ghash = genspec_hash(gen, render_settings(cfg))
    result = MultiGenResult(gen.prompt_id, [], prompt_en=prompt_en, genspec_hash=ghash)
    spec_ids = {se.entity_id for se in spec.entities}
    refs = list(ref_images or [])
    if not refs:
        refs = [se.reference_image for se in spec.entities if se.reference_image and se.kind == "object"]
    previous = _read_previous(out_dir)

    for key in model_keys:
        try:
            base_key, lora_scale = parse_key(key)
            variant = parse_variant(key)
            flags = parse_flags(key)
            mspec = get_model(base_key)
            gate_note = None
            if "ref" in flags:
                if mspec.family not in ("sdxl",):
                    raise KeyError(f"'+ref' chỉ dùng cho họ sdxl (khoá {key})")
                use, why = should_use_refs(spec, kb, cfg)
                if use:
                    mspec = replace(mspec, ip_adapter=True, ip_adapter_kind="plus",
                                    ip_adapter_scale=float(getattr(cfg, "ref_scale", 0.4)))
                    gate_note = f"auto_ref: {why}"
                else:
                    gate_note = f"auto_ref: {why} -> hàng chạy KHÔNG ảnh tham chiếu"
        except KeyError as exc:
            result.runs.append(ModelRun(key, "-", gen, error=str(exc)))
            _save(result, out_dir)
            if on_model_done:
                on_model_done(result.runs[-1])
            continue
        if mspec.only_if_entity and not (set(mspec.only_if_entity) & spec_ids):
            result.runs.append(ModelRun(key, mspec.repo, gen,
                                        error=f"bỏ qua: spec không có {mspec.only_if_entity}"))
            if on_model_done:
                on_model_done(result.runs[-1])
            continue

        gspec = adapt_spec(gen, mspec, cfg, spec=spec, prompt_en=prompt_en, variant=variant, t2i_cfg=t2i_cfg)
        safe_key = key.replace("@", "_s").replace("#", "_r").replace("+", "_")  # tên thư mục/file an toàn
        need = gspec.n_candidates
        if getattr(cfg, "adaptive", None) is not None and getattr(cfg.adaptive, "enabled", False):
            need = max(1, min(int(cfg.adaptive.min), gspec.n_candidates))
        prev = _load_previous(previous, key, ghash, need)
        if prev is not None:
            log(f"  [4b] {key}: dùng lại ảnh trên đĩa (GenSpec không đổi)")
            prev.gen_spec = gspec
            if any(c.itm_score is None or c.attr_contrast is None for c in prev.output.candidates) or any(
                    not c.clip_probs for c in prev.output.candidates):
                score_run(prev, spec, clip, itm, prompt_en)
            result.runs.append(prev)
            _save(result, out_dir)
            if on_model_done:
                on_model_done(prev)
            continue

        t0 = time.time()
        run_rec = ModelRun(key, mspec.repo, gspec)
        if gate_note:
            run_rec.notes.append(gate_note)
            log(f"  [4b] {key}: {gate_note}")
        pipe = None
        g = None
        try:
            # v1.4.3: kiểm VRAM trước khi nạp. v1.4 p001: 14,4 GB còn cấp phát từ hàng trước -> 4 hàng OOM ngay lúc nạp.
            held = model_loader.allocated_gb(cfg.device)
            if held is not None and held > 1.0:
                model_loader.free_vram()
                held2 = model_loader.allocated_gb(cfg.device)
                msg = f"VRAM còn giữ {held} GB trước khi nạp" + (f", sau gc {held2} GB" if held2 != held else "") + " (rò từ hàng trước?)"
                log(f"  [4b] {key}: {msg}")
                run_rec.notes.append(msg)
            model_loader.reset_peak(cfg.device)
            if mspec.family == "stub":
                g = StubGenerator(cfg)
                g.model_key = safe_key
            else:
                if mspec.ip_adapter and not refs:
                    raise RuntimeError("bỏ qua: spec không có ảnh tham chiếu đạt CLIP cho thực thể vật thể")
                log(f"  [4b] nạp {key} ({mspec.repo}) ...")
                try:
                    pipe = loader(mspec, cfg.device, cfg.cpu_offload, log=log, scheduler=getattr(cfg, "scheduler", None))
                except TypeError:  # loader tiêm từ test có chữ ký cũ
                    pipe = loader(mspec, cfg.device, cfg.cpu_offload)
                run_rec.notes.append(f"scheduler {type(pipe.scheduler).__name__}" if hasattr(pipe, "scheduler") else "")
                if gspec.render != gen.render:
                    run_rec.notes.append(f"render {gspec.render}")
                trigger = None
                if mspec.lora:
                    how = model_loader.attach_lora(pipe, mspec.lora, lora_dir or (out_dir.parent.parent / "_cache" / "lora"),
                                                   log=log, scale=lora_scale)
                    log(f"  [4b] {key}: LoRA gắn xong ({how})")
                    run_rec.notes.append(f"LoRA {how}")
                    trigger = mspec.lora.get("trigger")
                ref_imgs: list[str] | None = None
                if mspec.ip_adapter:
                    n_ref = int(getattr(cfg, "ref_images", 1)) if mspec.ip_adapter_kind == "plus" else 1
                    ref_imgs = refs[:max(1, n_ref)]
                    how = model_loader.load_ip_adapter(pipe, mspec.ip_adapter_kind, mspec.ip_adapter_scale, log=log)
                    log(f"  [4b] {key}: {how}, {len(ref_imgs)} ảnh tham chiếu: " + ", ".join(Path(r).name for r in ref_imgs))
                    run_rec.notes.append(f"{how}, {len(ref_imgs)} ảnh tham chiếu")
                hires = getattr(cfg, "hires", None) if mspec.hires_ok else None
                if hires is not None and mspec.ip_adapter and getattr(hires, "skip_ip_adapter", True):
                    hires = None  # IP-Adapter (+1,3 GB encoder) cộng img2img 1536 px vượt T4 15 GB (v1.4 p001)
                    run_rec.notes.append("hires tắt cho hàng IP-Adapter (VRAM)")
                g = DiffusersGenerator(pipe, safe_key, trigger=trigger, negative_ok=mspec.negative_ok,
                                       ip_adapter_image=ref_imgs, family=mspec.family,
                                       long_prompt=bool(getattr(cfg, "long_prompt", False)), hires=hires, log=log)
            ad = getattr(cfg, "adaptive", None)
            adaptive = bool(ad is not None and getattr(ad, "enabled", False) and clip is not None and mspec.family != "stub_never")
            if adaptive:
                # v1.5 best-of-N thích nghi: sinh `min`, verifier (CLIP attr tốt nhất) chưa đạt thì thêm `step` tới `max`.
                n_min = max(1, min(int(ad.min), gspec.n_candidates if gspec.n_candidates > 0 else int(ad.min)))
                first = replace(gspec, n_candidates=n_min)
                run_rec.output = g.generate(first, spec, kb, out_dir / safe_key)
                score_run(run_rec, spec, clip, itm, prompt_en)
                total = len(run_rec.output.candidates)
                rounds = 0
                while total < int(ad.max):
                    best_attr = max([c.attr_contrast for c in run_rec.output.candidates if c.attr_contrast is not None] or [None]) \
                        if any(c.attr_contrast is not None for c in run_rec.output.candidates) else None
                    if best_attr is not None and best_attr >= float(ad.target_attr):
                        break
                    step = max(1, min(int(ad.step), int(ad.max) - total))
                    more = g.generate(replace(gspec, n_candidates=step), spec, kb, out_dir / safe_key, start_index=total)
                    run_rec.output.candidates.extend(more.candidates)
                    total = len(run_rec.output.candidates)
                    rounds += 1
                    score_run(run_rec, spec, clip, itm, prompt_en)
                    log(f"  [4b] {key}: verifier chưa đạt (attr tốt nhất {best_attr}), sinh thêm {step} -> {total}")
                if rounds:
                    run_rec.notes.append(f"adaptive: {n_min} -> {total} ứng viên (mục tiêu attr {ad.target_attr})")
                else:
                    run_rec.notes.append(f"adaptive: đạt với {total} ứng viên")
            else:
                run_rec.output = g.generate(gspec, spec, kb, out_dir / safe_key)
                score_run(run_rec, spec, clip, itm, prompt_en)
            run_rec.seconds = round(time.time() - t0, 1)
            run_rec.peak_vram_gb = model_loader.peak_gb(cfg.device)
            run_rec.prompt_tokens = getattr(g, "prompt_tokens", None)
            run_rec.notes = [n for n in run_rec.notes + list(getattr(g, "notes", [])) if n]
            for n in getattr(g, "notes", []):
                log(f"  [4b] {key}: {n}")
            rechoose(MultiGenResult(gen.prompt_id, [run_rec]))
            log(f"  [4b] {key}: {len(run_rec.output.candidates)} ảnh, {run_rec.seconds}s"
                + (f", đỉnh {run_rec.peak_vram_gb} GB" if run_rec.peak_vram_gb else "")
                + (f", CLIP {max(c.clip_fidelity for c in run_rec.output.candidates):.2f}" if clip else ""))
        except Exception as exc:  # noqa: BLE001 - một model lỗi (OOM, gated, LoRA) không được làm hỏng grid
            run_rec.error = f"{type(exc).__name__}: {str(exc)[:400]}"
            run_rec.seconds = round(time.time() - t0, 1)
            log(f"  [4b] {key}: LỖI {run_rec.error}")
        finally:
            # Gỡ MỌI tham chiếu tới pipeline trước khi unload: generator (pipe, compel giữ text encoder, img2img dùng chung
            # trọng số). Biến `g` sống tới vòng lặp sau nếu không xoá -> UNet 5 GB không bao giờ được trả.
            if g is not None:
                for attr in ("pipe", "_compel", "_hires_pipe"):
                    try:
                        setattr(g, attr, None)
                    except Exception:  # noqa: BLE001
                        pass
                g = None
            if pipe is not None:
                model_loader.unload(pipe, log=log)
                pipe = None
            else:
                model_loader.free_vram()
            held = model_loader.allocated_gb(cfg.device)
            if held is not None:
                run_rec.notes.append(f"VRAM sau giải phóng {held} GB")
                if held > 1.0:
                    log(f"  [4b] {key}: CẢNH BÁO còn {held} GB cấp phát sau khi giải phóng")
        result.runs.append(run_rec)
        _save(result, out_dir)
        if on_model_done:
            on_model_done(run_rec)

    global COPY_THRESHOLD
    COPY_THRESHOLD = float(getattr(cfg, "copy_threshold", 0.88))
    score_ref_sim(result, clip, refs, log=log)
    if aesthetic is not None or any(c.pick_score is not None for r in result.runs if r.output for c in r.output.candidates):
        score_aesthetic(result, aesthetic, prompt_en, log=log)
    if getattr(cfg, "ensemble", True):
        ensemble_rank(result)
        rechoose(result)
    elif getattr(cfg, "aesthetic", None) is not None and getattr(cfg.aesthetic, "enabled", False):
        from .aesthetic import LAST_ERROR

        result.notes.append("PickScore không nạp được" + (f": {LAST_ERROR}" if LAST_ERROR else " (xem log cell bước 4)") + " -> cột 'đẹp' trống")
    rechoose(result)
    result.grid_path = str(draw_grid(result, spec, out_dir / "grid.png"))
    _save(result, out_dir)
    return result


def _save(result: MultiGenResult, out_dir: Path) -> None:
    (out_dir / "multigen.json").write_text(json.dumps(to_dict(result), ensure_ascii=False, indent=1), encoding="utf-8")


def best_run(result: MultiGenResult) -> ModelRun | None:
    """Model có ứng viên điểm tổng (danh tính + thuộc tính) cao nhất."""
    best, best_s = None, -1.0
    for r in result.runs:
        if not r.output:
            continue
        for c in r.output.candidates:
            s = score_key(c)
            if s > best_s:
                best, best_s = r, s
    return best


def draw_grid(result: MultiGenResult, spec: CulturalSpec, path: Path, cell: int = 512) -> Path:
    """PNG: hàng = model, cột = ứng viên, nhãn điểm dưới mỗi ảnh (tự xuống dòng theo bề rộng ô)."""
    from PIL import Image, ImageDraw

    rows = result.runs
    ncol = max([len(r.output.candidates) for r in rows if r.output] + [1])
    pad, left, line_h = 10, 200, 15
    f_b, f_s, f_h = _font(14, True), _font(11), _font(16, True)

    def caption(c) -> list[str]:  # noqa: D401
        parts = []
        if c.clip_probs:
            parts.append(f"CLIP id {c.clip_fidelity:.2f}")
        if c.attr_contrast is not None:
            parts.append(f"attr {c.attr_contrast:.2f}")
        if c.itm_score is not None:
            parts.append(f"ITM {c.itm_score:.2f}")
        if c.itm_attrs is not None:
            parts.append(f"ITMattr {c.itm_attrs:.2f}")
        if c.aesthetic is not None:
            parts.append(f"đẹp {c.aesthetic:.2f}")
        if c.clip_prompt_sim is not None:
            parts.append(f"sim {c.clip_prompt_sim:.2f}")
        if c.ref_sim is not None:
            parts.append(f"giống ref {c.ref_sim:.2f}" + ("!" if c.ref_sim > COPY_THRESHOLD else ""))
        if c.ensemble is not None:
            parts.append(f"hạng {c.ensemble:.2f}")
        if c.base_path:
            parts.append("hires")
        return _fit_lines(parts or [f"seed {c.seed}"], f_s, cell - 4)

    caps = {id(c): caption(c) for r in rows if r.output for c in r.output.candidates}
    max_lines = max([len(v) for v in caps.values()] + [1])
    label_h = line_h * max_lines + 10
    W = left + ncol * (cell + pad) + pad
    H = pad + 28 + len(rows) * (cell + label_h + pad) + 10
    img = Image.new("RGB", (W, H), (250, 249, 246))
    d = ImageDraw.Draw(img)
    d.text((pad, pad), f"{result.prompt_id} · {len(rows)} model", font=f_h, fill=(28, 30, 34))
    y = pad + 28
    for r in rows:
        d.text((pad, y + 4), r.model_key, font=f_b, fill=(28, 30, 34))
        meta = f"{r.gen_spec.steps} bước · g{r.gen_spec.guidance:g} · {r.gen_spec.width}px"
        d.text((pad, y + 24), meta, font=f_s, fill=(120, 124, 132))
        d.text((pad, y + 40), f"{r.seconds:.0f}s" + (f" · {r.peak_vram_gb} GB" if r.peak_vram_gb else ""), font=f_s, fill=(120, 124, 132))
        yy = y + 56
        if r.source == "disk":
            d.text((pad, yy), "ảnh từ lần trước", font=f_s, fill=(120, 124, 132)); yy += 16
        for n in (r.notes or [])[:4]:
            if n.startswith("scheduler"):
                continue
            for line in _fit_lines(n.split(), f_s, left - 2 * pad)[:2]:
                d.text((pad, yy), line, font=f_s, fill=(120, 124, 132)); yy += 14
        if r.error or not r.output:
            d.rectangle([left, y, left + cell, y + cell], fill=(238, 236, 230))
            for i, line in enumerate(_wrap(r.error or "không có ảnh", max(20, cell // 8))[:8]):
                d.text((left + 8, y + 8 + 16 * i), line, font=f_s, fill=(196, 48, 43))
        else:
            for j, c in enumerate(r.output.candidates):
                x = left + j * (cell + pad)
                try:
                    im = Image.open(c.path).convert("RGB")
                    im.thumbnail((cell, cell))
                    img.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
                except Exception:  # noqa: BLE001
                    d.rectangle([x, y, x + cell, y + cell], fill=(238, 236, 230))
                if j == r.output.chosen and len(r.output.candidates) > 1:
                    d.rectangle([x - 2, y - 2, x + cell + 1, y + cell + 1], outline=(22, 128, 82), width=3)
                for k, line in enumerate(caps[id(c)]):
                    d.text((x, y + cell + 4 + k * line_h), line, font=f_s, fill=(28, 30, 34))
        y += cell + label_h + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _fit_lines(parts: list[str], font, max_w: int, sep: str = " · ") -> list[str]:
    """Ghép các mảnh bằng ` · ` thành các dòng không rộng quá max_w (đo bằng font thật)."""
    def width(s: str) -> float:
        try:
            return font.getlength(s)
        except Exception:  # noqa: BLE001
            return 6.5 * len(s)
    lines, cur = [], ""
    for ptxt in parts:
        cand = f"{cur}{sep}{ptxt}" if cur else ptxt
        if cur and width(cand) > max_w:
            lines.append(cur); cur = ptxt
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _wrap(text: str, n: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines
