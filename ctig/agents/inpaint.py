"""
Refiner sửa CỤC BỘ (v1.8, theo GenArtist / SLD / Marmot): ảnh đã đúng phần lớn nhưng thiếu đúng MỘT thuộc tính có vùng xác định
được (cổ áo, đai, vành nón, mũi thuyền) -> tìm vùng bằng OWL-ViT, làm mặt nạ, inpaint vùng đó với prompt tập trung vào thuộc tính,
giữ nguyên phần còn lại. Thay cho sinh lại toàn ảnh (mất luôn phần đã đúng). Chỉ họ sdxl / sd15 (AutoPipelineForInpainting.from_pipe).

Đầu ra là một ModelRun như Refiner thường (ảnh vào revision/iter<n>/inpaint/), để Reviewer chấm lại và pool giữ hết.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..schema import Candidate, GenOutput, GenSpec, ModelRun

#: DANH TỪ NGẮN chỉ bộ phận (GenArtist dùng từ vựng 7.605 danh từ đơn; cụm mô tả dài bị CLIP của OWL-ViT nuốt còn tên thực thể).
_PARTS = ("collar", "neckline", "sleeve", "sleeves", "trousers", "pants", "sash", "belt", "waist", "panel", "panels", "skirt", "hem",
          "slit", "brim", "strap", "tip", "bow", "stern", "hull", "oar", "paddle", "button", "buttons", "placket",
          "cuff", "headscarf", "turban", "hat", "lantern", "tray", "envelope", "envelopes", "tree", "blossom", "blossoms")
#: bộ phận đứng riêng được (không cần gắn với thực thể cha)
_STANDALONE = {"tree", "lantern", "tray", "envelope", "envelopes", "blossom", "blossoms", "hat", "oar", "paddle"}


def part_noun(attr_en: str) -> str | None:
    """Danh từ bộ phận ngắn trong thuộc tính, hoặc None. 'high stand-up mandarin collar' -> 'collar'."""
    low = attr_en.lower()
    for p in sorted(_PARTS, key=len, reverse=True):
        if p in low:
            return p
    return None


def part_queries(attr_en: str, name_en: str) -> list[str]:
    """Thứ tự thử: danh từ bộ phận ngắn -> bộ phận gắn thực thể -> chính thực thể (SLD: hỏi ĐỐI TƯỢNG CHA vì bộ phận đang
    SAI hoặc THIẾU thì không tìm được; box cha chính là vùng cần sửa)."""
    short = name_en.split("(")[0].strip()
    p = part_noun(attr_en)
    if p is None:
        return [f"a {short}"]
    if p in _STANDALONE:
        return [f"a {p}", f"a {short}"]
    return [f"a {p}", f"the {p} of a {short}", f"a {short}"]


def part_query(attr_en: str, name_en: str) -> str:
    return part_queries(attr_en, name_en)[0]


def can_inpaint(model_key: str) -> bool:
    from ..models.registry import get as get_model, parse_key

    try:
        return get_model(parse_key(model_key)[0]).family in ("sdxl", "sd15")
    except KeyError:
        return False


def _mask_from_box(size: tuple[int, int], box: tuple[int, int, int, int], grow: float = 0.18):
    from PIL import Image, ImageDraw, ImageFilter

    W, H = size
    x0, y0, x1, y1 = box
    dx, dy = int((x1 - x0) * grow), int((y1 - y0) * grow)
    x0, y0, x1, y1 = max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle([x0, y0, x1, y1], fill=255)
    return m.filter(ImageFilter.GaussianBlur(max(4, (x1 - x0) // 24))), (x0, y0, x1, y1)


def inpaint_fix(best_path: str, attr_en: str, name_en: str, gen: GenSpec, model_key: str, cfg, out_dir: Path,
                iteration: int, clip=None, itm=None, spec=None, prompt_en: str = "", n: int = 3, log=print, agent=None) -> ModelRun | None:
    """Trả ModelRun với n ảnh đã inpaint vùng thuộc tính thiếu; None nếu không xác định được vùng hoặc họ model không hỗ trợ."""
    from PIL import Image

    from ..models import loader as model_loader
    from ..models.registry import get as get_model, parse_key
    from ..stages import multigen as mg
    from ..stages.refcrop import detect_owlvit

    if not can_inpaint(model_key):
        return None
    base_key = parse_key(model_key)[0]
    mspec = get_model(base_key)
    img = Image.open(best_path).convert("RGB")
    short = name_en.split("(")[0].strip()
    pn = part_noun(attr_en)
    det, q, how = None, "", ""
    # 1) VLM grounding (Qwen2.5-VL): hiểu cụm mô tả, không phải bag-of-words như CLIP của OWL-ViT
    if agent is not None and hasattr(agent, "locate"):
        labels = [x for x in ([pn] if pn else []) + [short] if x]
        rows = agent.locate(best_path, labels)
        if rows:
            W, H = img.size
            def _area(r):
                x0, y0, x1, y1 = r["bbox"]
                return (x1 - x0) * (y1 - y0) / (W * H)
            # ưu tiên hộp của BỘ PHẬN nếu có và không quá lớn; nếu không thì hộp thực thể cha
            part_rows = [r for r in rows if pn and pn in (r.get("label") or "").lower() and 0.002 <= _area(r) <= 0.6]
            pick = (part_rows or [r for r in rows if 0.01 <= _area(r) <= 0.98] or rows)[0]
            det, q, how = (tuple(pick["bbox"]), 0.99), pick.get("label") or labels[0], "VLM grounding"
    # 2) OWL-ViT với DANH TỪ NGẮN, cuối cùng là chính thực thể (SLD: box cha vẫn là vùng sửa được)
    if det is None:
        for q in part_queries(attr_en, name_en):
            det = detect_owlvit(img, q, device=getattr(cfg.multigen, "device", "cuda:0"), min_score=0.06, min_area=0.005)
            if det is not None:
                how = "OWL-ViT"
                break
    if det is None:
        log(f"  [inpaint] không tìm được vùng cho '{attr_en[:40]}' ({q}) -> bỏ nấc inpaint")
        return None
    box, sc = det
    mask, box2 = _mask_from_box(img.size, box)
    out = Path(out_dir) / "revision" / f"iter{iteration}" / "inpaint"
    out.mkdir(parents=True, exist_ok=True)
    mask.save(out / "mask.png")
    # DiffEdit/SLD: prompt vùng = thực thể + THUỘC TÍNH CẦN CÓ (thuộc tính nằm ở prompt, không ở câu truy vấn định vị)
    prompt = f"{short} with {attr_en}, {prompt_en}".strip(", ")
    negative = gen.negative_prompt if mspec.negative_ok else None
    t0 = time.time()
    run = ModelRun(model_key=f"{base_key}+inpaint", repo=mspec.repo, gen_spec=gen)
    pipe = None
    try:
        from diffusers import AutoPipelineForInpainting

        pipe = model_loader.load_pipeline(mspec, cfg.multigen.device, cfg.multigen.cpu_offload, log=log,
                                          scheduler=getattr(cfg.multigen, "scheduler", None))
        ip = AutoPipelineForInpainting.from_pipe(pipe)
        ip.set_progress_bar_config(disable=True)
        import torch

        W, H = img.size
        w8, h8 = (W // 8) * 8, (H // 8) * 8
        cands = []
        for i in range(n):
            seed = gen.seed + 1000 * iteration + 500 + i
            g = torch.Generator(device="cpu").manual_seed(seed)
            kw = dict(prompt=prompt, image=img.resize((w8, h8)), mask_image=mask.resize((w8, h8)), width=w8, height=h8,
                      num_inference_steps=max(20, gen.steps), guidance_scale=gen.guidance, strength=0.85, generator=g)
            if negative:
                kw["negative_prompt"] = negative
            res = ip(**kw).images[0]
            path = out / f"{gen.prompt_id}_{base_key}_inpaint_c{i}.png"
            res.save(path)
            cands.append(Candidate(str(path), seed, model_id=f"{base_key}+inpaint"))
        run.output = GenOutput(gen.prompt_id, iteration, cands, chosen=0, oracle=None)
        run.notes.append(f"inpaint vùng '{q}' box={box2} ({how} {sc:.2f}), strength 0.85, {n} ảnh từ {Path(best_path).name}")
        if spec is not None and clip is not None:
            mg.score_run(run, spec, clip, itm, prompt_en)
        log(f"  [inpaint] {n} ảnh, vùng '{q}' {box2} theo {how}, {time.time() - t0:.0f}s")
    except Exception as exc:  # noqa: BLE001
        run.error = f"inpaint lỗi: {type(exc).__name__}: {str(exc)[:120]}"
        log(f"  [inpaint] {run.error}")
    finally:
        if pipe is not None:
            try:
                model_loader.unload(pipe, log=log)
            except Exception:  # noqa: BLE001
                pass
    run.seconds = round(time.time() - t0, 1)
    return run
