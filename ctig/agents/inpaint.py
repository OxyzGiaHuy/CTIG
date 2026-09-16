"""
Refiner sửa CỤC BỘ (v1.9, theo SLD / GenArtist / Marmot): ảnh đã đúng phần lớn nhưng thiếu đúng MỘT thuộc tính -> tìm vùng, làm
mặt nạ, inpaint vùng đó với prompt chứa THUỘC TÍNH CẦN CÓ (DiffEdit), giữ nguyên phần còn lại.

Thang định vị (không hệ thống nào định vị được thuộc tính trừu tượng, tất cả định vị ĐỐI TƯỢNG rồi suy ra vùng):
  1. Qwen2.5-VL grounding hỏi cả bộ phận lẫn thực thể cha (bbox pixel, quy đổi image_grid_thw*14);
  2. crop-then-ground (Marmot / R-VLM): định vị thực thể cha -> crop + phóng to -> hỏi lại bộ phận TRONG crop -> map ngược;
     độ lệch giữa hộp toàn ảnh và hộp zoom là tín hiệu loại hộp rác;
  3. OWL-ViT với DANH TỪ NGẮN ("collar", không phải "the collar of a ao dai");
  4. box THỰC THỂ CHA (SLD: bộ phận đang thiếu thì không tìm được, box cha vẫn là vùng sửa được).
Hộp thô luôn được SAM tinh chỉnh thành mask (3 mask whole/part/subpart, chọn mask hợp lý nhất trong hộp).

Đầu ra là một ModelRun như Refiner thường (ảnh vào revision/iter<n>/inpaint/), để Reviewer chấm lại và pool giữ hết.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..schema import Candidate, GenOutput, GenSpec, ModelRun

_SAM: dict = {}

#: DANH TỪ NGẮN chỉ bộ phận (GenArtist dùng từ vựng 7.605 danh từ đơn; cụm mô tả dài bị CLIP của OWL-ViT nuốt còn tên thực thể).
_PARTS = ("collar", "neckline", "sleeve", "sleeves", "trousers", "pants", "sash", "belt", "waist", "panel", "panels", "skirt", "hem",
          "slit", "brim", "strap", "tip", "bow", "stern", "hull", "oar", "paddle", "button", "buttons", "placket",
          "cuff", "headscarf", "turban", "hat", "lantern", "tray", "envelope", "envelopes", "tree", "blossom", "blossoms")
#: bộ phận đứng riêng được (không cần gắn với thực thể cha)
_STANDALONE = {"tree", "lantern", "tray", "envelope", "envelopes", "blossom", "blossoms", "hat", "oar", "paddle"}

#: Bộ phận NHỎ và xác định rõ, vẽ lại một hộp là hợp lý (v1.9.5). Những phần còn lại của _PARTS (quần, tà, váy, gấu,
#: khe xẻ) là THÂN trang phục: sửa chúng nghĩa là vẽ lại gần hết bộ đồ, đó là việc của nấc sinh lại chứ không phải inpaint.
#: Bằng chứng S001/sdxl_base v192: thuộc tính "fitted bodice with flowing loose panels" -> định vị 'panels' ra vùng ống
#: chân -> vẽ lại biến khe xẻ và quần riêng (vốn ĐÚNG) thành váy liền. Ba ảnh inpaint đều thấp hơn ảnh gốc (+0,64 / +0,31
#: / +0,66 so với +0,84).
_LOCAL_PARTS = {"collar", "neckline", "sleeve", "sleeves", "cuff", "sash", "belt", "waist", "brim", "strap",
                "button", "buttons", "placket", "headscarf", "turban", "hat", "lantern", "tray", "envelope",
                "envelopes", "tree", "blossom", "blossoms", "oar", "paddle", "bow", "stern", "hull", "tip"}


def part_nouns_all(attr_en: str) -> set[str]:
    """MỌI danh từ bộ phận có trong thuộc tính. part_noun() chỉ trả cái dài nhất, không đủ cho cụm ghép."""
    low = attr_en.lower()
    return {p for p in _PARTS if p in low}


def locally_fixable(attr_en: str) -> bool:
    """Thuộc tính này có sửa được bằng cách vẽ lại MỘT vùng nhỏ không? Dáng tổng thể và thân áo thì không.

    Phải xét MỌI danh từ bộ phận, không chỉ cái đầu tiên: "long-sleeved tunic split at the hips into front and back
    panels" có cả 'sleeve' (nhỏ) lẫn 'panels' (thân áo); chỗ thiếu là tà chứ không phải tay áo, nên vẽ lại tay áo
    không sửa được gì. Chỉ nhận khi TẤT CẢ bộ phận được nhắc đều là bộ phận nhỏ.
    """
    ps = part_nouns_all(attr_en)
    return bool(ps) and ps <= _LOCAL_PARTS


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


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix, iy = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def crop_then_ground(agent, img, parent_box, part_noun_en: str, min_frac: float = 0.002, log=print):
    """Giai đoạn 2 (Marmot / R-VLM): crop theo hộp thực thể cha, phóng to, hỏi lại VỊ TRÍ BỘ PHẬN bên trong crop, rồi map
    toạ độ ngược về ảnh gốc. Trả (box_gốc, nguồn) hoặc None."""
    import tempfile

    if agent is None or not hasattr(agent, "locate") or not part_noun_en:
        return None
    x0, y0, x1, y1 = parent_box
    W, H = img.size
    pad_x, pad_y = int((x1 - x0) * 0.05), int((y1 - y0) * 0.05)
    cx0, cy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    cx1, cy1 = min(W, x1 + pad_x), min(H, y1 + pad_y)
    if cx1 - cx0 < 32 or cy1 - cy0 < 32:
        return None
    crop = img.crop((cx0, cy0, cx1, cy1))
    scale = max(1.0, 768 / max(crop.size))
    if scale > 1.0:
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        crop.save(f.name)
        rows = agent.locate(f.name, [part_noun_en])
    if not rows:
        return None
    cw, ch = crop.size
    best = None
    for r in rows:
        bx0, by0, bx1, by1 = r["bbox"]
        frac = ((bx1 - bx0) * (by1 - by0)) / max(1, cw * ch)
        if frac < min_frac or frac > 0.9:
            continue
        g = (cx0 + int(bx0 / scale), cy0 + int(by0 / scale), cx0 + int(bx1 / scale), cy0 + int(by1 / scale))
        if best is None or (g[2] - g[0]) * (g[3] - g[1]) > (best[2] - best[0]) * (best[3] - best[1]):
            best = g
    if best is None:
        return None
    log(f"  [inpaint] crop-then-ground tìm '{part_noun_en}' trong hộp cha -> {best}")
    return best, "crop-then-ground"


def sam_refine(img, box, device: str = "cuda:0", log=print):
    """Tinh chỉnh hộp thô thành mask bằng SAM (facebook/sam-vit-base, ~375 MB). SAM trả 3 mask whole/part/subpart; chọn mask có
    diện tích hợp lý nhất so với hộp (0,15-1,0 lần) để không lấy nhầm mức quá nhỏ hay tràn ra ngoài. Lỗi -> None (dùng hộp)."""
    try:
        import numpy as np
        import torch
        from transformers import SamModel, SamProcessor

        if "m" not in _SAM:
            dev = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
            _SAM["p"] = SamProcessor.from_pretrained("facebook/sam-vit-base")
            _SAM["m"] = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()
            _SAM["dev"] = dev
        proc, model, dev = _SAM["p"], _SAM["m"], _SAM["dev"]
        inputs = proc(img, input_boxes=[[list(box)]], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model(**inputs, multimask_output=True)
        masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                        inputs["reshaped_input_sizes"].cpu())[0][0].numpy()
        box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        best, best_score = None, -1.0
        for m in masks:
            a = float(m.sum())
            frac = a / box_area
            if not (0.15 <= frac <= 1.2):
                continue
            if frac > best_score:
                best, best_score = m, frac
        if best is None:
            return None
        from PIL import Image as _I

        return _I.fromarray((best.astype("uint8") * 255))
    except Exception as exc:  # noqa: BLE001
        log(f"  [inpaint] SAM không tinh chỉnh được ({type(exc).__name__}: {str(exc)[:60]}) -> dùng hộp")
        return None


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
    W, H = img.size
    short = name_en.split("(")[0].strip()
    pn = part_noun(attr_en)
    dev = getattr(cfg.multigen, "device", "cuda:0")
    box, q, how, parent_box, part_box_full = None, "", "", None, None

    def _frac(b):
        return (b[2] - b[0]) * (b[3] - b[1]) / (W * H)

    # --- nấc 1: VLM grounding toàn ảnh (bộ phận + thực thể cha trong một lượt) ---
    if agent is not None and hasattr(agent, "locate"):
        labels = [x for x in ([pn] if pn else []) + [short] if x]
        for r in agent.locate(best_path, labels):
            lab, b = (r.get("label") or "").lower(), tuple(r["bbox"])
            if pn and pn in lab and 0.002 <= _frac(b) <= 0.6 and part_box_full is None:
                part_box_full = b
            elif short.lower().split()[0] in lab and 0.01 <= _frac(b) <= 0.98 and parent_box is None:
                parent_box = b
        if parent_box is None and part_box_full is None:
            pass
        elif parent_box is None:
            parent_box = part_box_full
    # --- nấc 2: crop-then-ground trong hộp cha (Marmot / R-VLM) ---
    if pn and parent_box is not None:
        cg = crop_then_ground(agent, img, parent_box, pn, log=log)
        if cg is not None:
            zoom_box, _ = cg
            if part_box_full is not None and _iou(zoom_box, part_box_full) < 0.1:
                log(f"  [inpaint] hộp zoom lệch hẳn hộp toàn ảnh (IoU {_iou(zoom_box, part_box_full):.2f}) -> tin hộp zoom")
            box, q, how = zoom_box, pn, "crop-then-ground"
    if box is None and part_box_full is not None:
        box, q, how = part_box_full, pn or short, "VLM grounding"
    # --- nấc 3: OWL-ViT với danh từ ngắn ---
    if box is None:
        for qq in part_queries(attr_en, name_en):
            det = detect_owlvit(img, qq, device=dev, min_score=0.06, min_area=0.005)
            if det is not None:
                box, q, how = det[0], qq, "OWL-ViT"
                break
    # --- nấc 4: hộp thực thể cha (SLD) ---
    if box is None and parent_box is not None:
        box, q, how = parent_box, short, "hộp thực thể cha (SLD)"
    if box is None:
        log(f"  [inpaint] không tìm được vùng cho '{attr_en[:40]}' -> bỏ nấc inpaint")
        return None
    sc = 0.99
    # --- luôn cho SAM tinh chỉnh hộp thành mask ---
    mask = sam_refine(img, box, device=dev, log=log)
    if mask is not None:
        from PIL import ImageFilter

        bb = mask.getbbox() or box
        mask = mask.filter(ImageFilter.GaussianBlur(max(4, (bb[2] - bb[0]) // 24)))
        box2 = bb
        how += " + SAM"
    else:
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
