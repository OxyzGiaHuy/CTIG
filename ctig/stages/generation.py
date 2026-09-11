"""
Stage 4 - GEN.

Hai ô "D D" trong sơ đồ được hiện thực thành sinh N ứng viên (khác seed) rồi chọn
ứng viên tốt nhất theo CLIP ở stage review. Ba đòn can thiệp vòng review có thể
dùng, ánh xạ sang SDXL thật:

  negative prompt      -> `negative_prompt`
  tăng conditioning    -> đẩy tên thực thể lên đầu prompt, lặp lại, tăng guidance
  gắn LoRA             -> `load_lora_weights` (chỉ khi cấu hình có lora_path)
  ảnh tham chiếu       -> IP-Adapter với ảnh Commons đã qua kiểm CLIP

Bộ sinh stub giữ lại để test offline và làm baseline có "oracle".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..kb import KnowledgeBase, normalize
from ..schema import Candidate, CulturalSpec, GenOutput, GenSpec, Prompt, RevisionPlan

GENERIC_NEGATIVE = "blurry, deformed, extra limbs, text, watermark, logo, low quality, cartoon"
STYLE_SUFFIX = "photograph, natural lighting, highly detailed, realistic"


def _en(se) -> str:
    return se.name_en.split("(")[0].strip()


def build_initial_spec(prompt: Prompt, spec: CulturalSpec, prompt_en: str | None,
                       cfg, seed: int, init_negatives: bool = True) -> GenSpec:
    parts = [prompt_en or prompt.text_en]
    for se in spec.entities:
        parts.append(f"Vietnamese {_en(se)}")
        attrs = se.required_attrs_en or []
        if se.weight >= 0.8 and attrs:
            parts.extend(attrs[:2])
    parts.extend(spec.scene_notes[:2])
    parts.append(STYLE_SUFFIX)

    neg = [GENERIC_NEGATIVE]
    if init_negatives:
        from ..llm.shared import confusable_labels

        for se in spec.entities:
            for cf in se.confusables[:2]:
                neg.extend(confusable_labels(cf.get("name", "")))
    ref = next((se.reference_image for se in spec.entities if se.reference_image), None)
    return GenSpec(
        prompt_id=prompt.id, prompt=", ".join(dict.fromkeys(p for p in parts if p)),
        negative_prompt=", ".join(dict.fromkeys(n for n in neg if n)),
        conditioning={se.entity_id: 0.0 for se in spec.entities},
        lora=None, lora_scale=cfg.lora_scale,
        ip_adapter_image=ref if cfg.ip_adapter and getattr(cfg, "ip_adapter_from_start", False) else None,
        ip_adapter_scale=cfg.ip_adapter_scale,
        seed=seed, steps=cfg.steps, guidance=cfg.guidance,
        width=cfg.width, height=cfg.height, n_candidates=cfg.n_candidates, iteration=0,
    )


def apply_plan(gen: GenSpec, plan: RevisionPlan, spec: CulturalSpec, cfg, lora_id: str | None = None) -> GenSpec:
    """`lora_id` là LoRA mà bộ sinh thực sự có (SDXL: cfg.lora_path đã nạp; stub: LoRA ảo)."""
    cond = dict(gen.conditioning)
    for eid, d in plan.boost.items():
        cond[eid] = min(1.0, cond.get(eid, 0.0) + d)
    # Thực thể được nhấn mạnh đi lên đầu prompt và được lặp lại.
    emphasised = [spec.entity(eid) for eid, v in sorted(cond.items(), key=lambda kv: -kv[1]) if v >= 0.3]
    head = [f"Vietnamese {_en(se)}, {_en(se)}" for se in emphasised if se]
    prompt = ", ".join(dict.fromkeys(head + [gen.prompt] + plan.add_positive))
    negative = ", ".join(dict.fromkeys([gen.negative_prompt] + plan.add_negative))
    ref = gen.ip_adapter_image
    if plan.use_reference_image and cfg.ip_adapter:
        ref = next((se.reference_image for se in spec.entities if se.reference_image), None)
    return GenSpec(
        prompt_id=gen.prompt_id, prompt=prompt, negative_prompt=negative, conditioning=cond,
        lora=(lora_id if (plan.attach_lora and lora_id) else gen.lora),
        lora_scale=gen.lora_scale, ip_adapter_image=ref, ip_adapter_scale=gen.ip_adapter_scale,
        seed=gen.seed, steps=gen.steps,
        guidance=min(12.0, gen.guidance + plan.guidance_delta + 0.5 * max(cond.values(), default=0)),
        width=gen.width, height=gen.height, n_candidates=gen.n_candidates, iteration=gen.iteration + 1,
    )


# =============================================================== SDXL

class SDXLGenerator:
    name = "sdxl"

    def __init__(self, cfg):
        import torch
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        self.cfg = cfg
        self.torch = torch
        vae = AutoencoderKL.from_pretrained(cfg.vae, torch_dtype=torch.float16) if cfg.vae else None
        kwargs = dict(torch_dtype=torch.float16, use_safetensors=True)
        if vae is not None:
            kwargs["vae"] = vae
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model, variant="fp16", **kwargs)
        except (OSError, ValueError):
            self.pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model, **kwargs)
        if cfg.cpu_offload:
            try:
                self.pipe.enable_model_cpu_offload(device=cfg.device)
            except TypeError:
                self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(cfg.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.has_ip = False
        if cfg.ip_adapter:
            try:
                self.pipe.load_ip_adapter(cfg.ip_adapter_repo, subfolder="sdxl_models",
                                          weight_name=cfg.ip_adapter_weight)
                self.pipe.set_ip_adapter_scale(0.0)
                self.has_ip = True
            except Exception as exc:  # noqa: BLE001
                print(f"[gen] không tải được IP-Adapter, tắt: {type(exc).__name__}: {exc}")

        self.has_lora = False
        if cfg.lora_path:
            try:
                self.pipe.load_lora_weights(cfg.lora_path, adapter_name="culture")
                self.pipe.set_adapters(["culture"], adapter_weights=[0.0])
                self.has_lora = True
            except Exception as exc:  # noqa: BLE001
                print(f"[gen] không tải được LoRA, tắt: {type(exc).__name__}: {exc}")

    @property
    def lora_available(self) -> bool:
        return self.has_lora

    @property
    def lora_id(self) -> str | None:
        return self.cfg.lora_path if self.has_lora else None

    @property
    def reference_available(self) -> bool:
        return self.has_ip

    def generate(self, gen: GenSpec, spec: CulturalSpec, kb: KnowledgeBase, out_dir: Path) -> GenOutput:
        from PIL import Image

        out_dir.mkdir(parents=True, exist_ok=True)
        if self.has_lora:
            self.pipe.set_adapters(["culture"], adapter_weights=[gen.lora_scale if gen.lora else 0.0])
        ip_kwargs = {}
        if self.has_ip:
            if gen.ip_adapter_image:
                self.pipe.set_ip_adapter_scale(gen.ip_adapter_scale)
                ip_kwargs["ip_adapter_image"] = Image.open(gen.ip_adapter_image).convert("RGB")
            else:
                # IP-Adapter đã nạp thì pipeline đòi ảnh; đưa ảnh trung tính với scale 0.
                self.pipe.set_ip_adapter_scale(0.0)
                ip_kwargs["ip_adapter_image"] = Image.new("RGB", (224, 224), (128, 128, 128))

        cands = []
        for i in range(gen.n_candidates):
            seed = gen.seed + 1000 * gen.iteration + i
            g = self.torch.Generator(device="cpu").manual_seed(seed)
            img = self.pipe(
                prompt=gen.prompt, negative_prompt=gen.negative_prompt or None,
                num_inference_steps=gen.steps, guidance_scale=gen.guidance,
                width=gen.width, height=gen.height, generator=g, **ip_kwargs,
            ).images[0]
            path = out_dir / f"{gen.prompt_id}_iter{gen.iteration}_c{i}.png"
            img.save(path)
            cands.append(Candidate(str(path), seed))
        return GenOutput(gen.prompt_id, gen.iteration, cands, chosen=0, oracle=None)


# =============================================================== stub

T_CORRECT, T_DRIFT = 0.55, 0.30
STUB_LORA_COVERAGE = {
    "dan_bau", "dan_tranh", "tranh_dong_ho", "mua_roi_nuoc", "nha_rong", "thuyen_thung",
    "com_lang_vong", "banh_tet", "banh_chung", "non_quai_thao", "ao_tu_than", "quan_ho",
    "nha_nhac_hue", "cong_chieng_tay_nguyen", "trang_phuc_dao_do", "dinh_lang", "trung_thu", "tet_nguyen_dan",
}


class StubGenerator:
    """Mô phỏng thiên lệch văn hoá của T2I. Xem docs/ARCHITECTURE.md mục 'Stub'."""

    name = "stub"
    lora_available = True
    reference_available = True
    lora_id = "stub-culture-lora"

    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, gen: GenSpec, spec: CulturalSpec, kb: KnowledgeBase, out_dir: Path) -> GenOutput:
        out_dir.mkdir(parents=True, exist_ok=True)
        neg = normalize(gen.negative_prompt)
        pos = normalize(gen.prompt)
        oracle: dict[str, str] = {}
        strengths: dict[str, float] = {}
        for se in spec.entities:
            ent = kb.get(se.entity_id)
            if ent is None:
                continue
            s = ent.prior_strength + 0.55 * gen.conditioning.get(se.entity_id, 0.0)
            if gen.lora and se.entity_id in STUB_LORA_COVERAGE:
                s += 0.40
            if gen.ip_adapter_image and se.reference_image:
                s += 0.25
            if normalize(_en(se)) in pos:
                s += 0.10
            cf = ent.primary_confusable
            if cf:
                from ..llm.shared import confusable_labels

                blocked = any(normalize(v) in neg for v in confusable_labels(cf["name"]))
                s -= 0.0 if blocked else 0.25
            h = hashlib.sha256(f"{gen.seed}:{gen.iteration}:{se.entity_id}".encode()).digest()[0]
            s += (h / 255 - 0.5) * 0.12
            strengths[se.entity_id] = round(s, 3)
            if s >= T_CORRECT:
                oracle[se.entity_id] = se.name_vi
            elif s >= T_DRIFT and cf:
                oracle[se.entity_id] = cf["name"]
            else:
                oracle[se.entity_id] = "<không vẽ>"
        path = out_dir / f"{gen.prompt_id}_iter{gen.iteration}_c0.png"
        _draw_card(path, gen, spec, oracle, strengths)
        return GenOutput(gen.prompt_id, gen.iteration, [Candidate(str(path), gen.seed)], 0, oracle)


def _font(size, bold=False):
    from PIL import ImageFont

    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _draw_card(path, gen, spec, oracle, strengths):
    from PIL import Image, ImageDraw

    W, pad = 720, 24
    img = Image.new("RGB", (W, 200 + 100 * max(1, len(spec.entities))), (250, 249, 246))
    d = ImageDraw.Draw(img)
    y = pad
    d.text((pad, y), f"{gen.prompt_id} · vòng {gen.iteration} · STUB", font=_font(20, True), fill=(28, 30, 34)); y += 30
    d.text((pad, y), gen.prompt[:95], font=_font(12), fill=(120, 124, 132)); y += 18
    d.text((pad, y), f"LoRA: {'có' if gen.lora else 'không'} · ref: {'có' if gen.ip_adapter_image else 'không'} · guidance {gen.guidance:.1f}",
           font=_font(11), fill=(120, 124, 132)); y += 24
    for se in spec.entities:
        drawn = oracle.get(se.entity_id, "<không vẽ>")
        col = (22, 128, 82) if drawn == se.name_vi else (150, 105, 20) if drawn == "<không vẽ>" else (196, 48, 43)
        d.rectangle([pad, y, W - pad, y + 80], fill=(238, 236, 230)); d.rectangle([pad, y, pad + 4, y + 80], fill=col)
        d.text((pad + 14, y + 8), f"yêu cầu: {se.name_vi}", font=_font(14, True), fill=(28, 30, 34))
        d.text((pad + 14, y + 30), f"vẽ ra: {drawn}", font=_font(13), fill=col)
        s = strengths.get(se.entity_id, 0.0)
        bx0, bx1, by = pad + 14, W - pad - 120, y + 56
        d.rectangle([bx0, by, bx1, by + 8], fill=(214, 212, 206))
        d.rectangle([bx0, by, bx0 + int((bx1 - bx0) * min(1, max(0, s))), by + 8], fill=col)
        d.text((bx1 + 10, by - 3), f"strength {s:.2f}", font=_font(11), fill=(120, 124, 132))
        y += 90
    d.text((pad, y + 6), "Thẻ chẩn đoán của bộ sinh mô phỏng, không phải ảnh diffusion.", font=_font(11), fill=(120, 124, 132))
    img.crop((0, 0, W, y + 30)).save(path)


def get_generator(cfg):
    if cfg.backend == "sdxl":
        return SDXLGenerator(cfg)
    if cfg.backend == "stub":
        return StubGenerator(cfg)
    raise ValueError(f"t2i backend không rõ: {cfg.backend!r}")
