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


_VIET_CULTURE_MARKERS = ("việt", "viet", "vietnam")


def _forbidden_negatives(se, max_words: int = 8, limit: int = 3) -> list[str]:
    """must_not_en (KB viết tay) đưa vào negative prompt: "one-piece dress with no trousers underneath".

    Chỉ cụm tiếng Anh ngắn, ASCII. Đây là giả thuyết H9: negative theo thuộc tính giảm lỗi
    "áo dài không quần" mà tên confusable (kimono, qipao) không chặn được (v1.2.1 p001: 4/12 ảnh).
    """
    out = []
    for a in se.forbidden_attrs_en:
        a = (a or "").strip()
        if a and a.isascii() and len(a.split()) <= max_words:
            out.append(a)
    return out[:limit]


def _confusable_negatives(se) -> list[str]:
    """Tên confusable đưa vào negative prompt.

    Lần chạy v1.1: negative của p050 chứa "áo dài, vietnamese, tunic, trousers, shirt" vì
    (a) confusable của áo bà ba trong KB là "áo dài" (phân biệt NỘI BỘ Việt, hữu ích cho CLIP
    nhưng độc hại làm negative), (b) hàm này băm name_en thành từng từ. Nay:
      * chỉ confusable có culture KHÔNG phải Việt,
      * chỉ tên ASCII ngắn (kimono, qipao, zongzi, hanfu...), không băm câu mô tả.
    """
    from ..llm.shared import confusable_labels

    out = []
    for cf in se.confusables[:4]:
        culture = (cf.get("culture") or "").lower()
        if any(m in culture for m in _VIET_CULTURE_MARKERS):
            continue
        for lab in confusable_labels(cf.get("name", "")):
            lab = lab.strip()
            # tên trần ASCII, tối đa 3 từ; bỏ tên có dấu tiếng Việt (SDXL không đọc được, và có thể trùng tên Việt)
            if lab and lab.isascii() and len(lab.split()) <= 3:
                out.append(lab.lower())
    return out


def build_initial_spec(prompt: Prompt, spec: CulturalSpec, prompt_en: str | None,
                       cfg, seed: int, init_negatives: bool = True) -> GenSpec:
    n_attrs = int(getattr(cfg, "attrs_in_prompt", 3))
    terms = [prompt_en or prompt.text_en]
    for se in spec.entities:
        terms.append(f"Vietnamese {_en(se)}")
        en_attrs = [a for a in se.required_attrs_en if a]  # CHỈ tiếng Anh; cụm chưa dịch ("") bị bỏ
        if se.weight >= 0.8 and en_attrs:
            # v1.2.1 p001: chỉ 2 thuộc tính nên "worn over wide-legged trousers" (thứ 3) bị cắt,
            # và 4/12 ảnh ra váy xẻ tà không quần. Mặc định 3.
            terms.extend(en_attrs[:n_attrs])
    # scene_notes là tiếng Việt (từ analysis); SDXL không đọc được, prompt_en đã chứa bối cảnh.
    terms.extend(n for n in spec.scene_notes[:2] if n.isascii())
    terms.append(STYLE_SUFFIX)

    neg = GENERIC_NEGATIVE.split(", ")
    if init_negatives:
        for se in spec.entities:
            neg.extend(_confusable_negatives(se))
            if se.weight >= 0.8:
                neg.extend(_forbidden_negatives(se))
    fast = bool(getattr(cfg, "fast_iters", False))
    return GenSpec(
        prompt_id=prompt.id, prompt_terms=list(dict.fromkeys(t for t in terms if t)),
        negative_terms=list(dict.fromkeys(n for n in neg if n)), emphasis={},
        conditioning={se.entity_id: 0.0 for se in spec.entities},
        lora=None, lora_scale=cfg.lora_scale, ip_adapter_image=None, ip_adapter_scale=cfg.ip_adapter_scale,
        seed=seed, steps=(cfg.fast_steps if fast else cfg.steps),
        guidance=(cfg.fast_guidance if fast else cfg.guidance),
        width=cfg.width, height=cfg.height, n_candidates=cfg.n_candidates, iteration=0, fast=fast,
    )


def apply_plan(gen: GenSpec, plan: RevisionPlan, spec: CulturalSpec, cfg, lora_id: str | None = None) -> GenSpec:
    """`lora_id` là LoRA mà bộ sinh thực sự có (SDXL: cfg.lora_path đã nạp; stub: LoRA ảo).

    v1 nối chuỗi nên mỗi vòng lại thêm "Vietnamese Ao dai, Ao dai" -> nhấn quá tay, nón lá neon
    khổng lồ. v1.1 giữ danh sách cụm, mỗi thực thể chỉ được đẩy lên đầu MỘT lần.
    """
    cond = dict(gen.conditioning)
    for eid, d in plan.boost.items():
        cond[eid] = min(1.0, cond.get(eid, 0.0) + d)
    emphasis = dict(gen.emphasis)
    head: list[str] = []
    for eid, v in sorted(cond.items(), key=lambda kv: -kv[1]):
        se = spec.entity(eid)
        if se and v >= 0.3 and emphasis.get(eid, 0) < 1:
            head.append(f"Vietnamese {_en(se)}")
            emphasis[eid] = 1
    # Cụm đã nhấn được rút khỏi vị trí cũ rồi đặt lên đầu, không nhân đôi.
    body = [t for t in gen.prompt_terms if t not in head]
    terms = list(dict.fromkeys(head + body + [p for p in plan.add_positive if p]))
    negative = list(dict.fromkeys(gen.negative_terms + [n for n in plan.add_negative if n]))
    ref = gen.ip_adapter_image
    if plan.use_reference_image and cfg.ip_adapter:
        ref = next((se.reference_image for se in spec.entities if se.reference_image and se.kind == "object"), None)
    guidance = gen.guidance if gen.fast else min(12.0, gen.guidance + plan.guidance_delta + 0.5 * max(cond.values(), default=0))
    return GenSpec(
        prompt_id=gen.prompt_id, prompt_terms=terms, negative_terms=negative, emphasis=emphasis, conditioning=cond,
        lora=(lora_id if (plan.attach_lora and lora_id) else gen.lora),
        lora_scale=gen.lora_scale, ip_adapter_image=ref, ip_adapter_scale=gen.ip_adapter_scale,
        seed=gen.seed, steps=gen.steps, guidance=guidance,
        width=gen.width, height=gen.height, n_candidates=gen.n_candidates, iteration=gen.iteration + 1, fast=gen.fast,
    )


def to_final_render(gen: GenSpec, cfg) -> GenSpec:
    """Cùng prompt, render đủ bước với scheduler chuẩn (sau khi vòng nhanh đã đạt)."""
    from dataclasses import replace

    return replace(gen, fast=False, steps=cfg.steps, guidance=cfg.guidance, n_candidates=1)


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
        # Giải mã VAE ở 1024px là đỉnh VRAM của SDXL; slicing/tiling hạ đỉnh này đáng kể.
        for fn in ("enable_vae_slicing", "enable_vae_tiling"):
            try:
                getattr(self.pipe, fn)()
            except Exception:  # noqa: BLE001
                pass

        self.has_ip = False
        if cfg.ip_adapter:
            try:
                self.pipe.load_ip_adapter(cfg.ip_adapter_repo, subfolder="sdxl_models",
                                          weight_name=cfg.ip_adapter_weight)
                self.pipe.set_ip_adapter_scale(0.0)
                self.has_ip = True
            except Exception as exc:  # noqa: BLE001
                print(f"[gen] không tải được IP-Adapter, tắt: {type(exc).__name__}: {exc}")

        self.adapters: list[str] = []
        self.has_lora = False
        if cfg.lora_path:
            try:
                self.pipe.load_lora_weights(cfg.lora_path, adapter_name="culture")
                self.adapters.append("culture")
                self.has_lora = True
            except Exception as exc:  # noqa: BLE001
                print(f"[gen] không tải được LoRA, tắt: {type(exc).__name__}: {exc}")

        # LCM-LoRA (skill SD, Workflow 2 "fast prototyping"): 4-8 bước cho vòng sửa.
        self.has_lcm = False
        self._default_scheduler = self.pipe.scheduler
        if getattr(cfg, "fast_iters", False):
            try:
                from diffusers import LCMScheduler

                self.pipe.load_lora_weights(cfg.lcm_lora, adapter_name="lcm")
                self.adapters.append("lcm")
                self._lcm_scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
                self.has_lcm = True
            except Exception as exc:  # noqa: BLE001
                print(f"[gen] không tải được LCM-LoRA, chạy đủ bước: {type(exc).__name__}: {exc}")
        if self.adapters:
            self.pipe.set_adapters(self.adapters, adapter_weights=[0.0] * len(self.adapters))

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
        use_lcm = gen.fast and self.has_lcm
        if self.adapters:
            weights = {"culture": (gen.lora_scale if (gen.lora and self.has_lora) else 0.0), "lcm": (1.0 if use_lcm else 0.0)}
            self.pipe.set_adapters(self.adapters, adapter_weights=[weights[a] for a in self.adapters])
        self.pipe.scheduler = self._lcm_scheduler if use_lcm else self._default_scheduler
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
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return GenOutput(gen.prompt_id, gen.iteration, cands, chosen=0, oracle=None)


# =============================================================== bộ sinh diffusers tổng quát (multigen)

class DiffusersGenerator:
    """Bọc một pipeline diffusers bất kỳ (SDXL, SD1.5, Turbo, Playground, SD3) với cùng interface generate().

    v1.3 thêm ba việc cho "ảnh cuối đẹp và chuẩn", tất cả có đường lùi để một lỗi không làm hỏng hàng:
      * prompt dài: SDXL/SD1.5 cắt ở 77 token CLIP mà không báo; >75 token thì nối embedding bằng `compel`
        (không có compel hoặc lỗi -> dùng prompt thô và ghi chú "bị cắt").
      * hires fix: phóng ảnh `scale` lần rồi img2img `strength` thấp cùng prompt (OOM -> giữ ảnh gốc).
      * IP-Adapter nhiều ảnh tham chiếu (Plus): truyền list lồng [[img1, img2, ...]] cho một adapter.
    `trigger` là từ khoá LoRA thêm vào đầu prompt.
    """

    name = "diffusers"
    lora_available = False
    reference_available = False
    lora_id = None

    def __init__(self, pipe, model_key: str, trigger: str | None = None, negative_ok: bool = True,
                 ip_adapter_image: str | list[str] | None = None, family: str = "sdxl",
                 long_prompt: bool = True, hires=None, log=print):
        import torch

        self.torch = torch
        self.pipe = pipe
        self.model_key = model_key
        self.trigger = trigger
        self.negative_ok = negative_ok
        self.ip_adapter_image = ip_adapter_image
        self.family = family
        self.long_prompt = long_prompt
        self.hires = hires
        self.log = log
        self.notes: list[str] = []
        self.prompt_tokens: int | None = None
        self._compel = None

    # ---- prompt ------------------------------------------------------------------------
    def count_tokens(self, text: str) -> int | None:
        tok = getattr(self.pipe, "tokenizer", None)
        if tok is None:
            return None
        try:
            return len(tok(text, truncation=False)["input_ids"])
        except Exception:  # noqa: BLE001
            return None

    def _compel_embeds(self, prompt: str, negative: str | None):
        """Trả kwargs prompt_embeds(+pooled) cho SDXL / SD1.5 qua compel; None nếu không làm được."""
        if self.family not in ("sdxl", "sd15"):
            return None
        try:
            from compel import Compel, ReturnedEmbeddingsType
        except ImportError:
            self.notes.append("prompt > 75 token nhưng thiếu gói compel -> pipeline cắt bớt cuối prompt")
            return None
        try:
            if self._compel is None:
                if self.family == "sdxl":
                    self._compel = Compel(
                        tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                        text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True], truncate_long_prompts=False)
                else:
                    self._compel = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder,
                                          truncate_long_prompts=False)
            esc = lambda s: s.replace("(", "\\(").replace(")", "\\)")  # compel coi () là trọng số
            neg = negative or ""
            if self.family == "sdxl":
                cond, pooled = self._compel(esc(prompt))
                ncond, npooled = self._compel(esc(neg) if neg else "")
                cond, ncond = self._compel.pad_conditioning_tensors_to_same_length([cond, ncond])
                return {"prompt_embeds": cond, "pooled_prompt_embeds": pooled,
                        "negative_prompt_embeds": ncond, "negative_pooled_prompt_embeds": npooled}
            cond = self._compel(esc(prompt))
            ncond = self._compel(esc(neg) if neg else "")
            cond, ncond = self._compel.pad_conditioning_tensors_to_same_length([cond, ncond])
            return {"prompt_embeds": cond, "negative_prompt_embeds": ncond}
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"compel lỗi ({type(exc).__name__}: {str(exc)[:80]}) -> dùng prompt thô, có thể bị cắt")
            self._compel = None
            return None

    def _raw_prompt_kwargs(self, gen: GenSpec) -> dict:
        prompt = f"{self.trigger}, {gen.prompt}" if self.trigger else gen.prompt
        kw = {"prompt": prompt}
        if self.negative_ok and gen.negative_prompt:
            kw["negative_prompt"] = gen.negative_prompt
        return kw

    def _prompt_kwargs(self, gen: GenSpec) -> dict:
        prompt = f"{self.trigger}, {gen.prompt}" if self.trigger else gen.prompt
        negative = gen.negative_prompt if (self.negative_ok and gen.negative_prompt) else None
        n = self.count_tokens(prompt)
        self.prompt_tokens = n
        if n is not None and n > 75 and self.long_prompt:
            emb = self._compel_embeds(prompt, negative)
            if emb is not None:
                if not any(x.startswith("prompt dài") for x in self.notes):
                    self.notes.append(f"prompt dài ({n} token) -> nối embedding bằng compel")
                return emb
        elif n is not None and n > 75:
            self.notes.append(f"prompt {n} token > 75, pipeline sẽ cắt phần cuối (multigen.long_prompt=false)")
        kw = {"prompt": prompt}
        if negative:
            kw["negative_prompt"] = negative
        return kw

    def _ip_kwargs(self) -> dict:
        if not self.ip_adapter_image:
            return {}
        from PIL import Image

        paths = self.ip_adapter_image if isinstance(self.ip_adapter_image, list) else [self.ip_adapter_image]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        # Một adapter, nhiều ảnh: diffusers nhận list lồng [[...]]; một ảnh thì truyền thẳng.
        return {"ip_adapter_image": imgs[0] if len(imgs) == 1 else [imgs]}

    # ---- sinh --------------------------------------------------------------------------
    def generate(self, gen: GenSpec, spec: CulturalSpec, kb: KnowledgeBase, out_dir: Path) -> GenOutput:
        out_dir.mkdir(parents=True, exist_ok=True)
        pk = self._prompt_kwargs(gen)
        ipk = self._ip_kwargs()
        if isinstance(ipk.get("ip_adapter_image"), list):
            self.notes.append(f"IP-Adapter {len(ipk['ip_adapter_image'][0])} ảnh tham chiếu")
        hires_pipe = None
        cands = []
        for i in range(gen.n_candidates):
            seed = gen.seed + 1000 * gen.iteration + i
            g = self.torch.Generator(device="cpu").manual_seed(seed)
            kwargs = dict(num_inference_steps=gen.steps, guidance_scale=gen.guidance,
                          width=gen.width, height=gen.height, generator=g, **pk, **ipk)
            try:
                img = self.pipe(**kwargs).images[0]
            except Exception as exc:  # noqa: BLE001 - hai đường lùi trước khi coi là lỗi hàng
                if isinstance(ipk.get("ip_adapter_image"), list):
                    # Nhiều ảnh IP-Adapter không được pipeline này nhận -> lùi về ảnh đầu, thử lại.
                    self.notes.append(f"nhiều ảnh IP-Adapter bị từ chối ({type(exc).__name__}: {str(exc)[:60]}) -> dùng 1 ảnh")
                    ipk = {"ip_adapter_image": ipk["ip_adapter_image"][0][0]}
                elif "prompt_embeds" in pk:
                    # Embedding compel bị pipeline từ chối (device/dtype/độ dài) -> prompt thô.
                    self.notes.append(f"prompt_embeds bị từ chối ({type(exc).__name__}: {str(exc)[:60]}) -> prompt thô, có thể bị cắt")
                    pk = self._raw_prompt_kwargs(gen)
                else:
                    raise
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                kwargs = dict(num_inference_steps=gen.steps, guidance_scale=gen.guidance, width=gen.width, height=gen.height,
                              generator=self.torch.Generator(device="cpu").manual_seed(seed), **pk, **ipk)
                img = self.pipe(**kwargs).images[0]
            path = out_dir / f"{gen.prompt_id}_{self.model_key}_c{i}.png"
            img.save(path)
            cand = Candidate(str(path), seed, model_id=self.model_key)
            if self.hires is not None and getattr(self.hires, "enabled", False) and not getattr(self, "_hires_failed", False):
                hires_pipe, hr_path = self._hires(img, path, seed, pk, ipk, hires_pipe)
                if hr_path is not None:
                    cand.base_path = str(path)
                    cand.path = str(hr_path)
            cands.append(cand)
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return GenOutput(gen.prompt_id, gen.iteration, cands, chosen=0, oracle=None)

    def _hires(self, img, path: Path, seed: int, pk: dict, ipk: dict, hires_pipe):
        """Phóng to + img2img strength thấp. Trả (pipe img2img để dùng lại, đường dẫn ảnh hires hoặc None)."""
        from PIL import Image

        h = self.hires
        try:
            if hires_pipe is None:
                from ..models.loader import img2img_from

                hires_pipe = img2img_from(self.pipe)
            W = max(8, int(img.width * h.scale) // 8 * 8)
            H = max(8, int(img.height * h.scale) // 8 * 8)
            up = img.resize((W, H), Image.LANCZOS)
            g = self.torch.Generator(device="cpu").manual_seed(seed)
            out = hires_pipe(image=up, strength=h.strength, num_inference_steps=h.steps, generator=g, **pk, **ipk).images[0]
            hr_path = path.with_name(path.stem + "_hr.png")
            out.save(hr_path)
            if not any(x.startswith("hires") for x in self.notes):
                self.notes.append(f"hires fix x{h.scale:g} strength {h.strength:g} -> {W}px")
            return hires_pipe, hr_path
        except Exception as exc:  # noqa: BLE001 - OOM ở bước phụ không được làm mất ảnh gốc
            self._hires_failed = True  # một lần OOM là đủ, các ứng viên sau không thử lại (tiết kiệm thời gian)
            msg = f"hires bỏ qua ({type(exc).__name__}: {str(exc)[:80]}), giữ ảnh gốc cho các ứng viên còn lại"
            if msg not in self.notes:
                self.notes.append(msg)
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            return hires_pipe, None


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
        # Nhấn (emphasis) trong stub tính như +0.15 conditioning một lần.
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
            if gen.emphasis.get(se.entity_id):
                s += 0.15
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
        tag = getattr(self, "model_key", None)
        cands = []
        for i in range(max(1, gen.n_candidates)):  # stub tôn trọng n_candidates để test best-of-N/cache đĩa
            path = out_dir / (f"{gen.prompt_id}_{tag}_c{i}.png" if tag else f"{gen.prompt_id}_iter{gen.iteration}_c{i}.png")
            _draw_card(path, gen, spec, oracle, strengths)
            cands.append(Candidate(str(path), gen.seed + 1000 * gen.iteration + i, model_id=tag))
        return GenOutput(gen.prompt_id, gen.iteration, cands, 0, oracle)


def _font(size, bold=False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    cands = [f"/usr/share/fonts/truetype/dejavu/{name}", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"]
    try:  # Kaggle không có font hệ thống có dấu tiếng Việt (grid v1.2.1 ra ô vuông); matplotlib kèm DejaVu
        import matplotlib
        cands.insert(0, str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name))
    except Exception:  # noqa: BLE001
        pass
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
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
