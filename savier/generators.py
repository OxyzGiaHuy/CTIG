"""Generators G. Not agents: they make no decisions.

    SDXLGen  Stable Diffusion XL 1.0 base · fp16 VAE fix · DPM++ 2M Karras · compel for prompts > 77 CLIP tokens
             · optional IP-Adapter Plus (ViT-H) with up to two reference photographs
    FluxGen  FLUX.1-dev bf16 · model CPU offload · optional XLabs IP-Adapter (one reference photograph)

No negative prompt is used anywhere (FLUX has none; SDXL is run the same way so that arms differ only in
the prompt and the reference conditioning). The same seed is used for the baseline and the repaired image.
"""
from __future__ import annotations

import time
from pathlib import Path


class SDXLGen:
    REPO, VAE = "stabilityai/stable-diffusion-xl-base-1.0", "madebyollin/sdxl-vae-fp16-fix"

    def __init__(self, device="cuda:0", steps=30, guidance=5.0, width=1024, height=1024, ip_scale=0.5, log=print):
        import torch
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
        self.torch, self.log, self.device = torch, log, device
        self.steps, self.guidance, self.width, self.height, self.ip_scale = steps, guidance, width, height, ip_scale
        t0 = time.time()
        vae = AutoencoderKL.from_pretrained(self.VAE, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.REPO, vae=vae, torch_dtype=torch.float16, use_safetensors=True).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        self.pipe.set_progress_bar_config(disable=True)
        self._compel, self._ipa = None, False
        log(f"  [SDXLGen] loaded in {time.time() - t0:.0f}s")

    def _embeds(self, prompt: str):
        """compel concatenates chunks beyond 77 tokens (the pooled embedding still comes from the first chunk)."""
        try:
            from compel import Compel, ReturnedEmbeddingsType
        except ImportError:
            return {"prompt": prompt}
        if self._compel is None:
            self._compel = Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2], text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                                  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                                  requires_pooled=[False, True], truncate_long_prompts=False)
        esc = prompt.replace("(", "\\(").replace(")", "\\)")
        cond, pooled = self._compel(esc); ncond, npooled = self._compel("")
        cond, ncond = self._compel.pad_conditioning_tensors_to_same_length([cond, ncond])
        return {"prompt_embeds": cond, "pooled_prompt_embeds": pooled, "negative_prompt_embeds": ncond, "negative_pooled_prompt_embeds": npooled}

    def _attach_ipa(self):
        if not self._ipa:
            self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                                      image_encoder_folder="models/image_encoder")
            enc = getattr(self.pipe, "image_encoder", None)
            if enc is not None:
                enc.to(self.device, dtype=self.torch.float16)
            self._ipa = True

    def generate(self, prompt: str, seed: int, out_path: str | Path, refs: list[str] | None = None) -> str:
        from PIL import Image
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        kw = dict(num_inference_steps=self.steps, guidance_scale=self.guidance, width=self.width, height=self.height,
                  generator=self.torch.Generator("cpu").manual_seed(int(seed)), **self._embeds(prompt))
        if refs:
            self._attach_ipa(); self.pipe.set_ip_adapter_scale(self.ip_scale)
            kw["ip_adapter_image"] = [[Image.open(p).convert("RGB") for p in refs[:2]]]
        elif self._ipa:
            self.pipe.unload_ip_adapter(); self._ipa = False
        t0 = time.time(); self.pipe(**kw).images[0].save(out_path)
        self.log(f"  [SDXLGen] {'ref×' + str(len(refs[:2])) + ' ' if refs else ''}seed {seed} -> {out_path.name} ({time.time() - t0:.0f}s)")
        return str(out_path)


class FluxGen:
    REPO, IPA_REPO, IPA_FILE, IPA_ENCODER = "black-forest-labs/FLUX.1-dev", "XLabs-AI/flux-ip-adapter", "ip_adapter.safetensors", "openai/clip-vit-large-patch14"

    def __init__(self, device="cuda:0", offload=True, steps=28, guidance=3.5, width=1024, height=1024, ip_scale=0.6, log=print):
        import torch
        from diffusers import FluxPipeline
        self.torch, self.log, self.device, self.offload = torch, log, device, offload
        self.steps, self.guidance, self.width, self.height, self.ip_scale = steps, guidance, width, height, ip_scale
        t0 = time.time()
        self.pipe = FluxPipeline.from_pretrained(self.REPO, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device=device) if offload else self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True); self._ipa = False
        log(f"  [FluxGen] loaded bf16 {'(cpu offload)' if offload else device} in {time.time() - t0:.0f}s")

    def _attach_ipa(self):
        if self._ipa: return
        self.pipe.load_ip_adapter(self.IPA_REPO, weight_name=self.IPA_FILE, image_encoder_pretrained_model_name_or_path=self.IPA_ENCODER)
        enc = getattr(self.pipe, "image_encoder", None)
        if enc is not None: enc.to(dtype=self.torch.bfloat16)
        if self.offload: self.pipe.enable_model_cpu_offload(device=self.device)   # re-hook the newly added encoder
        elif enc is not None: enc.to(self.device)
        self._ipa = True

    def generate(self, prompt: str, seed: int, out_path: str | Path, refs: list[str] | None = None) -> str:
        from PIL import Image
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        kw = dict(prompt=prompt, num_inference_steps=self.steps, guidance_scale=self.guidance, width=self.width, height=self.height,
                  max_sequence_length=512, generator=self.torch.Generator("cpu").manual_seed(int(seed)))
        if refs:
            self._attach_ipa(); self.pipe.set_ip_adapter_scale(self.ip_scale)
            kw["ip_adapter_image"] = Image.open(refs[0]).convert("RGB").resize((self.width, self.height))
        elif self._ipa:   # a loaded adapter demands ip_adapter_image; unload instead of scale 0
            self.pipe.unload_ip_adapter(); self._ipa = False
            if self.offload: self.pipe.enable_model_cpu_offload(device=self.device)
        t0 = time.time(); self.pipe(**kw).images[0].save(out_path)
        self.log(f"  [FluxGen] {'ref ' if refs else ''}seed {seed} -> {out_path.name} ({time.time() - t0:.0f}s)")
        return str(out_path)
