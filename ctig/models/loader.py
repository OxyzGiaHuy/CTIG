"""
Nạp / giải phóng pipeline diffusers theo ModelSpec, cho một GPU T4 16 GB.

Quy tắc: một model trên GPU tại một thời điểm. `unload()` phải trả VRAM về mức nền; multigen ghi
`peak_vram_gb` từng model để bạn kiểm điều đó trong báo cáo.
"""

from __future__ import annotations

import gc
from pathlib import Path

from .registry import ModelSpec


SCHEDULERS = {
    # DPM++ 2M Karras: mặc định của phần lớn checkpoint SDXL thực ảnh; hội tụ ở 25-35 bước.
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"}),
    "dpmpp_2m": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++"}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
}


def set_scheduler(pipe, name: str | None, log=print) -> str | None:
    """Đổi scheduler theo tên trong SCHEDULERS. None/"keep" = giữ của repo. Lỗi thì giữ nguyên và báo."""
    if not name or name == "keep":
        return None
    if name not in SCHEDULERS:
        log(f"[loader] scheduler '{name}' không biết, giữ {type(pipe.scheduler).__name__}")
        return None
    cls_name, kwargs = SCHEDULERS[name]
    try:
        import diffusers

        cls = getattr(diffusers, cls_name)
        pipe.scheduler = cls.from_config(pipe.scheduler.config, **kwargs)
        return name
    except Exception as exc:  # noqa: BLE001
        log(f"[loader] không đổi được scheduler sang {name}: {type(exc).__name__}: {exc}")
        return None


#: Pipeline giữ thường trú trên GPU giữa các prompt, khoá theo (repo, family, dtype-ish).
#: Chỉ dùng cho pipeline KHÔNG trạng thái: không LoRA, không IP-Adapter, không scheduler lạ — nghĩa là đúng
#: các hàng '#bare'. Pipeline có LoRA hay IP-Adapter mang trạng thái dính vào trọng số, dùng lại sẽ rò sang
#: prompt sau, nên tuyệt đối không cache.
_RESIDENT: dict[str, object] = {}


def resident_key(spec: ModelSpec, device: str) -> str | None:
    """Khoá cache, hoặc None nếu pipeline này KHÔNG được phép giữ lại."""
    if spec.lora or spec.ip_adapter or getattr(spec, "init_image", False):
        return None
    return f"{spec.repo}|{spec.family}|{device}|{spec.vae}|{spec.scheduler}"


def resident_clear(log=None) -> int:
    """Trả hết pipeline thường trú. Gọi trước khi nạp model nặng khác (Mistral 48 GB, FLUX+ref 52 GB)."""
    n = len(_RESIDENT)
    for pipe in list(_RESIDENT.values()):
        _RESIDENT.clear()
        unload(pipe, log=log)
    _RESIDENT.clear()
    free_vram()
    if n and log:
        log(f"[loader] trả {n} pipeline thường trú")
    return n


def load_pipeline(spec: ModelSpec, device: str = "cuda:0", cpu_offload: bool = True, log=print, scheduler: str | None = None,
                  keep_loaded: int = 0):
    import torch
    from diffusers import AutoPipelineForText2Image

    rkey = resident_key(spec, device) if keep_loaded > 0 else None
    if rkey and rkey in _RESIDENT:
        log(f"[loader] dùng lại {spec.key} đang thường trú trên GPU")
        return _RESIDENT[rkey]

    # Họ DiT (sd3, flux) chạy bf16 khi GPU hỗ trợ (A100/H100): fp16 tràn số ra ảnh lỗi; SDXL/SD1.5 giữ fp16 (T4 không có bf16).
    dtype = torch.float16
    if spec.family in ("sd3", "flux") and device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    kwargs = dict(torch_dtype=dtype, use_safetensors=True, **spec.load_kwargs)
    if spec.vae:
        from diffusers import AutoencoderKL

        kwargs["vae"] = AutoencoderKL.from_pretrained(spec.vae, torch_dtype=dtype)
    try:
        if spec.variant:
            pipe = AutoPipelineForText2Image.from_pretrained(spec.repo, variant=spec.variant, **kwargs)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(spec.repo, **kwargs)
    except (OSError, ValueError) as exc:
        if spec.variant:
            log(f"[loader] {spec.key}: không có variant {spec.variant} ({type(exc).__name__}), nạp bản mặc định")
            pipe = AutoPipelineForText2Image.from_pretrained(spec.repo, **kwargs)
        else:
            raise
    if cpu_offload and device.startswith("cuda"):
        try:
            pipe.enable_model_cpu_offload(device=device)
        except TypeError:
            pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    for fn in ("enable_vae_slicing", "enable_vae_tiling"):
        try:
            getattr(pipe, fn)()
        except Exception:  # noqa: BLE001
            pass
    # Scheduler: model ghi "keep" thì giữ; None thì theo config multigen.scheduler.
    want = spec.scheduler if spec.scheduler is not None else scheduler
    if spec.family in ("sdxl", "sd15") and want and want != "keep":
        set_scheduler(pipe, want, log=log)
    if rkey:
        while len(_RESIDENT) >= keep_loaded:      # hết chỗ thì trả cái cũ nhất
            old_key = next(iter(_RESIDENT))
            old_pipe = _RESIDENT.pop(old_key)
            unload(old_pipe, log=log)
        _RESIDENT[rkey] = pipe
        log(f"[loader] giữ {spec.key} thường trú ({len(_RESIDENT)}/{keep_loaded} chỗ)")
    return pipe


def is_resident(pipe) -> bool:
    return any(p is pipe for p in _RESIDENT.values())


def img2img_from(pipe):
    """Pipeline img2img dùng CHUNG trọng số với pipe text2img (không tốn thêm VRAM) cho hires fix."""
    from diffusers import AutoPipelineForImage2Image

    p2 = AutoPipelineForImage2Image.from_pipe(pipe)
    p2.set_progress_bar_config(disable=True)
    return p2


def load_ip_adapter(pipe, kind: str, scale: float, log=print) -> str:
    """Gắn IP-Adapter SDXL. kind="base": ip-adapter_sdxl.bin (encoder ViT-bigG đi kèm sdxl_models);
    kind="plus": ip-adapter-plus_sdxl_vit-h (encoder ViT-H nằm ở models/image_encoder, PHẢI chỉ rõ)."""
    if kind == "flux":
        # XLabs IP-Adapter cho FLUX.1-dev (diffusers >= 0.32, FluxIPAdapterMixin); encoder CLIP ViT-L/14 nạp riêng.
        pipe.load_ip_adapter("XLabs-AI/flux-ip-adapter", weight_name="ip_adapter.safetensors",
                             image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")
        name = "IP-Adapter XLabs (FLUX, ViT-L/14)"
    elif kind == "plus":
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                             image_encoder_folder="models/image_encoder")
        name = "IP-Adapter Plus (ViT-H)"
    else:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        name = "IP-Adapter (ViT-bigG)"
    pipe.set_ip_adapter_scale(scale)
    # v1.5 p001: encoder ViT-H của bản Plus nằm lại CPU sau load_ip_adapter -> "index is on cpu, tensors on cuda:1".
    enc = getattr(pipe, "image_encoder", None)
    if enc is not None:
        try:
            unet = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
            dev = next(unet.parameters()).device if unet is not None else None
            dt = next(unet.parameters()).dtype if unet is not None else None
            if dev is not None and str(dev) != "cpu":
                enc.to(dev, dtype=dt)
        except Exception as exc:  # noqa: BLE001
            log(f"[loader] không đưa image_encoder lên GPU: {type(exc).__name__}: {exc}")
    return f"{name} scale {scale}"


class LoraError(RuntimeError):
    pass


def attach_lora(pipe, lora: dict, lora_dir: Path | str, log=print, scale: float | None = None) -> str:
    """Gắn LoRA theo mô tả trong ModelSpec.lora. Trả về mô tả cách đã gắn; raise LoraError kèm lý do gốc.

    Lần chạy v1.2 đầu: file Civitai tải đúng (80 MB, 2166 tensor kohya) nhưng gắn thất bại và thông
    điệp gốc bị nuốt. Nay: (1) thử đường PEFT (adapter_name + set_adapters), (2) không có peft thì
    load_lora_weights không adapter_name rồi fuse_lora(lora_scale), (3) lỗi gì cũng ném ra nguyên văn.
    """
    src = lora.get("source")
    scale = float(lora.get("scale", 0.8)) if scale is None else float(scale)
    if src == "civitai":
        from .civitai import download_civitai

        path = download_civitai(lora["version_id"], lora_dir, lora.get("file"), log=log)
        if not path:
            raise LoraError("không tải được file từ Civitai (cần CIVITAI_TOKEN hợp lệ, xem log [civitai])")
        folder, name = str(Path(path).parent), Path(path).name
    elif src == "path":
        path = Path(lora["path"])
        if not path.exists():
            raise LoraError(f"không thấy file LoRA {path}")
        folder, name = str(path.parent), path.name
    elif src == "hf":
        folder, name = lora["repo"], lora.get("file")
    else:
        raise LoraError(f"source LoRA không rõ: {src!r}")

    errors = []
    try:
        pipe.load_lora_weights(folder, weight_name=name, adapter_name="culture")
        pipe.set_adapters(["culture"], adapter_weights=[scale])
        return f"peft adapter, scale {scale}"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"peft: {type(exc).__name__}: {str(exc)[:160]}")
        try:
            pipe.unload_lora_weights()
        except Exception:  # noqa: BLE001
            pass
    try:
        pipe.load_lora_weights(folder, weight_name=name)
        pipe.fuse_lora(lora_scale=scale)
        return f"fused, scale {scale} (không có peft)"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"fuse: {type(exc).__name__}: {str(exc)[:160]}")
    raise LoraError(" | ".join(errors))


def unload(pipe, log=None) -> list[str]:
    """Giải phóng pipeline (kể cả hook offload của accelerate) và trả VRAM. Trả tên component không gỡ được."""
    failed: list[str] = []
    try:
        if hasattr(pipe, "remove_all_hooks"):
            pipe.remove_all_hooks()
    except Exception:  # noqa: BLE001
        pass
    try:
        comps = list(getattr(pipe, "components", {}).keys())
    except Exception:  # noqa: BLE001
        comps = []
    for name in comps + ["image_encoder", "feature_extractor"]:
        mod = getattr(pipe, name, None)
        if mod is None:
            continue
        try:
            if hasattr(mod, "to"):
                mod.to("cpu")  # đưa về CPU trước: dù còn ai giữ tham chiếu, VRAM vẫn được trả
        except Exception:  # noqa: BLE001
            pass
        try:
            setattr(pipe, name, None)
        except Exception:  # noqa: BLE001
            try:
                object.__setattr__(pipe, name, None)
            except Exception:  # noqa: BLE001
                failed.append(name)
    del pipe
    free_vram()
    if failed and log:
        log(f"[loader] không gỡ được component: {failed}")
    return failed


def allocated_gb(device: str = "cuda:0") -> float | None:
    """GB tensor còn sống trên GPU (không tính cache). Dùng để phát hiện rò giữa hai hàng multigen."""
    try:
        import torch

        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            return None
        torch.cuda.synchronize(device)
        return round(torch.cuda.memory_allocated(device) / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None


def free_gb(device: str = "cuda:0") -> float | None:
    try:
        import torch

        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            return None
        free, _total = torch.cuda.mem_get_info(torch.device(device))
        return round(free / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None


def free_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


def vram_report() -> str:
    """'GPU 0: 1.2 / 14.7 GB (đỉnh 9.8)' cho từng GPU; chuỗi rỗng nếu không có CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        parts = []
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / 2**30
            peak = torch.cuda.max_memory_allocated(i) / 2**30
            total = torch.cuda.get_device_properties(i).total_memory / 2**30
            parts.append(f"GPU {i}: {used:.1f} / {total:.1f} GB (đỉnh {peak:.1f})")
        return " | ".join(parts)
    except Exception:  # noqa: BLE001
        return ""


def reset_peak(device: str) -> None:
    try:
        import torch

        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:  # noqa: BLE001
        pass


def peak_gb(device: str) -> float | None:
    try:
        import torch

        if torch.cuda.is_available() and device.startswith("cuda"):
            return round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    return None
