"""
Nạp / giải phóng pipeline diffusers theo ModelSpec, cho một GPU T4 16 GB.

Quy tắc: một model trên GPU tại một thời điểm. `unload()` phải trả VRAM về mức nền; multigen ghi
`peak_vram_gb` từng model để bạn kiểm điều đó trong báo cáo.
"""

from __future__ import annotations

import gc
from pathlib import Path

from .registry import ModelSpec


def load_pipeline(spec: ModelSpec, device: str = "cuda:0", cpu_offload: bool = True, log=print):
    import torch
    from diffusers import AutoPipelineForText2Image

    kwargs = dict(torch_dtype=torch.float16, use_safetensors=True, **spec.load_kwargs)
    if spec.vae:
        from diffusers import AutoencoderKL

        kwargs["vae"] = AutoencoderKL.from_pretrained(spec.vae, torch_dtype=torch.float16)
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
    return pipe


class LoraError(RuntimeError):
    pass


def attach_lora(pipe, lora: dict, lora_dir: Path | str, log=print) -> str:
    """Gắn LoRA theo mô tả trong ModelSpec.lora. Trả về mô tả cách đã gắn; raise LoraError kèm lý do gốc.

    Lần chạy v1.2 đầu: file Civitai tải đúng (80 MB, 2166 tensor kohya) nhưng gắn thất bại và thông
    điệp gốc bị nuốt. Nay: (1) thử đường PEFT (adapter_name + set_adapters), (2) không có peft thì
    load_lora_weights không adapter_name rồi fuse_lora(lora_scale), (3) lỗi gì cũng ném ra nguyên văn.
    """
    src = lora.get("source")
    scale = float(lora.get("scale", 0.8))
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


def unload(pipe) -> None:
    """Giải phóng pipeline (kể cả hook offload của accelerate) và trả VRAM."""
    try:
        if hasattr(pipe, "remove_all_hooks"):
            pipe.remove_all_hooks()
    except Exception:  # noqa: BLE001
        pass
    try:
        for name in list(getattr(pipe, "components", {}).keys()):
            try:
                setattr(pipe, name, None)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    del pipe
    free_vram()


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
