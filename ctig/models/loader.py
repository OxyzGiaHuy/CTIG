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


def attach_lora(pipe, lora: dict, lora_dir: Path | str, log=print) -> bool:
    """Gắn LoRA theo mô tả trong ModelSpec.lora. Trả False nếu không tải được (không raise)."""
    path = None
    src = lora.get("source")
    try:
        if src == "civitai":
            from .civitai import download_civitai

            path = download_civitai(lora["version_id"], lora_dir, lora.get("file"), log=log)
        elif src == "hf":
            pipe.load_lora_weights(lora["repo"], weight_name=lora.get("file"), adapter_name="culture")
            pipe.set_adapters(["culture"], adapter_weights=[float(lora.get("scale", 0.8))])
            return True
        elif src == "path":
            path = Path(lora["path"])
        if not path or not Path(path).exists():
            return False
        pipe.load_lora_weights(str(Path(path).parent), weight_name=Path(path).name, adapter_name="culture")
        pipe.set_adapters(["culture"], adapter_weights=[float(lora.get("scale", 0.8))])
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"[loader] không gắn được LoRA {lora.get('version_id') or lora.get('repo') or lora.get('path')}: "
            f"{type(exc).__name__}: {exc}")
        return False


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
