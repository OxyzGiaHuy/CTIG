#!/bin/bash
# Tải đúng những file mà mã thật sự nạp, không tải thừa.
#
# Bài học 2026-09-17: bản đầu gọi snapshot_download() với allow_patterns=None cho 8/9 kho,
# kéo về cache 170 GB thay vì ~67 GB — cả fp32 lẫn fp16, cả .bin lẫn .safetensors, cả ONNX,
# OpenVINO và checkpoint single-file. Riêng SDXL 58 GB, FLUX 54 GB, RealVis 26 GB.
#
# Mã nạp model qua:
#   ctig/models/loader.py:53,59   use_safetensors=True, variant=spec.variant ("fp16" mặc định)
#   ctig/stages/generation.py:265,269  variant="fp16", use_safetensors=True
#   ctig/models/registry.py:125   flux_dev variant=None (FLUX chỉ có một bản, sharded)
# Không có chỗ nào gọi from_single_file, nên mọi checkpoint single-file đều thừa.
# VAE của SDXL/RealVis bị madebyollin/sdxl-vae-fp16-fix ghi đè (registry.py:15) nhưng vẫn giữ
# bản fp16 trong kho vì diffusers kiểm đủ file theo variant trước khi coi là đã cache.

set -uo pipefail
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
export HF_TOKEN=$(grep "^export HF_TOKEN" ~/.bashrc | cut -d= -f2)

/venv/main/bin/python - <<'PY'
import time
from huggingface_hub import snapshot_download

# Khung diffusers: file mô tả + trọng số theo đúng variant.
SDXL = ["model_index.json", "*/config.json", "*/*.fp16.safetensors",
        "scheduler/*", "tokenizer/*", "tokenizer_2/*"]

JOBS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", SDXL),
    ("SG161222/RealVisXL_V4.0", SDXL),
    # FLUX chỉ có một bản trọng số, nằm trong các thư mục con dạng diffusers. Mẫu "*/*" ở đây
    # cố tình KHÔNG khớp flux1-dev.safetensors và ae.safetensors ở thư mục gốc — đó là định dạng
    # BFL cho ComfyUI, FluxPipeline không đụng tới, mà nặng 22,5 GB.
    ("black-forest-labs/FLUX.1-dev",
     ["model_index.json", "*/config.json", "*/*.safetensors", "*/*.index.json",
      "scheduler/*", "tokenizer/*", "tokenizer_2/*"]),
    ("madebyollin/sdxl-vae-fp16-fix", ["config.json", "diffusion_pytorch_model.safetensors"]),
    ("Qwen/Qwen2.5-VL-7B-Instruct", ["*.json", "*.txt", "*.safetensors"]),
    # Bộ chấm của vòng sửa (nhánh model mở của T2I-Copilot). Kho có cả consolidated.safetensors 44,7 GB
    # định dạng mistral-inference mà transformers KHÔNG đọc -> chỉ lấy bản chia mảnh, tiết kiệm đúng nửa kho.
    ("mistralai/Mistral-Small-3.1-24B-Instruct-2503",
     ["*.json", "model-*-of-*.safetensors", "tokenizer*", "*.txt"]),
    # Kho này KHÔNG có bản safetensors, bắt buộc lấy .bin.
    ("openai/clip-vit-base-patch32", ["*.json", "*.txt", "*.bin"]),
    ("h94/IP-Adapter", ["sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
                        "sdxl_models/ip-adapter_sdxl.bin",
                        "sdxl_models/image_encoder/config.json",
                        "sdxl_models/image_encoder/model.safetensors",
                        "models/image_encoder/config.json",
                        "models/image_encoder/model.safetensors"]),
    ("facebook/sam-vit-base", ["*.json", "model.safetensors"]),
    ("google/owlvit-base-patch32", ["*.json", "*.txt", "model.safetensors"]),
]

# Lưới an toàn phòng khi allow_patterns lỡ rộng tay: *.onnx_data và openvino_model.bin mới là
# thứ nặng, còn *.onnx chỉ là phần mô tả đồ thị vài MB.
IGNORE = ["*.onnx", "*.onnx_data", "*openvino*", "*.msgpack", "*.h5", "*.ckpt", "*.pth"]

for repo, pats in JOBS:
    t = time.time()
    try:
        snapshot_download(repo, allow_patterns=pats, ignore_patterns=IGNORE, max_workers=8)
        print(f"OK   {repo}  {time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print(f"HỎNG {repo}  {type(e).__name__}: {str(e)[:110]}", flush=True)
PY

echo "DL_DONE $(date +%T)"
du -sh "$HF_HOME"
echo "--- theo từng kho (kỳ vọng tổng ~67 GB) ---"
du -sh "$HF_HOME"/hub/models--* | sort -h
