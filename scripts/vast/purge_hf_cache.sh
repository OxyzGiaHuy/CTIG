#!/usr/bin/env bash
# Dọn bản trùng trong cache HF. Mặc định DRY-RUN. Chạy thật: DRY=0 bash purge_hf.sh
set -uo pipefail
DRY="${DRY:-1}"
H=/workspace/.hf_home/hub
snap() { ls -d "$H/models--$1/snapshots"/*/ 2>/dev/null | head -1; }
S=$(snap "stabilityai--stable-diffusion-xl-base-1.0")
F=$(snap "black-forest-labs--FLUX.1-dev")
R=$(snap "SG161222--RealVisXL_V4.0")
I=$(snap "h94--IP-Adapter")
V=$(snap "madebyollin--sdxl-vae-fp16-fix")
M=$(snap "facebook--sam-vit-base")
O=$(snap "google--owlvit-base-patch32")
for v in S F R I V M O; do [ -n "${!v}" ] || { echo "LỖI: không tìm thấy snapshot cho \$$v"; exit 1; }; done

TOTAL=0
purge() {
  local p="$1" t sz
  if [ ! -e "$p" ] && [ ! -L "$p" ]; then echo "  bỏ qua (không có)  ${p#$H/}"; return 0; fi
  t="$(readlink -f "$p" 2>/dev/null)"; [ -n "$t" ] || t="$p"
  if [ -f "$t" ]; then sz=$(stat -c%s "$t"); else sz=0; fi
  TOTAL=$((TOTAL+sz))
  printf "  %-5s %9.1f MB  %s\n" "$([ "$DRY" = 1 ] && echo DRY || echo XOA)" \
         "$(awk -v b=$sz 'BEGIN{print b/1048576}')" "${p#$H/}"
  [ "$DRY" = 1 ] || rm -f -- "$t" "$p"
}

echo "== SDXL base 1.0 (bỏ fp32, ONNX, OpenVINO, single-file) =="
for f in \
  unet/diffusion_pytorch_model.safetensors unet/model.onnx_data \
  unet/openvino_model.bin unet/openvino_model.xml \
  text_encoder_2/model.safetensors text_encoder_2/model.onnx_data \
  text_encoder_2/openvino_model.bin text_encoder_2/openvino_model.xml \
  text_encoder/model.safetensors text_encoder/openvino_model.bin text_encoder/openvino_model.xml \
  vae/diffusion_pytorch_model.safetensors vae_1_0/diffusion_pytorch_model.safetensors \
  vae_encoder/openvino_model.bin vae_encoder/openvino_model.xml \
  vae_decoder/openvino_model.bin vae_decoder/openvino_model.xml \
  sd_xl_base_1.0.safetensors sd_xl_base_1.0_0.9vae.safetensors \
  sd_xl_offset_example-lora_1.0.safetensors 01.png comparison.png pipeline.png
do purge "$S$f"; done

echo "== FLUX.1-dev (bỏ bản single-file BFL, giữ transformer/ sharded) =="
for f in flux1-dev.safetensors ae.safetensors dev_grid.jpg; do purge "$F$f"; done

echo "== RealVisXL V4.0 (bỏ fp32, single-file) =="
for f in \
  unet/diffusion_pytorch_model.safetensors text_encoder_2/model.safetensors \
  text_encoder/model.safetensors vae/diffusion_pytorch_model.safetensors \
  RealVisXL_V4.0.safetensors
do purge "$R$f"; done

echo "== IP-Adapter (bỏ .bin trùng .safetensors) =="
for f in models/image_encoder/pytorch_model.bin sdxl_models/image_encoder/pytorch_model.bin
do purge "$I$f"; done

echo "== vae-fp16-fix / sam / owlvit (bỏ .bin và bản trùng) =="
purge "${V}diffusion_pytorch_model.bin"
purge "${V}sdxl.vae.safetensors"
purge "${V}sdxl_vae.safetensors"
purge "${M}pytorch_model.bin"
purge "${O}pytorch_model.bin"

printf "\n== %s: %.1f GB ==\n" "$([ "$DRY" = 1 ] && echo "SẼ GIẢI PHÓNG" || echo "ĐÃ GIẢI PHÓNG")" \
       "$(awk -v b=$TOTAL 'BEGIN{print b/1073741824}')"
if [ "$DRY" = 1 ]; then echo "(mới chỉ là thử — chạy thật: DRY=0 bash $0)"; else
  echo; du -sh $H/models--* | sort -h; df -h /workspace | tail -1
  echo "--- symlink gãy còn sót (phải rỗng) ---"; find $H -xtype l | head
fi
