# Chạy trên Kaggle

## Chuẩn bị

1. Push repo lên GitHub (public, hoặc private kèm token).
2. Tạo notebook mới trên Kaggle, hoặc upload `notebooks/kaggle_run.ipynb`.
3. Settings bên phải: **Accelerator = GPU T4 x2** (hoặc T4 / P100), **Internet = On**.
   Internet cần để tải model HF, gọi Wikipedia và Commons.

## Chạy

Notebook có sẵn các cell theo thứ tự: clone → pip install → smoke 2 prompt → xem ảnh → chạy đủ → nén kết quả.

Dòng lệnh tương đương:

```bash
git clone https://github.com/OxyzGiaHuy/CTIG.git && cd CTIG
pip install -q -r requirements.txt
python -m ctig.cli batch --config configs/kaggle_t4x2.yaml --limit 2 --run-name smoke
python -m ctig.cli batch --config configs/kaggle_t4x2.yaml --run-name v1-full
```

Dùng `configs/kaggle_t4.yaml` nếu chỉ có một GPU.

## v1.3: tối ưu ảnh cuối, thời gian và VRAM

Config `configs/kaggle_walkthrough_t4x2.yaml` mặc định 7 hàng × 4 ứng viên, hires fix bật: **~35–45 phút một prompt** sau khi
model đã tải (lần đầu tải thêm RealVisXL 7 GB, PickScore 3,9 GB, IP-Adapter Plus 1 GB + ViT-H 2,5 GB). Muốn nhanh: bỏ
`realvis_aodai`/`playground25` khỏi `models:`, hoặc `hires.enabled: false` (−40%), hoặc `n_candidates: 2`.

| việc | VRAM đỉnh ước lượng (GPU sinh ảnh) | ghi chú |
|---|---|---|
| SDXL 1024, 30 bước, không offload | ~7 GB | như v1.2 |
| + hires ×1.5 (img2img 1536px) | ~10–12 GB | OOM → code giữ ảnh gốc và tắt hires cho các ứng viên còn lại của hàng đó |
| `sdxl_refplus` (IP-Adapter Plus + ViT-H) | ~9–10 GB, +hires có thể chạm 14 GB | nếu OOM: `hires.enabled: false` hoặc `ref_images: 1` |
| PickScore | ~2 GB trên GPU 0 chỉ lúc chấm | offload CPU, không đụng GPU sinh ảnh |

Trên 1×T4 dùng `configs/kaggle_walkthrough.yaml`: offload, 3 ứng viên, hires tắt, `ref_images: 2`.

Sweep LoRA scale: thêm vào `models:` các khoá `sdxl_aodai@0.6, sdxl_aodai@0.8, sdxl_aodai@1.0` (mỗi khoá một hàng, cùng seed).

`sd35_medium` (SD3.5 Medium) là repo gated: vào trang model trên Hugging Face bấm chấp nhận điều khoản, tạo token Read,
đặt Secret `HF_TOKEN`. T4 không có bf16 nên chạy fp16; ra ảnh nhiễu/đen là do giới hạn số, không phải lỗi code.

## v1.4: agent loop review

Cell Bước 2c (Summary) tốn 1 lần gọi Qwen mỗi thực thể. Cell Bước 4c–4d: mô tả `k_candidates` (8) ảnh, mỗi ảnh ~5–8 s trên T4,
cộng một lần gọi so văn bản mỗi ảnh và một lần xếp hạng; nếu có vòng sửa thì thêm một lần sinh 4 ảnh trên model tốt nhất
(~4 phút với SDXL 1024 + hires). Trên 1×T4 Qwen được nạp lại sau bước 4 (~1 phút). Tắt bằng `agents.candidate_review: false`.

| hiện tượng | ý nghĩa |
|---|---|
| `[filter:candidate] giữ 8/8` và mọi ảnh "0/4 must_have thấy" | mô tả VLM quá chung, không nhắc collar/trousers → so văn bản không khớp; xem "mô tả VLM" trong bảng, cân nhắc model VLM lớn hơn |
| `[rank] top-1 KHÁC` | agent và metric bất đồng; xem lý do agent trong bảng, đây là dữ liệu cho H14 |
| `kế hoạch sửa rỗng` | thuộc tính thiếu đã có trong prompt và không có must_not → không sinh lại |

## v1.4.2: hàng +ref và cắt ảnh tham chiếu

Config 2×T4 mặc định 10 hàng × 4 ứng viên (~55–65 phút). Log `[3c] ảnh tham chiếu: cắt k/n về vùng '...'` cho biết bao nhiêu
ảnh được cắt; ảnh cắt nằm ở `runs/_cache/ref_crops/`. Cột "giống ref" > 0,88 (đỏ) là ảnh sinh chép ảnh tham chiếu.
Muốn nhanh: bỏ `sdxl_base`, `realvis_xl#legacy` hoặc `playground25`.

## Đổi backbone CLIP (v1.5.1)

`perception.clip_model` nhận CLIP hoặc SigLIP: `openai/clip-vit-base-patch32` (mặc định, 0,6 GB), `openai/clip-vit-large-patch14`
(1,7 GB, phân biệt chi tiết tốt hơn), `google/siglip-so400m-patch14-384` (3,5 GB fp16 ~1,8 GB, mạnh nhất zero-shot). Đổi rồi
chạy lại cùng prompt để so cột CLIP attr; các bước 1-3 vẫn lấy từ cache, chỉ ảnh Search và điểm chấm lại. CultureCLIP (COLM 2025)
chưa công bố trọng số; muốn dùng phải tự fine-tune theo công thức của họ với `confusable_with` trong KB làm cặp twin.

## v1.6: kho ảnh tham chiếu của nhóm (ImageRAG-style)

1. Upload `ref_images.zip` + `ref_images_complex.zip` (Drive/Data) thành một Dataset, Add Input. Kaggle tự giải nén thành
   `/kaggle/input/datasets/<user>/<ten>/evidence_images/` và `evidence_images_complex/`.
2. Đánh chỉ mục một lần (~2 phút cho 1.400 ảnh trên T4), lưu vào cache để đi theo `runs_cache.zip`:
   ```
   !cd /kaggle/working/CTIG && python -m ctig.stages.refindex build --root /kaggle/input/datasets/<user>/<ten> --out /kaggle/working/runs/_cache/ref_index.npz
   ```
3. Trong cell chọn prompt: `cfg.retrieval.ref_index = "/kaggle/working/runs/_cache/ref_index.npz"`. Log `[session] kho ảnh tham chiếu: N ảnh`.
   Từ đó ảnh tham chiếu lấy từ kho (tầng 0, log `[3b] ... tầng 0 (kho ...)`), web chỉ dùng khi kho không có; vòng sửa 4d truy hồi
   theo caption thuộc tính thiếu (`[4d] ảnh theo caption thuộc tính (kho|web): ...`).
4. Kho phải đánh chỉ mục bằng đúng `perception.clip_model` đang dùng; khác thì Session bỏ kho và báo.

Sweep scale IP-Adapter theo hàng: `multigen.overrides: {"realvis_xl+ref": {ip_scale: 0.5}}` (thêm hàng cùng khoá không được; dùng
hai config hoặc hai lần chạy).

## Ước lượng thời gian và dung lượng

| | 1×T4 (offload) | 2×T4 |
|---|---|---|
| tải model lần đầu | ~10 phút (SDXL 7 GB, Qwen 7.5 GB, IP-Adapter 0.7 GB, CLIP 0.6 GB) | như nhau |
| một prompt, một vòng, 2 ứng viên 768px | ~2–3 phút | ~1–1.5 phút (1024px) |
| 50 prompt, trung bình 2.5 vòng | ~5–6 giờ | ~2.5–3 giờ |

Phiên Kaggle tối đa 12 giờ, quota GPU 30 giờ/tuần. Pipeline ghi `summary.json`, `report.html`,
`user_study.csv` **sau mỗi prompt**, nên phiên bị ngắt vẫn giữ được kết quả đã chạy.

Kết quả nằm trong `/kaggle/working/runs/<run_name>/`. Kaggle chỉ cho tải từng file, nên gom thành
hai file tự chứa:

```bash
python -m ctig.bundle /kaggle/working/runs/<run_name>
```

ra `bundle.html` (mở bằng trình duyệt, ảnh đã nhúng, ~1–5 MB cho 50 prompt) và `bundle.json`
(summary, records, bằng chứng đã rút, review chi tiết của 8 prompt mẫu kèm ảnh thu nhỏ). Hai file này
đủ để người khác đánh giá pipeline. Cần ảnh gốc thì **Save Version** rồi tải zip từ tab Output của version.

## Sự cố thường gặp

| Triệu chứng | Nguyên nhân | Cách xử |
|---|---|---|
| `CUDA out of memory` khi khởi tạo | 1 GPU mà `cpu_offload: false` | dùng `kaggle_t4.yaml` hoặc `--set t2i.cpu_offload=true` |
| OOM giữa vòng review | ảnh 1024 + VLM cùng GPU | `--set t2i.width=768 --set t2i.height=768` |
| Ảnh đen hoàn toàn | VAE fp16 tràn số | đã dùng `madebyollin/sdxl-vae-fp16-fix`; kiểm `t2i.vae` không bị đặt null |
| `Không lấy được JSON sau 3 lần` | VLM 3B trả lời lan man | giảm `llm.temperature` về 0, tăng `llm.max_new_tokens`; hoặc đổi `kaggle_claude.yaml` |
| `retrieval_errors` không rỗng | Internet tắt hoặc Wikipedia chặn | bật Internet; pipeline vẫn chạy với KB offline |
| Qwen2.5-VL lỗi import | transformers cũ | `pip install -U "transformers>=4.51"` rồi restart kernel |
| NaN / rác từ Qwen fp16 | T4 không có bf16 gốc | `--set llm.model=Qwen/Qwen2-VL-2B-Instruct` |
| `sdxl_aodai: LoraError ... peft:` | thiếu gói `peft` (đã thêm vào requirements) | `pip install -U peft` rồi chạy lại; code tự lùi về `fuse_lora` nếu peft vẫn lỗi |
| `LoraError ... Found an incompatible version of torchao. Found version 0.10.0` | ảnh Kaggle có sẵn torchao 0.10, transformers mới đòi ≥ 0.16 nên peft import chết | `pip uninstall -y torchao` (cell cài đặt đã làm) rồi restart kernel |
| hàng ghi `prompt N token > 75, pipeline sẽ cắt` hoặc `thiếu gói compel` | prompt dài hơn 77 token CLIP | `pip install compel` (đã trong requirements) và `multigen.long_prompt: true` |
| hàng ghi `hires bỏ qua (OutOfMemoryError ...)` | img2img 1536px không vừa VRAM | ảnh gốc vẫn có; `hires.scale: 1.25` hoặc tắt hires |
| `[aesthetic] không nạp được PickScore` | mạng / VRAM | các cột khác vẫn có; `multigen.aesthetic.enabled: false` để tắt hẳn |
| `sdxl_refplus: bỏ qua: spec không có ảnh tham chiếu` | không ảnh Commons nào đạt `ref_image_min_clip` 0.75 | hạ ngưỡng xuống 0.7 hoặc chấp nhận hàng bị bỏ |
| `OutOfMemoryError` khi `[session] nạp agent qwen_vl` ở bước 4c | chạy lại cell chọn prompt mà không restart kernel: Session cũ vẫn giữ Qwen trên GPU 0 | restart kernel rồi chạy lại từ đầu (mọi bước lấy từ đĩa, ~2 phút tới bước 4c); từ v1.5.2 Session tự nạp VLM lên GPU còn trống (GPU 1 sau bước 4) |
| `playground25` treo/OOM sau cảnh báo `upcast_vae` | model ép VAE fp32 khi giải mã 1024px | registry đã dùng VAE fp16-fix cho hàng này; hoặc giảm `multigen.max_side` xuống 768 |
| `sdxl_ref: bỏ qua: spec không có ảnh tham chiếu` | không ảnh Commons nào đạt CLIP ≥ 0.75 cho thực thể vật thể | hạ `retrieval.ref_image_min_clip` hoặc bỏ hàng này |
| `[judge] không tải được BLIP-2 ITM` | transformers thiếu `Blip2ForImageTextRetrieval` hoặc hết VRAM | pipeline tự lùi về CLIP; hoặc `--set judge.backend=clip`; nâng transformers |
| `ddg text ...: RatelimitException` | DuckDuckGo giới hạn tần suất | pipeline bỏ qua web, vẫn có Wikipedia; chạy lại sau hoặc `--set retrieval.web_api=none` |
| checklist toàn `unsure` | VLM không thấy rõ, hoặc thuộc tính quá dài | xem `stage45_review.json → perception.checklist.note`; giảm `max_spec_entities` |

## Notebook v1.2: xem từng bước

`notebooks/ctig_walkthrough.ipynb` (config `configs/kaggle_walkthrough.yaml` cho 1×T4, `_t4x2.yaml` cho 2×T4): mỗi cell hiện
đầu ra một bước (keywords → search hai cột → bằng chứng → spec/GenSpec → grid nhiều model → bảng điểm). Badge trên mỗi bảng cho
biết kết quả lấy từ bộ nhớ, đĩa hay chạy mới; chạy lại cell không đổi gì thì không tốn API/model (cache gọi LLM ở
`runs/_cache/llm`, cache web ở `runs/_cache/web`, ảnh multigen dùng lại theo hash GenSpec).

## Key (tất cả tuỳ chọn)

| Nguồn | Biến | Lấy ở đâu |
|---|---|---|
| DuckDuckGo text + ảnh (mặc định) | không cần | — |
| LoRA áo dài Civitai (hàng `sdxl_aodai`) | `CIVITAI_TOKEN` | civitai.com → ảnh đại diện → Account settings → API Keys → Add API key |
| Serper (Google web + Google Images, 2.500 truy vấn trial) | `SERPER_API_KEY` | serper.dev → Sign up → Dashboard → API Key; rồi `--set retrieval.web_api=serper` |
| Model gated trên Hugging Face (`sd3_medium`) | `HF_TOKEN` | huggingface.co → Settings → Access Tokens; bấm chấp nhận điều khoản trên trang model |
| Claude API làm agent | `ANTHROPIC_API_KEY` | console.anthropic.com |

Trên Kaggle: Add-ons → Secrets → thêm tên và giá trị; notebook có cell đọc secret vào `os.environ`. Không có token
Civitai thì hàng `sdxl_aodai` bị bỏ qua với một dòng log rõ, các hàng khác vẫn chạy.

## Web search

Mặc định DuckDuckGo (`ddgs`, có trong requirements), không cần key, truy vấn tiếng Việt rồi tiếng Anh.
Tuỳ chọn Serper: Add-ons → Secrets → `SERPER_API_KEY`, rồi `--set retrieval.web_api=serper`.

## Ước lượng VRAM v1.1 trên một T4 16 GB

Qwen2.5-VL-3B fp16 ~7 GB, SDXL fp16 ~7 GB (cpu_offload nên chỉ giữ một phần trên GPU), CLIP ViT-B/32 0,6 GB,
BLIP-2 ITM ViT-g ~2,5 GB. Chật. Nếu OOM: `--set judge.backend=clip` trước, rồi giảm ảnh về 768.
Trên 2×T4 dùng `kaggle_t4x2.yaml`: Qwen ở GPU 0, mọi thứ khác ở GPU 1.

## Chạy lại nhanh

Stage 1–3 được cache theo prompt. Đổi tham số sinh ảnh (`t2i.*`, `review.*`) rồi chạy lại thì bỏ qua
analysis/search, chỉ sinh và review. Đổi model agent hoặc cấu hình truy hồi thì cache tự vô hiệu.
`--refresh` để ép chạy lại toàn bộ.

## Dùng Claude API thay Qwen

Add-ons → Secrets → thêm `ANTHROPIC_API_KEY`, rồi trong notebook:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["ANTHROPIC_API_KEY"] = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
```

và chạy với `configs/kaggle_claude.yaml` (`pip install anthropic` trước). Khi đó VLM local không được nạp,
SDXL có thể chạy 1024px không offload trên một T4.

## Khôi phục cache từ Dataset

**Ảnh đã sinh cũng phải nằm trong zip.** Kaggle xoá `/kaggle/working` giữa hai phiên; `multigen` chỉ dùng lại ảnh khi `runs/walkthrough/<pid>/multigen.json` và các PNG còn đó. Cell export từ v1.3.1 zip cả `runs/walkthrough`; upload zip đó thành version mới của Dataset. Chỉ zip `_cache` thì lần sau bước 4 sinh lại toàn bộ (v1.3 → v1.3.1 p001: 25 phút thay vì ~8).

Kaggle tự giải nén zip khi upload thành Dataset; trong dataset là thư mục `runs/_cache/...`. Cell khôi phục trong `ctig_walkthrough.ipynb` xử cả zip lẫn thư mục.
