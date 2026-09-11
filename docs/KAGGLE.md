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
