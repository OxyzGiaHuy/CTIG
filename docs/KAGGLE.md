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
git clone https://github.com/<user>/CTIG.git && cd CTIG
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

Kết quả nằm trong `/kaggle/working/runs/<run_name>/`. Nén rồi tải về:

```bash
zip -qr /kaggle/working/ctig_runs.zip /kaggle/working/runs -x "*/_cache/*"
```

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

## Web search API (tuỳ chọn)

Add-ons → Secrets → `SERPER_API_KEY` (serper.dev, 2.500 truy vấn/tháng miễn phí), rồi
`--set retrieval.web_api=serper`. Một lần chạy 50 prompt tốn khoảng 200–300 truy vấn. Không có key thì
pipeline dùng Wikipedia + Commons, vẫn chạy đủ.

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
